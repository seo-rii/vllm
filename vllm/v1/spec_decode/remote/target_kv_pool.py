# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sticky target-KV routing with one bounded socket driver per endpoint."""

import itertools
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.v1.spec_decode.proposal import ProposalHandle
from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.session import RemoteDraftError
from vllm.v1.spec_decode.remote.target_kv import AttentionSpec, TargetKVSnapshot
from vllm.v1.spec_decode.remote.target_kv_protocol import (
    TargetKVHello,
    snapshot_batch_wire_plan,
)
from vllm.v1.spec_decode.remote.target_kv_session import (
    TargetKVRemoteSession,
    TargetKVResult,
)

logger = init_logger(__name__)


@dataclass(eq=False)
class TargetKVPoolHandle(ProposalHandle):
    owner: object
    keys: tuple[SequenceKey, ...]
    deadline: float
    jobs: tuple[tuple[int, tuple[int, ...], Future], ...] = ()
    retained: tuple = ()
    active: bool = True


class TargetKVDraftPool:
    """Keep each sequence on one worker, including after that worker fails.

    Caller-owned GPU work must finish before dispatch: this inline pool only
    accepts CPU snapshots and hidden states. Drivers own their sockets and
    progress independently under one shared deadline. Recovered workers admit
    new identities; terminal requests never migrate or reactivate.
    """

    def __init__(
        self,
        endpoints: tuple[str, ...],
        hello: TargetKVHello,
        *,
        expected_specs: Mapping[str, AttentionSpec],
        failure_policy="error",
        request_timeout_ms=100,
        startup_timeout_ms=30000,
        clock=time.monotonic,
    ):
        if not 1 <= len(endpoints) <= 16 or len(set(endpoints)) != len(endpoints):
            raise ValueError("target-KV pool requires 1..16 distinct endpoints")
        self.hello = hello
        self.clock = clock
        self.failure_policy = failure_policy
        self.request_timeout_s = request_timeout_ms / 1000
        self.startup_timeout_s = startup_timeout_ms / 1000
        self._sessions = tuple(
            TargetKVRemoteSession(
                endpoint,
                hello,
                expected_specs=expected_specs,
                failure_policy=failure_policy,
                request_timeout_ms=request_timeout_ms,
                startup_timeout_ms=startup_timeout_ms,
                clock=clock,
            )
            for endpoint in endpoints
        )
        for i, session in enumerate(self._sessions):
            session._generations = itertools.count(i, len(endpoints))
        self.expected_specs = self._sessions[0].expected_specs
        self.budget = self._sessions[0].budget
        self._drivers = tuple(
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"draft-kv-{i}")
            for i in range(len(endpoints))
        )
        self.records: dict[SequenceKey, bool] = {}
        self._routes: dict[SequenceKey, int] = {}
        self._epochs: dict[SequenceKey, int] = {}
        self._counts = [0] * len(endpoints)
        self._next = 0
        self._owner = object()
        self._pending: TargetKVPoolHandle | None = None
        self._probes: dict[int, Future] = {}
        self._deferred_closes: set[SequenceKey] = set()
        self._last_round = -1
        self._closed = False

    def deadline(self) -> float:
        return self.clock() + self.request_timeout_s

    def _idle(self):
        if self._closed or self._pending is not None:
            raise RuntimeError("target-KV pool is closed or busy")

    def _call(self, index, method, *args, **kwargs):
        return self._drivers[index].submit(
            getattr(self._sessions[index], method), *args, **kwargs
        )

    @staticmethod
    def _finish_all(futures):
        error = None
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                if error is None:
                    error = exc
        return error

    def _sync_records(self):
        for key, index in self._routes.items():
            self.records[key] = key not in self._deferred_closes and self._sessions[
                index
            ].active(key)

    def connect(self, *, deadline=None):
        self._idle()
        deadline = (
            self.clock() + self.startup_timeout_s if deadline is None else deadline
        )
        self._reap_probes(wait=True)
        error = self._finish_all(
            [
                self._call(i, "connect", deadline=deadline)
                for i in range(len(self._sessions))
            ]
        )
        if error is not None and (
            self.failure_policy == "error"
            or all(session._conn is None for session in self._sessions)
        ):
            self._invalidate()
            raise error

    def _reap_probes(self, *, wait=False):
        for index, future in tuple(self._probes.items()):
            if not wait and not future.done():
                continue
            try:
                future.result()
            except Exception as error:
                logger.warning("Target-KV worker %d reconnect failed: %s", index, error)
            finally:
                del self._probes[index]

    def _workers(self, deadline):
        self._reap_probes()
        candidates = [
            i
            for i, session in enumerate(self._sessions)
            if i not in self._probes and session._conn is not None
        ]
        if candidates:
            for i, session in enumerate(self._sessions):
                if (
                    i not in self._probes
                    and session._conn is None
                    and session._reconnect.ready()
                ):
                    self._probes[i] = self._call(i, "connect", deadline=deadline)
            return candidates
        self._reap_probes(wait=True)
        connected = [
            i for i, session in enumerate(self._sessions) if session._conn is not None
        ]
        return connected or list(range(len(self._sessions)))

    def active(self, key: SequenceKey) -> bool:
        index = self._routes.get(key)
        return (
            index is not None
            and key not in self._deferred_closes
            and self._sessions[index].active(key)
        )

    def epoch_for(self, key: SequenceKey) -> int:
        if key not in self._epochs:
            raise ValueError("unknown target-KV pool sequence")
        return self._epochs[key]

    def open_sequences(
        self, sequence_ids: tuple[int, ...], *, deadline: float
    ) -> tuple[SequenceKey, ...]:
        self._idle()
        if (
            len(self.records) + len(sequence_ids) > self.hello.max_num_seqs
            or len(set(sequence_ids)) != len(sequence_ids)
            or any(i < 0 for i in sequence_ids)
            or any(key.sequence_id in sequence_ids for key in self.records)
        ):
            raise ValueError("invalid target-KV pool sequence IDs or capacity")
        if not sequence_ids:
            return ()
        candidates = self._workers(deadline)
        counts = self._counts.copy()
        groups: dict[int, list[int]] = {}
        for row in range(len(sequence_ids)):
            index = min(
                candidates,
                key=lambda i: (counts[i], (i - self._next) % len(self._sessions)),
            )
            groups.setdefault(index, []).append(row)
            counts[index] += 1
            self._next = (index + 1) % len(self._sessions)
        jobs = []
        try:
            for index, rows in groups.items():
                jobs.append(
                    (
                        index,
                        tuple(rows),
                        self._call(
                            index,
                            "open_sequences",
                            tuple(sequence_ids[row] for row in rows),
                            deadline=deadline,
                        ),
                    )
                )
            error = self._finish_all([job for _, _, job in jobs])
            if error is not None:
                raise error
            ordered = {}
            for index, selected_rows, job in jobs:
                for row, key in zip(selected_rows, job.result(), strict=True):
                    ordered[row] = key
                    self._routes[key] = index
                    self._epochs[key] = self._sessions[index].epoch_for(key)
                    self.records[key] = self._sessions[index].active(key)
                    self._counts[index] += 1
            self._sync_records()
            return tuple(ordered[row] for row in range(len(sequence_ids)))
        except BaseException:
            self._finish_all([job for _, _, job in jobs])
            # Strict failure may occur before a session returns its new keys.
            # Retire all identities created for this call, including those rows.
            cleanup = []
            for index in groups:
                keys = tuple(
                    key
                    for key in self._sessions[index].records
                    if key.sequence_id in sequence_ids
                )
                cleanup.append(
                    self._call(index, "close_sequences", keys, deadline=deadline)
                )
                for key in keys:
                    if key in self._routes:
                        self._counts[index] -= 1
                        self._routes.pop(key, None)
                        self._epochs.pop(key, None)
                        self.records.pop(key, None)
            self._finish_all(cleanup)
            self._sync_records()
            raise

    def dispatch_batch(
        self,
        snapshots: tuple[TargetKVSnapshot, ...],
        input_tokens: tuple[int, ...],
        hidden_states: torch.Tensor,
        *,
        k: int = 1,
        deadline: float,
    ) -> TargetKVPoolHandle:
        self._idle()
        keys = tuple(snapshot.identity.key for snapshot in snapshots)
        if (
            not 1 <= len(snapshots) <= self.hello.max_num_seqs
            or len(set(keys)) != len(keys)
            or any(key not in self._routes for key in keys)
            or k not in (0, 1)
            or any(
                s.identity.session_epoch != self.epoch_for(s.identity.key)
                or s.identity.round_id <= self._last_round
                for s in snapshots
            )
        ):
            raise ValueError("invalid target-KV pool proposal identity or budget")
        if (
            len(input_tokens) != len(snapshots)
            or any(not 0 <= t < self.hello.vocab_size for t in input_tokens)
            or hidden_states.shape != (len(snapshots), self.hello.hidden_size)
            or hidden_states.dtype != torch.bfloat16
        ):
            raise ValueError("invalid target-KV pool input tokens or hidden states")
        if hidden_states.device.type != "cpu" or any(
            layer.pages.device.type != "cpu"
            for snapshot in snapshots
            for layer in snapshot.layers
        ):
            raise ValueError(
                "target-KV pool inputs must be staged on the caller to CPU"
            )
        snapshot_batch_wire_plan(
            self.expected_specs,
            tuple(s.materialized_length for s in snapshots),
            self.budget,
        )
        self._last_round = max(s.identity.round_id for s in snapshots)
        groups: dict[int, list[int]] = {}
        for row, key in enumerate(keys):
            if self.active(key):
                groups.setdefault(self._routes[key], []).append(row)
        handle = TargetKVPoolHandle(
            self._owner, keys, deadline, retained=(snapshots, hidden_states)
        )
        self._pending = handle
        jobs = []
        try:
            for index, rows in groups.items():
                selected = tuple(snapshots[row] for row in rows)
                tokens = tuple(input_tokens[row] for row in rows)
                hidden = hidden_states[rows]

                def run(index=index, selected=selected, tokens=tokens, hidden=hidden):
                    session = self._sessions[index]
                    return session.collect_batch(
                        session.dispatch_batch(
                            selected, tokens, hidden, k=k, deadline=deadline
                        )
                    )

                jobs.append((index, tuple(rows), self._drivers[index].submit(run)))
            handle.jobs = tuple(jobs)
        except BaseException:
            handle.jobs = tuple(jobs)
            self.cancel(handle)
            raise
        return handle

    def _owned(self, handle: ProposalHandle) -> TargetKVPoolHandle:
        if (
            not isinstance(handle, TargetKVPoolHandle)
            or handle.owner is not self._owner
            or self._pending is not handle
            or not handle.active
        ):
            raise RuntimeError("stale, foreign or collected target-KV pool handle")
        return handle

    def collect_batch(self, handle: ProposalHandle) -> tuple[TargetKVResult, ...]:
        handle = self._owned(handle)
        try:
            results = [TargetKVResult() for _ in handle.keys]
            error = None
            for _, rows, future in handle.jobs:
                try:
                    for row, result in zip(rows, future.result(), strict=True):
                        results[row] = result
                except Exception as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error
            return tuple(results)
        finally:
            self._finish_all([future for _, _, future in handle.jobs])
            handle.active = False
            handle.retained = ()
            self._pending = None
            self._sync_records()
            self._flush_closes(handle.deadline)

    def _flush_closes(self, deadline):
        if self._deferred_closes:
            keys = tuple(self._deferred_closes)
            self._deferred_closes.clear()
            self.close_sequences(keys, deadline=deadline)

    def cancel(self, handle: ProposalHandle):
        handle = self._owned(handle)
        try:
            self._finish_all([future for _, _, future in handle.jobs])
            self._finish_all(
                [self._call(index, "_invalidate") for index, _, _ in handle.jobs]
            )
        finally:
            handle.active = False
            handle.retained = ()
            self._pending = None
            self._sync_records()
            self._flush_closes(handle.deadline)

    def _invalidate(self):
        try:
            if self._pending is not None:
                self.cancel(self._pending)
        finally:
            self._finish_all(
                [self._call(i, "_invalidate") for i in range(len(self._sessions))]
            )
            self._sync_records()

    def fail(self, detail: str):
        logger.warning("Target-KV draft pool failed: %s", detail)
        error = None
        try:
            self._invalidate()
        except Exception as exc:
            error = exc
            logger.warning("Target-KV draft pool retirement failed: %s", exc)
        if self.failure_policy == "error":
            raise RemoteDraftError(detail) from error

    def close_sequences(self, keys: tuple[SequenceKey, ...], *, deadline=None):
        keys = tuple(dict.fromkeys(key for key in keys if key in self._routes))
        if not keys:
            return
        if self._pending is not None:
            if not any(key in self._pending.keys for key in keys):
                self._deferred_closes.update(keys)
                self._sync_records()
                return
            self.cancel(self._pending)
            keys = tuple(key for key in keys if key in self._routes)
        deadline = self.deadline() if deadline is None else deadline
        groups: dict[int, list[SequenceKey]] = {}
        for key in keys:
            groups.setdefault(self._routes[key], []).append(key)
        try:
            error = self._finish_all(
                [
                    self._call(i, "close_sequences", tuple(selected), deadline=deadline)
                    for i, selected in groups.items()
                ]
            )
            if error is not None:
                raise error
        finally:
            for key in keys:
                index = self._routes.pop(key)
                self._counts[index] -= 1
                self._epochs.pop(key, None)
                self.records.pop(key, None)
                self._deferred_closes.discard(key)
            self._sync_records()

    def close(self):
        if self._closed:
            return
        try:
            try:
                self._invalidate()
            finally:
                error = self._finish_all(
                    [self._call(i, "close") for i in range(len(self._sessions))]
                )
            if error is not None:
                raise error
        finally:
            self._closed = True
            for driver in self._drivers:
                driver.shutdown(wait=True)
            self.records.clear()
            self._routes.clear()
            self._epochs.clear()
            self._counts = [0] * len(self._sessions)
            self._probes.clear()
            self._deferred_closes.clear()
