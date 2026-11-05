# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sticky same-host token routing with one socket driver per draft worker."""

import itertools
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.spec_decode.proposal import ProposalHandle
from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.runtime.token_state import TokenAdvanceRow
from vllm.v1.spec_decode.remote.session import RemoteDraftError
from vllm.v1.spec_decode.remote.token_protocol import TokenHello
from vllm.v1.spec_decode.remote.token_session import (
    TokenBatchResult,
    TokenClientProgress,
    TokenOnlyRemoteSession,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class TokenPoolRow:
    key: SequenceKey
    open_required: bool
    prefill_tokens: tuple[int, ...] | None
    final_prefill: bool
    advance: tuple[int, int, tuple[int, ...]] | None


@dataclass(eq=False)
class TokenPoolHandle(ProposalHandle):
    owner: object
    rows: tuple[TokenAdvanceRow, ...] | tuple[TokenPoolRow, ...]
    deadline: float
    jobs: tuple[tuple[int, tuple[int, ...], Future], ...]
    active: bool = True


class TokenDraftPool:
    """Route each identity once; an endpoint failure never migrates its KV.

    Sessions are driven exclusively by their own single-worker executors. Only
    one pool proposal may be pending, which also bounds each driver's queue.
    Every endpoint uses the same client deadline, including preparation.
    """

    def __init__(
        self,
        endpoints: tuple[str, ...],
        hello: TokenHello,
        *,
        failure_policy="error",
        request_timeout_ms=100,
        startup_timeout_ms=30000,
        clock=time.monotonic,
    ):
        if not 1 <= len(endpoints) <= 16 or len(set(endpoints)) != len(endpoints):
            raise ValueError("token pool requires 1..16 distinct endpoints")
        self.hello = hello
        self.clock = clock
        self.failure_policy = failure_policy
        self.request_timeout_s = request_timeout_ms / 1000
        self.startup_timeout_s = startup_timeout_ms / 1000
        self.records: dict[SequenceKey, TokenClientProgress] = {}
        self._sessions = tuple(
            TokenOnlyRemoteSession(
                endpoint,
                hello,
                failure_policy=failure_policy,
                request_timeout_ms=request_timeout_ms,
                startup_timeout_ms=startup_timeout_ms,
                clock=clock,
            )
            for endpoint in endpoints
        )
        # Disjoint generation ranges fence a reused sequence ID even when its
        # next incarnation is assigned to a different worker.
        for index, session in enumerate(self._sessions):
            session._generations = itertools.count(index, len(endpoints))
        self._drivers = tuple(
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"draft-io-{i}")
            for i in range(len(endpoints))
        )
        self._routes: dict[SequenceKey, int] = {}
        self._counts = [0] * len(endpoints)
        self._next = 0
        self._owner = object()
        self._pending: TokenPoolHandle | None = None
        self._probes: dict[int, Future] = {}
        self._deferred_closes: set[SequenceKey] = set()
        self._closed = False
        self._last_batch = -1

    def deadline(self):
        return self.clock() + self.request_timeout_s

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

    def connect(self, *, deadline=None):
        if self._closed or self._pending is not None:
            raise RuntimeError("token pool is closed or busy")
        deadline = (
            deadline if deadline is not None else self.clock() + self.startup_timeout_s
        )
        self._reap_probes(wait=True)
        jobs = [
            self._call(i, "connect", deadline=deadline)
            for i in range(len(self._sessions))
        ]
        error = self._finish_all(jobs)
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
                logger.warning(
                    "Remote draft worker %d reconnect failed: %s", index, error
                )
            finally:
                del self._probes[index]

    def active(self, key):
        index = self._routes.get(key)
        return (
            index is not None
            and key not in self._deferred_closes
            and self._sessions[index].active(key)
        )

    def _select_worker(self, deadline):
        self._reap_probes()
        candidates = [
            i
            for i, session in enumerate(self._sessions)
            if i not in self._probes and session._conn is not None
        ]
        if candidates:
            # Rejoin workers independently of this request. A probing socket
            # becomes eligible only after its handshake future has completed.
            for i, session in enumerate(self._sessions):
                if (
                    i not in self._probes
                    and session._conn is None
                    and session._reconnect.ready()
                ):
                    self._probes[i] = self._call(i, "connect", deadline=deadline)
        elif self._probes:
            self._reap_probes(wait=True)
            candidates = [
                i
                for i, session in enumerate(self._sessions)
                if session._conn is not None
            ]
        if not candidates:
            candidates = [
                i
                for i, session in enumerate(self._sessions)
                if session._reconnect.ready()
            ]
        if not candidates:
            # Keep a terminal identity even during an outage. The session's
            # reconnect backoff prevents another full network wait.
            candidates = list(range(len(self._sessions)))
        index = min(
            candidates,
            key=lambda i: (self._counts[i], (i - self._next) % len(self._sessions)),
        )
        self._next = (index + 1) % len(self._sessions)
        return index

    def reserve_sequence(self, sequence_id, *, deadline):
        if self._closed or self._pending is not None:
            raise RuntimeError("token pool is closed or busy")
        index = self._select_worker(deadline)
        key = self._call(index, "reserve_sequence", sequence_id).result()
        try:
            self.records[key] = self._sessions[index].records[key]
            self._routes[key] = index
            self._counts[index] += 1
        except BaseException:
            self._call(index, "_invalidate").result()
            self._call(index, "close_sequences", (key,), deadline=deadline).result()
            self.records.pop(key, None)
            self._routes.pop(key, None)
            raise
        return key

    def open_sequence(self, sequence_id, *, deadline):
        key = self.reserve_sequence(sequence_id, deadline=deadline)
        try:
            self._call(
                self._routes[key], "open_reserved", key, deadline=deadline
            ).result()
        except BaseException:
            try:
                self.close_sequences((key,), deadline=deadline)
            except Exception:
                logger.exception("Failed to retire an unsuccessful pool open")
            raise
        return key

    def prefill(self, key, tokens, *, is_final, deadline):
        if self._pending is not None:
            raise RuntimeError("cannot prefill while a pool proposal is pending")
        index = self._routes[key]
        self._call(
            index, "prefill", key, tokens, is_final=is_final, deadline=deadline
        ).result()

    def dispatch(self, batch_id, rows, k, *, deadline):
        return self._dispatch(batch_id, rows, k, deadline=deadline, prepared=False)

    def dispatch_prepared(self, batch_id, rows, k, *, deadline):
        return self._dispatch(batch_id, rows, k, deadline=deadline, prepared=True)

    @staticmethod
    def _prepare_and_collect(session, batch_id, rows, k, deadline):
        advances, indices = [], []
        for i, row in enumerate(rows):
            if not session.active(row.key):
                continue
            if row.open_required:
                session.open_reserved(row.key, deadline=deadline)
            if row.prefill_tokens is not None:
                session.prefill(
                    row.key,
                    row.prefill_tokens,
                    is_final=row.final_prefill,
                    deadline=deadline,
                )
            if row.advance is not None and session.active(row.key):
                state = session.records[row.key]
                offered, accepted, delta = row.advance
                advances.append(
                    TokenAdvanceRow(
                        row.key,
                        state.committed,
                        state.proposal_id,
                        offered,
                        accepted,
                        delta,
                    )
                )
                indices.append(i)
        tokens = [(0,) * session.hello.num_speculative_tokens for _ in rows]
        lengths = [0] * len(rows)
        if advances:
            result = session.collect(
                session.dispatch(batch_id, tuple(advances), k, deadline=deadline)
            )
            for local, original in enumerate(indices):
                tokens[original] = result.token_ids[local]
                lengths[original] = result.valid_lengths[local]
        return TokenBatchResult(tuple(tokens), tuple(lengths))

    def _dispatch(self, batch_id, rows, k, *, deadline, prepared):
        if self._closed or self._pending is not None:
            raise RuntimeError("token pool is closed or busy")
        if (
            batch_id <= self._last_batch
            or not 0 <= k <= self.hello.num_speculative_tokens
            or len(set(row.key for row in rows)) != len(rows)
            or any(not self.active(row.key) for row in rows)
        ):
            raise ValueError("invalid pool proposal identity, rows, or budget")
        self._last_batch = batch_id
        groups: dict[int, list[int]] = {}
        for row_index, row in enumerate(rows):
            groups.setdefault(self._routes[row.key], []).append(row_index)
        handle = TokenPoolHandle(self._owner, rows, deadline, ())
        self._pending = handle
        jobs = []
        try:
            for index, indices in groups.items():
                selected = tuple(rows[i] for i in indices)

                def run(index=index, selected=selected):
                    session = self._sessions[index]
                    if prepared:
                        return self._prepare_and_collect(
                            session, batch_id, selected, k, deadline
                        )
                    return session.collect(
                        session.dispatch(batch_id, selected, k, deadline=deadline)
                    )

                jobs.append((index, tuple(indices), self._drivers[index].submit(run)))
            handle.jobs = tuple(jobs)
        except BaseException:
            handle.jobs = tuple(jobs)
            self.cancel(handle)
            raise
        return handle

    def _owned(self, handle):
        if (
            not isinstance(handle, TokenPoolHandle)
            or handle.owner is not self._owner
            or self._pending is not handle
            or not handle.active
        ):
            raise RuntimeError("stale, foreign or collected token pool handle")
        return handle

    def collect(self, handle):
        handle = self._owned(handle)
        capacity = self.hello.num_speculative_tokens
        try:
            tokens = [(0,) * capacity for _ in handle.rows]
            lengths = [0] * len(handle.rows)
            error = None
            # All endpoint I/O is already progressing concurrently. Drain every
            # job even if an earlier endpoint failed, leaving no hidden handle.
            for _, indices, future in handle.jobs:
                try:
                    result = future.result()
                    for local, original in enumerate(indices):
                        tokens[original] = result.token_ids[local]
                        lengths[original] = result.valid_lengths[local]
                except Exception as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error
            return TokenBatchResult(tuple(tokens), tuple(lengths))
        finally:
            self._finish_all([future for _, _, future in handle.jobs])
            handle.active = False
            self._pending = None
            self._flush_closes(handle.deadline)

    def _flush_closes(self, deadline):
        if self._deferred_closes:
            keys = tuple(self._deferred_closes)
            self._deferred_closes.clear()
            self.close_sequences(keys, deadline=deadline)

    def cancel(self, handle):
        handle = self._owned(handle)
        try:
            self._finish_all([future for _, _, future in handle.jobs])
            self._finish_all(
                [self._call(index, "_invalidate") for index, _, _ in handle.jobs]
            )
        finally:
            handle.active = False
            self._pending = None
            self._flush_closes(handle.deadline)

    def _invalidate(self):
        try:
            if self._pending is not None:
                self.cancel(self._pending)
        finally:
            self._finish_all(
                [self._call(i, "_invalidate") for i in range(len(self._sessions))]
            )

    def fail(self, detail):
        self._invalidate()
        if self.failure_policy == "error":
            raise RemoteDraftError(detail)

    def close_sequences(self, keys, *, deadline=None):
        keys = tuple(dict.fromkeys(key for key in keys if key in self._routes))
        if not keys:
            return
        if self._pending is not None:
            if not any(row.key in keys for row in self._pending.rows):
                self._deferred_closes.update(keys)
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
                self.records.pop(key, None)
                self._deferred_closes.discard(key)

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
            self._deferred_closes.clear()
            self._probes.clear()
