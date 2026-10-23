# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sticky same-host token routing with one socket driver per draft worker."""

import itertools
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

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


@dataclass(eq=False)
class TokenPoolHandle(ProposalHandle):
    owner: object
    rows: tuple[TokenAdvanceRow, ...]
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

    def active(self, key):
        index = self._routes.get(key)
        return (
            index is not None
            and key not in self._deferred_closes
            and self._sessions[index].active(key)
        )

    def open_sequence(self, sequence_id, *, deadline):
        if self._closed or self._pending is not None:
            raise RuntimeError("token pool is closed or busy")
        candidates = [
            i for i, session in enumerate(self._sessions) if session._conn is not None
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
        key = self._call(
            index, "open_sequence", sequence_id, deadline=deadline
        ).result()
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

    def prefill(self, key, tokens, *, is_final, deadline):
        if self._pending is not None:
            raise RuntimeError("cannot prefill while a pool proposal is pending")
        index = self._routes[key]
        self._call(
            index, "prefill", key, tokens, is_final=is_final, deadline=deadline
        ).result()

    def dispatch(self, batch_id, rows, k, *, deadline):
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
            if self._deferred_closes:
                keys = tuple(self._deferred_closes)
                self._deferred_closes.clear()
                self.close_sequences(keys, deadline=handle.deadline)

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

    def _invalidate(self):
        if self._pending is not None:
            self.cancel(self._pending)
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
            self._invalidate()
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
