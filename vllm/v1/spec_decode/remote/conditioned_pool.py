# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sticky own-KV routing with independently progressing feature transports."""

import contextlib
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from itertools import chain

import torch

from vllm.logger import init_logger
from vllm.v1.spec_decode.proposal import ProposalHandle, SpeculatorOutput
from vllm.v1.spec_decode.remote.protocol import SequenceKey, SpeculatorStatusCode
from vllm.v1.spec_decode.remote.session import RemoteDraftError, RemoteDraftSession
from vllm.v1.spec_decode.remote.state import (
    FeatureBatch,
    RemoteProposalBatch,
    RemoteProposalResult,
    SequenceState,
)

logger = init_logger(__name__)


@dataclass(eq=False)
class ConditionedPoolHandle(ProposalHandle):
    owner: object
    batch: RemoteProposalBatch
    deadline: float
    jobs: tuple[tuple[tuple[int, ...], Future], ...] = ()
    retained: tuple = ()
    keys: tuple[SequenceKey, ...] = ()
    active: bool = True


class ConditionedDraftPool:
    """One driver and IPC arena per endpoint, with a shared round deadline.

    Feature partitioning runs on the producer stream. Driver streams wait on
    that producer event and retain inputs until their transport has completed.
    A failed endpoint never moves an existing sequence to a different worker.
    """

    def __init__(
        self,
        endpoints,
        identity,
        *,
        transport="auto",
        failure_policy="target_only",
        request_timeout_ms=100,
        startup_timeout_ms=30000,
        clock=time.monotonic,
    ):
        if not 1 <= len(endpoints) <= 16 or len(set(endpoints)) != len(endpoints):
            raise ValueError("conditioned pool requires 1..16 distinct endpoints")
        self._identity = identity
        self._clock = clock
        self._timeout = request_timeout_ms / 1000
        self._startup_timeout = startup_timeout_ms / 1000
        self._policy = failure_policy
        self._sessions = tuple(
            RemoteDraftSession(
                endpoint,
                replace(
                    identity,
                    verifier_instance_id=f"{identity.verifier_instance_id}:{i}",
                ),
                transport=transport,
                failure_policy=failure_policy,
                request_timeout_ms=request_timeout_ms,
                startup_timeout_ms=startup_timeout_ms,
                clock=clock,
            )
            for i, endpoint in enumerate(endpoints)
        )
        self._drivers = tuple(
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"draft-feature-{i}")
            for i in range(len(endpoints))
        )
        self._routes: dict[SequenceKey, int] = {}
        self._reserved: set[SequenceKey] = set()
        self._counts = [0] * len(endpoints)
        self._next = 0
        self._owner = object()
        self._pending: ConditionedPoolHandle | None = None
        self._deferred_closes: set[SequenceKey] = set()
        self._probes: dict[int, Future] = {}
        self._streams: tuple[torch.cuda.Stream, ...] = ()
        self._schema = None
        self._transport = ""
        self._last_batch = -1
        self._closed = False

    @property
    def feature_schema(self):
        if self._schema is None:
            raise RemoteDraftError("pool is not connected")
        return self._schema

    @property
    def selected_transport(self):
        return self._transport

    @property
    def cuda_ipc_reserved_bytes(self):
        return sum(session.cuda_ipc_reserved_bytes for session in self._sessions)

    def preparation_deadline(self):
        return self._clock() + self._timeout

    def _call(self, index, method, *args, **kwargs):
        return self._drivers[index].submit(
            getattr(self._sessions[index], method), *args, **kwargs
        )

    @staticmethod
    def _finish(futures):
        error = None
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                if error is None:
                    error = exc
        return error

    def _idle(self):
        if self._closed or self._pending is not None:
            raise RuntimeError("conditioned pool is closed or busy")

    def connect(self):
        self._idle()
        deadline = self._clock() + self._startup_timeout
        jobs = [
            self._call(i, "connect", timeout_s=max(0, deadline - self._clock()))
            for i in range(len(self._sessions))
        ]
        error = self._finish(jobs)
        connected = [session for session in self._sessions if session.connected]
        if (error is not None and self._policy == "error") or not connected:
            self.invalidate()
            raise error or RemoteDraftError("no draft pool worker connected")
        first = connected[0]
        self._schema, self._transport = first.feature_schema, first.selected_transport
        if any(
            session.feature_schema != self._schema
            or session.selected_transport != self._transport
            for session in connected
        ):
            self.invalidate()
            raise RemoteDraftError(
                "pool workers disagree on feature schema or transport"
            )
        # IPC arenas cannot be added after target memory profiling.
        if self._transport == "cuda_ipc" and len(connected) != len(self._sessions):
            self.invalidate()
            raise RemoteDraftError("all CUDA IPC pool workers must connect at startup")

    def prepare_cuda_ipc(self, device: torch.device):
        self._idle()
        if self._transport != "cuda_ipc" or self._streams:
            return
        try:
            with torch.accelerator.device_index(device.index):
                self._streams = tuple(
                    torch.cuda.Stream(device=device) for _ in self._sessions
                )
            for i, stream in enumerate(self._streams):

                def prepare(i=i, stream=stream):
                    with torch.cuda.stream(stream):
                        self._sessions[i].prepare_cuda_ipc(device)

                self._drivers[i].submit(prepare).result()
        except BaseException:
            self.invalidate()
            raise

    def _reap_probes(self):
        for i, future in tuple(self._probes.items()):
            if future.done():
                try:
                    future.result()
                    session = self._sessions[i]
                    if session.connected and session.feature_schema != self._schema:
                        self._call(i, "close").result()
                        raise RemoteDraftError(
                            "reconnected worker changed feature schema"
                        )
                finally:
                    del self._probes[i]

    def reserve_sequences(self, sequence_ids, *, deadline=None):
        self._idle()
        self._reap_probes()
        if len(set(sequence_ids)) != len(sequence_ids) or any(
            key.sequence_id in sequence_ids for key in self._routes
        ):
            raise ValueError("duplicate or already open pool sequence ids")
        deadline = self.preparation_deadline() if deadline is None else deadline
        keys = []
        try:
            for sequence_id in sequence_ids:
                candidates = [
                    i
                    for i, s in enumerate(self._sessions)
                    if i not in self._probes and s.connected
                ]
                if not candidates:
                    # Let each worker's prepared job perform its own reconnect.
                    candidates = [
                        i for i in range(len(self._sessions)) if i not in self._probes
                    ]
                if not candidates:
                    self._finish(list(self._probes.values()))
                    self._reap_probes()
                    candidates = list(range(len(self._sessions)))
                i = min(
                    candidates,
                    key=lambda i: (
                        self._counts[i],
                        (i - self._next) % len(self._sessions),
                    ),
                )
                key = self._call(i, "reserve_sequences", (sequence_id,)).result()[0]
                keys.append(key)
                self._routes[key] = i
                self._reserved.add(key)
                self._counts[i] += 1
                self._next = (i + 1) % len(self._sessions)
            if self._transport != "cuda_ipc":
                # Recovery probes never block a healthy endpoint's preparation.
                for i, session in enumerate(self._sessions):
                    if (
                        not session.connected
                        and i not in self._probes
                        and i not in {self._routes[key] for key in keys}
                    ):
                        self._probes[i] = self._call(
                            i, "maybe_reconnect", deadline=deadline
                        )
        except BaseException:
            self.invalidate()
            self.close_sequences(tuple(keys), deadline=deadline)
            raise
        return tuple(keys)

    def sequence_state(self, key):
        i = self._routes.get(key)
        if i is None:
            return None
        if key in self._deferred_closes:
            return SequenceState.CLOSING
        return self._sessions[i].sequence_state(key)

    def open_sequence(self, sequence_id):
        deadline = self.preparation_deadline()
        key = self.reserve_sequences((sequence_id,), deadline=deadline)[0]
        try:
            self._call(
                self._routes[key], "open_reserved", (key,), deadline=deadline
            ).result()
        finally:
            self._reserved.discard(key)
        return key

    def prefill(self, key, features, *, is_final=False):
        self._idle()
        self._call(
            self._routes[key], "prefill", key, features, is_final=is_final
        ).result()

    def partition(self, batch):
        """Split ragged token slots separately from per-sequence slots."""
        lengths = (
            [1] * len(batch.keys)
            if batch.query_lens is None
            else batch.query_lens.tolist()
        )
        if len(lengths) != len(batch.keys) or any(n < 0 for n in lengths):
            raise ValueError("invalid pool query lengths")
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        if len(batch.features.slots) != len(self.feature_schema.slots):
            raise ValueError("pool feature slots differ from the negotiated schema")
        if batch.features.schema_id != self.feature_schema.schema_id:
            raise ValueError("pool feature schema differs from the negotiated schema")
        for spec, tensor in zip(self.feature_schema.slots, batch.features.slots):
            expected = len(batch.keys) if spec.per_sequence else offsets[-1]
            if tensor is not None and tensor.shape != (expected, *spec.trailing_shape):
                raise ValueError("pool feature shape differs from its packed rows")
        if batch.accepted_counts.shape != (
            len(batch.keys),
        ) or batch.recovery_tokens.shape != (len(batch.keys),):
            raise ValueError("pool sequence metadata must have one value per row")
        groups: dict[int, list[int]] = {}
        for row, key in enumerate(batch.keys):
            groups.setdefault(self._routes[key], []).append(row)
        result = {}
        for i, rows in groups.items():
            token_rows = [
                t for row in rows for t in range(offsets[row], offsets[row + 1])
            ]
            slots = []
            for spec, tensor in zip(self.feature_schema.slots, batch.features.slots):
                indices = rows if spec.per_sequence else token_rows
                slots.append(
                    tensor
                    if tensor is None or indices == list(range(tensor.shape[0]))
                    else tensor[indices]
                )
            result[i] = (
                tuple(rows),
                replace(
                    batch,
                    keys=tuple(batch.keys[row] for row in rows),
                    accepted_counts=batch.accepted_counts[rows],
                    recovery_tokens=batch.recovery_tokens[rows],
                    query_lens=torch.tensor(
                        [lengths[row] for row in rows], dtype=torch.int32
                    ),
                    features=FeatureBatch(batch.features.schema_id, tuple(slots)),
                ),
            )
        return result

    @staticmethod
    def _cuda_tensors(batch, chunks):
        tensors = (tensor for _, features, _ in chunks for tensor in features.slots)
        if batch is not None:
            tensors = chain(
                tensors,
                batch.features.slots,
                (batch.accepted_counts, batch.recovery_tokens),
            )
        return (tensor for tensor in tensors if tensor is not None and tensor.is_cuda)

    def _run(self, index, batch, chunks, reserved, deadline, ready):
        session = self._sessions[index]
        stream = self._streams[index] if self._streams else None
        try:
            with (
                torch.cuda.stream(stream)
                if stream is not None
                else contextlib.nullcontext()
            ):
                for device, event in ready.items():
                    torch.cuda.current_stream(device).wait_event(event)
                for tensor in self._cuda_tensors(batch, chunks):
                    tensor.record_stream(torch.cuda.current_stream(tensor.device))
                if reserved:
                    session.open_reserved(reserved, deadline=deadline)
                session.prefill_many(chunks, deadline=deadline)
                if batch is not None:
                    return session.collect(session.dispatch(batch, deadline=deadline))
                return None
        except BaseException:
            # A preparation exception must not leave a hidden RPC or lease.
            session.close()
            raise

    def dispatch_prepared(self, batch, chunks=(), *, deadline=None):
        self._idle()
        if batch.batch_id <= self._last_batch or len(set(batch.keys)) != len(
            batch.keys
        ):
            raise ValueError("invalid pool batch identity")
        deadline = self.preparation_deadline() if deadline is None else deadline
        parts = self.partition(batch)
        grouped_chunks: dict[int, list[tuple[SequenceKey, FeatureBatch, bool]]] = {}
        for chunk in chunks:
            grouped_chunks.setdefault(self._routes[chunk[0]], []).append(chunk)
        indices = set(parts) | set(grouped_chunks)
        if any(self._routes[key] not in indices for key in self._reserved):
            raise ValueError(
                "reserved pool sequences require prefill in the same round"
            )
        ready = {}
        for tensor in self._cuda_tensors(batch, chunks):
            if tensor.device not in ready:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(tensor.device))
                ready[tensor.device] = event
        handle = ConditionedPoolHandle(
            self._owner,
            batch,
            deadline,
            retained=(parts, chunks, ready),
            keys=tuple(dict.fromkeys((*batch.keys, *(chunk[0] for chunk in chunks)))),
        )
        self._pending = handle
        self._last_batch = batch.batch_id
        jobs = []
        try:
            for i in sorted(indices):
                rows, part = parts.get(i, ((), None))
                reserved = tuple(
                    key for key in self._reserved if self._routes[key] == i
                )
                future = self._drivers[i].submit(
                    self._run,
                    i,
                    part,
                    tuple(grouped_chunks.get(i, ())),
                    reserved,
                    deadline,
                    ready,
                )
                jobs.append((rows, future))
                self._reserved.difference_update(reserved)
            handle.jobs = tuple(jobs)
        except BaseException:
            handle.jobs = tuple(jobs)
            self.cancel(handle)
            raise
        return handle

    def dispatch(self, batch, *, deadline=None):
        return self.dispatch_prepared(batch, deadline=deadline)

    def _owned(self, handle):
        if (
            not isinstance(handle, ConditionedPoolHandle)
            or handle.owner is not self._owner
            or self._pending is not handle
            or not handle.active
        ):
            raise RuntimeError("stale, foreign or collected conditioned pool handle")
        return handle

    def collect(self, handle):
        handle = self._owned(handle)
        try:
            device = handle.batch.recovery_tokens.device
            tokens = torch.zeros(
                (len(handle.batch.keys), self._identity.num_speculative_tokens),
                dtype=torch.int64,
                device=device,
            )
            lengths = torch.zeros(
                len(handle.batch.keys), dtype=torch.int32, device=device
            )
            statuses = [SpeculatorStatusCode.OK] * len(handle.batch.keys)
            error = None
            for rows, future in handle.jobs:
                try:
                    result = future.result()
                    if result is not None:
                        tokens[list(rows)] = result.output.token_ids
                        lengths[list(rows)] = result.output.valid_lengths
                        for row, status in zip(rows, result.row_statuses):
                            statuses[row] = status
                except Exception as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error
            return RemoteProposalResult(
                SpeculatorOutput(tokens, lengths), tuple(statuses)
            )
        finally:
            self._finish([future for _, future in handle.jobs])
            handle.active = False
            handle.retained = ()
            self._pending = None
            self._flush_closes(handle.deadline)

    def cancel(self, handle):
        handle = self._owned(handle)
        try:
            self._finish([future for _, future in handle.jobs])
            indices = {self._routes[key] for key in handle.keys}
            error = self._finish([self._call(i, "close") for i in indices])
            if error is not None:
                raise error
        finally:
            handle.active = False
            handle.retained = ()
            self._pending = None
            self._flush_closes(handle.deadline)

    def _flush_closes(self, deadline):
        if self._deferred_closes:
            self.close_sequences(tuple(self._deferred_closes), deadline=deadline)

    def close_sequences(self, keys, *, deadline=None):
        keys = tuple(dict.fromkeys(key for key in keys if key in self._routes))
        if not keys:
            return
        if self._pending is not None:
            if not set(keys).intersection(self._pending.keys):
                self._deferred_closes.update(keys)
                return
            self.cancel(self._pending)
            keys = tuple(key for key in keys if key in self._routes)
        deadline = self.preparation_deadline() if deadline is None else deadline
        groups: dict[int, list[SequenceKey]] = {}
        for key in keys:
            groups.setdefault(self._routes[key], []).append(key)
        error = self._finish(
            [
                self._call(i, "close_sequences", tuple(group), deadline=deadline)
                for i, group in groups.items()
            ]
        )
        if error is not None:
            raise error
        for key in keys:
            self._counts[self._routes.pop(key)] -= 1
            self._reserved.discard(key)
            self._deferred_closes.discard(key)

    def fail_sequences(
        self, keys, *, status=SpeculatorStatusCode.OUT_OF_MEMORY, detail, deadline=None
    ):
        if self._pending is not None:
            self.cancel(self._pending)
        groups: dict[int, list[SequenceKey]] = {}
        for key in keys:
            if key in self._routes:
                groups.setdefault(self._routes[key], []).append(key)
        error = self._finish(
            [
                self._call(
                    i,
                    "fail_sequences",
                    tuple(group),
                    status=status,
                    detail=detail,
                    deadline=deadline,
                )
                for i, group in groups.items()
            ]
        )
        self._reserved.difference_update(keys)
        if error is not None:
            raise error

    def invalidate(self):
        if self._pending is not None:
            self.cancel(self._pending)
        self._finish(list(self._probes.values()))
        self._probes.clear()
        error = self._finish(
            [self._call(i, "close") for i in range(len(self._sessions))]
        )
        self._reserved.clear()
        if error is not None:
            raise error

    def close(self):
        if self._closed:
            return
        try:
            self.invalidate()
        finally:
            self._closed = True
            for driver in self._drivers:
                driver.shutdown(wait=True)
            self._routes.clear()
            self._reserved.clear()
            self._deferred_closes.clear()
