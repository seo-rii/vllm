# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-owner v4 client; failures terminally fence the current request."""

import itertools
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.v1.spec_decode.proposal import ProposalHandle
from vllm.v1.spec_decode.remote.errors import (
    host_allocation_failure,
    runtime_unusable,
)
from vllm.v1.spec_decode.remote.protocol import (
    CloseSequence,
    ErrorReply,
    OpenSequence,
    ProtocolError,
    SequenceAck,
    SequenceKey,
    SpeculatorStatusCode,
)
from vllm.v1.spec_decode.remote.reconnect import ReconnectBackoff
from vllm.v1.spec_decode.remote.session import RemoteDraftError, tensor_to_frame
from vllm.v1.spec_decode.remote.target_kv import (
    AttentionSpec,
    SnapshotBudget,
    SnapshotIdentity,
    TargetKVSnapshot,
)
from vllm.v1.spec_decode.remote.target_kv_protocol import (
    CONTROL_BYTES,
    MAX_SNAPSHOT_BATCH_SIZE,
    TargetKVBatchQuery,
    TargetKVBatchReply,
    TargetKVHello,
    TargetKVHelloAck,
    TargetKVQuery,
    TargetKVReply,
    decode_target_kv_message,
    encode_target_kv_message,
    snapshot_batch_wire_plan,
    snapshot_wire_plan,
)
from vllm.v1.spec_decode.remote.transport import (
    FRAME_CONTROL,
    FramedConnection,
    TransportTimeout,
    connect,
    is_local_endpoint,
)

logger = init_logger(__name__)


@dataclass
class TargetKVProposalHandle(ProposalHandle):
    owner: object
    epoch: int
    request_id: int
    identities: tuple[SnapshotIdentity, ...]
    materialized_lengths: tuple[int, ...]
    deadline: float
    requested_k: int
    wire_rows: tuple[int, ...] = ()
    wire_batch: bool = False
    active: bool = True


@dataclass(frozen=True)
class TargetKVResult:
    token_ids: tuple[int, ...] = ()
    valid_length: int = 0


class TargetKVRemoteSession:
    def __init__(
        self,
        endpoint: str,
        hello: TargetKVHello,
        *,
        expected_specs: Mapping[str, AttentionSpec],
        failure_policy="error",
        request_timeout_ms=100,
        startup_timeout_ms=30000,
        clock=time.monotonic,
    ):
        if not is_local_endpoint(endpoint) or hello.transport != "inline":
            raise ValueError("target-KV snapshots require same-host inline transport")
        if failure_policy not in ("error", "target_only"):
            raise ValueError("invalid target-KV failure policy")
        if (
            hello.num_speculative_tokens != 1
            or not 1 <= hello.max_num_seqs <= MAX_SNAPSHOT_BATCH_SIZE
            or not 1 <= hello.max_model_len <= 8192
            or not hello.verifier_instance_id
            or not hello.contract_fingerprint
            or min(request_timeout_ms, startup_timeout_ms) <= 0
            or not all(
                math.isfinite(value)
                for value in (request_timeout_ms, startup_timeout_ms)
            )
        ):
            raise ValueError("invalid target-KV session contract")
        self.endpoint, self.hello = endpoint, hello
        self.expected_specs = dict(expected_specs)
        self.budget = SnapshotBudget(
            hello.max_snapshot_pages, hello.max_snapshot_bytes, hello.max_model_len
        )
        snapshot_wire_plan(self.expected_specs, 1, self.budget)
        self.failure_policy = failure_policy
        self.request_timeout_s = request_timeout_ms / 1000
        self.startup_timeout_s = startup_timeout_ms / 1000
        self.clock = clock
        self._reconnect = ReconnectBackoff(clock)
        self.records: dict[SequenceKey, bool] = {}
        self._owner = object()
        self._conn: FramedConnection | None = None
        self._sid = ""
        self._epoch = 0
        self._ids = itertools.count(1)
        self._generations = itertools.count()
        self._pending: TargetKVProposalHandle | None = None
        self._closed = False
        self._last_round = -1
        self._closing: set[SequenceKey] = set()

    @property
    def epoch(self) -> int:
        return self._epoch

    def deadline(self) -> float:
        return self.clock() + self.request_timeout_s

    def active(self, key: SequenceKey) -> bool:
        return self.records.get(key, False)

    def _invalidate(self):
        connection, self._conn = self._conn, None
        self._closing.clear()
        for key in self.records:
            self.records[key] = False
        if connection is not None:
            try:
                connection.close()
            except OSError as error:
                logger.warning("Target-KV connection cleanup failed: %s", error)

    def fail(self, detail: str) -> None:
        self._invalidate()
        if self.failure_policy == "error":
            raise RemoteDraftError(detail)

    def _send(self, message, request_id, deadline):
        assert self._conn is not None
        self._conn.send_control(
            encode_target_kv_message(
                message, session_id=self._sid, request_id=request_id
            ),
            deadline=deadline,
        )

    def _recv(self, request_id, deadline, *, handshake=False):
        assert self._conn is not None
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise TransportTimeout("target-KV result deadline expired")
        frame = self._conn.recv(timeout=remaining, max_frame_bytes=CONTROL_BYTES)
        if frame.kind != FRAME_CONTROL:
            raise ProtocolError("target-KV response must be control-only")
        envelope, message = decode_target_kv_message(frame.body)
        if envelope.request_id != request_id or (
            not handshake and envelope.session_id != self._sid
        ):
            raise ProtocolError("target-KV response request/session mismatch")
        if isinstance(message, ErrorReply):
            raise RemoteDraftError(message.detail)
        if handshake and (
            not isinstance(message, TargetKVHelloAck)
            or not message.session_id
            or envelope.session_id != message.session_id
        ):
            raise ProtocolError("invalid target-KV hello acknowledgment")
        return message

    def connect(self, *, deadline=None):
        if self._closed:
            raise RuntimeError("target-KV session is closed")
        if self._conn is not None:
            return
        if not self._reconnect.ready():
            raise RemoteDraftError("target-KV reconnect is in backoff")
        deadline = (
            deadline if deadline is not None else self.clock() + self.startup_timeout_s
        )
        try:
            self._conn = connect(self.endpoint, timeout=max(0, deadline - self.clock()))
            self._conn.set_receive_limits(
                max_control_bytes=CONTROL_BYTES,
                max_data_bytes=0,
                max_buffered_bytes=CONTROL_BYTES + 5,
            )
            self._sid = ""
            request_id = next(self._ids)
            self._send(self.hello, request_id, deadline)
            ack = self._recv(request_id, deadline, handshake=True)
            if (
                ack.contract_fingerprint != self.hello.contract_fingerprint
                or ack.max_snapshot_pages < self.budget.max_pages
                or ack.max_snapshot_bytes < self.budget.max_bytes
                or ack.max_num_seqs < self.hello.max_num_seqs
            ):
                raise ProtocolError("target-KV handshake contract/capacity mismatch")
            self._sid, self._epoch = ack.session_id, ack.session_epoch
            self._reconnect.succeeded()
        except Exception:
            self._reconnect.failed()
            self._invalidate()
            raise

    def open_sequence(self, sequence_id: int, *, deadline: float) -> SequenceKey:
        if self._pending is not None or len(self.records) >= self.hello.max_num_seqs:
            raise RuntimeError("target-KV open exceeds capacity or overlaps a proposal")
        if any(key.sequence_id == sequence_id for key in self.records):
            raise ValueError("target-KV sequence ID is already open")
        key = SequenceKey(
            self.hello.verifier_instance_id, sequence_id, next(self._generations)
        )
        self.records[key] = True
        try:
            self.connect(deadline=deadline)
            self._flush_closes(deadline)
            request_id = next(self._ids)
            self._send(OpenSequence(key), request_id, deadline)
            ack = self._recv(request_id, deadline)
            if (
                not isinstance(ack, SequenceAck)
                or ack.key != key
                or ack.status is not SpeculatorStatusCode.OK
            ):
                raise ProtocolError("invalid target-KV open acknowledgment")
        except Exception as error:
            self.fail(f"target-KV open failed: {error}")
        return key

    def open_sequences(
        self, sequence_ids: tuple[int, ...], *, deadline: float
    ) -> tuple[SequenceKey, ...]:
        if len(self.records) + len(sequence_ids) > self.hello.max_num_seqs:
            raise ValueError("target-KV open exceeds negotiated sequence capacity")
        if len(set(sequence_ids)) != len(sequence_ids) or any(
            key.sequence_id in sequence_ids for key in self.records
        ):
            raise ValueError("target-KV sequence IDs must be distinct and unopened")
        return tuple(self.open_sequence(i, deadline=deadline) for i in sequence_ids)

    def dispatch(
        self,
        snapshot: TargetKVSnapshot,
        input_token: int,
        hidden_states: torch.Tensor,
        *,
        k: int = 1,
        deadline: float,
    ) -> TargetKVProposalHandle:
        return self.dispatch_batch(
            (snapshot,), (input_token,), hidden_states, k=k, deadline=deadline
        )

    def dispatch_batch(
        self,
        snapshots: tuple[TargetKVSnapshot, ...],
        input_tokens: tuple[int, ...],
        hidden_states: torch.Tensor,
        *,
        k: int = 1,
        deadline: float,
    ) -> TargetKVProposalHandle:
        if self._pending is not None:
            raise RuntimeError("target-KV proposal is already in flight")
        if not 1 <= len(snapshots) <= self.hello.max_num_seqs:
            raise ValueError("target-KV proposal exceeds negotiated batch capacity")
        identities = tuple(s.identity for s in snapshots)
        if (
            k not in (0, 1)
            or any(i.round_id <= self._last_round for i in identities)
            or len({i.key for i in identities}) != len(identities)
        ):
            raise ValueError("invalid target-KV proposal round/budget")
        if any(i.session_epoch != self._epoch for i in identities):
            raise ValueError("stale target-KV session epoch")
        self._last_round = max(i.round_id for i in identities)
        wire_rows = tuple(
            i
            for i, identity in enumerate(identities)
            if k and self.active(identity.key)
        )
        handle = TargetKVProposalHandle(
            self._owner,
            self._epoch,
            next(self._ids),
            identities,
            tuple(s.materialized_length for s in snapshots),
            deadline,
            k,
            wire_rows,
            self.hello.max_num_seqs > 1,
        )
        self._pending = handle
        if not wire_rows:
            return handle
        try:
            self._flush_closes(deadline)
            if self._conn is None:
                return handle
            handle.request_id = next(self._ids)
            if len(input_tokens) != len(snapshots) or any(
                not 0 <= token < self.hello.vocab_size for token in input_tokens
            ):
                raise ValueError("target-KV recovery token is out of vocabulary")
            if hidden_states.shape != (len(snapshots), self.hello.hidden_size) or (
                hidden_states.dtype != torch.bfloat16
            ):
                raise ValueError(
                    "target-KV hidden state must be BF16 [batch, hidden_size]"
                )
            plans = snapshot_batch_wire_plan(
                self.expected_specs,
                tuple(snapshots[i].materialized_length for i in wire_rows),
                self.budget,
            )
            ordered_layers = []
            for i, row_plans in zip(wire_rows, plans):
                snapshot = snapshots[i]
                layers = {layer.layer_name: layer for layer in snapshot.layers}
                if len(layers) != len(snapshot.layers) or set(layers) != set(
                    self.expected_specs
                ):
                    raise ValueError(
                        "target-KV snapshot source layers differ from contract"
                    )
                row_layers = []
                for plan in row_plans:
                    layer = layers[plan.name]
                    if (
                        layer.spec != plan.spec
                        or layer.logical_pages != plan.logical_pages
                        or tuple(layer.pages.shape) != plan.shape
                        or layer.pages.dtype != torch.bfloat16
                        or not layer.pages.is_contiguous()
                    ):
                        raise ValueError("invalid target-KV snapshot layer")
                    row_layers.append(layer)
                ordered_layers.append(row_layers)
            remaining_ms = math.ceil((deadline - self.clock()) * 1000)
            if remaining_ms <= 0:
                raise TransportTimeout("target-KV preparation deadline expired")
            queries = tuple(
                TargetKVQuery(
                    identities[i].key,
                    self._epoch,
                    identities[i].round_id,
                    snapshots[i].materialized_length,
                    input_tokens[i],
                    remaining_ms,
                )
                for i in wire_rows
            )
            self._send(
                TargetKVBatchQuery(queries, remaining_ms)
                if handle.wire_batch
                else queries[0],
                handle.request_id,
                deadline,
            )
            assert self._conn is not None
            slot = 0
            for i, row_layers in zip(wire_rows, ordered_layers):
                self._conn.send_data(
                    tensor_to_frame(slot, hidden_states[i : i + 1]), deadline=deadline
                )
                slot += 1
                for layer in row_layers:
                    self._conn.send_data(
                        tensor_to_frame(slot, layer.pages), deadline=deadline
                    )
                    slot += 1
        except Exception as error:
            if runtime_unusable(error) or (
                isinstance(error, RuntimeError)
                and not isinstance(error, torch.OutOfMemoryError)
                and not host_allocation_failure(error)
            ):
                self._invalidate()
                handle.active = False
                self._pending = None
                raise
            try:
                self.fail(f"target-KV dispatch failed: {error}")
            except RemoteDraftError:
                handle.active = False
                self._pending = None
                raise
        return handle

    def collect(self, handle: TargetKVProposalHandle) -> TargetKVResult:
        return self.collect_batch(handle)[0]

    def collect_batch(
        self, handle: TargetKVProposalHandle
    ) -> tuple[TargetKVResult, ...]:
        if (
            handle.owner is not self._owner
            or self._pending is not handle
            or not handle.active
            or handle.epoch != self._epoch
        ):
            raise RuntimeError("stale, foreign or collected target-KV handle")
        results = [TargetKVResult() for _ in handle.identities]
        row_error = None
        try:
            if not handle.wire_rows or self._conn is None:
                return tuple(results)
            message = self._recv(handle.request_id, handle.deadline)
            if handle.wire_batch:
                if not isinstance(message, TargetKVBatchReply):
                    raise ProtocolError("target-KV expected a batch reply")
                replies = message.replies
            else:
                replies = (message,)
            if len(replies) != len(handle.wire_rows):
                raise ProtocolError("target-KV reply row count mismatch")
            for i, reply in zip(handle.wire_rows, replies):
                identity = handle.identities[i]
                if (
                    not isinstance(reply, TargetKVReply)
                    or reply.key != identity.key
                    or reply.session_epoch != identity.session_epoch
                    or reply.round_id != identity.round_id
                    or reply.materialized_length != handle.materialized_lengths[i]
                ):
                    raise ProtocolError("target-KV proposal identity/length mismatch")
                if reply.status is not SpeculatorStatusCode.OK:
                    self.records[identity.key] = False
                    row_error = reply.detail or reply.status.name
                    if not handle.wire_batch:
                        raise RemoteDraftError(row_error)
                elif (
                    reply.token_id is None
                    or not 0 <= reply.token_id < self.hello.vocab_size
                ):
                    raise ProtocolError("invalid target-KV proposal token")
                else:
                    results[i] = TargetKVResult((reply.token_id,), 1)
        except Exception as error:
            self.fail(f"target-KV collect failed: {error}")
            return tuple(TargetKVResult() for _ in handle.identities)
        finally:
            handle.active = False
            self._pending = None
            if self._closing:
                if self.clock() < handle.deadline:
                    self._flush_closes(handle.deadline)
                elif not any(self.records.values()):
                    self._invalidate()
        if row_error is not None and self.failure_policy == "error":
            raise RemoteDraftError(f"target-KV collect failed: {row_error}")
        return tuple(results)

    def cancel(self, handle: TargetKVProposalHandle):
        if (
            handle.owner is not self._owner
            or self._pending is not handle
            or not handle.active
        ):
            raise RuntimeError("cannot cancel stale target-KV handle")
        handle.active = False
        self._pending = None
        self._invalidate()

    def _flush_closes(self, deadline: float) -> None:
        if self._conn is None or not self._closing:
            return
        keys = tuple(self._closing)
        try:
            request_id = next(self._ids)
            self._send(CloseSequence(keys), request_id, deadline)
            for key in keys:
                ack = self._recv(request_id, deadline)
                if (
                    not isinstance(ack, SequenceAck)
                    or ack.key != key
                    or ack.status is not SpeculatorStatusCode.OK
                ):
                    raise ProtocolError("invalid target-KV close acknowledgment")
            self._closing.difference_update(keys)
        except Exception as error:
            self.fail(f"target-KV close failed: {error}")

    def close_sequences(self, keys: tuple[SequenceKey, ...]):
        keys = tuple(key for key in dict.fromkeys(keys) if key in self.records)
        if not keys:
            return
        if self._pending is not None and any(
            identity.key in keys for identity in self._pending.identities
        ):
            self.cancel(self._pending)
        if self.hello.max_num_seqs > 1 and self._conn is not None:
            self._closing.update(keys)
        for key in keys:
            self.records.pop(key, None)
        if self._pending is None:
            self._flush_closes(self.deadline())

    def close(self):
        self._closed = True
        if self._pending is not None:
            self._pending.active = False
            self._pending = None
        self._invalidate()
        self.records.clear()
