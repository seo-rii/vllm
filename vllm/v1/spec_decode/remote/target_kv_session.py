# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-owner v4 client; failures terminally fence the current request."""

import itertools
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from vllm.v1.spec_decode.proposal import ProposalHandle
from vllm.v1.spec_decode.remote.errors import (
    host_allocation_failure,
    runtime_unusable,
)
from vllm.v1.spec_decode.remote.protocol import (
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
    TargetKVHello,
    TargetKVHelloAck,
    TargetKVQuery,
    TargetKVReply,
    decode_target_kv_message,
    encode_target_kv_message,
    snapshot_wire_plan,
)
from vllm.v1.spec_decode.remote.transport import (
    FRAME_CONTROL,
    FramedConnection,
    TransportTimeout,
    connect,
    is_local_endpoint,
)


@dataclass
class TargetKVProposalHandle(ProposalHandle):
    owner: object
    epoch: int
    request_id: int
    identity: SnapshotIdentity
    materialized_length: int
    deadline: float
    requested_k: int
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
            or hello.max_num_seqs != 1
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

    @property
    def epoch(self) -> int:
        return self._epoch

    def deadline(self) -> float:
        return self.clock() + self.request_timeout_s

    def active(self, key: SequenceKey) -> bool:
        return self.records.get(key, False)

    def _invalidate(self):
        if self._conn is not None:
            self._conn.close()
        self._conn = None
        for key in self.records:
            self.records[key] = False

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
            ):
                raise ProtocolError("target-KV handshake contract/capacity mismatch")
            self._sid, self._epoch = ack.session_id, ack.session_epoch
            self._reconnect.succeeded()
        except Exception:
            self._reconnect.failed()
            self._invalidate()
            raise

    def open_sequence(self, sequence_id: int, *, deadline: float) -> SequenceKey:
        if self.records:
            raise RuntimeError("target-KV permits only one open request")
        key = SequenceKey(
            self.hello.verifier_instance_id, sequence_id, next(self._generations)
        )
        self.records[key] = True
        try:
            self.connect(deadline=deadline)
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

    def dispatch(
        self,
        snapshot: TargetKVSnapshot,
        input_token: int,
        hidden_states: torch.Tensor,
        *,
        k: int = 1,
        deadline: float,
    ) -> TargetKVProposalHandle:
        identity = snapshot.identity
        if self._pending is not None:
            raise RuntimeError("target-KV proposal is already in flight")
        if k not in (0, 1) or identity.round_id <= self._last_round:
            raise ValueError("invalid target-KV proposal round/budget")
        if identity.session_epoch != self._epoch:
            raise ValueError("stale target-KV session epoch")
        self._last_round = identity.round_id
        request_id = next(self._ids)
        handle = TargetKVProposalHandle(
            self._owner,
            self._epoch,
            request_id,
            identity,
            snapshot.materialized_length,
            deadline,
            k,
        )
        self._pending = handle
        if not k or not self.active(identity.key):
            return handle
        try:
            if not 0 <= input_token < self.hello.vocab_size:
                raise ValueError("target-KV recovery token is out of vocabulary")
            if (
                hidden_states.shape != (1, self.hello.hidden_size)
                or hidden_states.dtype != torch.bfloat16
            ):
                raise ValueError("target-KV hidden state must be BF16 [1, hidden_size]")
            plans = snapshot_wire_plan(
                self.expected_specs, snapshot.materialized_length, self.budget
            )
            layers = {layer.layer_name: layer for layer in snapshot.layers}
            if len(layers) != len(snapshot.layers) or set(layers) != set(
                self.expected_specs
            ):
                raise ValueError(
                    "target-KV snapshot source layers differ from contract"
                )
            for plan in plans:
                layer = layers[plan.name]
                if (
                    layer.spec != plan.spec
                    or layer.logical_pages != plan.logical_pages
                    or tuple(layer.pages.shape) != plan.shape
                    or layer.pages.dtype != torch.bfloat16
                    or not layer.pages.is_contiguous()
                ):
                    raise ValueError("invalid target-KV snapshot layer")
            remaining_ms = math.ceil((deadline - self.clock()) * 1000)
            if remaining_ms <= 0:
                raise TransportTimeout("target-KV preparation deadline expired")
            self._send(
                TargetKVQuery(
                    identity.key,
                    self._epoch,
                    identity.round_id,
                    snapshot.materialized_length,
                    input_token,
                    remaining_ms,
                ),
                request_id,
                deadline,
            )
            assert self._conn is not None
            self._conn.send_data(tensor_to_frame(0, hidden_states), deadline=deadline)
            for slot, plan in enumerate(plans, 1):
                self._conn.send_data(
                    tensor_to_frame(slot, layers[plan.name].pages), deadline=deadline
                )
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
        if (
            handle.owner is not self._owner
            or self._pending is not handle
            or not handle.active
            or handle.epoch != self._epoch
        ):
            raise RuntimeError("stale, foreign or collected target-KV handle")
        try:
            identity = handle.identity
            if (
                not handle.requested_k
                or self._conn is None
                or not self.active(identity.key)
            ):
                return TargetKVResult()
            response = self._recv(handle.request_id, handle.deadline)
            if (
                not isinstance(response, TargetKVReply)
                or response.key != identity.key
                or response.session_epoch != identity.session_epoch
                or response.round_id != identity.round_id
                or response.materialized_length != handle.materialized_length
            ):
                raise ProtocolError("target-KV proposal identity/length mismatch")
            if response.status is not SpeculatorStatusCode.OK:
                raise RemoteDraftError(response.detail or response.status.name)
            if (
                response.token_id is None
                or not 0 <= response.token_id < self.hello.vocab_size
            ):
                raise ProtocolError("invalid target-KV proposal token")
            return TargetKVResult((response.token_id,), 1)
        except Exception as error:
            self.fail(f"target-KV collect failed: {error}")
            return TargetKVResult()
        finally:
            handle.active = False
            self._pending = None

    def cancel(self, handle: TargetKVProposalHandle):
        if (
            handle.owner is not self._owner
            or self._pending is not handle
            or not handle.active
        ):
            raise RuntimeError("cannot cancel stale target-KV handle")
        self._invalidate()
        handle.active = False
        self._pending = None

    def close_sequences(self, keys: tuple[SequenceKey, ...]):
        if self._pending is not None and self._pending.identity.key in keys:
            self.cancel(self._pending)
        # The server retains only one identity fence, no per-request KV or tensors.
        for key in keys:
            self.records.pop(key, None)

    def close(self):
        self._invalidate()
        self._closed = True
        if self._pending is not None:
            self._pending.active = False
            self._pending = None
        self.records.clear()
