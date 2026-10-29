# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol v4: one authoritative target-KV snapshot per stateless K1 query."""

from collections.abc import Mapping
from dataclasses import dataclass

import msgspec

from vllm.v1.spec_decode.remote.protocol import (
    CloseSequence,
    ErrorReply,
    MessageEnvelope,
    NonNegativeInt,
    OpenSequence,
    Ping,
    Pong,
    PositiveInt,
    ProtocolError,
    ProtocolVersionError,
    SequenceAck,
    SequenceKey,
    SpeculatorStatusCode,
)
from vllm.v1.spec_decode.remote.target_kv import (
    AttentionSpec,
    SnapshotBudget,
    _page_shape,
    _range,
    _table_width,
    _validate_spec,
)

TARGET_KV_PROTOCOL_MAJOR = 4
CONTROL_BYTES = 16384
FRAME_METADATA_BYTES = 1024
MAX_SNAPSHOT_BATCH_SIZE = 16


class TargetKVHello(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    verifier_instance_id: str
    contract_fingerprint: str
    hidden_size: PositiveInt
    vocab_size: PositiveInt
    max_model_len: PositiveInt
    max_snapshot_pages: PositiveInt = 1024
    max_snapshot_bytes: PositiveInt = 128 << 20
    transport: str = "inline"
    num_speculative_tokens: PositiveInt = 1
    max_num_seqs: PositiveInt = 1


class TargetKVHelloAck(msgspec.Struct):
    session_id: str
    session_epoch: PositiveInt
    contract_fingerprint: str
    max_snapshot_pages: PositiveInt
    max_snapshot_bytes: PositiveInt
    max_num_seqs: PositiveInt = 1


class TargetKVQuery(msgspec.Struct):
    key: SequenceKey
    session_epoch: PositiveInt
    round_id: NonNegativeInt
    materialized_length: PositiveInt
    input_token: NonNegativeInt
    budget_ms: PositiveInt


class TargetKVReply(msgspec.Struct):
    key: SequenceKey
    session_epoch: PositiveInt
    round_id: NonNegativeInt
    materialized_length: PositiveInt
    token_id: NonNegativeInt | None = None
    status: SpeculatorStatusCode = SpeculatorStatusCode.OK
    detail: str = ""


class TargetKVBatchQuery(msgspec.Struct):
    queries: tuple[TargetKVQuery, ...]
    budget_ms: PositiveInt


class TargetKVBatchReply(msgspec.Struct):
    replies: tuple[TargetKVReply, ...]


_TYPES = {
    "hello": TargetKVHello,
    "hello_ack": TargetKVHelloAck,
    "open_sequence": OpenSequence,
    "close_sequence": CloseSequence,
    "sequence_ack": SequenceAck,
    "snapshot_query": TargetKVQuery,
    "snapshot_reply": TargetKVReply,
    "snapshot_batch_query": TargetKVBatchQuery,
    "snapshot_batch_reply": TargetKVBatchReply,
    "ping": Ping,
    "pong": Pong,
    "error": ErrorReply,
}
_NAMES = {kind: name for name, kind in _TYPES.items()}


def encode_target_kv_message(message, *, session_id="", request_id=0) -> bytes:
    name = _NAMES.get(type(message))
    if name is None:
        raise ProtocolError("unregistered target-KV message")
    result = msgspec.msgpack.encode(
        MessageEnvelope(
            4, 0, name, session_id, request_id, msgspec.msgpack.encode(message)
        )
    )
    if len(result) > CONTROL_BYTES:
        raise ProtocolError("target-KV control message exceeds the byte limit")
    return result


def decode_target_kv_message(data: bytes):
    if len(data) > CONTROL_BYTES:
        raise ProtocolError("target-KV control message exceeds the byte limit")
    try:
        envelope = msgspec.msgpack.decode(data, type=MessageEnvelope)
        if envelope.protocol_major != TARGET_KV_PROTOCOL_MAJOR:
            raise ProtocolVersionError("target-KV snapshots require protocol v4")
        if envelope.protocol_minor != 0:
            raise ProtocolVersionError("unsupported target-KV protocol minor")
        kind = _TYPES.get(envelope.message_type)
        if kind is None:
            raise ProtocolError("unknown target-KV message")
        message = msgspec.msgpack.decode(envelope.payload, type=kind)
    except msgspec.DecodeError as error:
        raise ProtocolError(f"malformed target-KV message: {error}") from error
    return envelope, message


@dataclass(frozen=True)
class LayerWirePlan:
    name: str
    spec: AttentionSpec
    logical_pages: tuple[int, ...]
    shape: tuple[int, ...]
    nbytes: int


def snapshot_wire_plan(
    specs: Mapping[str, AttentionSpec],
    length: int,
    budget: SnapshotBudget,
    receiver_budget: SnapshotBudget | None = None,
) -> tuple[LayerWirePlan, ...]:
    """Validate all export/import capacity before receiving or allocating pages."""
    budget.check(length, 0, 0)
    if not specs or len(specs) > 32:
        raise ValueError("target-KV requires between one and 32 source layers")
    plans = []
    pages = nbytes = receiver_pages = receiver_bytes = 0
    for name, spec in sorted(specs.items()):
        _validate_spec(spec)
        _, first, end = _range(spec, length)
        count = end - first
        size = count * spec.unpadded_page_size_bytes
        pages += count
        nbytes += size
        budget.check(length, pages, nbytes)
        receiver_pages += count + 1
        receiver_bytes += (count + 1) * spec.unpadded_page_size_bytes
        receiver_bytes += _table_width(spec, end) * 4
        if receiver_budget is not None:
            receiver_budget.check(length, receiver_pages, receiver_bytes)
        plans.append(
            LayerWirePlan(
                name, spec, tuple(range(first, end)), (count, *_page_shape(spec)), size
            )
        )
    return tuple(plans)


def snapshot_batch_wire_plan(
    specs: Mapping[str, AttentionSpec],
    lengths: tuple[int, ...],
    budget: SnapshotBudget,
    receiver_budget: SnapshotBudget | None = None,
) -> tuple[tuple[LayerWirePlan, ...], ...]:
    """Budget all rows and shared receiver caches before reading any payload."""
    if not 1 <= len(lengths) <= MAX_SNAPSHOT_BATCH_SIZE:
        raise ValueError("target-KV batch must contain between one and 16 rows")
    plans = tuple(snapshot_wire_plan(specs, n, budget) for n in lengths)
    pages = sum(len(p.logical_pages) for row in plans for p in row)
    nbytes = sum(p.nbytes for row in plans for p in row)
    budget.check(max(lengths), pages, nbytes)
    if receiver_budget is not None:
        receiver_bytes = nbytes
        for i, spec in enumerate(specs[name] for name in sorted(specs)):
            width = max(
                _table_width(spec, row[i].logical_pages[-1] + 1) for row in plans
            )
            receiver_bytes += spec.unpadded_page_size_bytes + len(lengths) * width * 4
        receiver_budget.check(max(lengths), pages + len(specs), receiver_bytes)
    return plans
