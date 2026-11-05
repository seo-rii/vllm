# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit v3 token-only messages; the target-conditioned v2 codec is unchanged."""

from typing import Literal

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
    RemoteServerLimits,
    SequenceAck,
    SequenceKey,
    SpeculatorStatusCode,
)

TOKEN_PROTOCOL_MAJOR = 3
TOKEN_PROTOCOL_MINOR = 2


class TokenHello(msgspec.Struct):
    verifier_instance_id: str
    target_fingerprint: str
    draft_fingerprint: str
    tokenizer_fingerprint: str
    num_speculative_tokens: PositiveInt
    vocab_size: PositiveInt
    max_num_seqs: PositiveInt
    max_num_batched_tokens: PositiveInt
    max_model_len: PositiveInt
    method: str = "draft_model"
    transport: str = "inline"
    policy: Literal["on_demand", "outcome_cache"] = "on_demand"


class TokenHelloAck(msgspec.Struct):
    session_id: str
    session_epoch: PositiveInt
    tokenizer_fingerprint: str
    limits: RemoteServerLimits
    policy: Literal["on_demand", "outcome_cache"] = "on_demand"


class TokenPrefill(msgspec.Struct):
    key: SequenceKey
    offset: NonNegativeInt
    num_tokens: PositiveInt
    token_slot: NonNegativeInt
    checksum: NonNegativeInt
    is_final: bool
    execution_timeout_ms: NonNegativeInt | None = None
    """Remaining relative budget; v3.0 peers do not enforce queued expiry."""


class TokenAdvance(msgspec.Struct):
    batch_id: NonNegativeInt
    keys: tuple[SequenceKey, ...]
    base_lengths_slot: NonNegativeInt
    previous_ids_slot: NonNegativeInt
    offered_slot: NonNegativeInt
    accepted_slot: NonNegativeInt
    delta_lengths_slot: NonNegativeInt
    token_slot: NonNegativeInt
    num_speculative_tokens: NonNegativeInt
    execution_timeout_ms: NonNegativeInt | None = None
    """Remaining relative budget; v3.0 peers do not enforce queued expiry."""


class TokenProposal(msgspec.Struct):
    batch_id: NonNegativeInt
    result_slot: NonNegativeInt | None = None
    status: SpeculatorStatusCode = SpeculatorStatusCode.OK
    origins: tuple[Literal["on_demand", "cache"], ...] = ()


_TYPES = {
    "hello": TokenHello,
    "hello_ack": TokenHelloAck,
    "open_sequence": OpenSequence,
    "prefill_tokens": TokenPrefill,
    "advance_tokens": TokenAdvance,
    "token_proposal": TokenProposal,
    "sequence_ack": SequenceAck,
    "close_sequence": CloseSequence,
    "ping": Ping,
    "pong": Pong,
    "error": ErrorReply,
}
_NAMES = {kind: name for name, kind in _TYPES.items()}


def encode_token_message(
    message: msgspec.Struct, *, session_id: str = "", request_id: int = 0
) -> bytes:
    name = _NAMES.get(type(message))
    if name is None:
        raise ProtocolError(f"unregistered token message: {type(message).__name__}")
    return msgspec.msgpack.encode(
        MessageEnvelope(
            TOKEN_PROTOCOL_MAJOR,
            TOKEN_PROTOCOL_MINOR,
            name,
            session_id,
            request_id,
            msgspec.msgpack.encode(message),
        )
    )


def decode_token_message(data: bytes) -> tuple[MessageEnvelope, msgspec.Struct]:
    try:
        envelope = msgspec.msgpack.decode(data, type=MessageEnvelope)
        if envelope.protocol_major != TOKEN_PROTOCOL_MAJOR:
            raise ProtocolVersionError("token-only connections require protocol v3")
        kind = _TYPES.get(envelope.message_type)
        if kind is None:
            raise ProtocolError(f"unknown token message: {envelope.message_type}")
        message = msgspec.msgpack.decode(envelope.payload, type=kind)
    except msgspec.DecodeError as error:
        raise ProtocolError(f"malformed token message: {error}") from error
    return envelope, message
