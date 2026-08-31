# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verifier-side session with one standalone speculator server.

The session owns the connection, the negotiated schema and limits, the
per-sequence client state machine, and the slot bookkeeping that keeps
stale data-plane frames from ever being read as current results. Runner
code hands it placement-neutral RemoteProposalBatch objects and receives
SpeculatorOutput back; it never sees protocol structs or frames.

Failure semantics follow the MVP policy: a row whose remote state is in
doubt (timeout, desync, stale generation) stops receiving remote proposals
for the rest of the request and reports ``valid_length=0`` every round.
Automatic re-prefill recovery is a later step; ``open_sequence`` on a
desynced sequence already bumps the generation so that step can build on
it.
"""

import math
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.v1.spec_decode.proposal import ProposalHandle, SpeculatorOutput
from vllm.v1.spec_decode.remote.capabilities import (
    SpeculatorPlacementCapabilities,
    placement_incompatibilities,
)
from vllm.v1.spec_decode.remote.protocol import (
    AdvanceAndPropose,
    CancelBatch,
    CloseSequence,
    ErrorReply,
    Hello,
    HelloAck,
    MessageEnvelope,
    OpenSequence,
    Ping,
    Pong,
    PrefillChunk,
    ProposalResponse,
    ProtocolError,
    RemoteServerLimits,
    SequenceAck,
    SequenceKey,
    SpeculatorStatusCode,
    TargetFeatureSchema,
    decode_envelope,
    decode_payload,
    encode_message,
)
from vllm.v1.spec_decode.remote.state import (
    FeatureBatch,
    HandleState,
    InvalidStateTransition,
    RemoteProposalBatch,
    RemoteProposalResult,
    SequenceState,
    can_transition_sequence,
    transition_handle,
    transition_sequence,
)
from vllm.v1.spec_decode.remote.transport import (
    FRAME_DATA,
    INLINE_TRANSPORT,
    ConnectionClosed,
    DataFrame,
    FramedConnection,
    TransportError,
    TransportTimeout,
    decode_data_frame,
    frame_to_ints,
)
from vllm.v1.spec_decode.remote.transport import connect as transport_connect

if TYPE_CHECKING:
    from vllm.config.speculative import RemoteDraftConfig

logger = init_logger(__name__)

# A completed round is delivered as three data frames at consecutive slots:
# draft tokens, per-row valid lengths, per-row status codes.
RESULT_FRAMES = 3

_DESYNC_STATUSES = frozenset(
    {
        SpeculatorStatusCode.STALE_GENERATION,
        SpeculatorStatusCode.SEQUENCE_DESYNC,
        SpeculatorStatusCode.ROUND_MISMATCH,
    }
)
_RETRYABLE_STATUSES = frozenset(
    {SpeculatorStatusCode.OK, SpeculatorStatusCode.QUEUE_FULL}
)


class RemoteDraftError(Exception):
    """Remote speculator failure the configured policy does not absorb."""


def tensor_to_frame(slot: int, tensor: torch.Tensor) -> DataFrame:
    """Serialize a tensor into an inline data frame (host copy)."""
    flat = tensor.detach().reshape(-1)
    if flat.device.type != "cpu":
        flat = flat.cpu()
    flat = flat.contiguous()
    payload = flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b""
    return DataFrame(
        slot=slot,
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=tuple(tensor.shape),
        payload=payload,
    )


def frame_to_tensor(
    frame: DataFrame, device: torch.device | str | None = None
) -> torch.Tensor:
    """Materialize an inline data frame as a tensor on ``device``."""
    dtype = getattr(torch, frame.dtype, None)
    if not isinstance(dtype, torch.dtype):
        raise TransportError(f"unknown dtype {frame.dtype!r} in data frame")
    numel = math.prod(frame.shape)
    if numel == 0:
        tensor = torch.empty(frame.shape, dtype=dtype)
    else:
        expected = numel * dtype.itemsize
        if len(frame.payload) != expected:
            raise TransportError(
                f"data frame {frame.slot} has {len(frame.payload)} payload "
                f"bytes, expected {expected} for {frame.dtype} {frame.shape}"
            )
        tensor = torch.frombuffer(bytearray(frame.payload), dtype=dtype)
        tensor = tensor.reshape(frame.shape)
    return tensor if device is None else tensor.to(device)


@dataclass(frozen=True)
class VerifierIdentity:
    """What the verifier reports about itself in HELLO."""

    verifier_instance_id: str
    target_fingerprint: str
    tokenizer_fingerprint: str
    method: str
    num_speculative_tokens: int
    parallel_drafting: bool
    provided_features: tuple[str, ...]
    """TargetFeatureKind values the verifier can transport every round."""
    draft_sample_method: str = "greedy"
    draft_checkpoint_fingerprint: str = ""
    uses_multi_module: bool = False


@dataclass
class _SequenceRecord:
    key: SequenceKey
    state: SequenceState
    prefilled_tokens: int = 0
    last_status: SpeculatorStatusCode = SpeculatorStatusCode.OK


@dataclass
class RemoteProposalHandle(ProposalHandle):
    """Dispatch receipt for one remote round; collect exactly once."""

    session_epoch: int
    batch_id: int
    keys: tuple[SequenceKey, ...]
    active_rows: tuple[int, ...]
    """Rows that were actually sent; the rest report valid_length=0."""
    num_speculative_tokens: int
    device: torch.device
    deadline: float
    session_id: str = ""
    request_id: int = -1
    input_slot: int = -1
    state: HandleState = HandleState.CREATED


class RemoteDraftSession:
    """One verifier's connection to one standalone speculator server.

    Not thread-safe: the model runner drives it from its own thread. The
    ``clock`` is injectable so timeout behaviour is testable without
    sleeping; it must be monotonic and in seconds.
    """

    def __init__(
        self,
        endpoint: str,
        identity: VerifierIdentity,
        *,
        transport: str = "auto",
        failure_policy: str = "target_only",
        request_timeout_ms: int = 100,
        startup_timeout_ms: int = 30_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if transport != "auto":
            raise NotImplementedError(
                f"remote_draft.transport={transport!r} has no data-plane "
                "implementation yet; use 'auto'"
            )
        if failure_policy not in ("error", "target_only"):
            raise ValueError(f"unknown failure_policy {failure_policy!r}")
        self._endpoint = endpoint
        self._identity = identity
        self._offered_transports = (INLINE_TRANSPORT,)
        self._failure_policy = failure_policy
        self._request_timeout_s = request_timeout_ms / 1000.0
        self._startup_timeout_s = startup_timeout_ms / 1000.0
        self._clock = clock

        self._conn: FramedConnection | None = None
        self._alive = False
        self._session_id = ""
        self._session_epoch = -1
        self._capabilities: SpeculatorPlacementCapabilities | None = None
        self._feature_schema: TargetFeatureSchema | None = None
        self._limits: RemoteServerLimits | None = None
        self._selected_transport = ""

        self._next_request_id = 1
        self._next_slot = 0
        self._last_batch_id = -1
        self._last_sequence_number = -1
        self._inflight_batch_id: int | None = None
        self._inbound: dict[int, DataFrame] = {}
        self._records: dict[int, _SequenceRecord] = {}
        self._generations: dict[int, int] = {}
        self._abandoned_batches: set[int] = set()

    @classmethod
    def from_config(
        cls,
        config: "RemoteDraftConfig",
        identity: VerifierIdentity,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> "RemoteDraftSession":
        return cls(
            config.endpoint,
            identity,
            transport=config.transport,
            failure_policy=config.failure_policy,
            request_timeout_ms=config.request_timeout_ms,
            startup_timeout_ms=config.startup_timeout_ms,
            clock=clock,
        )

    # ------------------------------------------------------------------
    # Negotiated state
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._alive

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def session_epoch(self) -> int:
        return self._session_epoch

    @property
    def selected_transport(self) -> str:
        return self._selected_transport

    @property
    def capabilities(self) -> SpeculatorPlacementCapabilities:
        if self._capabilities is None:
            raise RemoteDraftError("session is not connected")
        return self._capabilities

    @property
    def feature_schema(self) -> TargetFeatureSchema:
        if self._feature_schema is None:
            raise RemoteDraftError("session is not connected")
        return self._feature_schema

    @property
    def limits(self) -> RemoteServerLimits:
        if self._limits is None:
            raise RemoteDraftError("session is not connected")
        return self._limits

    def sequence_state(self, key: SequenceKey) -> SequenceState | None:
        """Current client-side state of ``key``, or None if unknown/stale."""
        record = self._records.get(key.sequence_id)
        if record is None or record.key != key:
            return None
        return record.state

    @property
    def pending_frames(self) -> int:
        """Buffered inbound data frames not yet consumed (diagnostic)."""
        return len(self._inbound)

    # ------------------------------------------------------------------
    # Handshake
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect and run the HELLO handshake; raises on any rejection.

        May be called again after the connection is lost; sequence numbers,
        slots, and batch ids restart with the new connection, and handles
        dispatched on the old one are fenced off at collect time.
        """
        if self._conn is not None:
            raise RemoteDraftError("session is already connected")
        deadline = self._clock() + self._startup_timeout_s
        try:
            self._conn = transport_connect(
                self._endpoint, timeout=self._startup_timeout_s
            )
        except TransportError as e:
            raise RemoteDraftError(
                f"cannot connect to remote draft server at {self._endpoint}: {e}"
            ) from e
        self._reset_connection_state()
        identity = self._identity
        hello = Hello(
            verifier_instance_id=identity.verifier_instance_id,
            target_fingerprint=identity.target_fingerprint,
            tokenizer_fingerprint=identity.tokenizer_fingerprint,
            method=identity.method,
            num_speculative_tokens=identity.num_speculative_tokens,
            parallel_drafting=identity.parallel_drafting,
            supported_transports=self._offered_transports,
            draft_checkpoint_fingerprint=identity.draft_checkpoint_fingerprint,
        )
        try:
            request_id = self._send(hello)
            _, ack = self._await(
                lambda env, msg: (
                    isinstance(msg, HelloAck) and env.request_id == request_id
                ),
                deadline,
            )
        except TransportTimeout:
            self._drop_connection()
            raise RemoteDraftError(
                f"HELLO handshake with {self._endpoint} timed out after "
                f"{self._startup_timeout_s:.1f}s"
            ) from None
        except (TransportError, RemoteDraftError) as e:
            self._drop_connection()
            raise RemoteDraftError(f"HELLO handshake failed: {e}") from e
        reasons = self._handshake_rejections(ack)
        if reasons:
            self._drop_connection()
            raise RemoteDraftError(
                "remote draft placement rejected: " + "; ".join(reasons)
            )
        self._session_id = ack.session_id
        self._session_epoch = ack.session_epoch
        self._capabilities = ack.capabilities
        self._feature_schema = ack.feature_schema
        self._limits = ack.limits
        self._selected_transport = ack.selected_transport
        self._alive = True
        logger.info(
            "Remote draft session %s (epoch %d) established with server %s "
            "over %s; schema %d with %d feature slots",
            ack.session_id,
            ack.session_epoch,
            ack.server_id,
            ack.selected_transport,
            ack.feature_schema.schema_id,
            len(ack.feature_schema.slots),
        )

    def _reset_connection_state(self) -> None:
        """Drop bookkeeping scoped to the previous connection.

        The server numbers proposal responses and reads data slots per
        connection; carrying the old counters across a reconnect would make
        every fresh response look stale and be discarded.
        """
        self._next_request_id = 1
        self._next_slot = 0
        self._last_batch_id = -1
        self._last_sequence_number = -1
        self._inflight_batch_id = None
        self._inbound.clear()
        self._abandoned_batches.clear()

    def _handshake_rejections(self, ack: HelloAck) -> list[str]:
        identity = self._identity
        reasons: list[str] = []
        if ack.selected_transport not in self._offered_transports:
            reasons.append(
                f"server selected transport {ack.selected_transport!r} which "
                f"the verifier did not offer {self._offered_transports}"
            )
        reasons.extend(
            placement_incompatibilities(
                ack.capabilities,
                parallel_drafting=identity.parallel_drafting,
                draft_sample_method=identity.draft_sample_method,
                provided_features=identity.provided_features,
                uses_multi_module=identity.uses_multi_module,
            )
        )
        provided = set(identity.provided_features)
        for slot in ack.feature_schema.slots:
            if not slot.optional and slot.kind not in provided:
                reasons.append(
                    f"feature schema requires {slot.kind!r} which the verifier "
                    "cannot provide"
                )
        return reasons

    # ------------------------------------------------------------------
    # Sequence lifecycle
    # ------------------------------------------------------------------

    def open_sequence(self, sequence_id: int) -> SequenceKey:
        """Register draft state for a request; returns its SequenceKey.

        Re-opening a desynced or invalidated sequence bumps the generation
        so the server discards anything it still holds for the old one.
        """
        record = self._records.get(sequence_id)
        if record is not None and record.state not in (
            SequenceState.DESYNCED,
            SequenceState.INVALID,
        ):
            raise RemoteDraftError(
                f"sequence {sequence_id} is already open ({record.state.value})"
            )
        generation = self._generations.get(sequence_id, -1) + 1
        self._generations[sequence_id] = generation
        key = SequenceKey(
            verifier_instance_id=self._identity.verifier_instance_id,
            sequence_id=sequence_id,
            generation=generation,
        )
        record = _SequenceRecord(key, SequenceState.OPENING)
        self._records[sequence_id] = record
        if not self._alive:
            record.state = SequenceState.INVALID
            record.last_status = SpeculatorStatusCode.INTERNAL_ERROR
            self._raise_if_strict(
                f"cannot open sequence {sequence_id}: session is not connected"
            )
            return key
        status, detail = self._sequence_rpc(OpenSequence(key=key), key)
        if status is SpeculatorStatusCode.OK:
            record.state = transition_sequence(record.state, SequenceState.PREFILLING)
        else:
            if status is SpeculatorStatusCode.QUEUE_FULL:
                # A rejected open left no server state to retry against;
                # latch target-only so prefill and dispatch stay quiet
                # no-ops instead of wedging the sequence in OPENING.
                record.last_status = status
                record.state = transition_sequence(
                    record.state, SequenceState.TARGET_ONLY
                )
            else:
                self._apply_status(record, status)
            self._raise_if_strict(
                f"open_sequence({sequence_id}) failed: {status.name} {detail}"
            )
        return key

    def prefill(
        self, key: SequenceKey, features: FeatureBatch, *, is_final: bool
    ) -> None:
        """Send one prompt chunk; ``is_final`` makes the sequence READY.

        Chunks are applied in order with client-tracked offsets. A sequence
        that is already latched target-only silently ignores prefill so the
        request keeps running without speculation.
        """
        record = self._current_record(key)
        if record.state is not SequenceState.PREFILLING:
            if record.state in (
                SequenceState.TARGET_ONLY,
                SequenceState.INVALID,
                SequenceState.DESYNCED,
            ):
                return
            raise RemoteDraftError(
                f"sequence {key.sequence_id} cannot accept prefill in state "
                f"{record.state.value}"
            )
        schema = self.feature_schema
        limits = self.limits
        num_tokens = self._chunk_length(features)
        if num_tokens > limits.max_feature_tokens:
            raise ValueError(
                f"prefill chunk of {num_tokens} tokens exceeds the negotiated "
                f"max_feature_tokens={limits.max_feature_tokens}"
            )
        if record.prefilled_tokens + num_tokens > limits.max_model_len:
            raise ValueError(
                f"sequence {key.sequence_id} would exceed the server "
                f"max_model_len={limits.max_model_len}"
            )
        base = self._reserve_slots(len(schema.slots))
        frames = self._feature_frames(base, features, num_tokens, rows=None)
        checksum = _checksum(frames)
        chunk = PrefillChunk(
            key=key,
            offset=record.prefilled_tokens,
            num_tokens=num_tokens,
            is_final=is_final,
            feature_slot=base,
            checksum=checksum,
        )
        status, detail = self._sequence_rpc(chunk, key, frames=frames)
        if status is SpeculatorStatusCode.OK:
            record.prefilled_tokens += num_tokens
            if is_final:
                record.state = transition_sequence(record.state, SequenceState.READY)
            return
        self._apply_status(record, status)
        self._raise_if_strict(
            f"prefill of sequence {key.sequence_id} at offset "
            f"{chunk.offset} failed: {status.name} {detail}"
        )

    def close_sequences(self, keys: tuple[SequenceKey, ...]) -> None:
        """Release remote state; unknown, stale, and closed keys are no-ops."""
        pending: list[SequenceKey] = []
        for key in keys:
            record = self._records.get(key.sequence_id)
            if record is None or record.key != key:
                continue
            if record.state in (SequenceState.CLOSING, SequenceState.CLOSED):
                continue
            if record.state is SequenceState.INVALID:
                del self._records[key.sequence_id]
                continue
            record.state = transition_sequence(record.state, SequenceState.CLOSING)
            pending.append(key)
        if pending and self._alive:
            self._close_remote(tuple(pending))
        for key in pending:
            record = self._records.pop(key.sequence_id, None)
            if record is not None and record.state is SequenceState.CLOSING:
                record.state = transition_sequence(record.state, SequenceState.CLOSED)

    def _close_remote(self, keys: tuple[SequenceKey, ...]) -> None:
        deadline = self._clock() + self._request_timeout_s
        remaining = set(keys)
        try:
            request_id = self._send(CloseSequence(keys=keys))
            while remaining:
                _, ack = self._await(
                    lambda env, msg: (
                        isinstance(msg, SequenceAck) and env.request_id == request_id
                    ),
                    deadline,
                )
                remaining.discard(ack.key)
        except TransportTimeout:
            logger.warning(
                "Remote draft server did not acknowledge closing %d sequences "
                "before the deadline; releasing them locally",
                len(remaining),
            )
        except TransportError as e:
            self._invalidate(f"connection lost while closing sequences: {e}")
        except RemoteDraftError as e:
            logger.warning("Remote draft server rejected close: %s", e)

    # ------------------------------------------------------------------
    # Proposal rounds
    # ------------------------------------------------------------------

    def dispatch(self, batch: RemoteProposalBatch) -> RemoteProposalHandle:
        """Send one round to the server without waiting for the result.

        Rows whose sequence is not READY (latched target-only, desynced,
        stale key, unknown) are skipped locally and report
        ``valid_length=0`` at collect time; they generate no wire traffic.

        At most one proposal may be in flight per session: collect the
        previous handle before dispatching the next round.
        """
        if self._inflight_batch_id is not None:
            raise RemoteDraftError(
                f"batch {batch.batch_id} dispatched while batch "
                f"{self._inflight_batch_id} is still in flight; collect it "
                "first"
            )
        if batch.batch_id <= self._last_batch_id:
            raise ValueError(
                f"batch_id {batch.batch_id} is not newer than "
                f"{self._last_batch_id}; batch ids must increase per session"
            )
        self._last_batch_id = batch.batch_id
        batch_size = len(batch.keys)
        if batch.accepted_counts.shape != (batch_size,):
            raise ValueError(
                f"accepted_counts must be [{batch_size}], got "
                f"{tuple(batch.accepted_counts.shape)}"
            )
        if batch.recovery_tokens.shape[:1] != (batch_size,):
            raise ValueError(
                f"recovery_tokens must have {batch_size} rows, got "
                f"{tuple(batch.recovery_tokens.shape)}"
            )
        handle = RemoteProposalHandle(
            session_epoch=self._session_epoch,
            session_id=self._session_id,
            batch_id=batch.batch_id,
            keys=batch.keys,
            active_rows=tuple(
                i for i, key in enumerate(batch.keys) if self._is_ready(key)
            ),
            num_speculative_tokens=self._identity.num_speculative_tokens,
            device=batch.recovery_tokens.device,
            deadline=self._clock() + self._request_timeout_s,
        )
        if not handle.active_rows or not self._alive:
            handle.state = transition_handle(handle.state, HandleState.COMPLETED)
            return handle
        schema = self.feature_schema
        if batch.features.schema_id != schema.schema_id:
            raise ValueError(
                f"features carry schema {batch.features.schema_id}, session "
                f"negotiated schema {schema.schema_id}"
            )
        num_active = len(handle.active_rows)
        if num_active > self.limits.max_batch_size:
            raise RemoteDraftError(
                f"{num_active} active rows exceed the negotiated "
                f"max_batch_size={self.limits.max_batch_size}"
            )
        rows = None
        if num_active != batch_size:
            rows = torch.tensor(handle.active_rows, dtype=torch.long)
        base = self._reserve_slots(2 + len(schema.slots))
        frames = [
            tensor_to_frame(base, _select_rows(batch.accepted_counts, rows)),
            tensor_to_frame(base + 1, _select_rows(batch.recovery_tokens, rows)),
        ]
        frames.extend(self._feature_frames(base + 2, batch.features, batch_size, rows))
        message = AdvanceAndPropose(
            batch_id=batch.batch_id,
            keys=tuple(batch.keys[i] for i in handle.active_rows),
            accepted_counts_slot=base,
            recovery_tokens_slot=base + 1,
            feature_slot=base + 2,
        )
        try:
            handle.request_id = self._send(message, frames=frames)
        except TransportError as e:
            self._invalidate(f"connection lost while dispatching: {e}")
            handle.state = transition_handle(handle.state, HandleState.COMPLETED)
            return handle
        for i in handle.active_rows:
            record = self._records[batch.keys[i].sequence_id]
            record.state = transition_sequence(record.state, SequenceState.IN_FLIGHT)
        handle.input_slot = base
        handle.state = transition_handle(handle.state, HandleState.DISPATCHED)
        self._inflight_batch_id = batch.batch_id
        return handle

    def collect(self, handle: RemoteProposalHandle) -> RemoteProposalResult:
        """Wait for the round dispatched as ``handle``; one-shot.

        Timeouts cancel the batch on the server, latch the affected
        sequences target-only, and return an all-zero output; under
        ``failure_policy="error"`` they raise instead (after the same state
        updates).
        """
        if (
            handle.session_id != self._session_id
            or handle.session_epoch != self._session_epoch
        ):
            raise RemoteDraftError(
                f"handle for batch {handle.batch_id} belongs to session "
                f"{handle.session_id!r} (epoch {handle.session_epoch}); the "
                f"current session is {self._session_id!r} "
                f"(epoch {self._session_epoch})"
            )
        if handle.state is HandleState.COMPLETED:
            # Nothing was dispatched: every row was skipped locally.
            handle.state = transition_handle(handle.state, HandleState.COLLECTED)
            return self._finalize(handle, {}, None, None)
        if handle.state is not HandleState.DISPATCHED:
            raise InvalidStateTransition(
                f"cannot collect a {handle.state.value} handle "
                f"(batch {handle.batch_id})"
            )
        tokens = valid = None
        outcome = HandleState.COMPLETED
        try:
            _, response = self._await(
                lambda env, msg: (
                    isinstance(msg, ProposalResponse)
                    and msg.batch_id == handle.batch_id
                ),
                handle.deadline,
            )
        except TransportTimeout:
            self._cancel(handle)
            outcome = HandleState.TIMED_OUT
            statuses = self._fail_rows(handle, SpeculatorStatusCode.TIMEOUT)
        except TransportError as e:
            self._invalidate(
                f"connection lost while collecting batch {handle.batch_id}: {e}"
            )
            statuses = self._fail_rows(handle, SpeculatorStatusCode.INTERNAL_ERROR)
        except RemoteDraftError as e:
            logger.warning(
                "Batch %d failed with a session-level error: %s", handle.batch_id, e
            )
            statuses = self._fail_rows(handle, SpeculatorStatusCode.INTERNAL_ERROR)
        else:
            statuses, tokens, valid = self._apply_response(handle, response)
        if self._inflight_batch_id == handle.batch_id:
            self._inflight_batch_id = None
        handle.state = transition_handle(handle.state, outcome)
        handle.state = transition_handle(handle.state, HandleState.COLLECTED)
        return self._finalize(handle, statuses, tokens, valid)

    def _finalize(
        self,
        handle: RemoteProposalHandle,
        statuses: dict[int, SpeculatorStatusCode],
        tokens: torch.Tensor | None,
        valid: torch.Tensor | None,
    ) -> RemoteProposalResult:
        result = self._build_result(handle, statuses, tokens, valid)
        failed = [
            (i, status.name)
            for i, status in enumerate(result.row_statuses)
            if status is not SpeculatorStatusCode.OK
        ]
        if failed:
            self._raise_if_strict(f"batch {handle.batch_id} has failed rows: {failed}")
        return result

    def _apply_response(
        self, handle: RemoteProposalHandle, response: ProposalResponse
    ) -> tuple[
        dict[int, SpeculatorStatusCode], torch.Tensor | None, torch.Tensor | None
    ]:
        if response.status is not SpeculatorStatusCode.OK:
            return self._fail_rows(handle, response.status), None, None
        if response.result_slot is None:
            logger.warning("Batch %d succeeded without a result slot", handle.batch_id)
            return (
                self._fail_rows(handle, SpeculatorStatusCode.INTERNAL_ERROR),
                None,
                None,
            )
        frames = self._take_frames(response.result_slot, RESULT_FRAMES)
        if any(frame is None for frame in frames):
            logger.warning(
                "Batch %d response references missing result frames at slot %d",
                handle.batch_id,
                response.result_slot,
            )
            return (
                self._fail_rows(handle, SpeculatorStatusCode.INTERNAL_ERROR),
                None,
                None,
            )
        num_active = len(handle.active_rows)
        expected = (num_active, handle.num_speculative_tokens)
        try:
            tokens = frame_to_tensor(frames[0])
            valid = frame_to_tensor(frames[1])
            status_codes = frame_to_ints(frames[2])
        except TransportError as e:
            logger.warning(
                "Batch %d result frames are malformed: %s", handle.batch_id, e
            )
            return (
                self._fail_rows(handle, SpeculatorStatusCode.INTERNAL_ERROR),
                None,
                None,
            )
        if (
            tuple(tokens.shape) != expected
            or tuple(valid.shape) != (num_active,)
            or len(status_codes) != num_active
            or tokens.is_floating_point()
            or valid.is_floating_point()
        ):
            logger.warning(
                "Batch %d result has unexpected layout: tokens %s, valid %s, "
                "%d statuses for %d rows",
                handle.batch_id,
                tuple(tokens.shape),
                tuple(valid.shape),
                len(status_codes),
                num_active,
            )
            return (
                self._fail_rows(handle, SpeculatorStatusCode.INTERNAL_ERROR),
                None,
                None,
            )
        statuses: dict[int, SpeculatorStatusCode] = {}
        for row, code in zip(handle.active_rows, status_codes):
            status = _status_from_wire(code)
            statuses[row] = status
            record = self._records.get(handle.keys[row].sequence_id)
            if record is not None and record.key == handle.keys[row]:
                self._apply_status(record, status)
        return statuses, tokens, valid

    def _build_result(
        self,
        handle: RemoteProposalHandle,
        statuses: dict[int, SpeculatorStatusCode],
        tokens: torch.Tensor | None,
        valid: torch.Tensor | None,
    ) -> RemoteProposalResult:
        batch_size = len(handle.keys)
        num_tokens = handle.num_speculative_tokens
        token_ids = torch.zeros(
            batch_size, num_tokens, dtype=torch.int64, device=handle.device
        )
        valid_lengths = torch.zeros(batch_size, dtype=torch.int32, device=handle.device)
        row_statuses = tuple(
            statuses[i] if i in statuses else self._skip_status(key)
            for i, key in enumerate(handle.keys)
        )
        if tokens is not None and valid is not None:
            rows = torch.tensor(handle.active_rows, dtype=torch.long)
            ok = torch.tensor(
                [statuses[i] is SpeculatorStatusCode.OK for i in handle.active_rows]
            )
            valid = torch.where(ok, valid.to(torch.int32), 0).clamp_(0, num_tokens)
            token_ids.index_copy_(
                0, rows.to(handle.device), tokens.to(handle.device, torch.int64)
            )
            valid_lengths.index_copy_(
                0, rows.to(handle.device), valid.to(handle.device)
            )
        return RemoteProposalResult(
            output=SpeculatorOutput(token_ids, valid_lengths),
            row_statuses=row_statuses,
        )

    def _fail_rows(
        self, handle: RemoteProposalHandle, status: SpeculatorStatusCode
    ) -> dict[int, SpeculatorStatusCode]:
        for i in handle.active_rows:
            key = handle.keys[i]
            record = self._records.get(key.sequence_id)
            if record is not None and record.key == key:
                self._apply_status(record, status)
        return dict.fromkeys(handle.active_rows, status)

    def _cancel(self, handle: RemoteProposalHandle) -> None:
        self._abandoned_batches.add(handle.batch_id)
        try:
            self._send(CancelBatch(batch_id=handle.batch_id))
        except TransportError as e:
            self._invalidate(f"connection lost while cancelling: {e}")

    # ------------------------------------------------------------------
    # Health and teardown
    # ------------------------------------------------------------------

    def ping(self) -> Pong:
        """Round-trip liveness probe."""
        nonce = self._next_request_id
        deadline = self._clock() + self._request_timeout_s
        try:
            self._send(Ping(nonce=nonce))
            _, pong = self._await(
                lambda env, msg: isinstance(msg, Pong) and msg.nonce == nonce,
                deadline,
            )
        except TransportTimeout:
            raise RemoteDraftError("ping timed out") from None
        except TransportError as e:
            self._invalidate(f"connection lost during ping: {e}")
            raise RemoteDraftError(f"ping failed: {e}") from e
        return pong

    def close(self) -> None:
        """Close the connection; every sequence becomes INVALID."""
        self._invalidate("session closed")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _send(self, message: Any, *, frames: list[DataFrame] | None = None) -> int:
        conn = self._require_connection()
        request_id = self._next_request_id
        self._next_request_id += 1
        for frame in frames or ():
            conn.send_data(frame)
        conn.send_control(
            encode_message(message, session_id=self._session_id, request_id=request_id)
        )
        return request_id

    def _await(
        self,
        matches: Callable[[MessageEnvelope, Any], bool],
        deadline: float,
    ) -> tuple[MessageEnvelope, Any]:
        """Read frames until a control message satisfies ``matches``.

        Data frames are buffered by slot. Stale proposal responses are
        discarded together with their result frames so a later round can
        never read them by accident.
        """
        conn = self._require_connection()
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise TransportTimeout("deadline elapsed")
            try:
                frame = conn.recv(timeout=remaining)
            except TransportTimeout:
                continue
            if frame.kind == FRAME_DATA:
                data = decode_data_frame(frame.body)
                self._inbound[data.slot] = data
                continue
            try:
                envelope = decode_envelope(frame.body)
                message = decode_payload(envelope)
            except ProtocolError as e:
                self._invalidate(f"protocol error from server: {e}")
                raise ConnectionClosed(f"protocol error from server: {e}") from e
            if isinstance(message, ErrorReply):
                raise RemoteDraftError(
                    f"server error {message.status.name}: {message.detail}"
                )
            if isinstance(message, ProposalResponse) and not self._accept_response(
                message
            ):
                continue
            if matches(envelope, message):
                return envelope, message
            logger.debug(
                "Ignoring unexpected %s (request %d)",
                type(message).__name__,
                envelope.request_id,
            )

    def _accept_response(self, response: ProposalResponse) -> bool:
        if response.sequence_number <= self._last_sequence_number:
            self._purge_result(response)
            return False
        self._last_sequence_number = response.sequence_number
        if response.batch_id in self._abandoned_batches:
            self._abandoned_batches.discard(response.batch_id)
            self._purge_result(response)
            return False
        return True

    def _purge_result(self, response: ProposalResponse) -> None:
        if response.result_slot is not None:
            self._take_frames(response.result_slot, RESULT_FRAMES)

    def _take_frames(self, base: int, count: int) -> list[DataFrame | None]:
        return [self._inbound.pop(base + i, None) for i in range(count)]

    def _sequence_rpc(
        self,
        message: Any,
        key: SequenceKey,
        *,
        frames: list[DataFrame] | None = None,
    ) -> tuple[SpeculatorStatusCode, str]:
        deadline = self._clock() + self._request_timeout_s
        try:
            request_id = self._send(message, frames=frames)
            _, ack = self._await(
                lambda env, msg: (
                    isinstance(msg, SequenceAck)
                    and env.request_id == request_id
                    and msg.key == key
                ),
                deadline,
            )
        except TransportTimeout:
            return SpeculatorStatusCode.TIMEOUT, "no acknowledgement before deadline"
        except TransportError as e:
            self._invalidate(f"connection lost: {e}")
            return SpeculatorStatusCode.INTERNAL_ERROR, str(e)
        except RemoteDraftError as e:
            return SpeculatorStatusCode.INTERNAL_ERROR, str(e)
        return ack.status, ack.detail

    def _apply_status(
        self, record: _SequenceRecord, status: SpeculatorStatusCode
    ) -> None:
        record.last_status = status
        if status in _RETRYABLE_STATUSES:
            # Nothing on the server changed for a rejected-as-busy row, so
            # it may be proposed again next round.
            if record.state is SequenceState.IN_FLIGHT:
                record.state = SequenceState.READY
            return
        if status in _DESYNC_STATUSES and can_transition_sequence(
            record.state, SequenceState.DESYNCED
        ):
            record.state = SequenceState.DESYNCED
            return
        if can_transition_sequence(record.state, SequenceState.TARGET_ONLY):
            record.state = SequenceState.TARGET_ONLY

    def _is_ready(self, key: SequenceKey) -> bool:
        record = self._records.get(key.sequence_id)
        return (
            record is not None
            and record.key == key
            and record.state is SequenceState.READY
        )

    def _skip_status(self, key: SequenceKey) -> SpeculatorStatusCode:
        record = self._records.get(key.sequence_id)
        if record is None:
            return SpeculatorStatusCode.SEQUENCE_DESYNC
        if record.key != key:
            return SpeculatorStatusCode.STALE_GENERATION
        if record.last_status is not SpeculatorStatusCode.OK:
            return record.last_status
        return SpeculatorStatusCode.SEQUENCE_DESYNC

    def _current_record(self, key: SequenceKey) -> _SequenceRecord:
        record = self._records.get(key.sequence_id)
        if record is None:
            raise ValueError(f"sequence {key.sequence_id} is not open")
        if record.key != key:
            raise ValueError(
                f"key generation {key.generation} is not current for sequence "
                f"{key.sequence_id} (current generation {record.key.generation})"
            )
        return record

    def _feature_frames(
        self,
        base: int,
        features: FeatureBatch,
        expected_rows: int,
        rows: torch.Tensor | None,
    ) -> list[DataFrame]:
        schema = self.feature_schema
        if len(features.slots) != len(schema.slots):
            raise ValueError(
                f"features carry {len(features.slots)} slots, schema "
                f"{schema.schema_id} has {len(schema.slots)}"
            )
        frames: list[DataFrame] = []
        for i, (spec, tensor) in enumerate(zip(schema.slots, features.slots)):
            if tensor is None:
                if not spec.optional:
                    raise ValueError(
                        f"required feature slot {i} ({spec.kind}) is missing"
                    )
                continue
            if tensor.shape[0] != expected_rows:
                raise ValueError(
                    f"feature slot {i} ({spec.kind}) has {tensor.shape[0]} rows, "
                    f"expected {expected_rows}"
                )
            frames.append(tensor_to_frame(base + i, _select_rows(tensor, rows)))
        return frames

    def _chunk_length(self, features: FeatureBatch) -> int:
        lengths = {t.shape[0] for t in features.slots if t is not None}
        if len(lengths) != 1:
            raise ValueError(
                "prefill features must all share one token dimension, got "
                f"{sorted(lengths)}"
            )
        return lengths.pop()

    def _reserve_slots(self, count: int) -> int:
        base = self._next_slot
        self._next_slot += count
        return base

    def _require_connection(self) -> FramedConnection:
        if self._conn is None or self._conn.closed:
            raise ConnectionClosed("session is not connected")
        return self._conn

    def _raise_if_strict(self, detail: str) -> None:
        if self._failure_policy == "error":
            raise RemoteDraftError(detail)
        logger.warning("Remote draft falling back to target-only: %s", detail)

    def _invalidate(self, reason: str) -> None:
        if not self._alive and self._conn is None:
            return
        if self._alive:
            logger.warning(
                "Remote draft session %s invalidated: %s", self._session_id, reason
            )
        self._alive = False
        for record in self._records.values():
            if can_transition_sequence(record.state, SequenceState.INVALID):
                record.state = SequenceState.INVALID
                record.last_status = SpeculatorStatusCode.INTERNAL_ERROR
        self._inflight_batch_id = None
        self._inbound.clear()
        self._drop_connection()

    def _drop_connection(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()


def _select_rows(tensor: torch.Tensor, rows: torch.Tensor | None) -> torch.Tensor:
    if rows is None:
        return tensor
    return tensor.index_select(0, rows.to(tensor.device))


def _checksum(frames: list[DataFrame]) -> int:
    value = 0
    for frame in frames:
        value = zlib.crc32(frame.payload, value)
    return value & 0x7FFFFFFF


def _status_from_wire(code: int) -> SpeculatorStatusCode:
    try:
        return SpeculatorStatusCode(code)
    except ValueError:
        return SpeculatorStatusCode.INTERNAL_ERROR
