# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-owner v3 token RPCs, with one outstanding proposal and no replay."""

import itertools
import math
import time
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.spec_decode.proposal import ProposalHandle
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
from vllm.v1.spec_decode.remote.runtime.token_state import TokenAdvanceRow
from vllm.v1.spec_decode.remote.session import RemoteDraftError
from vllm.v1.spec_decode.remote.token_protocol import (
    TokenAdvance,
    TokenHello,
    TokenHelloAck,
    TokenPrefill,
    TokenProposal,
    decode_token_message,
    encode_token_message,
)
from vllm.v1.spec_decode.remote.transport import (
    FRAME_CONTROL,
    FRAME_DATA,
    MAX_CONTROL_FRAME_BYTES,
    MAX_HANDSHAKE_BYTES,
    DataFrame,
    FramedConnection,
    TransportTimeout,
    check_data_frame_size,
    connect,
    decode_data_frame,
    frame_checksum,
    frame_to_ints,
    ints_to_frame,
    is_local_endpoint,
)

OK = SpeculatorStatusCode.OK
logger = init_logger(__name__)


@dataclass
class TokenClientProgress:
    committed: int = 0
    proposal_id: int | None = None
    proposal_tokens: tuple[int, ...] = ()
    terminal: bool = False


@dataclass
class TokenProposalHandle(ProposalHandle):
    owner: object
    epoch: int
    request_id: int
    batch_id: int
    requested_k: int
    rows: tuple[TokenAdvanceRow, ...]
    deadline: float
    active: bool = True


@dataclass(frozen=True)
class TokenBatchResult:
    token_ids: tuple[tuple[int, ...], ...]
    valid_lengths: tuple[int, ...]


class TokenOnlyRemoteSession:
    def __init__(
        self,
        endpoint: str,
        hello: TokenHello,
        *,
        failure_policy="error",
        request_timeout_ms=100,
        startup_timeout_ms=30000,
        clock=time.monotonic,
    ):
        if not is_local_endpoint(endpoint) or hello.transport != "inline":
            raise ValueError("token-only v3 requires same-host inline transport")
        if failure_policy not in ("error", "target_only"):
            raise ValueError("invalid token-only failure policy")
        for name, value in (
            ("request_timeout_ms", request_timeout_ms),
            ("startup_timeout_ms", startup_timeout_ms),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.endpoint, self.hello = endpoint, hello
        self.failure_policy = failure_policy
        self.request_timeout_s = request_timeout_ms / 1000
        self.startup_timeout_s = startup_timeout_ms / 1000
        self.clock = clock
        self._reconnect = ReconnectBackoff(clock)
        self.records: dict[SequenceKey, TokenClientProgress] = {}
        self._owner = object()
        self._conn: FramedConnection | None = None
        self._sid = ""
        self._epoch = 0
        self._ids = itertools.count(1)
        self._generations = itertools.count()
        self._pending: TokenProposalHandle | None = None
        self._deferred_closes: set[SequenceKey] = set()
        self._closed = False
        self._last_batch = -1

    def deadline(self) -> float:
        return self.clock() + self.request_timeout_s

    def _invalidate(self) -> None:
        connection, self._conn = self._conn, None
        for state in self.records.values():
            state.terminal = True
        for key in self._deferred_closes:
            self.records.pop(key, None)
        self._deferred_closes.clear()
        if connection is not None:
            try:
                connection.close()
            except OSError as error:
                logger.warning("Token-only connection cleanup failed: %s", error)

    def fail(self, detail: str) -> None:
        self._invalidate()
        if self.failure_policy == "error":
            raise RemoteDraftError(detail)

    def _send(self, message, request_id, deadline, frames=()) -> None:
        assert self._conn is not None
        for frame in frames:
            self._conn.send_data(frame, deadline=deadline)
        if isinstance(message, (TokenPrefill, TokenAdvance)):
            message.execution_timeout_ms = max(0, int((deadline - self.clock()) * 1000))
        self._conn.send_control(
            encode_token_message(message, session_id=self._sid, request_id=request_id),
            deadline=deadline,
        )

    def _recv(self, request_id, deadline, *, handshake=False):
        assert self._conn is not None
        frames: dict[int, DataFrame] = {}
        slot = (1 << 64) - 1
        rows = self.hello.max_num_seqs
        sizes = [check_data_frame_size(slot, "int64", (rows,))] * 4
        sizes.append(
            check_data_frame_size(
                slot, "int64", (rows, self.hello.num_speculative_tokens)
            )
        )
        remaining_bytes = 0 if handshake else sum(sizes)
        max_frame = 0 if handshake else max(sizes)
        control_limit = MAX_HANDSHAKE_BYTES if handshake else MAX_CONTROL_FRAME_BYTES
        while True:
            self._conn.set_receive_limits(
                max_control_bytes=control_limit,
                max_data_bytes=min(max_frame, remaining_bytes),
                max_buffered_bytes=control_limit + remaining_bytes + 30,
            )
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise TransportTimeout("token result deadline expired")
            frame = self._conn.recv(timeout=remaining)
            if frame.kind != FRAME_CONTROL:
                if frame.kind != FRAME_DATA:
                    raise ProtocolError("unexpected token result frame kind")
                data = decode_data_frame(frame.body)
                if data.slot in frames or data.cuda_ipc is not None or len(frames) >= 5:
                    raise ProtocolError("invalid token result frame")
                remaining_bytes -= len(frame.body)
                if remaining_bytes < 0:
                    raise ProtocolError("token result exceeds receive byte budget")
                frames[data.slot] = data
                continue
            envelope, message = decode_token_message(frame.body)
            if isinstance(message, ErrorReply):
                raise RemoteDraftError(message.detail)
            if envelope.request_id != request_id or (
                not handshake and envelope.session_id != self._sid
            ):
                raise ProtocolError("token response request/session mismatch")
            if (
                handshake
                and isinstance(message, TokenHelloAck)
                and (
                    not message.session_id or envelope.session_id != message.session_id
                )
            ):
                raise ProtocolError("token hello session identity mismatch")
            return message, frames

    def connect(self, *, deadline=None) -> None:
        if self._closed:
            raise RuntimeError("token session is closed")
        if self._conn is not None:
            return
        if not self._reconnect.ready():
            raise RemoteDraftError("token reconnect is in backoff")
        if deadline is None:
            deadline = self.clock() + self.startup_timeout_s
        try:
            self._conn = connect(self.endpoint, timeout=max(0, deadline - self.clock()))
            self._sid = ""
            request_id = next(self._ids)
            self._send(self.hello, request_id, deadline)
            ack, frames = self._recv(request_id, deadline, handshake=True)
            if (
                not isinstance(ack, TokenHelloAck)
                or frames
                or ack.tokenizer_fingerprint != self.hello.tokenizer_fingerprint
            ):
                raise ProtocolError("invalid token hello acknowledgment")
            if (
                ack.limits.max_batch_size < self.hello.max_num_seqs
                or ack.limits.max_feature_tokens < self.hello.max_num_batched_tokens
                or ack.limits.max_model_len < self.hello.max_model_len
            ):
                raise ProtocolError("token server limits are too small")
            self._sid, self._epoch = ack.session_id, ack.session_epoch
            self._reconnect.succeeded()
        except Exception:
            self._reconnect.failed()
            self._invalidate()
            raise

    def active(self, key: SequenceKey) -> bool:
        return (
            key in self.records
            and not self.records[key].terminal
            and key not in self._deferred_closes
        )

    def open_sequence(self, sequence_id: int, *, deadline: float) -> SequenceKey:
        key = SequenceKey(
            self.hello.verifier_instance_id, sequence_id, next(self._generations)
        )
        self.records[key] = TokenClientProgress()
        try:
            self.connect(deadline=deadline)
            request_id = next(self._ids)
            self._send(OpenSequence(key), request_id, deadline)
            message, frames = self._recv(request_id, deadline)
            self._ack(message, frames, key)
        except Exception as error:
            self.fail(f"token open failed: {error}")
        return key

    def _ack(self, message, frames, key):
        if not isinstance(message, SequenceAck) or message.key != key or frames:
            raise ProtocolError("invalid token lifecycle acknowledgment")
        if message.status is not OK:
            self.records[key].terminal = True
            if self.failure_policy == "error":
                raise RemoteDraftError(message.detail or message.status.name)

    def prefill(
        self,
        key: SequenceKey,
        tokens: tuple[int, ...],
        *,
        is_final: bool,
        deadline: float,
    ) -> None:
        if not self.active(key):
            return
        state = self.records[key]
        try:
            if not tokens:
                raise ValueError("token prefill must not be empty")
            capacity = self.hello.max_num_batched_tokens
            for start in range(0, len(tokens), capacity):
                end = min(start + capacity, len(tokens))
                check_data_frame_size(0, "int32", (end - start,))
                frame = ints_to_frame(0, tokens[start:end])
                message = TokenPrefill(
                    key,
                    state.committed,
                    end - start,
                    0,
                    frame_checksum([frame]),
                    is_final and end == len(tokens),
                )
                request_id = next(self._ids)
                self._send(message, request_id, deadline, (frame,))
                ack, frames = self._recv(request_id, deadline)
                self._ack(ack, frames, key)
                if state.terminal:
                    return
                state.committed += end - start
        except Exception as error:
            self.fail(f"token prefill failed: {error}")

    def dispatch(
        self,
        batch_id: int,
        rows: tuple[TokenAdvanceRow, ...],
        k: int,
        *,
        deadline: float,
    ) -> TokenProposalHandle:
        if self._pending is not None:
            raise RuntimeError("token proposal is already in flight")
        if (
            batch_id <= self._last_batch
            or not 0 <= k <= self.hello.num_speculative_tokens
        ):
            raise ValueError("invalid token proposal identity/budget")
        if len(set(row.key for row in rows)) != len(rows):
            raise ValueError("duplicate token proposal rows")
        self._last_batch = batch_id
        request_id = next(self._ids)
        handle = TokenProposalHandle(
            self._owner, self._epoch, request_id, batch_id, k, rows, deadline
        )
        self._pending = handle
        if not rows:
            return handle
        try:
            if any(not self.active(row.key) for row in rows):
                raise ProtocolError("terminal token request cannot resume")
            arrays = (
                [row.base_length for row in rows],
                [
                    -1 if row.previous_proposal_id is None else row.previous_proposal_id
                    for row in rows
                ],
                [row.offered for row in rows],
                [row.accepted for row in rows],
                [len(row.delta) for row in rows],
                [token for row in rows for token in row.delta],
            )
            frames = [
                ints_to_frame(i, values, dtype="int64")
                for i, values in enumerate(arrays)
            ]
            self._send(
                TokenAdvance(
                    batch_id, tuple(row.key for row in rows), 0, 1, 2, 3, 4, 5, k
                ),
                request_id,
                deadline,
                frames,
            )
        except Exception as error:
            try:
                self.fail(f"token dispatch failed: {error}")
            except BaseException:
                # A caller cannot collect or cancel a handle dispatch never
                # returned. Target-only still returns a collectable empty result.
                handle.active = False
                self._pending = None
                raise
        return handle

    def collect(self, handle: TokenProposalHandle) -> TokenBatchResult:
        if (
            handle.owner is not self._owner
            or self._pending is not handle
            or not handle.active
            or handle.epoch != self._epoch
        ):
            raise RuntimeError("stale, foreign or collected token handle")
        empty = None
        try:
            n, capacity = len(handle.rows), self.hello.num_speculative_tokens
            empty = TokenBatchResult(
                tuple((0,) * capacity for _ in handle.rows), (0,) * n
            )
            if not n or self._conn is None:
                return empty
            response, frames = self._recv(handle.request_id, handle.deadline)
            if (
                not isinstance(response, TokenProposal)
                or response.batch_id != handle.batch_id
                or response.status is not OK
                or response.result_slot is None
            ):
                raise ProtocolError("invalid token proposal response")
            slot = response.result_slot
            if set(frames) != set(range(slot, slot + 5)):
                raise ProtocolError("incomplete token proposal result")
            values = [frame_to_ints(frames[slot + i]) for i in range(5)]
            if frames[slot].shape != (n, capacity) or any(
                frames[slot + i].shape != (n,) for i in range(1, 5)
            ):
                raise ProtocolError("token proposal result shape mismatch")
            tokens, valid, committed, identities, statuses = values
            output = []
            for i, row in enumerate(handle.rows):
                status = SpeculatorStatusCode(statuses[i])
                state = self.records[row.key]
                row_tokens = tuple(tokens[i * capacity : (i + 1) * capacity])
                remaining = max(
                    0, self.hello.max_model_len - row.base_length - len(row.delta)
                )
                if not 0 <= valid[i] <= min(handle.requested_k, remaining) or any(
                    not 0 <= t < self.hello.vocab_size for t in row_tokens[: valid[i]]
                ):
                    raise ProtocolError("invalid token proposal values")
                if status is not OK:
                    state.terminal = True
                    valid[i] = 0
                    if self.failure_policy == "error":
                        raise RemoteDraftError(f"token proposal failed: {status.name}")
                else:
                    if identities[i] != handle.batch_id or committed[
                        i
                    ] != row.base_length + len(row.delta):
                        raise ProtocolError(
                            "token proposal committed-prefix acknowledgment mismatch"
                        )
                    state.committed, state.proposal_id = committed[i], identities[i]
                    state.proposal_tokens = row_tokens[: valid[i]]
                output.append(row_tokens)
            return TokenBatchResult(tuple(output), tuple(valid))
        except Exception as error:
            self.fail(f"token collect failed: {error}")
            if empty is None:
                raise
            return empty
        finally:
            handle.active = False
            self._pending = None
            if self._deferred_closes:
                keys = tuple(self._deferred_closes)
                self._deferred_closes.clear()
                self.close_sequences(keys, deadline=handle.deadline)

    def cancel(self, handle: TokenProposalHandle) -> None:
        if (
            handle.owner is not self._owner
            or self._pending is not handle
            or not handle.active
        ):
            raise RuntimeError("cannot cancel a stale token handle")
        handle.active = False
        self._pending = None
        self._invalidate()

    def close_sequences(
        self, keys: tuple[SequenceKey, ...], *, deadline: float | None = None
    ) -> None:
        keys = tuple(dict.fromkeys(key for key in keys if key in self.records))
        if not keys:
            return
        if self._pending is not None:
            if not any(row.key in keys for row in self._pending.rows):
                self._deferred_closes.update(keys)
                return
            self.cancel(self._pending)
        live = tuple(key for key in keys if self.active(key))
        try:
            if live and self._conn is not None:
                if deadline is None:
                    deadline = self.deadline()
                request_id = next(self._ids)
                self._send(CloseSequence(live), request_id, deadline)
                for key in live:
                    ack, frames = self._recv(request_id, deadline)
                    self._ack(ack, frames, key)
        except Exception as error:
            self.fail(f"token close failed: {error}")
        finally:
            for key in keys:
                self.records.pop(key, None)

    def close(self) -> None:
        self._closed = True
        if self._pending is not None:
            self._pending.active = False
            self._pending = None
        self._invalidate()
        self.records.clear()
