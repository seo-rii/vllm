# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-host v3 token server using the existing framed transport/lifecycle."""

import threading
import time
import uuid

from vllm.logger import init_logger
from vllm.v1.spec_decode.remote.protocol import (
    CloseSequence,
    ErrorReply,
    OpenSequence,
    Ping,
    Pong,
    ProtocolError,
    SpeculatorStatusCode,
    limit_shortfalls,
)
from vllm.v1.spec_decode.remote.runtime.token_adapter import TokenOnlyDraftAdapter
from vllm.v1.spec_decode.remote.runtime.token_state import TokenAdvanceRow
from vllm.v1.spec_decode.remote.server import (
    AdapterError,
    AdapterQueueFull,
    PrefillInputs,
    RemoteDraftServer,
    SequenceCleanupFailed,
    ServerShutdownTimeout,
    _ServerSession,
)
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
    MAX_HANDSHAKE_BYTES,
    FramedConnection,
    TransportError,
    check_data_frame_size,
    frame_checksum,
    frame_to_ints,
    ints_to_frame,
    is_local_endpoint,
    listen,
)

OK = SpeculatorStatusCode.OK
logger = init_logger(__name__)


class TokenDraftServer(RemoteDraftServer):
    adapter: TokenOnlyDraftAdapter

    def start(self, endpoint: str) -> str:
        if self._listener is not None or not self.healthy:
            raise RuntimeError("token runtime is already started or unhealthy")
        if not is_local_endpoint(endpoint):
            raise ValueError("token-only placement currently requires the same host")
        try:
            self.adapter.warmup()
        except Exception as error:
            self._adapter_failed(error)
            raise
        self._listener = listen(endpoint)
        self._stopping.clear()
        self._acceptor = threading.Thread(
            target=self._accept_loop, name="token-draft-acceptor", daemon=True
        )
        self._acceptor.start()
        return self._listener.endpoint

    def _serve_connection(self, conn: FramedConnection) -> None:
        session: _TokenServerSession | None = None
        try:
            session = _TokenServerSession(self, conn)
            session.run()
        except (
            TransportError,
            ProtocolError,
            AdapterQueueFull,
            ServerShutdownTimeout,
            SequenceCleanupFailed,
        ):
            pass
        except Exception:
            logger.exception(
                "Remote token draft session %s crashed",
                getattr(session, "session_id", "<initializing>"),
            )
        finally:
            try:
                if (
                    session is not None
                    and not self._shutdown_failed
                    and not self._cleanup_failed
                ):
                    session.cleanup()
            except SequenceCleanupFailed:
                pass
            except Exception:
                logger.exception(
                    "Failed to clean up remote token draft session %s",
                    getattr(session, "session_id", "<initializing>"),
                )
            finally:
                conn.close()
                with self._connections_lock:
                    self._connections.discard(conn)
                    self._handshaking.discard(conn)


class _TokenServerSession(_ServerSession):
    server: TokenDraftServer

    def _send(self, message, request_id, *, deadline=None, frames=None):
        if deadline is None:
            deadline = time.monotonic() + self.server.send_timeout_s
        for frame in frames or ():
            self.conn.send_data(frame, deadline=deadline)
        self.conn.send_control(
            encode_token_message(
                message, session_id=self.session_id, request_id=request_id
            ),
            deadline=deadline,
        )

    def _handshake(self) -> bool:
        self.conn.set_receive_limits(
            max_control_bytes=MAX_HANDSHAKE_BYTES,
            max_data_bytes=0,
            max_buffered_bytes=MAX_HANDSHAKE_BYTES + 5,
        )
        deadline = time.monotonic() + self.server.handshake_timeout_s
        frame = self.conn.recv(timeout=max(0, deadline - time.monotonic()))
        if frame.kind != FRAME_CONTROL:
            return False
        try:
            envelope, hello = decode_token_message(frame.body)
            if not isinstance(hello, TokenHello):
                raise ProtocolError("expected v3 token hello")
            contract = self.server.adapter.contract
            state = self.server.adapter.state
            reasons = limit_shortfalls(hello, self.server.limits)
            if (
                hello.method != "draft_model"
                or hello.transport != "inline"
                or hello.target_fingerprint != contract.target_fingerprint
                or hello.draft_fingerprint != contract.draft_fingerprint
                or hello.tokenizer_fingerprint != contract.tokenizer_fingerprint
                or hello.vocab_size != state.vocab_size
                or hello.num_speculative_tokens != state.max_k
                or hello.max_model_len != state.max_model_len
                or self.server.limits.max_model_len != state.max_model_len
            ):
                reasons.append(
                    "token-only checkpoint/tokenizer/method contract mismatch"
                )
            if reasons:
                raise ProtocolError("; ".join(reasons))
        except ProtocolError as error:
            self._send(
                ErrorReply(SpeculatorStatusCode.UNSUPPORTED_FEATURE, str(error)),
                0,
                deadline=deadline,
            )
            return False
        self.session_id = uuid.uuid4().hex
        self.session_epoch = self.server.next_epoch()
        self.verifier_instance_id = hello.verifier_instance_id
        self.num_speculative_tokens = hello.num_speculative_tokens
        self.selected_transport = "inline"
        self.max_pending_frames = 8
        slot = (1 << 64) - 1
        sizes = [check_data_frame_size(slot, "int64", (hello.max_num_seqs,))] * 5
        sizes.append(
            check_data_frame_size(slot, "int64", (hello.max_num_batched_tokens,))
        )
        self.max_data_frame_bytes = max(sizes)
        self.max_pending_bytes = sum(sizes)
        if self.max_pending_bytes > self.server.max_pending_bytes:
            raise ProtocolError("token input bytes exceed the server receive budget")
        self._set_input_budget()
        self._send(
            TokenHelloAck(
                self.session_id,
                self.session_epoch,
                contract.tokenizer_fingerprint,
                self.server.limits,
            ),
            envelope.request_id,
            deadline=deadline,
        )
        return True

    def _handle_control(self, body: bytes) -> None:
        envelope, message = decode_token_message(body)
        if envelope.session_id != self.session_id:
            raise ProtocolError("token session identity mismatch")
        request_id = envelope.request_id
        if isinstance(message, OpenSequence):
            self._open(message, request_id)
        elif isinstance(message, CloseSequence):
            self._close(message, request_id)
        elif isinstance(message, TokenPrefill):
            self._prefill_tokens(message, request_id)
        elif isinstance(message, TokenAdvance):
            self._advance_tokens(message, request_id)
        elif isinstance(message, Ping):
            self._send(
                Pong(
                    message.nonce,
                    queue_depth=self.server.queue_depth,
                    active_sequences=self.server.registry.count(
                        self.verifier_instance_id
                    ),
                    healthy=self.server.healthy,
                ),
                request_id,
            )
        else:
            raise ProtocolError("unexpected token command")

    def _prefill_tokens(self, message: TokenPrefill, request_id: int) -> None:
        budget = message.execution_timeout_ms
        deadline = None if budget is None else time.monotonic() + budget / 1000
        frame = self.inbound.pop(message.token_slot, None)
        if (
            not self._owns(message.key)
            or frame is None
            or frame.shape != (message.num_tokens,)
        ):
            raise ProtocolError("invalid token prefill frame/key")
        if message.num_tokens > self.server.limits.max_feature_tokens:
            raise ProtocolError("token prefill exceeds the batch token budget")
        frame_to_ints(frame)
        if frame_checksum([frame]) != message.checksum:
            raise ProtocolError("token prefill checksum mismatch")
        with self.server.adapter_turn():
            if not self.server.healthy:
                status, detail, apply = (
                    SpeculatorStatusCode.INTERNAL_ERROR,
                    "unhealthy runtime",
                    False,
                )
            elif self._expire_owned_sequences(deadline, (message.key,)):
                status, detail, apply = (
                    SpeculatorStatusCode.TIMEOUT,
                    "execution budget expired before token prefill",
                    False,
                )
            else:
                status, detail, apply = self.server.registry.prefill(
                    message.key,
                    owner=self.session_id,
                    offset=message.offset,
                    num_tokens=message.num_tokens,
                    checksum=message.checksum,
                    is_final=message.is_final,
                )
            if apply:
                failure = self._call_adapter(
                    self.server.adapter.prefill,
                    message.key,
                    PrefillInputs(
                        message.offset, message.num_tokens, message.is_final, (frame,)
                    ),
                )
                if failure is not None:
                    status, detail = failure
            if status is not OK:
                self._discard_owned_sequences((message.key,))
        self._ack(message.key, request_id, status, detail)

    def _advance_tokens(self, message: TokenAdvance, request_id: int) -> None:
        budget = message.execution_timeout_ms
        deadline = None if budget is None else time.monotonic() + budget / 1000
        if message.batch_id <= self.last_batch_id:
            raise ProtocolError("token batch identity did not advance")
        self.last_batch_id = message.batch_id
        keys, n = message.keys, len(message.keys)
        if (
            not 0 < n <= self.server.limits.max_batch_size
            or len(set(keys)) != n
            or any(not self._owns(key) for key in keys)
        ):
            raise ProtocolError("invalid token batch keys")
        slots = (
            message.base_lengths_slot,
            message.previous_ids_slot,
            message.offered_slot,
            message.accepted_slot,
            message.delta_lengths_slot,
            message.token_slot,
        )
        if len(set(slots)) != len(slots):
            raise ProtocolError("token frame slots overlap")
        frames = []
        for slot in slots:
            frame = self.inbound.pop(slot, None)
            if frame is None:
                raise ProtocolError("missing token batch metadata")
            frames.append(frame)
        arrays = [frame_to_ints(frame) for frame in frames]
        if any(frame.shape != (n,) for frame in frames[:-1]) or frames[-1].shape != (
            len(arrays[-1]),
        ):
            raise ProtocolError("token batch metadata shape mismatch")
        base, previous, offered, accepted, lengths, tokens = arrays
        if (
            any(length < 1 for length in lengths)
            or sum(lengths) != len(tokens)
            or len(tokens) > self.server.limits.max_feature_tokens
        ):
            raise ProtocolError("token delta lengths exceed the token budget")
        if not 0 <= message.num_speculative_tokens <= self.num_speculative_tokens:
            raise ProtocolError("token draft budget exceeds the negotiated maximum")
        if any(
            value < 0 for array in (base, offered, accepted) for value in array
        ) or any(value < -1 for value in previous):
            raise ProtocolError("negative token state metadata")
        statuses = [SpeculatorStatusCode.INTERNAL_ERROR] * n
        results = [None] * n
        valid, committed, identities = [], [], []
        drafts: list[int] = []
        with self.server.adapter_turn():
            updates = []
            selected = []
            offset = 0
            for i, key in enumerate(keys):
                if not self.server.healthy:
                    status = SpeculatorStatusCode.INTERNAL_ERROR
                elif self._expire_owned_sequences(deadline, (key,)):
                    status = SpeculatorStatusCode.TIMEOUT
                else:
                    status, _ = self.server.registry.begin_round(
                        key, message.batch_id, owner=self.session_id
                    )
                statuses[i] = status
                if status is OK:
                    selected.append(i)
                    updates.append(
                        (
                            message.batch_id,
                            TokenAdvanceRow(
                                key,
                                base[i],
                                None if previous[i] == -1 else previous[i],
                                offered[i],
                                accepted[i],
                                tuple(tokens[offset : offset + lengths[i]]),
                            ),
                            message.num_speculative_tokens,
                        )
                    )
                offset += lengths[i]
            if updates:
                outcomes = self.server.adapter.state.advance_many(
                    tuple(updates), deadlines=(deadline,) * len(updates)
                )
                for i, outcome in zip(selected, outcomes):
                    if isinstance(outcome, Exception):
                        self.server._adapter_failed(outcome)
                        statuses[i] = (
                            outcome.status
                            if isinstance(outcome, AdapterError)
                            else SpeculatorStatusCode.INTERNAL_ERROR
                        )
                    else:
                        results[i] = outcome
            for key, status, result in zip(keys, statuses, results):
                if status is not OK:
                    self._discard_owned_sequences((key,))
                elif result is not None:
                    entry = self.server.registry.get(key)
                    assert entry is not None
                    entry.prefix_length = result.committed
                row_tokens = result.token_ids if result is not None else ()
                drafts.extend(
                    row_tokens + (0,) * (self.num_speculative_tokens - len(row_tokens))
                )
                valid.append(len(row_tokens))
                committed.append(result.committed if result is not None else -1)
                identities.append(result.proposal_id if result is not None else -1)
        statuses = [int(status) for status in statuses]
        slot = self.next_result_slot
        self.next_result_slot += 5
        self._send(
            TokenProposal(message.batch_id, slot),
            request_id,
            frames=[
                ints_to_frame(
                    slot, drafts, dtype="int64", shape=(n, self.num_speculative_tokens)
                ),
                ints_to_frame(slot + 1, valid),
                ints_to_frame(slot + 2, committed, dtype="int64"),
                ints_to_frame(slot + 3, identities, dtype="int64"),
                ints_to_frame(slot + 4, statuses),
            ],
        )
