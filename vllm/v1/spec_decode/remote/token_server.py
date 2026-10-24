# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-host v3 token server using the existing framed transport/lifecycle."""

import threading
import time
import uuid
from dataclasses import dataclass

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
from vllm.v1.spec_decode.remote.runtime.token_state import (
    TokenAdvanceRow,
    TokenRowResult,
)
from vllm.v1.spec_decode.remote.runtime.work_queue import DraftWork, DraftWorkQueue
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


@dataclass(frozen=True)
class _AdvanceInput:
    session: "_TokenServerSession"
    batch_id: int
    rows: tuple[TokenAdvanceRow, ...]
    k: int
    deadline: float | None


class TokenDraftServer(RemoteDraftServer):
    adapter: TokenOnlyDraftAdapter

    def __init__(
        self,
        adapter: TokenOnlyDraftAdapter,
        *,
        max_pending_jobs: int = 64,
        max_queued_token_bytes: int = 64 << 20,
        max_jobs_per_verifier: int = 8,
        max_token_bytes_per_verifier: int = 16 << 20,
        max_sequences_per_verifier: int | None = None,
        batch_wait_ms: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(adapter, **kwargs)
        if max_sequences_per_verifier is None:
            max_sequences_per_verifier = self.limits.max_sequences
        if (
            type(max_sequences_per_verifier) is not int
            or not 0 < max_sequences_per_verifier <= self.limits.max_sequences
        ):
            raise ValueError("per-verifier sequence credit must fit the server limit")
        self.max_sequences_per_verifier = max_sequences_per_verifier
        self.work_queue = DraftWorkQueue(
            self._execute_advances,
            max_jobs=max_pending_jobs,
            max_bytes=max_queued_token_bytes,
            max_jobs_per_owner=max_jobs_per_verifier,
            max_bytes_per_owner=max_token_bytes_per_verifier,
            max_cleanup_jobs=self.max_connections,
            max_batch_rows=self.limits.max_batch_size,
            max_batch_tokens=self.limits.max_feature_tokens,
            batch_wait_s=batch_wait_ms / 1000,
        )

    @property
    def queue_depth(self) -> int:
        return max(super().queue_depth, self.work_queue.depth)

    def _warmup_owner(self) -> None:
        device = getattr(self.adapter.state.executor, "device", None)
        if device is not None:
            import torch

            if device.type != "cpu":
                torch.accelerator.set_device_index(device)
                # The constructor may enqueue weight/KV initialization elsewhere.
                torch.accelerator.synchronize(device)
        self.adapter.warmup()

    def stop(self) -> None:
        deadline = time.monotonic() + self.shutdown_timeout_s
        try:
            super().stop()
        finally:
            try:
                self.work_queue.stop(deadline)
            except ServerShutdownTimeout:
                self._shutdown_failed = True
                raise

    def _execute_advances(self, work: tuple[DraftWork, ...]) -> None:
        with self.adapter_turn():
            updates, deadlines, locations = [], [], []
            results: list[list[tuple[SpeculatorStatusCode, TokenRowResult | None]]] = []
            for job in work:
                item = job.payload
                rows: list[tuple[SpeculatorStatusCode, TokenRowResult | None]] = []
                for row in item.rows:
                    if not self.healthy:
                        status = SpeculatorStatusCode.INTERNAL_ERROR
                    elif item.session._expire_owned_sequences(
                        item.deadline, (row.key,)
                    ):
                        status = SpeculatorStatusCode.TIMEOUT
                    else:
                        status, _ = self.registry.begin_round(
                            row.key, item.batch_id, owner=item.session.session_id
                        )
                    rows.append((status, None))
                    if status is OK:
                        locations.append((len(results), len(rows) - 1))
                        updates.append((item.batch_id, row, item.k))
                        deadlines.append(item.deadline)
                results.append(rows)
            if updates:
                outcomes = self.adapter.state.advance_many(
                    tuple(updates), deadlines=tuple(deadlines)
                )
                if len(outcomes) != len(updates):
                    raise RuntimeError("token executor returned an invalid row count")
                for (batch, index), outcome in zip(locations, outcomes):
                    if isinstance(outcome, Exception):
                        self._adapter_failed(outcome)
                        status = (
                            outcome.status
                            if isinstance(outcome, AdapterError)
                            else SpeculatorStatusCode.INTERNAL_ERROR
                        )
                        logger.error(
                            "Remote token batch %d row failed: %s",
                            work[batch].payload.batch_id,
                            outcome,
                        )
                        results[batch][index] = (status, None)
                    else:
                        results[batch][index] = (OK, outcome)
            for job, rows in zip(work, results):
                item = job.payload
                for row, (status, result) in zip(item.rows, rows):
                    if status is not OK:
                        item.session._discard_owned_sequences((row.key,))
                    else:
                        entry = self.registry.get(row.key)
                        assert entry is not None and result is not None
                        entry.prefix_length = result.committed
                job.result = rows

    def start(self, endpoint: str) -> str:
        if self._listener is not None or not self.healthy:
            raise RuntimeError("token runtime is already started or unhealthy")
        if not is_local_endpoint(endpoint):
            raise ValueError("token-only placement currently requires the same host")
        try:
            self.work_queue.start(self._warmup_owner)
        except Exception as error:
            self._adapter_failed(error)
            raise
        try:
            self._listener = listen(endpoint)
        except BaseException:
            self.work_queue.stop(time.monotonic() + self.shutdown_timeout_s)
            raise
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
        except AdapterQueueFull:
            conn.close()
        except (
            TransportError,
            ProtocolError,
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

    def cleanup(self) -> None:
        self.server.work_queue.call(
            self.verifier_instance_id,
            super().cleanup,
            cleanup=True,
        )

    def _open(self, message: OpenSequence, request_id: int) -> None:
        key = message.key
        if not self._owns(key):
            self._ack(
                key,
                request_id,
                SpeculatorStatusCode.PROTOCOL_ERROR,
                "key belongs to another verifier",
            )
            return

        def open_sequence():
            server = self.server
            with server.adapter_turn():
                registry = server.registry
                if not server.healthy:
                    return SpeculatorStatusCode.INTERNAL_ERROR, "runtime is unhealthy"
                if not registry.has_identity(key) and (
                    len(registry) >= server.limits.max_sequences
                    or registry.count(self.verifier_instance_id)
                    >= server.max_sequences_per_verifier
                ):
                    return SpeculatorStatusCode.QUEUE_FULL, "sequence credit exhausted"
                replaced = registry.replaced_by(key)
                if replaced is not None:
                    server._retire_sequences((replaced,))
                outcome = registry.open(key, owner=self.session_id)
                if outcome.created:
                    failure = self._call_adapter(server.adapter.open_sequence, key)
                    if failure is not None:
                        self._discard_owned_sequences((key,))
                        return failure
                return outcome.status, outcome.detail

        status, detail = self.server.work_queue.call(
            self.verifier_instance_id, open_sequence
        )
        self._ack(key, request_id, status, detail)

    def _close(self, message: CloseSequence, request_id: int) -> None:
        def close_sequences():
            with self.server.adapter_turn(cleanup=True):
                self._discard_owned_sequences(
                    tuple(key for key in message.keys if self._owns(key))
                )

        self.server.work_queue.call(
            self.verifier_instance_id, close_sequences, cleanup=True
        )
        deadline = time.monotonic() + self.server.send_timeout_s
        for key in message.keys:
            status = OK if self._owns(key) else SpeculatorStatusCode.PROTOCOL_ERROR
            self._ack(key, request_id, status, deadline=deadline)

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

        def prefill():
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
                            message.offset,
                            message.num_tokens,
                            message.is_final,
                            (frame,),
                        ),
                    )
                    if failure is not None:
                        status, detail = failure
                if status is not OK:
                    self._discard_owned_sequences((message.key,))
            return status, detail

        status, detail = self.server.work_queue.call(
            self.verifier_instance_id, prefill, num_bytes=len(frame.payload)
        )
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
        rows = []
        offset = 0
        for i, key in enumerate(keys):
            rows.append(
                TokenAdvanceRow(
                    key,
                    base[i],
                    None if previous[i] == -1 else previous[i],
                    offered[i],
                    accepted[i],
                    tuple(tokens[offset : offset + lengths[i]]),
                )
            )
            offset += lengths[i]
        work = DraftWork(
            self.verifier_instance_id,
            None,
            _AdvanceInput(
                self,
                message.batch_id,
                tuple(rows),
                message.num_speculative_tokens,
                deadline,
            ),
            num_bytes=sum(len(frame.payload) for frame in frames),
            num_rows=n,
            num_tokens=len(tokens),
        )
        outcomes = self.server.work_queue.submit(work).wait()
        statuses, valid, committed, identities = [], [], [], []
        drafts: list[int] = []
        for status, result in outcomes:
            row_tokens = result.token_ids if result is not None else ()
            drafts.extend(
                row_tokens + (0,) * (self.num_speculative_tokens - len(row_tokens))
            )
            statuses.append(int(status))
            valid.append(len(row_tokens))
            committed.append(result.committed if result is not None else -1)
            identities.append(result.proposal_id if result is not None else -1)
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
