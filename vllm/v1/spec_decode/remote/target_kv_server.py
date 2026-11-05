# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded v4 snapshot server with one active connection and GPU owner."""

import threading
import time
import uuid
from collections.abc import Mapping

import torch

from vllm.logger import init_logger
from vllm.v1.spec_decode.remote.protocol import (
    CloseSequence,
    OpenSequence,
    Ping,
    Pong,
    ProtocolError,
    RemoteServerLimits,
    SequenceAck,
    SpeculatorStatusCode,
)
from vllm.v1.spec_decode.remote.server import RemoteDraftServer
from vllm.v1.spec_decode.remote.session import frame_to_tensor
from vllm.v1.spec_decode.remote.state import SequenceState
from vllm.v1.spec_decode.remote.target_kv import (
    AttentionSpec,
    SnapshotBudget,
    SnapshotCompletionError,
    SnapshotIdentity,
    SnapshotLayer,
    TargetKVSnapshot,
)
from vllm.v1.spec_decode.remote.target_kv_protocol import (
    CONTROL_BYTES,
    FRAME_METADATA_BYTES,
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
    FRAME_DATA,
    FramedConnection,
    TransportError,
    TransportTimeout,
    decode_data_frame,
    is_local_endpoint,
    validate_frame,
)

logger = init_logger(__name__)


class TargetKVDraftServer(RemoteDraftServer):
    def __init__(
        self,
        adapter,
        *,
        contract_fingerprint: str,
        hidden_size: int,
        vocab_size: int,
        expected_specs: Mapping[str, AttentionSpec],
        snapshot_budget: SnapshotBudget,
        receiver_budget: SnapshotBudget,
        max_num_seqs: int = 1,
        **kwargs,
    ):
        if (
            not contract_fingerprint
            or not 1 <= max_num_seqs <= MAX_SNAPSHOT_BATCH_SIZE
            or min(hidden_size, vocab_size) < 1
            or not 1 <= snapshot_budget.max_model_len <= 8192
            or snapshot_budget.max_model_len != receiver_budget.max_model_len
        ):
            raise ValueError("invalid target-KV server contract")
        snapshot_wire_plan(expected_specs, 1, snapshot_budget, receiver_budget)
        super().__init__(
            adapter,
            limits=RemoteServerLimits(
                max_num_seqs, max_num_seqs, max_num_seqs, snapshot_budget.max_model_len
            ),
            **kwargs,
        )
        self.contract_fingerprint = contract_fingerprint
        self.hidden_size, self.vocab_size = hidden_size, vocab_size
        self.expected_specs = dict(expected_specs)
        self.snapshot_budget, self.receiver_budget = snapshot_budget, receiver_budget
        self.max_num_seqs = max_num_seqs
        self._connection_turn = threading.Lock()

    def _adapter_failed(self, error):
        if isinstance(error, SnapshotCompletionError):
            self._unhealthy.set()
            self._stopping.set()
        super()._adapter_failed(error)

    def start(self, endpoint: str) -> str:
        if self._listener is not None or not self.healthy:
            raise RuntimeError("target-KV runtime is already started or unhealthy")
        if not is_local_endpoint(endpoint):
            raise ValueError("target-KV snapshots require same-host placement")
        try:
            self.adapter.warmup()
        except Exception as error:
            self._adapter_failed(error)
            raise
        return self._start_listener(endpoint, name="target-kv-acceptor")

    def _serve_connection(self, conn: FramedConnection):
        if not self._connection_turn.acquire(blocking=False):
            try:
                conn.close()
            except BaseException as error:
                self._lifecycle_failed(error)
                logger.warning("Target-KV connection cleanup failed: %s", error)
            else:
                with self._connections_lock:
                    self._connections.discard(conn)
            finally:
                with self._connections_lock:
                    self._handshaking.discard(conn)
            return
        session: _TargetKVServerSession | None = None
        try:
            session = _TargetKVServerSession(self, conn)
            session.run()
        except (TransportError, ProtocolError, ValueError):
            pass
        except Exception as error:
            self._adapter_failed(error)
            logger.warning("Target-KV session failed: %s", error)
        finally:
            try:
                if session is not None and not self._shutdown_failed:
                    with self.adapter_turn(cleanup=True):
                        self.registry.close_all(session.session_id)
            finally:
                try:
                    conn.close()
                except BaseException as error:
                    self._lifecycle_failed(error)
                    logger.warning("Target-KV connection cleanup failed: %s", error)
                else:
                    with self._connections_lock:
                        self._connections.discard(conn)
                finally:
                    with self._connections_lock:
                        self._handshaking.discard(conn)
                    self._connection_turn.release()


class _TargetKVServerSession:
    def __init__(self, server, conn):
        self.server, self.conn = server, conn
        self.session_id = ""
        self.epoch = 0
        self.verifier_id = ""
        self.keys = set()
        self.max_num_seqs = 1
        self.last_generation = -1
        self.last_round = -1
        self.last_request = -1
        self.budget = server.snapshot_budget

    def _send(self, message, request_id):
        self.conn.send_control(
            encode_target_kv_message(
                message, session_id=self.session_id, request_id=request_id
            ),
            deadline=time.monotonic() + self.server.send_timeout_s,
        )

    def _control(self, *, timeout=None):
        self.conn.set_receive_limits(
            max_control_bytes=CONTROL_BYTES,
            max_data_bytes=0,
            max_buffered_bytes=CONTROL_BYTES + 5,
            partial_frame_timeout=self.server.partial_frame_timeout_s,
        )
        frame = self.conn.recv(timeout=timeout, max_frame_bytes=CONTROL_BYTES)
        if frame.kind != FRAME_CONTROL:
            raise ProtocolError("target-KV requires a control manifest before data")
        envelope, message = decode_target_kv_message(frame.body)
        if envelope.request_id <= self.last_request:
            raise ProtocolError("target-KV request identity did not advance")
        self.last_request = envelope.request_id
        return envelope, message

    def run(self):
        envelope, hello = self._control(timeout=self.server.handshake_timeout_s)
        server = self.server
        if (
            not isinstance(hello, TargetKVHello)
            or envelope.session_id
            or not hello.verifier_instance_id
            or hello.contract_fingerprint != server.contract_fingerprint
            or hello.hidden_size != server.hidden_size
            or hello.vocab_size != server.vocab_size
            or hello.max_model_len != server.snapshot_budget.max_model_len
            or hello.max_snapshot_pages > server.snapshot_budget.max_pages
            or hello.max_snapshot_bytes > server.snapshot_budget.max_bytes
            or hello.transport != "inline"
            or hello.num_speculative_tokens != 1
            or not 1 <= hello.max_num_seqs <= server.max_num_seqs
        ):
            raise ProtocolError("target-KV handshake contract/capacity mismatch")
        self.budget = SnapshotBudget(
            hello.max_snapshot_pages, hello.max_snapshot_bytes, hello.max_model_len
        )
        self.session_id, self.epoch = uuid.uuid4().hex, server.next_epoch()
        self.verifier_id = hello.verifier_instance_id
        self.max_num_seqs = hello.max_num_seqs
        self._send(
            TargetKVHelloAck(
                self.session_id,
                self.epoch,
                server.contract_fingerprint,
                self.budget.max_pages,
                self.budget.max_bytes,
                self.max_num_seqs,
            ),
            envelope.request_id,
        )
        server._handshake_complete(self.conn)
        while not server._stopping.is_set():
            envelope, message = self._control()
            if envelope.session_id != self.session_id:
                raise ProtocolError("target-KV session identity mismatch")
            if isinstance(message, OpenSequence):
                key = message.key
                if (
                    key.verifier_instance_id != self.verifier_id
                    or key.generation <= self.last_generation
                ):
                    raise ProtocolError(
                        "stale or foreign target-KV sequence generation"
                    )
                with server.adapter_turn():
                    if not server.healthy:
                        raise ProtocolError("target-KV runtime is unhealthy")
                    if self.max_num_seqs == 1:
                        server.registry.close_all(self.session_id)
                        self.keys.clear()
                    elif len(self.keys) >= self.max_num_seqs:
                        raise ProtocolError("target-KV sequence capacity exceeded")
                    elif any(k.sequence_id == key.sequence_id for k in self.keys):
                        raise ProtocolError("target-KV sequence ID is already open")
                    server.registry.open(key, owner=self.session_id)
                    entry = server.registry.get(key)
                    assert entry is not None
                    entry.state = SequenceState.READY
                    self.last_generation = key.generation
                    self.keys.add(key)
                self._send(
                    SequenceAck(key, SpeculatorStatusCode.OK), envelope.request_id
                )
            elif isinstance(message, TargetKVQuery):
                self._query(message, envelope.request_id)
            elif isinstance(message, TargetKVBatchQuery):
                if self.max_num_seqs == 1:
                    raise ProtocolError("target-KV batch support was not negotiated")
                self._queries(
                    message.queries, envelope.request_id, message.budget_ms, batch=True
                )
            elif isinstance(message, CloseSequence):
                if len(message.keys) > self.max_num_seqs or any(
                    key.verifier_instance_id != self.verifier_id for key in message.keys
                ):
                    raise ProtocolError("invalid target-KV close identities")
                with server.adapter_turn():
                    for key in message.keys:
                        server.registry.close(key, owner=self.session_id)
                        self.keys.discard(key)
                for key in message.keys:
                    self._send(
                        SequenceAck(key, SpeculatorStatusCode.OK), envelope.request_id
                    )
            elif isinstance(message, Ping):
                self._send(
                    Pong(message.nonce, healthy=server.healthy), envelope.request_id
                )
            else:
                raise ProtocolError("unsupported target-KV command")

    def _data(self, slot, shape, nbytes, deadline):
        self.conn.set_receive_limits(
            max_control_bytes=CONTROL_BYTES,
            max_data_bytes=nbytes + FRAME_METADATA_BYTES,
            max_buffered_bytes=nbytes + FRAME_METADATA_BYTES + 5,
            partial_frame_timeout=self.server.partial_frame_timeout_s,
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TransportTimeout("target-KV snapshot receive budget expired")
        frame = self.conn.recv(
            timeout=remaining, max_frame_bytes=nbytes + FRAME_METADATA_BYTES
        )
        if frame.kind != FRAME_DATA:
            raise ProtocolError("missing target-KV snapshot frame")
        data = decode_data_frame(frame.body)
        if (
            data.slot != slot
            or data.dtype != "bfloat16"
            or data.shape != shape
            or data.cuda_ipc is not None
        ):
            raise ProtocolError("target-KV snapshot frame differs from contract")
        validate_frame(data)
        return frame_to_tensor(data)

    def _query(self, message: TargetKVQuery, request_id: int):
        self._queries((message,), request_id, message.budget_ms, batch=False)

    def _queries(
        self,
        queries: tuple[TargetKVQuery, ...],
        request_id: int,
        budget_ms: int,
        *,
        batch: bool,
    ):
        server = self.server
        if (
            not 1 <= len(queries) <= self.max_num_seqs
            or len({q.key for q in queries}) != len(queries)
            or any(
                q.key not in self.keys
                or q.session_epoch != self.epoch
                or q.round_id <= self.last_round
                or q.budget_ms != budget_ms
                or not 0 <= q.input_token < server.vocab_size
                for q in queries
            )
        ):
            raise ProtocolError("stale or invalid target-KV query")
        plans = snapshot_batch_wire_plan(
            server.expected_specs,
            tuple(q.materialized_length for q in queries),
            self.budget,
            server.receiver_budget,
        )
        self.last_round = max(q.round_id for q in queries)
        identities = tuple(
            SnapshotIdentity(q.key, self.epoch, q.round_id) for q in queries
        )
        deadline = time.monotonic() + min(budget_ms / 1000, server.send_timeout_s)
        snapshots, hidden_rows = [], []
        slot = 0
        for query, identity, row_plans in zip(queries, identities, plans):
            hidden_rows.append(
                self._data(
                    slot, (1, server.hidden_size), 2 * server.hidden_size, deadline
                )
            )
            slot += 1
            layers = []
            for plan in row_plans:
                layers.append(
                    SnapshotLayer(
                        plan.name,
                        plan.spec,
                        plan.logical_pages,
                        self._data(slot, plan.shape, plan.nbytes, deadline),
                    )
                )
                slot += 1
            snapshots.append(
                TargetKVSnapshot(identity, query.materialized_length, tuple(layers))
            )
        hidden = torch.cat(hidden_rows) if batch else hidden_rows[0]
        status, detail = SpeculatorStatusCode.OK, ""
        tokens: tuple[int | None, ...] = (None,) * len(queries)
        with server.adapter_turn():
            if time.monotonic() >= deadline:
                raise TransportTimeout("target-KV execution budget expired in queue")
            if not server.healthy:
                raise ProtocolError("target-KV runtime is unhealthy")
            try:
                if batch:
                    tokens = server.adapter.execute_batch(
                        tuple(snapshots),
                        expected_identities=identities,
                        expected_materialized_lengths=tuple(
                            q.materialized_length for q in queries
                        ),
                        recovery_tokens=tuple(q.input_token for q in queries),
                        hidden_states=hidden,
                    )
                else:
                    tokens = (
                        server.adapter.execute(
                            snapshots[0],
                            expected_identity=identities[0],
                            expected_materialized_length=queries[0].materialized_length,
                            recovery_token=queries[0].input_token,
                            hidden_state=hidden,
                        ),
                    )
                if len(tokens) != len(queries) or any(
                    not isinstance(token, int) or not 0 <= token < server.vocab_size
                    for token in tokens
                ):
                    raise ValueError("target-KV executor returned invalid tokens")
            except Exception as error:
                server._adapter_failed(error)
                status = (
                    SpeculatorStatusCode.OUT_OF_MEMORY
                    if isinstance(error, torch.OutOfMemoryError)
                    else SpeculatorStatusCode.INTERNAL_ERROR
                )
                detail = f"{type(error).__name__}: {error}".encode()[
                    : min(1024, CONTROL_BYTES // (2 * len(queries)))
                ].decode("utf-8", errors="ignore")
                tokens = (None,) * len(queries)
                for query in queries:
                    server.registry.close(query.key, owner=self.session_id)
                    self.keys.discard(query.key)
        # The executor fences reads or quarantines ownership before returning.
        del snapshots, layers, hidden_rows, hidden
        replies = tuple(
            TargetKVReply(
                q.key,
                self.epoch,
                q.round_id,
                q.materialized_length,
                token,
                status,
                detail,
            )
            for q, token in zip(queries, tokens)
        )
        self._send(TargetKVBatchReply(replies) if batch else replies[0], request_id)
