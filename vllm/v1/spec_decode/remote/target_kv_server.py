# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded v4 snapshots with a shared GPU owner and verifier admission."""

import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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
from vllm.v1.spec_decode.remote.runtime.work_queue import DraftWork, DraftWorkQueue
from vllm.v1.spec_decode.remote.server import (
    AdapterQueueFull,
    RemoteDraftServer,
    ServerShutdownTimeout,
)
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


@dataclass(frozen=True)
class _SnapshotInput:
    session: "_TargetKVServerSession"
    queries: tuple[TargetKVQuery, ...]
    snapshots: tuple[TargetKVSnapshot, ...]
    hidden: tuple[torch.Tensor, ...]
    deadline: float
    batch: bool


class TargetKVDraftServer(RemoteDraftServer):
    adapter: Any

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
        max_connections: int = 1,
        max_sequences: int | None = None,
        max_sequences_per_verifier: int | None = None,
        max_pending_jobs: int = 64,
        max_queued_snapshot_bytes: int = 128 << 20,
        max_jobs_per_verifier: int = 8,
        max_snapshot_bytes_per_verifier: int = 128 << 20,
        batch_wait_ms: float = 1.0,
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
        max_sequences = max_num_seqs if max_sequences is None else max_sequences
        max_sequences_per_verifier = (
            max_num_seqs
            if max_sequences_per_verifier is None
            else max_sequences_per_verifier
        )
        if (
            type(max_sequences) is not int
            or max_sequences < max_num_seqs
            or type(max_sequences_per_verifier) is not int
            or not 0 < max_sequences_per_verifier <= max_sequences
        ):
            raise ValueError("invalid target-KV sequence credits")
        super().__init__(
            adapter,
            limits=RemoteServerLimits(
                max_num_seqs, max_num_seqs, max_sequences, snapshot_budget.max_model_len
            ),
            max_connections=max_connections,
            **kwargs,
        )
        self.contract_fingerprint = contract_fingerprint
        self.hidden_size, self.vocab_size = hidden_size, vocab_size
        self.expected_specs = dict(expected_specs)
        self.snapshot_budget, self.receiver_budget = snapshot_budget, receiver_budget
        self.max_num_seqs = max_num_seqs
        self.max_sequences_per_verifier = max_sequences_per_verifier
        self._admission_lock = threading.Lock()
        self._verifiers: dict[str, str] = {}
        self._snapshot_credits: dict[str, tuple[int, int]] = {}
        self._snapshot_jobs = self._snapshot_bytes = 0
        self.work_queue = DraftWorkQueue(
            self._execute_snapshots,
            max_jobs=max_pending_jobs,
            max_bytes=max_queued_snapshot_bytes,
            max_jobs_per_owner=max_jobs_per_verifier,
            max_bytes_per_owner=max_snapshot_bytes_per_verifier,
            max_cleanup_jobs=self.max_connections,
            max_batch_rows=max_num_seqs,
            max_batch_tokens=max_num_seqs,
            batch_wait_s=batch_wait_ms / 1000,
            on_failure=self._lifecycle_failed,
        )

    @property
    def pending_snapshot_bytes(self) -> int:
        with self._admission_lock:
            return self._snapshot_bytes

    @property
    def pending_snapshot_jobs(self) -> int:
        with self._admission_lock:
            return self._snapshot_jobs

    @property
    def queue_depth(self) -> int:
        return max(super().queue_depth, self.work_queue.depth)

    def _reserve_snapshot(self, owner: str, nbytes: int) -> None:
        with self._admission_lock:
            jobs, size = self._snapshot_credits.get(owner, (0, 0))
            queue = self.work_queue
            if (
                not self.healthy
                or self._stopping.is_set()
                or self._snapshot_jobs >= queue.max_jobs
                or self._snapshot_bytes + nbytes > queue.max_bytes
                or jobs >= queue.max_jobs_per_owner
                or size + nbytes > queue.max_bytes_per_owner
            ):
                raise AdapterQueueFull("target-KV snapshot admission credit exhausted")
            self._snapshot_credits[owner] = jobs + 1, size + nbytes
            self._snapshot_jobs += 1
            self._snapshot_bytes += nbytes

    def _release_snapshot(self, owner: str, nbytes: int) -> None:
        with self._admission_lock:
            jobs, size = self._snapshot_credits[owner]
            if jobs == 1:
                del self._snapshot_credits[owner]
            else:
                self._snapshot_credits[owner] = jobs - 1, size - nbytes
            self._snapshot_jobs -= 1
            self._snapshot_bytes -= nbytes

    def _warmup_owner(self) -> None:
        device = getattr(self.adapter, "device", None)
        if device is not None and device.type != "cpu":
            torch.accelerator.set_device_index(device)
            torch.accelerator.synchronize(device)
        self.adapter.warmup()

    def stop(self, *, deadline: float | None = None) -> None:
        self._stop_with_owner(self.work_queue.stop, deadline=deadline)

    def _failed_job(self, job: DraftWork, status, detail: str) -> None:
        item = job.payload
        detail = detail.encode()[
            : min(1024, CONTROL_BYTES // (2 * len(item.queries)))
        ].decode("utf-8", errors="ignore")
        for query in item.queries:
            self.registry.close(query.key, owner=item.session.session_id)
            item.session.keys.discard(query.key)
        job.result = (None,) * len(item.queries), status, detail

    def _execute_group(self, jobs: tuple[DraftWork, ...]) -> None:
        current = []
        for job in jobs:
            if not self.healthy:
                self._failed_job(
                    job, SpeculatorStatusCode.INTERNAL_ERROR, "runtime is unhealthy"
                )
            elif time.monotonic() >= job.payload.deadline:
                self._failed_job(
                    job, SpeculatorStatusCode.TIMEOUT, "snapshot expired in queue"
                )
            else:
                current.append(job)
        if not current:
            return
        queries = tuple(q for job in current for q in job.payload.queries)
        snapshots = tuple(s for job in current for s in job.payload.snapshots)
        identities = tuple(snapshot.identity for snapshot in snapshots)
        try:
            hidden = torch.cat(tuple(h for job in current for h in job.payload.hidden))
            if len(current) == 1 and not current[0].payload.batch:
                tokens = (
                    self.adapter.execute(
                        snapshots[0],
                        expected_identity=identities[0],
                        expected_materialized_length=queries[0].materialized_length,
                        recovery_token=queries[0].input_token,
                        hidden_state=hidden,
                    ),
                )
            else:
                tokens = self.adapter.execute_batch(
                    snapshots,
                    expected_identities=identities,
                    expected_materialized_lengths=tuple(
                        q.materialized_length for q in queries
                    ),
                    recovery_tokens=tuple(q.input_token for q in queries),
                    hidden_states=hidden,
                )
            if len(tokens) != len(queries) or any(
                not isinstance(token, int) or not 0 <= token < self.vocab_size
                for token in tokens
            ):
                raise ValueError("target-KV executor returned invalid tokens")
        except Exception as error:
            self._adapter_failed(error)
            status = (
                SpeculatorStatusCode.OUT_OF_MEMORY
                if isinstance(error, torch.OutOfMemoryError)
                else SpeculatorStatusCode.INTERNAL_ERROR
            )
            logger.warning("Target-KV batch failed: %s", error)
            for job in current:
                self._failed_job(job, status, f"{type(error).__name__}: {error}")
            return
        offset = 0
        for job in current:
            end = offset + len(job.payload.queries)
            if time.monotonic() >= job.payload.deadline:
                self._failed_job(
                    job,
                    SpeculatorStatusCode.TIMEOUT,
                    "snapshot execution exceeded deadline",
                )
            else:
                job.result = tuple(tokens[offset:end]), SpeculatorStatusCode.OK, ""
            offset = end

    def _execute_snapshots(self, jobs: tuple[DraftWork, ...]) -> None:
        try:
            with self.adapter_turn():
                group: list[DraftWork] = []
                lengths: tuple[int, ...] = ()
                for job in jobs:
                    candidate = lengths + tuple(
                        q.materialized_length for q in job.payload.queries
                    )
                    try:
                        snapshot_batch_wire_plan(
                            self.expected_specs,
                            candidate,
                            self.snapshot_budget,
                            self.receiver_budget,
                        )
                    except ValueError:
                        # Each job was admitted independently; a rectangular combined
                        # receiver may need more space than those independent plans.
                        self._execute_group(tuple(group))
                        group = []
                        candidate = tuple(
                            q.materialized_length for q in job.payload.queries
                        )
                    group.append(job)
                    lengths = candidate
                self._execute_group(tuple(group))
        finally:
            # Completed work must not keep payloads alive after the handler
            # releases its resident-byte reservation.
            for job in jobs:
                job.payload = None

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
            self.work_queue.start(self._warmup_owner)
        except Exception as error:
            self._adapter_failed(error)
            raise
        return self._start_listener(endpoint, name="target-kv-acceptor")

    def _serve_connection(self, conn: FramedConnection):
        session: _TargetKVServerSession | None = None
        try:
            session = _TargetKVServerSession(self, conn)
            session.run()
        except (
            TransportError,
            ProtocolError,
            ValueError,
            AdapterQueueFull,
            ServerShutdownTimeout,
        ):
            pass
        except Exception as error:
            self._adapter_failed(error)
            logger.warning("Target-KV session failed: %s", error)
        finally:
            try:
                conn.close()
            except OSError as error:
                logger.warning("Target-KV connection cleanup failed: %s", error)
            finally:
                try:
                    if session is not None and not self._shutdown_failed:
                        session.cleanup()
                except ServerShutdownTimeout:
                    self._shutdown_failed = True
                finally:
                    with self._connections_lock:
                        self._connections.discard(conn)
                        self._handshaking.discard(conn)
                    if session is not None:
                        with self._admission_lock:
                            if (
                                self._verifiers.get(session.verifier_id)
                                == session.session_id
                            ):
                                del self._verifiers[session.verifier_id]


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

    def cleanup(self):
        def release():
            with self.server.adapter_turn(cleanup=True):
                self.server.registry.close_all(self.session_id)
                self.keys.clear()

        self.server.work_queue.call(self.verifier_id, release, cleanup=True)

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
        with server._admission_lock:
            if self.verifier_id in server._verifiers:
                raise ProtocolError("target-KV verifier is already connected")
            server._verifiers[self.verifier_id] = self.session_id
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

                def open_sequence(key=key):
                    with server.adapter_turn():
                        if not server.healthy:
                            return (
                                SpeculatorStatusCode.INTERNAL_ERROR,
                                "runtime is unhealthy",
                            )
                        if self.max_num_seqs == 1:
                            server.registry.close_all(self.session_id)
                            self.keys.clear()
                        if (
                            len(self.keys) >= self.max_num_seqs
                            or len(server.registry) >= server.limits.max_sequences
                            or server.registry.count(self.verifier_id)
                            >= server.max_sequences_per_verifier
                        ):
                            return (
                                SpeculatorStatusCode.QUEUE_FULL,
                                "sequence credit exhausted",
                            )
                        if any(k.sequence_id == key.sequence_id for k in self.keys):
                            raise ProtocolError("target-KV sequence ID is already open")
                        server.registry.open(key, owner=self.session_id)
                        entry = server.registry.get(key)
                        assert entry is not None
                        entry.state = SequenceState.READY
                        self.keys.add(key)
                        return SpeculatorStatusCode.OK, ""

                self.last_generation = key.generation
                status, detail = server.work_queue.call(self.verifier_id, open_sequence)
                self._send(SequenceAck(key, status, detail), envelope.request_id)
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

                def close_sequences(keys=message.keys):
                    with server.adapter_turn(cleanup=True):
                        for key in keys:
                            server.registry.close(key, owner=self.session_id)
                            self.keys.discard(key)

                server.work_queue.call(self.verifier_id, close_sequences, cleanup=True)
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
        # Credits cover receive, queued/active work, and blocked reply writes.
        # Hidden input and its combined copy are charged as well as snapshot pages.
        nbytes = sum(
            p.nbytes for row in plans for p in row
        ) + 4 * server.hidden_size * len(queries)
        server._reserve_snapshot(self.verifier_id, nbytes)
        try:
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
            work = DraftWork(
                self.verifier_id,
                None,
                payload=_SnapshotInput(
                    self, queries, tuple(snapshots), tuple(hidden_rows), deadline, batch
                ),
                num_bytes=nbytes,
                num_rows=len(queries),
                num_tokens=len(queries),
            )
            tokens, status, detail = server.work_queue.submit(work).wait()
            # The executor fenced all reads or quarantined ownership before completion.
            del work, snapshots, layers, hidden_rows
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
        finally:
            server._release_snapshot(self.verifier_id, nbytes)
