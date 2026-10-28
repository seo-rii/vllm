# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual v4 socket boundaries with CPU snapshots; no GPU/model claims."""

import errno
import os
import socket
import struct
import threading
import time
from dataclasses import replace

import pytest
import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
from vllm.v1.spec_decode.remote.protocol import (
    Ping,
    ProtocolVersionError,
    encode_message,
)
from vllm.v1.spec_decode.remote.session import RemoteDraftError
from vllm.v1.spec_decode.remote.target_kv import (
    SnapshotBudget,
    SnapshotCompletionError,
    SnapshotIdentity,
    SnapshotLayer,
    TargetKVSnapshot,
    import_target_kv,
    import_target_kv_batch,
)
from vllm.v1.spec_decode.remote.target_kv_protocol import (
    TargetKVBatchQuery,
    TargetKVHello,
    TargetKVQuery,
    decode_target_kv_message,
    encode_target_kv_message,
    snapshot_wire_plan,
)
from vllm.v1.spec_decode.remote.target_kv_server import TargetKVDraftServer
from vllm.v1.spec_decode.remote.target_kv_session import TargetKVRemoteSession
from vllm.v1.spec_decode.remote.token_protocol import encode_token_message
from vllm.v1.spec_decode.remote.transport import (
    FRAME_CONTROL,
    ConnectionClosed,
    FramedConnection,
    TransportError,
    connect,
)


def until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline
        threading.Event().wait(0.001)


class SnapshotOracle:
    def __init__(self, specs, budget):
        self.specs, self.budget = specs, budget
        self.calls = []
        self.batch_sizes = []
        self.warmed = False
        self.error = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def warmup(self):
        self.warmed = True

    def execute(
        self,
        snapshot,
        *,
        expected_identity,
        expected_materialized_length,
        recovery_token,
        hidden_state,
    ):
        self.entered.set()
        assert self.release.wait(2)
        if self.error is not None:
            raise self.error
        imported = import_target_kv(
            snapshot,
            expected_identity=expected_identity,
            expected_materialized_length=expected_materialized_length,
            expected_specs=self.specs,
            budget=self.budget,
            device=torch.device("cpu"),
        )
        self.calls.append((expected_identity, expected_materialized_length))
        assert imported.identity == expected_identity
        return (recovery_token + int(hidden_state.sum())) % 128

    def execute_batch(
        self,
        snapshots,
        *,
        expected_identities,
        expected_materialized_lengths,
        recovery_tokens,
        hidden_states,
    ):
        self.entered.set()
        assert self.release.wait(2)
        if self.error is not None:
            raise self.error
        imported = import_target_kv_batch(
            snapshots,
            expected_identities=expected_identities,
            expected_materialized_lengths=expected_materialized_lengths,
            expected_specs=self.specs,
            budget=self.budget,
            device=torch.device("cpu"),
        )
        assert imported.identities == expected_identities
        self.batch_sizes.append(len(snapshots))
        self.calls.extend(zip(expected_identities, expected_materialized_lengths))
        return tuple(
            (token + int(hidden.sum())) % 128
            for token, hidden in zip(recovery_tokens, hidden_states)
        )


@pytest.fixture
def wire_runtime(tmp_path, request):
    max_num_seqs = getattr(request, "param", 1)
    specs = {
        "target.layers.1": SlidingWindowSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.bfloat16,
            sliding_window=5,
        ),
        "target.layers.3": FullAttentionSpec(
            block_size=4, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
        ),
    }
    budget = SnapshotBudget(128, 1 << 20, 64)
    executor = SnapshotOracle(specs, budget)
    server = TargetKVDraftServer(
        executor,
        contract_fingerprint="contract",
        hidden_size=8,
        vocab_size=128,
        expected_specs=specs,
        snapshot_budget=budget,
        receiver_budget=budget,
        max_num_seqs=max_num_seqs,
    )
    endpoint = server.start(f"unix://{tmp_path}/snapshot.sock")
    hello = TargetKVHello(
        "verifier",
        "contract",
        8,
        128,
        64,
        128,
        1 << 20,
        max_num_seqs=max_num_seqs,
    )
    client = TargetKVRemoteSession(
        endpoint, hello, expected_specs=specs, request_timeout_ms=1000
    )
    client.connect()
    try:
        yield server, executor, client
    finally:
        executor.release.set()
        client.close()
        server.stop()
        assert len(server.registry) == 0
        assert not server._connections
        assert not server._handshaking


def snapshot(client, key, round_id=0, length=9):
    plans = snapshot_wire_plan(client.expected_specs, length, client.budget)
    return TargetKVSnapshot(
        SnapshotIdentity(key, client.epoch, round_id),
        length,
        tuple(
            SnapshotLayer(
                p.name,
                p.spec,
                p.logical_pages,
                torch.ones(p.shape, dtype=torch.bfloat16),
            )
            for p in plans
        ),
    )


def propose(client, snap, *, k=1):
    return client.collect(
        client.dispatch(
            snap,
            10,
            torch.ones((1, 8), dtype=torch.bfloat16),
            k=k,
            deadline=client.deadline(),
        )
    )


def test_v4_is_separate_from_v2_and_v3():
    assert (
        decode_target_kv_message(encode_target_kv_message(Ping()))[0].protocol_major
        == 4
    )
    for old in (encode_message(Ping()), encode_token_message(Ping())):
        with pytest.raises(ProtocolVersionError):
            decode_target_kv_message(old)


def test_declared_frame_limit_rejects_header_without_waiting_for_body():
    left, right = socket.socketpair()
    conn = FramedConnection(left)
    try:
        right.sendall(struct.pack("!BI", FRAME_CONTROL, 1 << 20))
        with pytest.raises(TransportError, match="exceeds 64"):
            conn.recv(timeout=0.1, max_frame_bytes=64)
        assert len(conn._buffer) == 5
    finally:
        conn.close()
        right.close()


def test_authoritative_rounds_k0_and_sequential_requests_use_same_connection(
    wire_runtime,
):
    server, executor, client = wire_runtime
    assert executor.warmed
    key = client.open_sequence(0, deadline=client.deadline())
    assert propose(client, snapshot(client, key)).token_ids == (18,)
    assert propose(client, snapshot(client, key, 1), k=0).valid_length == 0
    assert propose(client, snapshot(client, key, 2, 12)).token_ids == (18,)
    assert len(executor.calls) == 2
    conn, epoch = client._conn, client.epoch
    client.close_sequences((key,))
    second = client.open_sequence(0, deadline=client.deadline())
    assert second.generation > key.generation
    assert client._conn is conn and client.epoch == epoch
    assert propose(client, snapshot(client, second, 3, 1)).valid_length == 1
    assert len(server.registry) == 1


def test_stale_round_epoch_and_double_collect_fail_before_execution(wire_runtime):
    _, executor, client = wire_runtime
    key = client.open_sequence(0, deadline=client.deadline())
    snap = snapshot(client, key)
    handle = client.dispatch(
        snap, 10, torch.ones((1, 8), dtype=torch.bfloat16), deadline=client.deadline()
    )
    client.collect(handle)
    with pytest.raises(RuntimeError, match="collected"):
        client.collect(handle)
    with pytest.raises(ValueError, match="round"):
        propose(client, snap)
    with pytest.raises(ValueError, match="epoch"):
        propose(
            client, replace(snap, identity=SnapshotIdentity(key, client.epoch + 1, 1))
        )
    assert len(executor.calls) == 1


@pytest.mark.parametrize("policy", ["error", "target_only"])
def test_executor_oom_fences_request_and_releases_connection(wire_runtime, policy):
    server, executor, client = wire_runtime
    client.failure_policy = policy
    executor.error = torch.OutOfMemoryError("injected allocation failure")
    key = client.open_sequence(0, deadline=client.deadline())
    if policy == "error":
        with pytest.raises(RemoteDraftError, match="OutOfMemoryError"):
            propose(client, snapshot(client, key))
    else:
        assert propose(client, snapshot(client, key)).valid_length == 0
    assert not client.active(key)
    assert client._conn is None
    assert server.healthy
    assert not executor.calls


def test_timeout_and_cancel_never_resume_terminal_request(wire_runtime):
    _, executor, client = wire_runtime
    client.failure_policy = "target_only"
    key = client.open_sequence(0, deadline=client.deadline())
    executor.release.clear()
    client.request_timeout_s = 0.03
    handle = client.dispatch(
        snapshot(client, key),
        10,
        torch.ones((1, 8), dtype=torch.bfloat16),
        deadline=client.deadline(),
    )
    assert executor.entered.wait(1)
    assert client.collect(handle).valid_length == 0
    assert not client.active(key) and client._conn is None
    assert propose(client, snapshot(client, key, 1)).valid_length == 0
    executor.release.set()


def test_cancel_invalidates_handle_and_connection(wire_runtime):
    _, _, client = wire_runtime
    key = client.open_sequence(0, deadline=client.deadline())
    handle = client.dispatch(
        snapshot(client, key),
        10,
        torch.ones((1, 8), dtype=torch.bfloat16),
        deadline=client.deadline(),
    )
    client.cancel(handle)
    assert not client.active(key) and client._conn is None
    with pytest.raises(RuntimeError, match="collected"):
        client.collect(handle)


@pytest.mark.parametrize(
    "error",
    [
        SnapshotCompletionError("unknown completion"),
        RuntimeError("CUDA error: an illegal memory access was encountered"),
    ],
)
def test_unknown_or_fatal_completion_marks_runtime_unhealthy(wire_runtime, error):
    server, executor, client = wire_runtime
    executor.error = error
    key = client.open_sequence(0, deadline=client.deadline())
    with pytest.raises(RemoteDraftError):
        propose(client, snapshot(client, key))
    assert not server.healthy


def test_snapshot_capacity_rejects_before_receiving_data_or_executor(wire_runtime):
    from vllm.v1.spec_decode.remote.target_kv_protocol import TargetKVQuery

    _, executor, client = wire_runtime
    key = client.open_sequence(0, deadline=client.deadline())
    request_id = next(client._ids)
    client._send(
        TargetKVQuery(key, client.epoch, 0, 65, 10, 1000), request_id, client.deadline()
    )
    with pytest.raises(TransportError):
        client._recv(request_id, client.deadline())
    assert not executor.calls


def test_second_connection_is_refused_without_running_adapter(wire_runtime):
    _, executor, client = wire_runtime
    second = TargetKVRemoteSession(
        client.endpoint, client.hello, expected_specs=client.expected_specs
    )
    try:
        with pytest.raises(TransportError):
            second.connect()
        assert not executor.calls
    finally:
        second.close()


def test_server_close_error_returns_connection_and_sequence_credit(
    wire_runtime, monkeypatch
):
    """A failed socket close must not prevent a fresh verifier connection."""
    server, executor, client = wire_runtime
    key = client.open_sequence(0, deadline=client.deadline())
    assert propose(client, snapshot(client, key)).token_ids == (18,)
    assert len(server.registry) == 1
    with server._connections_lock:
        connection = next(iter(server._connections))
        worker = server._workers[0]
    errors: list[threading.ExceptHookArgs] = []
    monkeypatch.setattr(threading, "excepthook", errors.append)
    replacement = TargetKVRemoteSession(
        client.endpoint,
        client.hello,
        expected_specs=client.expected_specs,
        startup_timeout_ms=1000,
        request_timeout_ms=1000,
    )
    try:
        # Fail the actual accepted socket's close, after a completed snapshot.
        os.close(connection._sock.fileno())
        client.close()
        worker.join(timeout=3)
        assert not worker.is_alive()
        assert connection.closed
        assert not errors, [error.exc_value for error in errors]
        assert not server._connections and not server._handshaking
        assert not server._connection_turn.locked()
        assert len(server.registry) == 0 and server.healthy
        replacement.connect()
        key = replacement.open_sequence(0, deadline=replacement.deadline())
        assert propose(replacement, snapshot(replacement, key)).token_ids == (18,)
        assert len(executor.calls) == 2
    finally:
        replacement.close()


def test_session_construction_failure_releases_exclusive_connection(
    wire_runtime, monkeypatch
):
    from vllm.v1.spec_decode.remote import target_kv_server

    server, executor, client = wire_runtime
    client.close()
    assert server._connection_turn.acquire(timeout=2)
    server._connection_turn.release()
    construct = target_kv_server._TargetKVServerSession
    injected = threading.Event()
    workers: list[threading.Thread] = []
    thread_errors: list[threading.ExceptHookArgs] = []
    rejected = replacement = None

    def fail_once(*args, **kwargs):
        if not injected.is_set():
            workers.append(threading.current_thread())
            injected.set()
            raise MemoryError("injected target-KV session allocation failure")
        return construct(*args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(target_kv_server, "_TargetKVServerSession", fail_once)
            patch.setattr(threading, "excepthook", thread_errors.append)
            rejected = connect(client.endpoint, timeout=1)
            assert injected.wait(1)
            workers[0].join(timeout=1)
            assert not workers[0].is_alive()
            assert not thread_errors, [error.exc_value for error in thread_errors]
            with pytest.raises(ConnectionClosed):
                rejected.recv(timeout=1)
            assert not server._connection_turn.locked()
            assert not server._connections and not server._handshaking
            assert not len(server.registry)
            assert server.healthy and server._acceptor.is_alive()

            replacement = TargetKVRemoteSession(
                client.endpoint,
                client.hello,
                expected_specs=client.expected_specs,
                startup_timeout_ms=1000,
                request_timeout_ms=1000,
            )
            replacement.connect()
            key = replacement.open_sequence(0, deadline=replacement.deadline())
            assert propose(replacement, snapshot(replacement, key)).token_ids == (18,)
            assert len(executor.calls) == 1
    finally:
        if rejected is not None:
            rejected.close()
        if replacement is not None:
            replacement.close()


@pytest.mark.parametrize("policy", ["error", "target_only"])
def test_connection_backoff_bounds_new_target_kv_requests(
    wire_runtime, monkeypatch, policy
):
    import time

    from vllm.v1.spec_decode.remote import target_kv_session
    from vllm.v1.spec_decode.remote.reconnect import ReconnectBackoff

    server, _, client = wire_runtime
    client.close()
    assert server._connection_turn.acquire(timeout=2)
    server._connection_turn.release()
    now = [time.monotonic()]
    session = TargetKVRemoteSession(
        client.endpoint,
        client.hello,
        expected_specs=client.expected_specs,
        failure_policy=policy,
        clock=lambda: now[0],
    )
    attempts = []
    original = target_kv_session.connect

    def unavailable(*args, **kwargs):
        attempts.append(now[0])
        raise TransportError("injected connection outage")

    monkeypatch.setattr(target_kv_session, "connect", unavailable)
    try:
        for i in range(32):
            if policy == "error":
                with pytest.raises(RemoteDraftError):
                    session.open_sequence(i, deadline=session.deadline())
            else:
                key = session.open_sequence(i, deadline=session.deadline())
                assert not session.active(key)
            session.close_sequences(tuple(session.records))
        assert len(attempts) == 1
        now[0] += 1
        monkeypatch.setattr(target_kv_session, "connect", original)
        key = session.open_sequence(32, deadline=session.deadline())
        assert session.active(key)
        assert isinstance(session._reconnect, ReconnectBackoff)
    finally:
        session.close()


def test_wire_order_is_independent_of_mapping_insertion_order(wire_runtime):
    _, _, client = wire_runtime
    client.expected_specs = dict(reversed(tuple(client.expected_specs.items())))
    key = client.open_sequence(0, deadline=client.deadline())
    assert propose(client, snapshot(client, key)).token_ids == (18,)


@pytest.mark.parametrize("mutation", ["slot", "dtype", "shape", "payload"])
def test_malformed_first_data_frame_rejected_before_tensor_allocation(
    wire_runtime, monkeypatch, mutation
):
    from vllm.v1.spec_decode.remote import target_kv_server
    from vllm.v1.spec_decode.remote.target_kv_protocol import TargetKVQuery
    from vllm.v1.spec_decode.remote.transport import DataFrame

    _, executor, client = wire_runtime
    key = client.open_sequence(0, deadline=client.deadline())
    request_id = next(client._ids)
    client._send(
        TargetKVQuery(key, client.epoch, 0, 9, 10, 1000), request_id, client.deadline()
    )
    fields = dict(slot=0, dtype="bfloat16", shape=(1, 8), payload=bytes(16))
    fields[mutation] = {
        "slot": 1,
        "dtype": "int64",
        "shape": (8, 1),
        "payload": bytes(14),
    }[mutation]
    allocations = []
    monkeypatch.setattr(
        target_kv_server, "frame_to_tensor", lambda _: allocations.append(1)
    )
    client._conn.send_data(DataFrame(**fields), deadline=client.deadline())
    with pytest.raises(TransportError):
        client._recv(request_id, client.deadline())
    assert not allocations and not executor.calls


def test_handle_does_not_retain_completed_inline_snapshot(wire_runtime):
    import weakref

    _, _, client = wire_runtime
    key = client.open_sequence(0, deadline=client.deadline())
    snap = snapshot(client, key)
    ref = weakref.ref(snap.layers[0].pages)
    handle = client.dispatch(
        snap, 10, torch.ones((1, 8), dtype=torch.bfloat16), deadline=client.deadline()
    )
    del snap
    assert ref() is None
    assert client.collect(handle).valid_length == 1


def test_pre_send_deadline_failure_is_terminal_and_clears_strict_handle(wire_runtime):
    _, executor, client = wire_runtime
    key = client.open_sequence(0, deadline=client.deadline())
    with pytest.raises(RemoteDraftError, match="deadline"):
        client.dispatch(
            snapshot(client, key),
            10,
            torch.ones((1, 8), dtype=torch.bfloat16),
            deadline=client.clock() - 1,
        )
    assert not client.active(key) and client._conn is None and client._pending is None
    assert not executor.calls


def test_bounded_receive_does_not_read_oversized_payload_before_header_check():
    left, right = socket.socketpair()
    conn = FramedConnection(left)
    try:
        right.sendall(struct.pack("!BI", FRAME_CONTROL, 4096) + bytes(4096))
        with pytest.raises(TransportError, match="exceeds 64"):
            conn.recv(timeout=0.1, max_frame_bytes=64)
        assert len(conn._buffer) == 5
    finally:
        conn.close()
        right.close()


def test_receive_allocation_oom_disconnects_without_unhealthy_gpu(
    wire_runtime, monkeypatch
):
    from vllm.v1.spec_decode.remote import target_kv_server

    server, executor, client = wire_runtime
    client.failure_policy = "target_only"
    key = client.open_sequence(0, deadline=client.deadline())

    def fail_allocation(_):
        raise torch.OutOfMemoryError("injected snapshot CPU allocation")

    monkeypatch.setattr(target_kv_server, "frame_to_tensor", fail_allocation)
    assert propose(client, snapshot(client, key)).valid_length == 0
    assert not client.active(key) and client._conn is None
    assert not executor.calls and server.healthy


@pytest.mark.parametrize("field", ["key", "session_epoch", "round_id"])
def test_server_rejects_stale_snapshot_identity_before_receiving_pages(
    wire_runtime, field
):
    from vllm.v1.spec_decode.remote.target_kv_protocol import TargetKVQuery

    _, executor, client = wire_runtime
    key = client.open_sequence(0, deadline=client.deadline())
    assert propose(client, snapshot(client, key)).valid_length == 1
    values = dict(
        key=key,
        session_epoch=client.epoch,
        round_id=1,
        materialized_length=9,
        input_token=10,
        budget_ms=1000,
    )
    if field == "key":
        values[field] = type(key)(
            key.verifier_instance_id, key.sequence_id, key.generation + 1
        )
    elif field == "session_epoch":
        values[field] += 1
    else:
        values[field] = 0
    request_id = next(client._ids)
    client._send(TargetKVQuery(**values), request_id, client.deadline())
    with pytest.raises(TransportError):
        client._recv(request_id, client.deadline())
    assert len(executor.calls) == 1


@pytest.mark.parametrize(
    "kind,wrapper",
    [
        ("ordinary", None),
        ("fatal", None),
        ("unknown", None),
        ("fatal", MemoryError),
        ("fatal", torch.OutOfMemoryError),
        ("unknown", MemoryError),
        ("unknown", torch.OutOfMemoryError),
    ],
)
def test_local_copy_runtime_errors_never_become_target_only(
    wire_runtime, monkeypatch, kind, wrapper
):
    from vllm.v1.spec_decode.remote import target_kv_session

    _, _, client = wire_runtime
    client.failure_policy = "target_only"
    key = client.open_sequence(0, deadline=client.deadline())
    error = {
        "ordinary": RuntimeError("unclassified CUDA copy failure"),
        "fatal": RuntimeError("CUDA illegal memory access"),
        "unknown": SnapshotCompletionError("unknown source completion"),
    }[kind]
    if wrapper is not None:
        wrapped = wrapper("staging allocation after unsafe completion")
        if wrapper is MemoryError:
            wrapped.__cause__ = error
        else:
            wrapped.__context__ = error
        error = wrapped

    def fail_copy(*args):
        raise error

    monkeypatch.setattr(target_kv_session, "tensor_to_frame", fail_copy)
    with pytest.raises(type(error), match=str(error)):
        propose(client, snapshot(client, key))
    assert not client.active(key) and client._conn is None and client._pending is None


@pytest.mark.parametrize("policy", ["target_only", "error"])
@pytest.mark.parametrize("failure_slot", [0, 1, 2])
@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("source staging allocation"),
        MemoryError("source staging allocation"),
        OSError(errno.ENOMEM, "source staging allocation"),
        RuntimeError(
            "DefaultCPUAllocator: can't allocate memory: you tried to allocate "
            "4096 bytes. Error code 12 (Cannot allocate memory)"
        ),
    ],
)
def test_local_copy_oom_obeys_policy_and_invalidates_partial_request(
    wire_runtime, monkeypatch, policy, failure_slot, error
):
    from vllm.v1.spec_decode.remote import target_kv_session

    _, executor, client = wire_runtime
    client.failure_policy = policy
    key = client.open_sequence(0, deadline=client.deadline())
    tensor_to_frame = target_kv_session.tensor_to_frame
    visited = []

    def fail_copy(slot, tensor):
        visited.append(slot)
        if slot == failure_slot:
            raise error
        return tensor_to_frame(slot, tensor)

    monkeypatch.setattr(target_kv_session, "tensor_to_frame", fail_copy)
    if policy == "error":
        with pytest.raises(RemoteDraftError, match="target-KV dispatch failed"):
            propose(client, snapshot(client, key))
    else:
        assert propose(client, snapshot(client, key)).valid_length == 0
    assert visited == list(range(failure_slot + 1))
    assert not executor.calls
    assert not client.active(key) and client._conn is None and client._pending is None


@pytest.mark.parametrize("wire_runtime", [4], indirect=True)
def test_batch_rounds_preserve_row_order_and_reuse_after_k0(wire_runtime):
    server, executor, client = wire_runtime
    keys = client.open_sequences((1, 2, 3), deadline=client.deadline())
    with pytest.raises(ValueError, match="already open"):
        client.open_sequence(1, deadline=client.deadline())
    assert len(server.registry) == 3
    ordered = (keys[2], keys[0], keys[1])
    hidden = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8)
    for round_id, k in ((0, 0), (1, 1), (2, 1)):
        snapshots = tuple(
            snapshot(client, key, round_id, length)
            for key, length in zip(ordered, (17, 4, 33))
        )
        result = client.collect_batch(
            client.dispatch_batch(
                snapshots, (10, 20, 30), hidden, k=k, deadline=client.deadline()
            )
        )
        assert [r.valid_length for r in result] == [k] * 3
        if k:
            assert [r.token_ids[0] for r in result] == [38, 112, 58]
        assert all(client.active(key) for key in keys)
    assert executor.batch_sizes == [3, 3]
    client.close_sequences(keys)
    assert len(server.registry) == 0


@pytest.mark.parametrize("wire_runtime", [4], indirect=True)
def test_unrelated_batch_close_waits_for_reply_without_cancelling(wire_runtime):
    server, executor, client = wire_runtime
    a, b, c = client.open_sequences((1, 2, 3), deadline=client.deadline())
    executor.release.clear()
    handle = client.dispatch_batch(
        (snapshot(client, a), snapshot(client, b, length=17)),
        (10, 20),
        torch.ones((2, 8), dtype=torch.bfloat16),
        deadline=client.deadline(),
    )
    try:
        assert executor.entered.wait(1)
        client.close_sequences(())
        client.close_sequences((c,))
        assert len(server.registry) == 3
    finally:
        executor.release.set()
    assert [r.token_ids for r in client.collect_batch(handle)] == [(18,), (28,)]
    assert len(server.registry) == 2
    replacement = client.open_sequence(3, deadline=client.deadline())
    assert replacement.generation > c.generation
    assert propose(client, snapshot(client, replacement, 1)).token_ids == (18,)
    assert client.active(a) and client.active(b)


@pytest.mark.parametrize("wire_runtime", [4], indirect=True)
@pytest.mark.parametrize("policy", ["error", "target_only"])
def test_batch_execution_failure_fences_only_submitted_requests(wire_runtime, policy):
    server, executor, client = wire_runtime
    client.failure_policy = policy
    a, b, healthy = client.open_sequences((1, 2, 3), deadline=client.deadline())
    executor.error = torch.OutOfMemoryError("batch allocation failure")
    handle = client.dispatch_batch(
        (snapshot(client, a), snapshot(client, b)),
        (10, 20),
        torch.ones((2, 8), dtype=torch.bfloat16),
        deadline=client.deadline(),
    )
    if policy == "error":
        with pytest.raises(RemoteDraftError, match="batch allocation failure"):
            client.collect_batch(handle)
    else:
        assert [r.valid_length for r in client.collect_batch(handle)] == [0, 0]
    assert not client.active(a) and not client.active(b)
    assert client.active(healthy) and server.healthy and len(server.registry) == 1
    executor.error = None
    assert propose(client, snapshot(client, healthy, 1)).token_ids == (18,)


@pytest.mark.parametrize("wire_runtime", [16], indirect=True)
@pytest.mark.parametrize("diagnostic", ["allocation failure " * 100, "🧠" * 1024])
def test_large_batch_error_reply_preserves_unsubmitted_sibling(
    wire_runtime, diagnostic
):
    """Repeated per-row diagnostics must fit the bounded control frame."""
    server, executor, client = wire_runtime
    client.failure_policy = "target_only"
    *failed, healthy = client.open_sequences(
        tuple(range(16)), deadline=client.deadline()
    )
    executor.error = torch.OutOfMemoryError(diagnostic)
    handle = client.dispatch_batch(
        tuple(snapshot(client, key) for key in failed),
        (10,) * len(failed),
        torch.ones((len(failed), 8), dtype=torch.bfloat16),
        deadline=client.deadline(),
    )
    assert [r.valid_length for r in client.collect_batch(handle)] == [0] * len(failed)
    assert not any(client.active(key) for key in failed)
    assert client.active(healthy) and server.healthy and len(server.registry) == 1
    executor.error = None
    assert propose(client, snapshot(client, healthy, 1)).token_ids == (18,)


@pytest.mark.parametrize("wire_runtime", [4], indirect=True)
def test_batch_total_page_limit_rejects_manifest_before_any_data(wire_runtime):
    server, executor, client = wire_runtime
    keys = client.open_sequences((1, 2), deadline=client.deadline())
    # Each row fits; together they exceed the negotiated receive page budget.
    server.receiver_budget = SnapshotBudget(20, 1 << 20, 64)
    queries = tuple(TargetKVQuery(key, client.epoch, 0, 64, 1, 1000) for key in keys)
    client._send(
        TargetKVBatchQuery(queries, 1000), next(client._ids), client.deadline()
    )
    with pytest.raises(TransportError):
        client._conn.recv(timeout=1)
    assert not executor.calls


@pytest.mark.parametrize("wire_runtime", [2], indirect=True)
@pytest.mark.parametrize("policy", ["error", "target_only"])
def test_cleanup_error_preserves_failure_policy_and_terminal_state(
    wire_runtime, monkeypatch, policy
):
    _, _, client = wire_runtime
    client.failure_policy = policy
    key, deferred = client.open_sequences((0, 1), deadline=client.deadline())
    handle = client.dispatch(
        snapshot(client, key),
        10,
        torch.ones((1, 8), dtype=torch.bfloat16),
        deadline=client.deadline(),
    )
    client.close_sequences((deferred,))
    connection = client._conn
    close_socket = socket.socket.close

    def close_then_fail(sock):
        close_socket(sock)
        if sock is connection._sock:
            raise OSError(errno.EIO, "injected socket cleanup failure")

    with monkeypatch.context() as patch:
        patch.setattr(socket.socket, "close", close_then_fail)
        if policy == "error":
            with pytest.raises(RemoteDraftError, match="original target-KV failure"):
                client.fail("original target-KV failure")
        else:
            client.fail("original target-KV failure")
    assert connection.closed and client._conn is None
    assert not client.active(key)
    assert deferred not in client.records and not client._closing
    assert client.collect(handle).valid_length == 0
    assert not handle.active and client._pending is None


@pytest.mark.parametrize("operation", ["cancel", "close"])
def test_cleanup_error_finishes_cancelled_handle(wire_runtime, monkeypatch, operation):
    _, _, client = wire_runtime
    key = client.open_sequence(0, deadline=client.deadline())
    handle = client.dispatch(
        snapshot(client, key),
        10,
        torch.ones((1, 8), dtype=torch.bfloat16),
        deadline=client.deadline(),
    )
    connection = client._conn
    close_socket = socket.socket.close

    def close_then_fail(sock):
        close_socket(sock)
        if sock is connection._sock:
            raise OSError(errno.EIO, "injected socket cleanup failure")

    with monkeypatch.context() as patch:
        patch.setattr(socket.socket, "close", close_then_fail)
        if operation == "cancel":
            client.cancel(handle)
        else:
            client.close()
    assert connection.closed and client._conn is None
    assert not client.active(key)
    assert not handle.active and client._pending is None
    with pytest.raises(RuntimeError, match="collected"):
        client.collect(handle)
    if operation == "close":
        assert not client.records
        with pytest.raises(RuntimeError, match="closed"):
            client.connect()
    else:
        assert propose(client, snapshot(client, key, 1)).valid_length == 0
