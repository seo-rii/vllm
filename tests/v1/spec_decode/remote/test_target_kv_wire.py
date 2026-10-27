# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual v4 socket boundaries with CPU snapshots; no GPU/model claims."""

import errno
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace

import msgspec
import pytest
import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
from vllm.v1.spec_decode.remote.protocol import (
    Ping,
    ProtocolVersionError,
    encode_message,
)
from vllm.v1.spec_decode.remote.server import ServerShutdownTimeout
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
        self.threads = set()
        self.batches = []
        self.batch_sizes = []
        self.attempts = []
        self.warmed = False
        self.error = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def warmup(self):
        self.threads.add(threading.get_ident())
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
        self.attempts.append((expected_identity,))
        self.threads.add(threading.get_ident())
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
        self.attempts.append(expected_identities)
        self.threads.add(threading.get_ident())
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
        self.batches.append(expected_identities)
        self.batch_sizes.append(len(snapshots))
        self.calls.extend(zip(expected_identities, expected_materialized_lengths))
        return tuple(
            (token + int(hidden.sum())) % 128
            for token, hidden in zip(recovery_tokens, hidden_states)
        )


@pytest.fixture
def wire_runtime(tmp_path, request):
    options = getattr(request, "param", 1)
    options = dict(options) if isinstance(options, dict) else {"max_num_seqs": options}
    max_num_seqs = options.pop("max_num_seqs", 1)
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
    receiver_budget = options.pop("receiver_budget", budget)
    executor = SnapshotOracle(specs, receiver_budget)
    server = TargetKVDraftServer(
        executor,
        contract_fingerprint="contract",
        hidden_size=8,
        vocab_size=128,
        expected_specs=specs,
        snapshot_budget=budget,
        receiver_budget=receiver_budget,
        max_num_seqs=max_num_seqs,
        **options,
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
        if not server._shutdown_failed:
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


def test_session_construction_failure_releases_connection_admission(
    wire_runtime, monkeypatch
):
    from vllm.v1.spec_decode.remote import target_kv_server

    server, executor, client = wire_runtime
    client.close()
    until(lambda: not server._connections)
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
    until(lambda: not server._connections)
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


@contextmanager
def peer(client, identity):
    other = TargetKVRemoteSession(
        client.endpoint,
        msgspec.structs.replace(client.hello, verifier_instance_id=identity),
        expected_specs=client.expected_specs,
        request_timeout_ms=2000,
    )
    try:
        other.connect()
        yield other
    finally:
        other.close()


_POOL_LIMITS = {
    "max_num_seqs": 4,
    "max_connections": 4,
    "max_sequences": 12,
    "batch_wait_ms": 0,
}


@pytest.mark.parametrize("wire_runtime", [_POOL_LIMITS], indirect=True)
def test_snapshot_pool_combines_distinct_verifiers_on_one_gpu_owner(wire_runtime):
    server, oracle, a = wire_runtime
    with peer(a, "b") as b, peer(a, "blocker") as blocker:
        ka, kb, kc = (
            c.open_sequence(0, deadline=c.deadline()) for c in (a, b, blocker)
        )
        oracle.release.clear()
        with ThreadPoolExecutor(max_workers=3) as threads:
            blocked = threads.submit(propose, blocker, snapshot(blocker, kc))
            assert oracle.entered.wait(2)
            pa = threads.submit(propose, a, snapshot(a, ka, length=9))
            pb = threads.submit(propose, b, snapshot(b, kb, length=13))
            until(lambda: server.work_queue.depth == 3)
            assert server.pending_snapshot_jobs == 3
            oracle.release.set()
            assert blocked.result().token_ids == (18,)
            assert pa.result().token_ids == pb.result().token_ids == (18,)
        assert oracle.batch_sizes == [1, 2]
        assert {i.key.verifier_instance_id for i in oracle.batches[-1]} == {
            "verifier",
            "b",
        }
        assert len(oracle.threads) == 1
        assert threading.get_ident() not in oracle.threads
        until(lambda: server.pending_snapshot_jobs == 0)
        assert server.pending_snapshot_bytes == server.work_queue.pending_bytes == 0
        assert len(server.registry) == 3


@pytest.mark.parametrize(
    "wire_runtime",
    [{**_POOL_LIMITS, "receiver_budget": SnapshotBudget(128, 1000, 64)}],
    indirect=True,
)
def test_snapshot_pool_splits_legal_jobs_when_combined_receiver_cannot_fit(
    wire_runtime,
):
    server, oracle, a = wire_runtime
    with peer(a, "b") as b, peer(a, "blocker") as blocker:
        ka, kb, kc = (
            c.open_sequence(0, deadline=c.deadline()) for c in (a, b, blocker)
        )
        oracle.release.clear()
        with ThreadPoolExecutor(max_workers=3) as threads:
            held = threads.submit(propose, blocker, snapshot(blocker, kc))
            assert oracle.entered.wait(2)
            pa = threads.submit(propose, a, snapshot(a, ka))
            pb = threads.submit(propose, b, snapshot(b, kb))
            until(lambda: server.work_queue.depth == 3)
            oracle.release.set()
            assert (
                held.result().valid_length
                == pa.result().valid_length
                == pb.result().valid_length
                == 1
            )
        assert oracle.batch_sizes == [1, 1, 1]
        assert server.healthy
        assert len(server.registry) == 3


@pytest.mark.parametrize("wire_runtime", [_POOL_LIMITS], indirect=True)
def test_snapshot_pool_rejects_duplicate_live_verifier_without_affecting_owner(
    wire_runtime,
):
    server, oracle, client = wire_runtime
    key = client.open_sequence(0, deadline=client.deadline())
    with pytest.raises(TransportError), peer(client, client.hello.verifier_instance_id):
        pytest.fail("duplicate verifier was admitted")
    assert propose(client, snapshot(client, key)).token_ids == (18,)
    assert len(server.registry) == 1
    assert server.healthy
    client.close()
    until(lambda: not server._connections)
    with peer(client, client.hello.verifier_instance_id) as replacement:
        key = replacement.open_sequence(0, deadline=replacement.deadline())
        assert propose(replacement, snapshot(replacement, key)).token_ids == (18,)


@pytest.mark.parametrize(
    "wire_runtime",
    [{**_POOL_LIMITS, "max_queued_snapshot_bytes": 1100}],
    indirect=True,
)
def test_snapshot_pool_resident_credit_includes_active_and_queued_jobs(wire_runtime):
    server, oracle, a = wire_runtime
    with peer(a, "b") as b, peer(a, "excess") as excess:
        ka, kb, kc = (c.open_sequence(0, deadline=c.deadline()) for c in (a, b, excess))
        oracle.release.clear()
        with ThreadPoolExecutor(max_workers=2) as threads:
            pa = threads.submit(propose, a, snapshot(a, ka))
            assert oracle.entered.wait(2)
            pb = threads.submit(propose, b, snapshot(b, kb))
            until(lambda: server.work_queue.depth == 2)
            assert server.pending_snapshot_jobs == 2
            assert server.pending_snapshot_bytes == 1088
            with pytest.raises(RemoteDraftError):
                propose(excess, snapshot(excess, kc))
            assert server.pending_snapshot_jobs == 2
            assert server.pending_snapshot_bytes == 1088
            oracle.release.set()
            assert pa.result().valid_length == pb.result().valid_length == 1
        until(lambda: not server.pending_snapshot_jobs)
        assert server.pending_snapshot_bytes == 0 and server.healthy


@pytest.mark.parametrize(
    "wire_runtime",
    [{**_POOL_LIMITS, "max_sequences": 4, "max_sequences_per_verifier": 2}],
    indirect=True,
)
def test_snapshot_pool_sequence_credits_preserve_other_verifier_progress(wire_runtime):
    server, oracle, a = wire_runtime
    with peer(a, "b") as b:
        keys = a.open_sequences((0, 1), deadline=a.deadline())
        kb = b.open_sequence(0, deadline=b.deadline())
        assert len(server.registry) == 3
        with pytest.raises(RemoteDraftError, match="open"):
            a.open_sequence(2, deadline=a.deadline())
        assert all(a.active(key) for key in keys)
        rejected = tuple(key for key in a.records if key not in keys)
        assert len(rejected) == 1 and not a.active(rejected[0])
        assert propose(a, snapshot(a, keys[0])).token_ids == (18,)
        assert propose(b, snapshot(b, kb)).token_ids == (18,)
        assert server.registry.count("verifier") == 2
        assert len(server.registry) == 3 and server.healthy


@pytest.mark.parametrize(
    "wire_runtime",
    [{"max_num_seqs": n, "max_connections": 2, "max_sequences": n} for n in (1, 2)],
    indirect=True,
)
def test_snapshot_pool_close_returns_sequence_credit_without_disconnect(wire_runtime):
    server, _, a = wire_runtime
    with peer(a, "b") as b:
        keys = a.open_sequences(
            tuple(range(a.hello.max_num_seqs)), deadline=a.deadline()
        )
        assert propose(a, snapshot(a, keys[0])).token_ids == (18,)
        epoch = a.epoch
        a.close_sequences(keys)
        assert not a.records
        assert len(server.registry) == 0

        key = b.open_sequence(0, deadline=b.deadline())
        assert propose(b, snapshot(b, key)).token_ids == (18,)
        b.close_sequences((key,))
        assert len(server.registry) == 0

        reopened = a.open_sequence(0, deadline=a.deadline())
        assert a.epoch == epoch and reopened.generation > keys[0].generation
        assert propose(a, snapshot(a, reopened, 1)).token_ids == (18,)
        a.close_sequences((reopened,))
        assert len(server.registry) == 0 and server.healthy


@pytest.mark.parametrize(
    "wire_runtime",
    [{**_POOL_LIMITS, "max_queued_snapshot_bytes": 600}],
    indirect=True,
)
def test_snapshot_pool_reserves_credit_before_receiving_any_payload(wire_runtime):
    server, oracle, a = wire_runtime
    with peer(a, "b") as b:
        ka, kb = (c.open_sequence(0, deadline=c.deadline()) for c in (a, b))
        q = TargetKVQuery(ka, a.epoch, 0, 9, 10, 1000)
        a._send(TargetKVBatchQuery((q,), 1000), next(a._ids), a.deadline())
        until(lambda: server.pending_snapshot_jobs == 1)
        assert server.pending_snapshot_bytes == 544
        assert server.work_queue.depth == 0 and not oracle.calls
        with pytest.raises(RemoteDraftError):
            propose(b, snapshot(b, kb))
        assert server.pending_snapshot_bytes == 544
        a.close()
        until(lambda: server.pending_snapshot_jobs == 0)
        assert server.pending_snapshot_bytes == 0
    with peer(a, "replacement") as replacement:
        key = replacement.open_sequence(0, deadline=replacement.deadline())
        assert propose(replacement, snapshot(replacement, key)).token_ids == (18,)


@pytest.mark.parametrize("wire_runtime", [_POOL_LIMITS], indirect=True)
def test_snapshot_pool_expired_queued_job_does_not_fail_healthy_sibling(wire_runtime):
    server, oracle, a = wire_runtime
    with peer(a, "b") as b, peer(a, "blocker") as blocker:
        ka, kb, kc = (
            c.open_sequence(0, deadline=c.deadline()) for c in (a, b, blocker)
        )
        oracle.release.clear()
        with ThreadPoolExecutor(max_workers=2) as threads:
            held = threads.submit(propose, blocker, snapshot(blocker, kc))
            assert oracle.entered.wait(2)
            handle = a.dispatch(
                snapshot(a, ka),
                10,
                torch.ones((1, 8), dtype=torch.bfloat16),
                deadline=time.monotonic() + 0.05,
            )
            healthy = threads.submit(propose, b, snapshot(b, kb))
            until(lambda: server.work_queue.depth == 3)
            with pytest.raises(RemoteDraftError):
                a.collect(handle)
            oracle.release.set()
            assert held.result().token_ids == healthy.result().token_ids == (18,)
        assert {i.key.verifier_instance_id for i, _ in oracle.calls} == {"blocker", "b"}
        until(lambda: not server.pending_snapshot_jobs)
        assert server.pending_snapshot_bytes == 0 and server.healthy


@pytest.mark.parametrize("wire_runtime", [_POOL_LIMITS], indirect=True)
def test_snapshot_pool_unknown_completion_fences_queued_verifier_work(wire_runtime):
    server, oracle, a = wire_runtime
    with peer(a, "b") as b:
        ka, kb = (c.open_sequence(0, deadline=c.deadline()) for c in (a, b))
        oracle.release.clear()
        with ThreadPoolExecutor(max_workers=2) as threads:
            pa = threads.submit(propose, a, snapshot(a, ka))
            assert oracle.entered.wait(2)
            pb = threads.submit(propose, b, snapshot(b, kb))
            until(lambda: server.work_queue.depth == 2)
            oracle.error = SnapshotCompletionError("injected unknown GPU completion")
            oracle.release.set()
            for future in (pa, pb):
                with pytest.raises(RemoteDraftError):
                    future.result()
        assert not server.healthy
        assert len(oracle.attempts) == 1
        until(lambda: server.pending_snapshot_jobs == 0)
        assert server.pending_snapshot_bytes == 0


@pytest.mark.parametrize(
    "wire_runtime",
    [{**_POOL_LIMITS, "shutdown_timeout_s": 0.05}],
    indirect=True,
)
def test_snapshot_pool_shutdown_retains_running_owner_until_process_restart(
    wire_runtime,
):
    server, oracle, client = wire_runtime
    key = client.open_sequence(0, deadline=client.deadline())
    oracle.release.clear()
    with ThreadPoolExecutor(max_workers=1) as threads:
        pending = threads.submit(propose, client, snapshot(client, key))
        assert oracle.entered.wait(2)
        try:
            with pytest.raises(ServerShutdownTimeout):
                server.stop()
            assert not server.healthy
            assert server.pending_snapshot_jobs == len(server.registry) == 1
            with pytest.raises(RuntimeError, match="unhealthy"):
                server.start(client.endpoint)
        finally:
            oracle.release.set()
        with pytest.raises(RemoteDraftError):
            pending.result()
    until(lambda: server.pending_snapshot_jobs == 0)
    assert len(server.registry) == 1  # Logical state retained after unsafe shutdown.
    assert len(oracle.attempts) == 1
    until(lambda: not server._connections and not server._acceptor.is_alive())


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


@pytest.mark.parametrize(
    "wire_runtime",
    [{"max_num_seqs": 3, "max_sequences_per_verifier": 1}],
    indirect=True,
)
def test_partial_batch_open_retires_unreturned_sequences(wire_runtime):
    """A strict rejection after the first open returns every local credit."""
    server, _, client = wire_runtime
    with pytest.raises(RemoteDraftError, match="sequence credit exhausted"):
        client.open_sequences((0, 1, 2), deadline=client.deadline())
    assert not client.records
    until(lambda: len(server.registry) == 0)
    key = client.open_sequence(3, deadline=client.deadline())
    assert client.active(key)
    client.close_sequences((key,))
    assert not client.records
