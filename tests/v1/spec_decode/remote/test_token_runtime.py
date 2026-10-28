# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU KV oracle and actual framed socket checks; no CUDA/model assertions."""

from copy import deepcopy
from dataclasses import replace

import pytest
from huggingface_hub.errors import (
    StrictDataclassClassValidationError,
    StrictDataclassFieldValidationError,
)

from vllm.v1.spec_decode.remote.capabilities import SpeculatorPlacementCapabilities
from vllm.v1.spec_decode.remote.protocol import (
    FeatureSlot,
    Ping,
    ProtocolVersionError,
    RemoteServerLimits,
    SequenceKey,
    SpeculatorStatusCode,
    TargetFeatureSchema,
    decode_envelope,
    encode_message,
)
from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
from vllm.v1.spec_decode.remote.runtime.schema import RuntimeContract
from vllm.v1.spec_decode.remote.runtime.token_adapter import TokenOnlyDraftAdapter
from vllm.v1.spec_decode.remote.runtime.token_schema import (
    tokenizer_artifact_fingerprint,
)
from vllm.v1.spec_decode.remote.runtime.token_state import (
    TokenAdvanceRow,
    TokenStateMachine,
    TokenStepExecutor,
)
from vllm.v1.spec_decode.remote.server import AdapterError
from vllm.v1.spec_decode.remote.session import RemoteDraftError
from vllm.v1.spec_decode.remote.token_protocol import (
    TokenHello,
    decode_token_message,
    encode_token_message,
)
from vllm.v1.spec_decode.remote.token_server import TokenDraftServer
from vllm.v1.spec_decode.remote.token_session import TokenOnlyRemoteSession


@pytest.mark.parametrize("handshake", [False, True])
def test_token_client_rejects_oversized_header_before_waiting_for_body(
    socket_runtime, handshake
):
    import socket
    import struct
    import time

    from vllm.v1.spec_decode.remote.transport import (
        FRAME_DATA,
        FramedConnection,
        TransportError,
    )

    _, _, _, original = socket_runtime
    client = TokenOnlyRemoteSession(original.endpoint, original.hello)
    ours, peer = socket.socketpair()
    client._conn = FramedConnection(ours)
    try:
        peer.sendall(struct.pack("!BI", FRAME_DATA, 1 << 20))
        with pytest.raises(TransportError, match="exceeds"):
            client._recv(1, time.monotonic() + 2, handshake=handshake)
        assert client._conn.closed
        assert len(client._conn._buffer) == 5
    finally:
        peer.close()
        client.close()


def test_token_server_releases_handshake_admission_and_bounds_data(socket_runtime):
    import struct

    from vllm.v1.spec_decode.remote.transport import FRAME_DATA, TransportError

    _, _, server, session = socket_runtime
    session.open_sequence(99, deadline=session.deadline())
    assert not server._handshaking
    assert session._conn is not None
    session._conn._sock.sendall(struct.pack("!BI", FRAME_DATA, 1 << 20))
    with pytest.raises(TransportError):
        session._conn.recv(timeout=2)


def test_token_session_construction_failure_releases_admission(
    socket_runtime, monkeypatch
):
    import threading

    from vllm.v1.spec_decode.remote import token_server
    from vllm.v1.spec_decode.remote.transport import ConnectionClosed, connect

    _, _, server, sibling = socket_runtime
    baseline = sibling.open_sequence(99, deadline=sibling.deadline())
    sibling.close_sequences((baseline,))
    server.max_handshakes = 1
    construct = token_server._TokenServerSession
    attempted = threading.Event()
    failed_workers: list[threading.Thread] = []
    failed_connections = []
    thread_errors: list[threading.ExceptHookArgs] = []

    def fail_once(owner, conn):
        if not failed_workers:
            failed_workers.append(threading.current_thread())
            failed_connections.append(conn)
            attempted.set()
            raise MemoryError("injected token session construction failure")
        return construct(owner, conn)

    replacement = TokenOnlyRemoteSession(
        sibling.endpoint, sibling.hello, request_timeout_ms=5000
    )
    failed_peer = None
    try:
        with monkeypatch.context() as patch:
            patch.setattr(threading, "excepthook", thread_errors.append)
            patch.setattr(token_server, "_TokenServerSession", fail_once)
            failed_peer = connect(sibling.endpoint, timeout=2)
            assert attempted.wait(2)
            failed_workers[0].join(timeout=2)
            assert not failed_workers[0].is_alive()
            assert not thread_errors
            assert failed_connections[0].closed
            with pytest.raises(ConnectionClosed):
                failed_peer.recv(timeout=2)
            assert len(server._connections) == 1
            assert not server._handshaking
            assert server.healthy

            replacement.connect()
            for sequence_id, client in enumerate((sibling, replacement)):
                key = client.open_sequence(sequence_id, deadline=client.deadline())
                client.prefill(key, (1,), is_final=True, deadline=client.deadline())
                result = client.collect(
                    client.dispatch(
                        0,
                        (TokenAdvanceRow(key, 1, None, 0, 0, (2,)),),
                        1,
                        deadline=client.deadline(),
                    )
                )
                assert result.valid_lengths == (1,)
                client.close_sequences((key,))
            assert not thread_errors
    finally:
        if failed_peer is not None:
            failed_peer.close()
        replacement.close()


@pytest.mark.parametrize("operation", ["close", "disconnect"])
@pytest.mark.parametrize("failure", [False, True])
def test_token_server_cleanup_fence_retains_owners_without_thread_errors(
    socket_runtime, monkeypatch, caplog, operation, failure
):
    import threading

    state, _, server, session = socket_runtime
    owner_thread = server._workers[0]
    key = session.open_sequence(0, deadline=session.deadline())
    session.prefill(key, (1,), is_final=True, deadline=session.deadline())
    sibling = TokenOnlyRemoteSession(session.endpoint, session.hello)
    thread_errors: list[threading.ExceptHookArgs] = []
    releases = []
    release = state.block_pool.release

    def return_blocks(req_index):
        releases.append(req_index)
        if failure:
            raise MemoryError("injected token block return failure")
        release(req_index)

    try:
        other = sibling.open_sequence(1, deadline=sibling.deadline())
        sibling.prefill(other, (2,), is_final=True, deadline=sibling.deadline())
        owners = {key: session._sid, other: sibling._sid}
        free_before = state.block_pool.num_free_blocks
        with monkeypatch.context() as patch:
            patch.setattr(threading, "excepthook", thread_errors.append)
            patch.setattr(state.block_pool, "release", return_blocks)
            if operation == "close":
                if failure:
                    with pytest.raises(RemoteDraftError, match="token close failed"):
                        session.close_sequences((key,))
                else:
                    session.close_sequences((key,))
            session.close()
            owner_thread.join(timeout=2)
            assert not owner_thread.is_alive()
            assert len(releases) == 1

            if failure:
                assert server._cleanup_failed and not server.healthy
                assert server.wait(timeout=0)
                assert set(state.sequences) == {key, other}
                assert state.block_pool.num_free_blocks == free_before
                for sequence, owner in owners.items():
                    entry = server.registry.get(sequence)
                    assert entry is not None and entry.owner == owner
                with pytest.raises(RemoteDraftError):
                    sibling.collect(
                        sibling.dispatch(
                            0,
                            (TokenAdvanceRow(other, 1, None, 0, 0, (3,)),),
                            1,
                            deadline=sibling.deadline(),
                        )
                    )
            else:
                assert server.healthy and not server.wait(timeout=0)
                assert server.registry.get(key) is None
                assert set(state.sequences) == {other}
                assert state.block_pool.num_free_blocks == free_before + 1
                assert sibling.collect(
                    sibling.dispatch(
                        0,
                        (TokenAdvanceRow(other, 1, None, 0, 0, (3,)),),
                        1,
                        deadline=sibling.deadline(),
                    )
                ).valid_lengths == (1,)
            sibling.close()
            server.stop()
            assert not thread_errors
            assert len(releases) == (1 if failure else 2)
            if failure:
                assert "injected token block return failure" in caplog.text
                assert len(server.registry) == 2
                assert set(state.sequences) == {key, other}
            else:
                assert not state.sequences and not len(server.registry)
            assert not server._connections and not server._handshaking
    finally:
        sibling.close()
        server.stop()
        # A fenced real runtime exits; this CPU oracle can explicitly free its pool.
        state.close(tuple(state.sequences))


class KVOracle(TokenStepExecutor):
    """Store per-slot position tokens and sample a prefix-dependent checksum."""

    def __init__(self):
        self.kv = {}
        self.calls = []
        self.warmed = False

    def warmup(self):
        self.warmed = True

    @staticmethod
    def predict(prefix):
        return sum((i + 1) * token for i, token in enumerate(prefix)) % 127

    def forward(self, key, req_index, start, tokens, blocks, *, sample):
        kv = self.kv.setdefault(req_index, {})
        assert all(i in kv for i in range(start))
        assert start + len(tokens) <= len(blocks) * 4
        for i, token in enumerate(tokens, start):
            kv[i] = token
        self.calls.append((key, start, tokens, sample))
        if sample:
            return self.predict([kv[i] for i in range(start + len(tokens))])
        return None


def machine(*, max_len=64, blocks=33, k=3, executor=None):
    oracle = KVOracle() if executor is None else executor
    pool = DraftBlockPool(blocks, 4)
    return TokenStateMachine(
        oracle, pool, max_num_reqs=4, max_model_len=max_len, vocab_size=128, max_k=k
    ), oracle


class BatchedKVOracle(KVOracle):
    def __init__(self, *, token_limit=None, row_limit=None, failure=None):
        super().__init__()
        self.batches = []
        self.token_limit, self.row_limit = token_limit, row_limit
        self.failure = failure

    @property
    def max_num_tokens(self):
        return self.token_limit

    @property
    def max_num_reqs(self):
        return self.row_limit

    def forward_batch(self, rows):
        self.batches.append(rows)
        if self.failure is not None:
            raise self.failure
        return super().forward_batch(rows)


def test_batched_token_state_matches_independent_rounds_with_ragged_corrections():
    oracle = BatchedKVOracle()
    batched, _ = machine(executor=oracle)
    reference, _ = machine()
    keys = tuple(SequenceKey("batch", i, 0) for i in range(3))
    for state in (batched, reference):
        for i, key in enumerate(keys):
            state.open(key)
            state.prefill(key, 0, tuple(range(1, i + 2)), is_final=True)
    updates = tuple(
        (10 + i, TokenAdvanceRow(key, i + 1, None, 0, 0, (7, 8)), k)
        for i, (key, k) in enumerate(zip(keys, (3, 0, 2)))
    )
    expected = tuple(reference.advance(*update) for update in updates)
    assert batched.advance_many(updates) == expected
    assert [len(rows) for rows in oracle.batches] == [3, 2, 1]
    assert [sum(len(row.tokens) for row in rows) for rows in oracle.batches] == [
        6,
        2,
        1,
    ]
    # Distinct client batch IDs, reordering and all/partial rejection retain the
    # same prefix alignment as the established one-sequence execution path.
    following = []
    for i in (2, 0, 1):
        previous = expected[i]
        accepted = len(previous.token_ids) // 2
        following.append(
            (
                20 + i,
                TokenAdvanceRow(
                    keys[i],
                    previous.committed,
                    previous.proposal_id,
                    len(previous.token_ids),
                    accepted,
                    previous.token_ids[:accepted] + (5,),
                ),
                (0, 3, 1)[i],
            )
        )
    assert batched.advance_many(tuple(following)) == tuple(
        reference.advance(*update) for update in following
    )
    for key in keys:
        assert batched.sequences[key] == reference.sequences[key]
        index = batched.sequences[key].req_index
        assert oracle.kv[index] == reference.executor.kv[index]


def test_batched_token_admission_failure_and_expired_row_do_not_delay_sibling():
    oracle = BatchedKVOracle()
    state, _ = machine(max_len=8, executor=oracle)
    keys = tuple(SequenceKey("batch", i, 0) for i in range(3))
    for key in keys:
        state.open(key)
        state.prefill(key, 0, (1,), is_final=True)
    bad = TokenAdvanceRow(keys[0], 1, None, 0, 0, (2,) * 8)
    expired = TokenAdvanceRow(keys[1], 1, None, 0, 0, (2,))
    valid = TokenAdvanceRow(keys[2], 1, None, 0, 0, (3,))
    failed, timed_out, result = state.advance_many(
        ((1, bad, 1), (2, expired, 3), (3, valid, 3)),
        deadlines=(None, 0, None),
    )
    assert isinstance(failed, AdapterError)
    assert failed.status == SpeculatorStatusCode.OUT_OF_MEMORY
    assert isinstance(timed_out, AdapterError)
    assert timed_out.status == SpeculatorStatusCode.TIMEOUT
    assert result.committed == 2 and len(result.token_ids) == 3
    assert state.sequences[keys[0]].committed == state.sequences[keys[1]].committed == 1
    assert all(row.key == keys[2] for batch in oracle.batches for row in batch)


def test_batched_token_execution_stays_within_both_scratch_budgets():
    oracle = BatchedKVOracle(token_limit=3, row_limit=2)
    state, _ = machine(executor=oracle)
    keys = tuple(SequenceKey("batch", i, 0) for i in range(4))
    for key in keys:
        state.open(key)
        state.prefill(key, 0, (1,), is_final=True)
    results = state.advance_many(
        tuple(
            (i, TokenAdvanceRow(key, 1, None, 0, 0, (2, 3)), 3)
            for i, key in enumerate(keys)
        )
    )
    assert all(len(result.token_ids) == 3 for result in results)
    assert all(
        len(rows) <= 2 and sum(len(row.tokens) for row in rows) <= 3
        for rows in oracle.batches
    )
    assert [len(rows) for rows in oracle.batches] == [1, 1, 1, 1, 2, 2, 2, 2]


def test_batched_token_fatal_forward_stops_all_subsequent_gpu_submissions():
    failure = RuntimeError("CUDA error: device-side assert triggered")
    oracle = BatchedKVOracle(row_limit=1, failure=failure)
    state, _ = machine(executor=oracle)
    keys = tuple(SequenceKey("batch", i, 0) for i in range(2))
    for key in keys:
        state.open(key)
        state.prefill(key, 0, (1,), is_final=True)
    results = state.advance_many(
        tuple(
            (i, TokenAdvanceRow(key, 1, None, 0, 0, (2,)), 3)
            for i, key in enumerate(keys)
        )
    )
    assert all(result is failure for result in results)
    assert len(oracle.batches) == 1
    assert all(state.sequences[key].committed == 1 for key in keys)
    assert not state.sequences[keys[0]].ready


def test_token_close_reuses_all_slots_without_growing_free_storage():
    class NoGrowth(list):
        def append(self, value):
            raise MemoryError("injected slot-list growth failure")

    state, _ = machine(blocks=5)
    state._free = NoGrowth(state._free)
    for generation in range(3):
        keys = tuple(SequenceKey("v", i, generation) for i in range(4))
        for key in keys:
            state.open(key)
            state.prefill(key, 0, (1,), is_final=True)
        assert state.block_pool.num_free_blocks == 0
        state.close(keys)
        state.close(keys)
        assert not state.sequences and state.block_pool.num_free_blocks == 4


def test_token_failed_close_preserves_slot_for_retry(monkeypatch):
    state, _ = machine(blocks=5)
    keys = tuple(SequenceKey("v", i, 0) for i in range(4))
    for key in keys:
        state.open(key)
        state.prefill(key, 0, (1,), is_final=True)

    def fail_release(req_index):
        raise MemoryError("injected block return failure")

    with monkeypatch.context() as patch:
        patch.setattr(state.block_pool, "release", fail_release)
        with pytest.raises(MemoryError):
            state.close(keys)
    assert set(state.sequences) == set(keys)
    state.close(keys)
    assert not state.sequences and state.block_pool.num_free_blocks == 4
    for key in keys:
        state.open(key)
    assert {state.sequences[key].req_index for key in keys} == {0, 1, 2, 3}


@pytest.mark.parametrize("failure", ["record", "ownership"])
def test_token_open_metadata_oom_preserves_all_slots(monkeypatch, failure):
    import vllm.v1.spec_decode.remote.runtime.token_state as token_state

    state, _ = machine()
    failed_key = SequenceKey("v", 99, 0)

    def fail_record(*args):
        raise MemoryError("injected request allocation failure")

    class FailInsert(dict):
        def __setitem__(self, key, value):
            raise MemoryError("injected ownership allocation failure")

    with monkeypatch.context() as patch:
        if failure == "record":
            patch.setattr(token_state, "TokenProgress", fail_record)
        else:
            patch.setattr(state, "sequences", FailInsert())
        with pytest.raises(MemoryError):
            state.open(failed_key)
    state.close((failed_key,))
    keys = tuple(SequenceKey("v", i, 0) for i in range(4))
    for key in keys:
        state.open(key)
    assert {state.sequences[key].req_index for key in keys} == {0, 1, 2, 3}


@pytest.fixture
def socket_runtime(tmp_path, request):
    state, oracle = machine(executor=BatchedKVOracle())
    contract = RuntimeContract(
        TargetFeatureSchema(3, (FeatureSlot("token_ids", "int32", ()),)),
        SpeculatorPlacementCapabilities(
            state_dependency="own_kv",
            required_features=("token_ids",),
            standalone_weights="complete",
        ),
        "target",
        "tokenizer",
        "draft",
    )
    server = TokenDraftServer(
        TokenOnlyDraftAdapter(state, contract),
        limits=RemoteServerLimits(4, 64, 4, 64),
        max_connections=getattr(request, "param", 16),
    )
    endpoint = server.start(f"unix://{tmp_path}/t.sock")
    hello = TokenHello("verifier", "target", "draft", "tokenizer", 3, 128, 4, 64, 64)
    session = TokenOnlyRemoteSession(endpoint, hello, request_timeout_ms=5000)
    try:
        session.connect()
        yield state, oracle, server, session
    finally:
        session.close()
        server.stop()
        assert not state.sequences
        assert state.block_pool.num_free_blocks == state.block_pool.num_blocks - 1


@pytest.mark.parametrize("socket_runtime", [1], indirect=True)
def test_token_server_socket_close_error_releases_connection_admission(
    socket_runtime, monkeypatch, caplog
):
    import os
    import threading

    state, oracle, server, session = socket_runtime
    key = session.open_sequence(0, deadline=session.deadline())
    session.prefill(key, (1,), is_final=True, deadline=session.deadline())
    with server._connections_lock:
        (connection,) = server._connections
        worker = server._workers[0]
    thread_errors: list[threading.ExceptHookArgs] = []
    monkeypatch.setattr(threading, "excepthook", thread_errors.append)
    os.close(connection._sock.fileno())
    session.close()
    worker.join(timeout=2)
    assert not worker.is_alive()
    with server._connections_lock:
        assert not server._connections and not server._handshaking
    assert not state.sequences and server.healthy
    assert not thread_errors
    assert "Token connection cleanup failed" in caplog.text

    replacement = TokenOnlyRemoteSession(
        session.endpoint, session.hello, request_timeout_ms=5000
    )
    try:
        key = replacement.open_sequence(1, deadline=replacement.deadline())
        replacement.prefill(key, (2,), is_final=True, deadline=replacement.deadline())
        result = replacement.collect(
            replacement.dispatch(
                0,
                (TokenAdvanceRow(key, 1, None, 0, 0, (3,)),),
                1,
                deadline=replacement.deadline(),
            )
        )
        assert result.valid_lengths == (1,)
        assert result.token_ids[0][0] == oracle.predict([2, 3])
    finally:
        replacement.close()


@pytest.mark.parametrize("started", [False, True])
def test_token_acceptor_start_failure_rolls_back_and_allows_restart(
    socket_runtime, monkeypatch, tmp_path, started
):
    import threading

    _, _, server, original = socket_runtime
    original.close()
    server.stop()
    path = tmp_path / "restart.sock"
    endpoint = f"unix://{path}"

    class FailingThread(threading.Thread):
        def start(self):
            if self.name == "token-draft-acceptor":
                if started:
                    super().start()
                raise RuntimeError("injected token acceptor start failure")
            return super().start()

    with monkeypatch.context() as patch:
        patch.setattr(threading, "Thread", FailingThread)
        with pytest.raises(RuntimeError, match="injected token acceptor"):
            server.start(endpoint)
    assert not path.exists()
    assert server.endpoint is None
    assert server.healthy
    server.start(endpoint)
    replacement = TokenOnlyRemoteSession(endpoint, original.hello)
    try:
        key = replacement.open_sequence(55, deadline=replacement.deadline())
        replacement.prefill(key, (1,), is_final=True, deadline=replacement.deadline())
        assert key in server.adapter.state.sequences
    finally:
        replacement.close()
        server.stop()


@pytest.mark.parametrize("policy", ["error", "target_only"])
def test_token_cleanup_error_preserves_failure_policy_and_terminal_state(
    socket_runtime, monkeypatch, policy
):
    import errno
    import socket

    _, _, _, session = socket_runtime
    session.failure_policy = policy
    key, deferred = (
        session.open_sequence(i, deadline=session.deadline()) for i in range(2)
    )
    session.prefill(key, (1,), is_final=True, deadline=session.deadline())
    handle = session.dispatch(
        0, (TokenAdvanceRow(key, 1, None, 0, 0, (2,)),), 1, deadline=session.deadline()
    )
    session.close_sequences((deferred,))
    connection = session._conn
    close_socket = socket.socket.close

    def close_then_fail(sock):
        close_socket(sock)
        if sock is connection._sock:
            raise OSError(errno.EIO, "injected socket cleanup failure")

    with monkeypatch.context() as patch:
        patch.setattr(socket.socket, "close", close_then_fail)
        if policy == "error":
            with pytest.raises(RemoteDraftError, match="original token failure"):
                session.fail("original token failure")
        else:
            session.fail("original token failure")
    assert connection.closed and session._conn is None
    assert not session.active(key)
    assert deferred not in session.records and not session._deferred_closes
    assert session.collect(handle).valid_lengths == (0,)
    assert not handle.active and session._pending is None


@pytest.mark.parametrize("operation", ["cancel", "close"])
def test_token_cleanup_error_finishes_cancelled_handle(
    socket_runtime, monkeypatch, operation
):
    import errno
    import socket

    _, _, _, session = socket_runtime
    key = session.open_sequence(0, deadline=session.deadline())
    session.prefill(key, (1,), is_final=True, deadline=session.deadline())
    handle = session.dispatch(
        0, (TokenAdvanceRow(key, 1, None, 0, 0, (2,)),), 1, deadline=session.deadline()
    )
    connection = session._conn
    close_socket = socket.socket.close

    def close_then_fail(sock):
        close_socket(sock)
        if sock is connection._sock:
            raise OSError(errno.EIO, "injected socket cleanup failure")

    with monkeypatch.context() as patch:
        patch.setattr(socket.socket, "close", close_then_fail)
        if operation == "cancel":
            session.cancel(handle)
        else:
            session.close()
    assert connection.closed and session._conn is None
    assert not session.active(key)
    assert not handle.active and session._pending is None
    with pytest.raises(RuntimeError, match="stale, foreign or collected"):
        session.collect(handle)
    if operation == "close":
        assert not session.records
        with pytest.raises(RuntimeError, match="closed"):
            session.connect()
    else:
        next_handle = session.dispatch(1, (), 0, deadline=session.deadline())
        assert session.collect(next_handle).valid_lengths == ()


def test_token_socket_batches_rows_at_each_draft_depth(socket_runtime):
    state, oracle, _, session = socket_runtime
    keys = tuple(
        session.open_sequence(i, deadline=session.deadline()) for i in range(3)
    )
    prompts = ((1,), (2, 3), (4, 5, 6))
    for key, prompt in zip(keys, prompts):
        session.prefill(key, prompt, is_final=True, deadline=session.deadline())
    rows = tuple(
        TokenAdvanceRow(key, len(prompt), None, 0, 0, (7, 8))
        for key, prompt in zip(keys, prompts)
    )
    result = session.collect(session.dispatch(7, rows, 3, deadline=session.deadline()))
    expected = []
    for prompt in prompts:
        prefix = list(prompt) + [7, 8]
        draft = []
        for _ in range(3):
            token = KVOracle.predict(prefix)
            draft.append(token)
            prefix.append(token)
        expected.append(tuple(draft))
    assert result.token_ids == tuple(expected)
    assert result.valid_lengths == (3, 3, 3)
    assert [len(batch) for batch in oracle.batches] == [3, 3, 3]
    assert all(
        state.sequences[key].committed == len(prompt) + 2
        for key, prompt in zip(keys, prompts)
    )


def test_protocol_majors_are_explicit_and_v2_unchanged():
    v2 = encode_message(Ping(1))
    v3 = encode_token_message(Ping(1))
    assert decode_envelope(v2).protocol_major == 2
    assert decode_token_message(v3)[0].protocol_major == 3
    assert decode_token_message(v3)[0].protocol_minor == 1
    with pytest.raises(ProtocolVersionError):
        decode_token_message(v2)
    with pytest.raises(ProtocolVersionError):
        decode_envelope(v3)


@pytest.mark.parametrize("phase", ["prefill", "proposal"])
def test_token_budget_refresh_after_data_and_legacy_minor_decode(
    socket_runtime, monkeypatch, phase
):
    from types import SimpleNamespace

    import msgspec

    from vllm.v1.spec_decode.remote.token_protocol import TokenAdvance, TokenPrefill

    _, _, _, session = socket_runtime
    key = SequenceKey("verifier", 0, 0)
    message = (
        TokenPrefill(key, 0, 1, 0, 0, True)
        if phase == "prefill"
        else TokenAdvance(0, (key,), 0, 1, 2, 3, 4, 5, 1)
    )
    now, controls = [0.0], []

    def send_data(frame, *, deadline):
        now[0] += 0.25

    with monkeypatch.context() as patch:
        patch.setattr(session, "clock", lambda: now[0])
        patch.setattr(
            session,
            "_conn",
            SimpleNamespace(
                send_data=send_data,
                send_control=lambda data, **_: controls.append(data),
            ),
        )
        session._send(message, 1, 1.0, frames=(None, None))
    envelope, decoded = decode_token_message(controls[0])
    assert decoded.execution_timeout_ms == 500
    # Old commands omit the optional field; newer peers retain their behavior.
    payload = msgspec.msgpack.decode(envelope.payload)
    payload.pop("execution_timeout_ms")
    envelope.protocol_minor = 0
    envelope.payload = msgspec.msgpack.encode(payload)
    legacy = decode_token_message(msgspec.msgpack.encode(envelope))[1]
    assert legacy.execution_timeout_ms is None


@pytest.mark.parametrize("phase", ["prefill", "proposal"])
@pytest.mark.parametrize("expired", [False, True])
@pytest.mark.parametrize("policy", ["target_only", "error"])
def test_token_queued_expiry_skips_forward_and_preserves_other_client(
    socket_runtime, phase, expired, policy
):
    """An expired client must not start work when another adapter turn ends."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import msgspec

    state, oracle, server, healthy = socket_runtime
    waiting = TokenOnlyRemoteSession(
        server.endpoint,
        msgspec.structs.replace(healthy.hello, verifier_instance_id="waiting"),
        failure_policy=policy,
        request_timeout_ms=1000,
    )
    try:
        healthy_key = healthy.open_sequence(0, deadline=healthy.deadline())
        healthy.prefill(healthy_key, (1,), is_final=True, deadline=healthy.deadline())
        key = waiting.open_sequence(0, deadline=waiting.deadline())
        if phase == "proposal":
            waiting.prefill(key, (1,), is_final=True, deadline=waiting.deadline())
        before = len(oracle.calls)
        if expired:
            waiting.request_timeout_s = 0.05

        def request():
            deadline = waiting.deadline()
            if phase == "prefill":
                waiting.prefill(key, (1,), is_final=True, deadline=deadline)
                return None
            return waiting.collect(
                waiting.dispatch(
                    0,
                    (TokenAdvanceRow(key, 1, None, 0, 0, (2,)),),
                    1,
                    deadline=deadline,
                )
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            with server.adapter_turn():
                future = executor.submit(request)
                limit = time.monotonic() + 1
                while server.queue_depth != 2:
                    assert time.monotonic() < limit
                    threading.Event().wait(0.001)
                if expired:
                    # The server starts its relative budget after receipt.
                    # Client timeout alone does not prove server expiry.
                    server_expiry_bound = time.monotonic() + waiting.request_timeout_s
                    if policy == "error":
                        with pytest.raises(
                            RemoteDraftError, match="timed out|deadline"
                        ):
                            future.result(timeout=1)
                    else:
                        result = future.result(timeout=1)
                        if phase == "proposal":
                            assert result.valid_lengths == (0,)
                    assert waiting._conn is None
                    assert waiting.records[key].terminal
                    threading.Event().wait(
                        max(0, server_expiry_bound - time.monotonic())
                    )
            if not expired:
                result = future.result(timeout=1)
                if phase == "proposal":
                    assert result.valid_lengths == (1,)

        # Closing the client waits for neither server execution nor cleanup.
        # Wait for the queued handler to finish before inspecting its calls.
        limit = time.monotonic() + 1
        while server.queue_depth or (expired and key in state.sequences):
            assert time.monotonic() < limit
            threading.Event().wait(0.001)
        calls = [call for call in oracle.calls[before:] if call[0] == key]
        assert bool(calls) is not expired
        assert server.healthy
        result = healthy.collect(
            healthy.dispatch(
                0,
                (TokenAdvanceRow(healthy_key, 1, None, 0, 0, (2,)),),
                1,
                deadline=healthy.deadline(),
            )
        )
        assert result.valid_lengths == (1,)
    finally:
        waiting.close()


def test_token_row_expiry_preserves_completed_and_unrelated_rows(
    socket_runtime, monkeypatch
):
    import time
    from types import SimpleNamespace

    from vllm.v1.spec_decode.remote import server as server_module
    from vllm.v1.spec_decode.remote import token_server
    from vllm.v1.spec_decode.remote.runtime import token_state

    state, oracle, server, session = socket_runtime
    session.failure_policy = "target_only"
    keys = [session.open_sequence(i, deadline=session.deadline()) for i in range(3)]
    for key in keys:
        session.prefill(key, (1,), is_final=True, deadline=session.deadline())
    oracle.row_limit = 1
    before = len(oracle.calls)
    now = [time.monotonic()]
    forward = oracle.forward

    def finish_first_row(key, *args, **kwargs):
        result = forward(key, *args, **kwargs)
        now[0] += 10
        return result

    with monkeypatch.context() as patch:
        clock = SimpleNamespace(monotonic=lambda: now[0])
        patch.setattr(token_server, "time", clock)
        patch.setattr(server_module, "time", clock)
        patch.setattr(token_state, "time", clock)
        patch.setattr(oracle, "forward", finish_first_row)
        result = session.collect(
            session.dispatch(
                0,
                tuple(TokenAdvanceRow(key, 1, None, 0, 0, (2,)) for key in keys[:2]),
                1,
                deadline=session.deadline(),
            )
        )
    assert result.valid_lengths == (1, 0)
    assert [call[0] for call in oracle.calls[before:]] == [keys[0]]
    assert set(state.sequences) == {keys[0], keys[2]}
    assert session.active(keys[0]) and session.active(keys[2])
    assert not session.active(keys[1])
    assert server.healthy
    result = session.collect(
        session.dispatch(
            1,
            (TokenAdvanceRow(keys[2], 1, None, 0, 0, (2,)),),
            1,
            deadline=session.deadline(),
        )
    )
    assert result.valid_lengths == (1,)


@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
@pytest.mark.parametrize("next_k", [0, 1, 3])
def test_real_state_rejection_all_acceptance_and_k0(accepted, next_k):
    state, oracle = machine()
    key = SequenceKey("v", 1, 0)
    state.open(key)
    state.prefill(key, 0, (1, 2), is_final=False)
    state.prefill(key, 2, (3,), is_final=True)
    first = state.advance(1, TokenAdvanceRow(key, 3, None, 0, 0, (7,)), 3)
    prefix = [1, 2, 3, 7]
    for token in first.token_ids:
        assert token == oracle.predict(prefix)
        prefix.append(token)
    oracle.row_limit = 1
    before = len(oracle.calls)
    delta = first.token_ids[:accepted] + (9,)
    second = state.advance(2, TokenAdvanceRow(key, 4, 1, 3, accepted, delta), next_k)
    expected = [1, 2, 3, 7, *delta]
    current = state.sequences[key]
    assert current.committed == current.materialized == len(expected)
    assert [oracle.kv[current.req_index][i] for i in range(len(expected))] == expected
    assert current.speculative_written_end == len(expected) + max(next_k - 1, 0)
    assert len(oracle.calls[before][2]) == (2 if accepted == 3 else 1)
    assert len(second.token_ids) == next_k
    assert current.pending_last_token == (
        (current.speculative_written_end, second.token_ids[-1]) if next_k else None
    )
    if next_k == 0:
        assert len(oracle.calls) == before + 1
        third = state.advance(
            3, TokenAdvanceRow(key, len(expected), 2, 0, 0, (11, 12)), 1
        )
        assert third.token_ids == (oracle.predict(expected + [11, 12]),)


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_length", 3),
        ("previous_proposal_id", 0),
        ("accepted", 4),
        ("offered", 4),
        ("delta", (99, 9)),
    ],
)
def test_stale_identity_and_bad_acceptance_reject_before_mutation(field, value):
    state, oracle = machine()
    key = SequenceKey("v", 1, 0)
    state.open(key)
    state.prefill(key, 0, (1,), is_final=True)
    first = state.advance(1, TokenAdvanceRow(key, 1, None, 0, 0, (2,)), 3)
    row = TokenAdvanceRow(key, 2, 1, 3, 1, first.token_ids[:1] + (9,))
    before = deepcopy(
        (state.sequences, oracle.kv, oracle.calls, state.block_pool.blocks(0))
    )
    with pytest.raises(AdapterError):
        state.advance(2, replace(row, **{field: value}), 1)
    assert (
        state.sequences,
        oracle.kv,
        oracle.calls,
        state.block_pool.blocks(0),
    ) == before


def test_actual_socket_prefill_full_acceptance_k0_and_close(socket_runtime):
    state, oracle, server, session = socket_runtime
    assert oracle.warmed
    key = session.open_sequence(0, deadline=session.deadline())
    session.prefill(key, (1, 2), is_final=False, deadline=session.deadline())
    session.prefill(key, (3,), is_final=True, deadline=session.deadline())
    first = session.collect(
        session.dispatch(
            0,
            (TokenAdvanceRow(key, 3, None, 0, 0, (7,)),),
            3,
            deadline=session.deadline(),
        )
    )
    delta = first.token_ids[0] + (9,)
    second = session.collect(
        session.dispatch(
            1,
            (TokenAdvanceRow(key, 4, 0, 3, 3, delta),),
            0,
            deadline=session.deadline(),
        )
    )
    assert second.valid_lengths == (0,)
    assert session.records[key].committed == 8
    assert state.sequences[key].committed == state.sequences[key].materialized == 8
    third = session.collect(
        session.dispatch(
            2,
            (TokenAdvanceRow(key, 8, 1, 0, 0, (10,)),),
            1,
            deadline=session.deadline(),
        )
    )
    assert third.valid_lengths == (1,)
    session.close_sequences((key,))
    assert not state.sequences
    assert len(server.registry) == 0


def test_actual_socket_stale_proposal_terminal_and_sibling_survives(socket_runtime):
    state, _, _, session = socket_runtime
    session.failure_policy = "target_only"
    keys = tuple(
        session.open_sequence(i, deadline=session.deadline()) for i in range(2)
    )
    for key in keys:
        session.prefill(key, (1,), is_final=True, deadline=session.deadline())
    first = session.collect(
        session.dispatch(
            0,
            tuple(TokenAdvanceRow(key, 1, None, 0, 0, (2,)) for key in keys),
            1,
            deadline=session.deadline(),
        )
    )
    assert first.valid_lengths == (1, 1)
    rows = (
        TokenAdvanceRow(keys[0], 2, None, 0, 0, (3,)),
        TokenAdvanceRow(keys[1], 2, 0, 0, 0, (3,)),
    )
    result = session.collect(session.dispatch(1, rows, 1, deadline=session.deadline()))
    assert result.valid_lengths == (0, 1)
    assert not session.active(keys[0]) and session.active(keys[1])
    assert keys[0] not in state.sequences and keys[1] in state.sequences


@pytest.mark.parametrize(
    "semantic_field",
    ["add_prefix_space", "do_lower_case", "extra_ids", "chat_template"],
)
def test_tokenizer_identity_hashes_all_configuration(tmp_path, semantic_field):
    import json

    (tmp_path / "tokenizer.json").write_text('{"vocab":{"a":0}}')
    config = tmp_path / "tokenizer_config.json"
    config.write_text("{}")
    before = tokenizer_artifact_fingerprint(str(tmp_path), None)
    config.write_text(json.dumps({semantic_field: True}))
    assert tokenizer_artifact_fingerprint(str(tmp_path), None) != before


def test_remote_tokenizer_mutable_revision_rejected():
    with pytest.raises(ValueError, match="immutable"):
        tokenizer_artifact_fingerprint("Qwen/Qwen3-8B", "main")


def test_actual_socket_malformed_result_exceeding_k0_is_fenced(
    socket_runtime, monkeypatch
):
    from vllm.v1.spec_decode.remote.token_protocol import TokenProposal
    from vllm.v1.spec_decode.remote.token_server import _TokenServerSession
    from vllm.v1.spec_decode.remote.transport import ints_to_frame

    _, _, _, session = socket_runtime
    send = _TokenServerSession._send

    def corrupt(self, message, request_id, **kwargs):
        if isinstance(message, TokenProposal):
            assert message.result_slot is not None
            frames = list(kwargs["frames"])
            frames[1] = ints_to_frame(message.result_slot + 1, [1])
            kwargs["frames"] = frames
        return send(self, message, request_id, **kwargs)

    monkeypatch.setattr(_TokenServerSession, "_send", corrupt)
    key = session.open_sequence(0, deadline=session.deadline())
    session.prefill(key, (1,), is_final=True, deadline=session.deadline())
    handle = session.dispatch(
        0, (TokenAdvanceRow(key, 1, None, 0, 0, (2,)),), 0, deadline=session.deadline()
    )
    with pytest.raises(RemoteDraftError, match="invalid token proposal values"):
        session.collect(handle)
    assert session.records[key].terminal and session._conn is None
    assert session.records[key].committed == 1
    with pytest.raises(RuntimeError, match="collected"):
        session.collect(handle)


def test_expired_buffered_socket_result_is_not_consumed(socket_runtime):
    import time

    from vllm.v1.spec_decode.remote.transport import _HEADER, TransportTimeout

    _, _, _, session = socket_runtime
    session._send(Ping(999), 999, session.deadline())
    conn = session._conn
    deadline = time.monotonic() + 5
    conn._fill(_HEADER.size, deadline)
    _, length = _HEADER.unpack_from(conn._buffer)
    conn._fill(_HEADER.size + length, deadline)
    buffered = bytes(conn._buffer)
    session.clock = lambda: 2.0
    with pytest.raises(TransportTimeout, match="deadline expired"):
        session._recv(999, 1.0)
    assert bytes(conn._buffer) == buffered


@pytest.mark.parametrize("kind", ["ordinary", "oom", "wrapped_fatal"])
def test_actual_socket_error_health_fencing(socket_runtime, monkeypatch, kind):
    import torch

    state, oracle, server, session = socket_runtime
    session.failure_policy = "target_only"
    keys = tuple(
        session.open_sequence(i, deadline=session.deadline()) for i in range(2)
    )
    for key in keys:
        session.prefill(key, (1,), is_final=True, deadline=session.deadline())
    forward = oracle.forward
    calls = []

    def fail_once(key, *args, **kwargs):
        calls.append(key)
        if len(calls) == 1:
            if kind == "wrapped_fatal":
                try:
                    raise RuntimeError(
                        "CUDA error: an illegal memory access was encountered"
                    )
                except RuntimeError as cause:
                    failure = AdapterError(
                        SpeculatorStatusCode.INTERNAL_ERROR, "wrapped"
                    )
                    assert isinstance(failure, AdapterError)
                    raise failure from cause
            if kind == "oom":
                raise torch.OutOfMemoryError("allocation failed")
            raise RuntimeError("ordinary model failure")
        return forward(key, *args, **kwargs)

    monkeypatch.setattr(oracle, "forward", fail_once)
    result = session.collect(
        session.dispatch(
            0,
            tuple(TokenAdvanceRow(key, 1, None, 0, 0, (2,)) for key in keys),
            1,
            deadline=session.deadline(),
        )
    )
    assert result.valid_lengths == ((0, 0) if kind == "wrapped_fatal" else (0, 1))
    assert server.healthy == (kind != "wrapped_fatal")
    assert len(calls) == (1 if kind == "wrapped_fatal" else 2)
    if kind == "wrapped_fatal":
        assert server.wait(timeout=0)
        # Fatal cleanup must avoid further adapter/CUDA work. The owner process
        # exits; this synthetic CPU test may explicitly release its fake pool.
        state.close(tuple(state.sequences))


def worker_for(session):
    import itertools
    from types import SimpleNamespace

    import torch

    from vllm.v1.worker.gpu.spec_decode.token_remote import TokenOnlyRemoteSpeculator

    worker = object.__new__(TokenOnlyRemoteSpeculator)
    worker.device = torch.device("cpu")
    worker.num_speculative_steps = 3
    worker._tp = SimpleNamespace(
        rank_in_group=0, broadcast_object=lambda value, src: value
    )
    worker._owner = object()
    worker._rounds = itertools.count()
    worker._sequences = itertools.count()
    worker._requests = {}
    worker._terminal_requests = set()
    worker._pending = None
    worker._closed = False
    worker._session = session
    return worker


def target_context(
    rows, *, physical=(3, 1), sampled=None, rejected=None, recovery=None
):
    """Actual postprocess shapes; extra physical slots/padded tokens are poison."""
    from types import SimpleNamespace

    import numpy as np
    import torch

    from vllm.v1.spec_decode.contracts import LocalExecutionContext

    lengths = [len(row[1]) for row in rows]
    starts = np.array([0, *np.cumsum(lengths)], dtype=np.int32)
    size = len(rows)
    last = torch.full((4,), 127, dtype=torch.int64)
    for i, slot in enumerate(physical[:size]):
        last[slot] = recovery[i]
    batch = SimpleNamespace(
        req_ids=[row[0] for row in rows],
        num_reqs=size,
        num_tokens=sum(lengths),
        query_start_loc_np=starts,
        is_prefilling_np=np.array([row[3] for row in rows]),
        prefill_len_np=np.array([row[4] for row in rows]),
        input_ids=torch.tensor(
            [token for row in rows for token in row[1]] + [999, 999], dtype=torch.int32
        ),
        positions=torch.tensor(
            [position for row in rows for position in row[2]] + [999, 999],
            dtype=torch.int64,
        ),
        idx_mapping=torch.tensor(physical[:size], dtype=torch.int64),
    )
    return LocalExecutionContext(
        batch,
        None,
        None,
        None,
        None,
        torch.tensor(sampled),
        torch.tensor(rejected),
        last,
        torch.empty((1, 4), dtype=torch.int32),
        torch.zeros(4),
        torch.zeros(4, dtype=torch.int64),
    )


def step(worker, context, k):
    from vllm.v1.spec_decode.contracts import DraftStep

    return worker.collect(
        worker.dispatch(DraftStep.from_batch(context.input_batch, k), context)
    )


@pytest.mark.parametrize("k", [0, 1, 3])
def test_worker_empty_batch_preserves_result_shape_without_rpc(k):
    from unittest.mock import Mock, call

    from vllm.v1.spec_decode.contracts import DraftStep

    session = Mock()
    worker = worker_for(session)
    context = target_context([], sampled=[], rejected=[], recovery=[])
    handle = worker.dispatch(DraftStep.from_batch(context.input_batch, k), context)
    output = worker.collect(handle)

    assert output.token_ids.shape == (0, worker.num_speculative_steps)
    assert output.valid_lengths.shape == (0,)
    assert output.to_legacy_tokens().shape == (0, worker.num_speculative_steps)
    assert session.mock_calls == [call.deadline()]
    assert worker._pending is None
    assert worker._requests == {}
    assert handle.context is None
    assert handle.output is None
    assert handle.rpc is None
    with pytest.raises(RuntimeError, match="already collected"):
        worker.collect(handle)


def test_worker_actual_socket_ragged_chunks_reorder_rejection_and_k0(socket_runtime):
    state, oracle, _, session = socket_runtime
    worker = worker_for(session)
    first = target_context(
        [("a", [1, 2], [0, 1], True, 4), ("b", [3, 4, 5], [0, 1, 2], True, 3)],
        sampled=[0, 1],
        rejected=[0, 0],
        recovery=[0, 7],
    )
    initial = step(worker, first, 3)
    assert initial.valid_lengths.tolist() == [0, 3]
    b = initial.token_ids[1].tolist()
    # Reorder physical/logical rows: b accepts every draft (needs final draft
    # plus bonus materialized), a completes its second prompt chunk.
    second = target_context(
        [("b", [7, *b], [3, 4, 5, 6], False, 3), ("a", [8, 9], [2, 3], True, 4)],
        physical=(1, 3),
        sampled=[4, 1],
        rejected=[0, 0],
        recovery=[10, 11],
    )
    output = step(worker, second, 0)
    assert output.valid_lengths.tolist() == [0, 0]
    for req_id, expected in (("b", [3, 4, 5, 7, *b, 10]), ("a", [1, 2, 8, 9, 11])):
        request = worker._requests[req_id]
        progress = state.sequences[request.key]
        assert progress.committed == progress.materialized == len(expected)
        assert [
            oracle.kv[progress.req_index][i] for i in range(len(expected))
        ] == expected
    # K0 has no proposal yet both rows still advance authoritative context.
    third = target_context(
        [("a", [11], [4], False, 4), ("b", [10], [7], False, 3)],
        sampled=[1, 1],
        rejected=[0, 0],
        recovery=[12, 13],
    )
    proposed = step(worker, third, 3)
    assert proposed.valid_lengths.tolist() == [3, 3]
    a, b = proposed.token_ids.tolist()
    fourth = target_context(
        [
            ("a", [12, *a], [5, 6, 7, 8], False, 4),
            ("b", [13, *b], [8, 9, 10, 11], False, 3),
        ],
        sampled=[1, 3],
        rejected=[3, 1],
        recovery=[14, 15],
    )
    assert step(worker, fourth, 1).valid_lengths.tolist() == [1, 1]
    worker.close_requests(["a", "b"])
    assert not worker._requests and not state.sequences


def test_worker_cancel_fences_live_request_and_handle(socket_runtime):
    from vllm.v1.spec_decode.contracts import DraftStep

    _, _, _, session = socket_runtime
    worker = worker_for(session)
    context = target_context(
        [("a", [1, 2], [0, 1], True, 2)], sampled=[1], rejected=[0], recovery=[3]
    )
    handle = worker.dispatch(DraftStep.from_batch(context.input_batch, 3), context)
    worker.cancel(handle, "cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        worker.collect(handle)
    key = worker._requests["a"].key
    assert not session.active(key)
    later = target_context(
        [("a", [3], [2], False, 2)], sampled=[1], rejected=[0], recovery=[4]
    )
    assert step(worker, later, 3).valid_lengths.tolist() == [0]
    worker.close_requests(["a"])


@pytest.mark.parametrize("failure", ["timeout", "row_error"])
def test_worker_terminal_batch_skips_metadata_and_admits_fresh_request(
    socket_runtime, monkeypatch, failure
):
    import torch

    from vllm.v1.spec_decode.remote.transport import TransportTimeout

    _, oracle, _, session = socket_runtime
    session.failure_policy = "target_only"
    worker = worker_for(session)
    first = target_context(
        [("a", [1, 2], [0, 1], True, 2)], sampled=[1], rejected=[0], recovery=[3]
    )
    assert step(worker, first, 3).valid_lengths.tolist() == [3]
    key = worker._requests["a"].key
    sid = session._sid
    later = target_context(
        [("a", [3], [2], False, 2)], sampled=[1], rejected=[0], recovery=[4]
    )

    def fail(*args, **kwargs):
        if failure == "timeout":
            raise TransportTimeout("injected token timeout")
        raise RuntimeError("injected row failure")

    with monkeypatch.context() as patch:
        if failure == "timeout":
            patch.setattr(session, "_recv", fail)
        else:
            patch.setattr(oracle, "forward", fail)
        assert step(worker, later, 3).valid_lengths.tolist() == [0]
    assert not session.active(key)
    assert (session._conn is None) == (failure == "timeout")

    def unexpected(*args, **kwargs):
        pytest.fail("terminal batches must skip token metadata copies and RPCs")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", unexpected)
        patch.setattr(session, "connect", unexpected)
        patch.setattr(session, "dispatch", unexpected)
        for _ in range(2):
            assert step(worker, later, 3).valid_lengths.tolist() == [0]

    mixed = target_context(
        [("a", [4], [3], False, 2), ("b", [5, 6], [0, 1], True, 2)],
        sampled=[1, 1],
        rejected=[0, 0],
        recovery=[7, 8],
    )
    assert step(worker, mixed, 3).valid_lengths.tolist() == [0, 3]
    assert worker._requests["a"].key == key and not session.active(key)
    assert session.active(worker._requests["b"].key)
    assert (session._sid != sid) == (failure == "timeout")
    worker.close_requests(["a", "b"])


@pytest.mark.parametrize("policy", ["target_only", "error"])
@pytest.mark.parametrize(
    "phase", ["prepare", "result_tokens", "result_lengths", "serialize"]
)
@pytest.mark.parametrize("failure", ["memory", "allocator", "torch"])
def test_worker_allocation_failure_obeys_policy_and_fences_state(
    socket_runtime, monkeypatch, policy, phase, failure
):
    import torch

    from vllm.v1.spec_decode.contracts import DraftStep

    _, _, _, session = socket_runtime
    session.failure_policy = policy
    worker = worker_for(session)
    first = target_context(
        [("a", [1, 2], [0, 1], True, 2)], sampled=[1], rejected=[0], recovery=[3]
    )
    assert step(worker, first, 3).valid_lengths.tolist() == [3]
    later = target_context(
        [("a", [3], [2], False, 2)], sampled=[1], rejected=[0], recovery=[4]
    )
    errors = {
        "memory": MemoryError("host allocation failed"),
        "allocator": RuntimeError(
            "[enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: "
            "can't allocate memory: you tried to allocate 4096 bytes. "
            "Error code 12 (Cannot allocate memory)"
        ),
        "torch": torch.OutOfMemoryError("allocation failed"),
    }
    injected = False
    tensor, tolist = torch.tensor, torch.Tensor.tolist

    def allocate(*args, **kwargs):
        nonlocal injected
        dtype = torch.int64 if phase == "result_tokens" else torch.int32
        if not injected and kwargs.get("dtype") == dtype and "device" not in kwargs:
            injected = True
            raise errors[failure]
        return tensor(*args, **kwargs)

    def unpack(self):
        nonlocal injected
        matches = (
            self.dtype == torch.int32 and self.shape == (1,)
            if phase == "prepare"
            else self.dtype == torch.int64 and self.shape == (1, 3)
        )
        if not injected and matches:
            injected = True
            raise errors[failure]
        return tolist(self)

    with monkeypatch.context() as patch:
        if phase == "prepare":
            patch.setattr(torch.Tensor, "tolist", unpack)
        handle = worker.dispatch(DraftStep.from_batch(later.input_batch, 3), later)
        if phase == "serialize":
            patch.setattr(torch.Tensor, "tolist", unpack)
        elif phase != "prepare":
            patch.setattr(torch, "tensor", allocate)
        patch.setattr(
            worker,
            "_empty",
            lambda *args, **kwargs: pytest.fail("reuse the allocated fallback output"),
        )
        if policy == "target_only":
            output = worker.collect(handle)
            assert output.to_legacy_tokens().tolist() == [[-1, -1, -1]]
        else:
            with pytest.raises(RuntimeError, match="token metadata allocation failed"):
                worker.collect(handle)
    assert injected
    assert not session.active(worker._requests["a"].key)
    assert session._conn is None
    assert session._pending is None and worker._pending is None
    assert step(worker, later, 3).valid_lengths.tolist() == [0]


@pytest.mark.parametrize("policy", ["target_only", "error"])
@pytest.mark.parametrize("fail_at", [0, 1, 2], ids=["control", "empty", "result"])
def test_worker_collect_result_allocation_consumes_rpc_and_recovers(
    socket_runtime, monkeypatch, policy, fail_at
):
    from vllm.v1.spec_decode.contracts import DraftStep
    from vllm.v1.spec_decode.remote import token_session

    _, _, _, session = socket_runtime
    session.failure_policy = policy
    worker = worker_for(session)
    first = target_context(
        [("a", [1, 2], [0, 1], True, 2)], sampled=[1], rejected=[0], recovery=[3]
    )
    handle = worker.dispatch(DraftStep.from_batch(first.input_batch, 3), first)
    rpc = handle.rpc
    assert rpc is not None
    key, sid = worker._requests["a"].key, session._sid
    result_type = token_session.TokenBatchResult
    calls = 0

    def allocate(*args):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise MemoryError("one-shot result allocation failure")
        return result_type(*args)

    with monkeypatch.context() as patch:
        patch.setattr(token_session, "TokenBatchResult", allocate)
        patch.setattr(
            worker,
            "_empty",
            lambda *args, **kwargs: pytest.fail("reuse the allocated fallback output"),
        )
        if fail_at and policy == "error":
            with pytest.raises(
                RuntimeError, match="one-shot result allocation failure"
            ):
                worker.collect(handle)
        else:
            output = worker.collect(handle)
            assert output.valid_lengths.tolist() == ([0] if fail_at else [3])
    assert calls == (1 if fail_at == 1 else 2)
    assert session._pending is None and worker._pending is None
    assert not rpc.active and handle.context is None
    assert session.active(key) == (fail_at == 0)
    assert (session._conn is None) == bool(fail_at)

    fresh = target_context(
        [("b", [5, 6], [0, 1], True, 2)], sampled=[1], rejected=[0], recovery=[7]
    )
    assert step(worker, fresh, 3).valid_lengths.tolist() == [3]
    assert (session._sid != sid) == bool(fail_at)
    later = target_context(
        [("a", [3], [2], False, 2)], sampled=[1], rejected=[0], recovery=[4]
    )
    assert step(worker, later, 3).valid_lengths.tolist() == ([0] if fail_at else [3])
    assert worker._requests["a"].key == key
    worker.close_requests(["a", "b"])


@pytest.mark.parametrize("phase", ["prepare", "result"])
@pytest.mark.parametrize(
    "failure", ["ordinary", "fatal", "wrapped_torch", "wrapped_memory"]
)
def test_worker_nonrecoverable_error_still_raises(
    socket_runtime, monkeypatch, phase, failure
):
    import torch

    from vllm.v1.spec_decode.contracts import DraftStep

    _, _, _, session = socket_runtime
    session.failure_policy = "target_only"
    worker = worker_for(session)
    first = target_context(
        [("a", [1, 2], [0, 1], True, 2)], sampled=[1], rejected=[0], recovery=[3]
    )
    assert step(worker, first, 3).valid_lengths.tolist() == [3]
    context = target_context(
        [("a", [3], [2], False, 2)], sampled=[1], rejected=[0], recovery=[4]
    )
    fatal = RuntimeError("CUDA error: an illegal memory access was encountered")
    errors = {
        "ordinary": RuntimeError("ordinary failure"),
        "fatal": fatal,
        "wrapped_torch": torch.OutOfMemoryError("allocation failed"),
        "wrapped_memory": MemoryError("allocation failed"),
    }
    if failure.startswith("wrapped"):
        errors[failure].__cause__ = fatal

    def fail(*args, **kwargs):
        raise errors[failure]

    if phase == "prepare":
        monkeypatch.setattr(worker, "_dispatch_tokens", fail)
    handle = worker.dispatch(DraftStep.from_batch(context.input_batch, 3), context)
    if phase == "result":
        monkeypatch.setattr(torch, "tensor", fail)
    with pytest.raises(RuntimeError, match="token drafter proposal failed"):
        worker.collect(handle)
    assert not session.active(worker._requests["a"].key)
    assert worker._pending is None


def test_worker_persistent_result_allocation_failure_broadcasts_error(
    socket_runtime, monkeypatch
):
    import torch

    from vllm.v1.spec_decode.contracts import DraftStep

    _, _, _, session = socket_runtime
    session.failure_policy = "target_only"
    worker = worker_for(session)
    context = target_context(
        [("a", [1, 2], [0, 1], True, 2)], sampled=[1], rejected=[0], recovery=[3]
    )
    handle = worker.dispatch(DraftStep.from_batch(context.input_batch, 3), context)
    attempts = 0
    tolist = torch.Tensor.tolist
    payloads = []

    def fail(self):
        nonlocal attempts
        if self.shape == (1, 3):
            attempts += 1
            raise MemoryError("persistent host allocation failure")
        return tolist(self)

    monkeypatch.setattr(torch.Tensor, "tolist", fail)

    def broadcast(value, src):
        payloads.append(value)
        return value

    worker._tp.broadcast_object = broadcast
    with pytest.raises(RuntimeError, match="MemoryError: persistent host allocation"):
        worker.collect(handle)
    assert attempts == 2
    assert len(payloads) == 1 and payloads[0][1:] == (None, None)
    assert not session.active(worker._requests["a"].key)
    assert worker._pending is None and session._pending is None


@pytest.mark.parametrize("policy", ["target_only", "error"])
def test_worker_defers_unrelated_close_until_proposal_collected(socket_runtime, policy):
    from vllm.v1.spec_decode.contracts import DraftStep

    state, _, _, session = socket_runtime
    session.failure_policy = policy
    worker = worker_for(session)
    for name in ("a", "b"):
        context = target_context(
            [(name, [1, 2], [0, 1], True, 2)],
            sampled=[1],
            rejected=[0],
            recovery=[3],
        )
        assert step(worker, context, 3).valid_lengths.tolist() == [3]
    b = worker._requests["b"].key
    context = target_context(
        [("a", [3], [2], False, 2)], sampled=[1], rejected=[0], recovery=[4]
    )
    handle = worker.dispatch(DraftStep.from_batch(context.input_batch, 3), context)
    connection = session._conn
    session.close_sequences(())
    worker.close_requests(["unknown"])
    assert session._conn is connection and handle.rpc.active
    worker.close_requests(["b"])
    assert b in state.sequences and b in session.records
    assert not session.active(b) and handle.rpc.active
    assert worker.collect(handle).valid_lengths.tolist() == [3]
    assert b not in state.sequences and b not in session.records
    assert not session._deferred_closes
    assert session.active(worker._requests["a"].key)
    assert session._conn is connection


@pytest.mark.parametrize("policy", ["target_only", "error"])
@pytest.mark.parametrize("failure", ["deadline", "collect"])
def test_deferred_token_close_uses_pending_deadline(
    socket_runtime, monkeypatch, policy, failure
):
    import time

    from vllm.v1.spec_decode.contracts import DraftStep
    from vllm.v1.spec_decode.remote.protocol import ProtocolError
    from vllm.v1.spec_decode.remote.token_protocol import TokenProposal

    state, _, server, session = socket_runtime
    session.failure_policy = policy
    worker = worker_for(session)
    for name in ("a", "b"):
        context = target_context(
            [(name, [1, 2], [0, 1], True, 2)],
            sampled=[1],
            rejected=[0],
            recovery=[3],
        )
        assert step(worker, context, 3).valid_lengths.tolist() == [3]
    b = worker._requests["b"].key
    context = target_context(
        [("a", [3], [2], False, 2)], sampled=[1], rejected=[0], recovery=[4]
    )
    handle = worker.dispatch(DraftStep.from_batch(context.input_batch, 3), context)
    worker.close_requests(["b"])
    receive = session._recv

    def expired_after_response(*args, **kwargs):
        response = receive(*args, **kwargs)
        if isinstance(response[0], TokenProposal):
            if failure == "collect":
                raise ProtocolError("injected response failure")
            handle.rpc.deadline = time.monotonic() - 1
        return response

    monkeypatch.setattr(session, "_recv", expired_after_response)
    monkeypatch.setattr(
        session, "deadline", lambda: pytest.fail("close must not renew the budget")
    )
    if failure == "collect":
        monkeypatch.setattr(
            session,
            "close_sequences",
            lambda *args, **kwargs: pytest.fail("do not mask the collection failure"),
        )
    if policy == "target_only":
        assert worker.collect(handle).valid_lengths.tolist() == (
            [3] if failure == "deadline" else [0]
        )
    else:
        expected = (
            "token close failed"
            if failure == "deadline"
            else "token collect failed: injected response failure"
        )
        with pytest.raises(RuntimeError, match=expected):
            worker.collect(handle)
    assert b not in session.records and not session._deferred_closes
    assert session._conn is None and not session.active(worker._requests["a"].key)
    assert worker._pending is None and session._pending is None
    server.stop()
    assert not state.sequences


def test_cancel_pending_token_proposal_clears_deferred_closes(socket_runtime):
    from vllm.v1.spec_decode.contracts import DraftStep

    state, _, server, session = socket_runtime
    worker = worker_for(session)
    for name in ("a", "b"):
        context = target_context(
            [(name, [1, 2], [0, 1], True, 2)],
            sampled=[1],
            rejected=[0],
            recovery=[3],
        )
        step(worker, context, 3)
    b = worker._requests["b"].key
    context = target_context(
        [("a", [3], [2], False, 2)], sampled=[1], rejected=[0], recovery=[4]
    )
    handle = worker.dispatch(DraftStep.from_batch(context.input_batch, 3), context)
    worker.close_requests(["b"])
    worker.close_requests(["a"])
    assert not session._deferred_closes and b not in session.records
    assert worker._pending is None and session._pending is None
    with pytest.raises(RuntimeError, match="stale, foreign, cancelled"):
        worker.collect(handle)
    server.stop()
    assert not state.sequences


@pytest.mark.parametrize("offered", [0, 1, 2])
def test_offered_prefix_truncation_discards_unoffered_drafts(offered):
    state, oracle = machine()
    key = SequenceKey("v", 0, 0)
    state.open(key)
    state.prefill(key, 0, (1,), is_final=True)
    first = state.advance(0, TokenAdvanceRow(key, 1, None, 0, 0, (2,)), 3)
    delta = first.token_ids[:offered] + (5,)
    second = state.advance(1, TokenAdvanceRow(key, 2, 0, offered, offered, delta), 1)
    expected = [1, 2, *delta]
    assert second.token_ids == (oracle.predict(expected),)
    progress = state.sequences[key]
    assert [oracle.kv[progress.req_index][i] for i in range(len(expected))] == expected


@pytest.mark.parametrize("remaining", [0, 1, 2])
def test_context_budget_shortens_before_reserving(remaining):
    state, _ = machine(max_len=4, blocks=2)
    key = SequenceKey("v", 0, 0)
    state.open(key)
    prompt = (1,) * (3 - remaining)
    state.prefill(key, 0, prompt, is_final=True)
    result = state.advance(0, TokenAdvanceRow(key, len(prompt), None, 0, 0, (2,)), 3)
    assert len(result.token_ids) == remaining
    progress = state.sequences[key]
    assert progress.committed + max(remaining - 1, 0) <= 4
    assert state.block_pool.num_free_blocks == 0
    before = deepcopy(progress)
    with pytest.raises(AdapterError) as error:
        state.advance(1, TokenAdvanceRow(key, progress.committed, 0, 0, 0, (1,) * 5), 0)
    assert error.value.status in (
        SpeculatorStatusCode.SEQUENCE_DESYNC,
        SpeculatorStatusCode.OUT_OF_MEMORY,
    )
    assert progress == before


def test_actual_socket_tiny_kv_oom_releases_slot_and_all_blocks(socket_runtime):
    state, _, _, session = socket_runtime
    session.failure_policy = "target_only"
    state.block_pool = DraftBlockPool(2, 4)
    key = session.open_sequence(0, deadline=session.deadline())
    session.prefill(key, (1, 2, 3), is_final=True, deadline=session.deadline())
    result = session.collect(
        session.dispatch(
            0,
            (TokenAdvanceRow(key, 3, None, 0, 0, (4,)),),
            3,
            deadline=session.deadline(),
        )
    )
    assert result.valid_lengths == (0,)
    assert not session.active(key) and key not in state.sequences
    assert state.block_pool.num_free_blocks == 1
    replacement = session.open_sequence(1, deadline=session.deadline())
    session.prefill(replacement, (1,), is_final=True, deadline=session.deadline())
    assert session.collect(
        session.dispatch(
            1,
            (TokenAdvanceRow(replacement, 1, None, 0, 0, (2,)),),
            1,
            deadline=session.deadline(),
        )
    ).valid_lengths == (1,)
    session.close_sequences((key, replacement))
    assert not state.sequences and state.block_pool.num_free_blocks == 1


@pytest.mark.parametrize("policy", ["target_only", "error"])
@pytest.mark.parametrize("failure", ["frame", "deadline", "partial"])
def test_dispatch_failure_leaves_only_returned_handles_collectable(
    socket_runtime, monkeypatch, policy, failure
):
    import vllm.v1.spec_decode.remote.token_session as token_session
    from vllm.v1.spec_decode.remote.transport import ConnectionClosed

    _, _, _, session = socket_runtime
    session.failure_policy = policy
    key = session.open_sequence(0, deadline=session.deadline())
    session.prefill(key, (1,), is_final=True, deadline=session.deadline())
    handles = []
    handle_type = token_session.TokenProposalHandle

    def create_handle(*args):
        handle = handle_type(*args)
        handles.append(handle)
        return handle

    monkeypatch.setattr(token_session, "TokenProposalHandle", create_handle)
    deadline = session.deadline()
    if failure == "frame":

        def fail_frame(*args, **kwargs):
            raise MemoryError("injected frame allocation failure")

        monkeypatch.setattr(token_session, "ints_to_frame", fail_frame)
    elif failure == "deadline":
        deadline = session.clock() - 1
    else:
        send_data = session._conn.send_data

        def fail_after_frame(*args, **kwargs):
            send_data(*args, **kwargs)
            raise ConnectionClosed("injected failure after first data frame")

        monkeypatch.setattr(session._conn, "send_data", fail_after_frame)

    rows = (TokenAdvanceRow(key, 1, None, 0, 0, (2,)),)
    if policy == "error":
        with pytest.raises(RemoteDraftError, match="token dispatch failed"):
            session.dispatch(0, rows, 1, deadline=deadline)
        assert not handles[0].active
        assert session._pending is None
    else:
        handle = session.dispatch(0, rows, 1, deadline=deadline)
        assert handle is handles[0] and session._pending is handle
        assert session.collect(handle).valid_lengths == (0,)
        assert session._pending is None and not handle.active
    assert session._conn is None and not session.active(key)


@pytest.mark.parametrize("policy", ["target_only", "error"])
def test_socket_reconnect_backoff_bounds_new_requests_and_preserves_terminal_state(
    socket_runtime, monkeypatch, tmp_path, policy
):
    import time

    import vllm.v1.spec_decode.remote.token_session as token_session
    from vllm.v1.spec_decode.remote.reconnect import ReconnectBackoff

    _, _, _, session = socket_runtime
    session.failure_policy = policy
    old_key = session.open_sequence(0, deadline=session.deadline())
    session._invalidate()
    endpoint, session.endpoint = session.endpoint, f"unix://{tmp_path}/absent.sock"
    now = [time.monotonic()]
    session.clock = lambda: now[0]
    session._reconnect = ReconnectBackoff(session.clock)
    connect, attempts = token_session.connect, []

    def counted_connect(*args, **kwargs):
        attempts.append(now[0])
        return connect(*args, **kwargs)

    monkeypatch.setattr(token_session, "connect", counted_connect)

    def unavailable(sequence_id):
        if policy == "error":
            with pytest.raises(RemoteDraftError):
                session.open_sequence(sequence_id, deadline=session.deadline())
        else:
            key = session.open_sequence(sequence_id, deadline=session.deadline())
            assert not session.active(key)

    for sequence_id in range(1, 65):
        unavailable(sequence_id)
    assert len(attempts) == 1
    now[0] += 1
    unavailable(65)
    assert len(attempts) == 2
    now[0] += 1
    unavailable(66)
    assert len(attempts) == 2
    now[0] += 1
    session.endpoint = endpoint
    fresh = session.open_sequence(67, deadline=session.deadline())
    assert len(attempts) == 3 and session.active(fresh)
    assert not session.active(old_key)
    assert all(state.terminal for key, state in session.records.items() if key != fresh)


def test_reconnect_backoff_has_finite_cap_and_resets_after_success():
    from vllm.v1.spec_decode.remote.reconnect import ReconnectBackoff

    now = [0.0]
    backoff = ReconnectBackoff(lambda: now[0])
    for delay in [1, 2, 4, 8, 16, 30, 30, 30]:
        assert backoff.ready()
        backoff.failed()
        now[0] += delay - 0.01
        assert not backoff.ready()
        now[0] += 0.01
        assert backoff.ready()
    backoff.succeeded()
    backoff.failed()
    now[0] += 1
    assert backoff.ready()


@pytest.mark.parametrize("name", ["request_timeout_ms", "startup_timeout_ms"])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("-inf"), float("nan")])
def test_token_session_rejects_unbounded_or_nonpositive_timeouts(name, value):
    hello = TokenHello("verifier", "target", "draft", "tokenizer", 3, 128, 4, 64, 64)
    with pytest.raises(ValueError, match=f"{name} must be finite and positive"):
        TokenOnlyRemoteSession("unix:///unused.sock", hello, **{name: value})


@pytest.mark.parametrize("policy", ["target_only", "error"])
def test_actual_socket_timeout_fences_late_completion(
    socket_runtime, monkeypatch, policy
):
    import threading

    state, oracle, _, session = socket_runtime
    session.failure_policy = policy
    key = session.open_sequence(0, deadline=session.deadline())
    session.prefill(key, (1,), is_final=True, deadline=session.deadline())
    entered, release = threading.Event(), threading.Event()
    forward = oracle.forward

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return forward(*args, **kwargs)

    monkeypatch.setattr(oracle, "forward", delayed)
    session.request_timeout_s = 0.05
    handle = session.dispatch(
        0, (TokenAdvanceRow(key, 1, None, 0, 0, (2,)),), 1, deadline=session.deadline()
    )
    try:
        assert entered.wait(5)
        if policy == "error":
            with pytest.raises(RemoteDraftError, match="timed out|deadline"):
                session.collect(handle)
        else:
            assert session.collect(handle).valid_lengths == (0,)
        assert session.records[key].terminal and session._conn is None
    finally:
        release.set()


def token_config(
    tmp_path,
    *,
    target_type="qwen3",
    draft_type="qwen3",
    target_options=None,
    **draft_options,
):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        AutoModelForCausalLM,
        LlamaConfig,
        MistralConfig,
        Qwen3Config,
        TokenizersBackend,
    )

    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        SpeculativeConfig,
        VllmConfig,
    )

    families = {
        "qwen3": (Qwen3Config, "Qwen3ForCausalLM"),
        "llama": (LlamaConfig, "LlamaForCausalLM"),
        "mistral": (MistralConfig, "MistralForCausalLM"),
    }
    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(WordLevel({f"token{i}": i for i in range(128)})),
        bos_token="token0",
        eos_token="token1",
        pad_token="token2",
    )

    def model(path, family, *, hidden_size, **options):
        hf_class, architecture = families[family]
        hf = hf_class(
            architectures=[architecture],
            hidden_size=hidden_size,
            head_dim=8,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=2,
            num_attention_heads=hidden_size // 8,
            num_key_value_heads=2,
            vocab_size=128,
            max_position_embeddings=64,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            **(({"sliding_window": None} if family == "mistral" else {}) | options),
        )
        with torch.random.fork_rng():
            torch.manual_seed(0)
            AutoModelForCausalLM.from_config(hf).save_pretrained(path)
        tokenizer.save_pretrained(path)
        return ModelConfig(
            model=str(path),
            tokenizer_mode="hf",
            dtype="float32",
            max_model_len=64,
            skip_tokenizer_init=True,
        )

    target = model(
        tmp_path / "target", target_type, hidden_size=32, **(target_options or {})
    )
    draft = model(tmp_path / "draft", draft_type, hidden_size=16, **draft_options)
    parallel = ParallelConfig()
    spec = SpeculativeConfig(
        model=draft.model,
        method="draft_model",
        num_speculative_tokens=3,
        target_model_config=target,
        target_parallel_config=parallel,
    )
    return VllmConfig(
        model_config=target,
        parallel_config=parallel,
        speculative_config=spec,
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=64,
            max_model_len=64,
            async_scheduling=False,
            is_encoder_decoder=False,
        ),
        cache_config=CacheConfig(
            enable_prefix_caching=False,
            sliding_window=(
                target.get_sliding_window()
                if getattr(target.hf_text_config, "layer_types", None) is None
                or all(
                    layer == "sliding_attention"
                    for layer in target.hf_text_config.layer_types
                )
                else None
            ),
        ),
        device_config=DeviceConfig("cpu"),
    )


@pytest.mark.parametrize(
    "target_type,draft_type",
    [("qwen3", "llama"), ("llama", "mistral"), ("mistral", "qwen3")],
)
def test_token_contract_accepts_native_full_attention_with_different_geometry(
    tmp_path, target_type, draft_type
):
    from vllm.v1.spec_decode.remote.runtime.token_schema import (
        create_token_runtime_contract,
    )

    config = token_config(tmp_path, target_type=target_type, draft_type=draft_type)
    contract = create_token_runtime_contract(config)
    assert contract.schema.slots == (FeatureSlot("token_ids", "int32", ()),)
    assert contract.capabilities.state_dependency == "own_kv"
    assert (
        config.model_config.get_hidden_size()
        != config.speculative_config.draft_model_config.get_hidden_size()
    )


@pytest.mark.parametrize(
    "family,target_options,draft_options",
    [
        ("mistral", {}, {"sliding_window": 16}),
        ("mistral", {"sliding_window": 16}, {}),
        ("mistral", {"sliding_window": 16}, {"sliding_window": 32}),
        (
            "qwen3",
            {"use_sliding_window": True, "max_window_layers": 0, "sliding_window": 16},
            {"use_sliding_window": True, "max_window_layers": 0, "sliding_window": 32},
        ),
    ],
)
def test_token_contract_accepts_independent_uniform_attention_windows(
    tmp_path, family, target_options, draft_options
):
    from vllm.v1.spec_decode.remote.runtime.token_gpu import (
        create_token_executor_config,
    )
    from vllm.v1.spec_decode.remote.runtime.token_schema import (
        create_token_runtime_contract,
    )

    config = token_config(
        tmp_path,
        target_type=family,
        draft_type=family,
        target_options=target_options,
        **draft_options,
    )
    contract = create_token_runtime_contract(config)
    assert contract.schema.slots == (FeatureSlot("token_ids", "int32", ()),)
    assert contract.capabilities.state_dependency == "own_kv"
    resource = create_token_executor_config(config)
    assert resource.cache_config.sliding_window == draft_options.get("sliding_window")
    assert config.cache_config.sliding_window == target_options.get("sliding_window")
    assert resource.cache_config is not config.cache_config
    assert resource.model_config is config.speculative_config.draft_model_config


def test_full_attention_layer_declarations_ignore_unused_window(tmp_path):
    from vllm.v1.spec_decode.remote.runtime.token_gpu import (
        create_token_executor_config,
    )
    from vllm.v1.spec_decode.remote.runtime.token_schema import (
        create_token_runtime_contract,
    )

    config = token_config(
        tmp_path,
        draft_type="llama",
        sliding_window=16,
        layer_types=["full_attention", "full_attention"],
    )
    assert (
        create_token_runtime_contract(config).capabilities.state_dependency == "own_kv"
    )
    assert create_token_executor_config(config).cache_config.sliding_window is None


@pytest.mark.parametrize("family", ["llama", "mistral"])
def test_token_contract_preserves_independent_mixed_attention_order(tmp_path, family):
    from vllm.v1.spec_decode.remote.runtime.token_gpu import (
        create_token_executor_config,
    )
    from vllm.v1.spec_decode.remote.runtime.token_schema import (
        create_token_runtime_contract,
    )

    config = token_config(
        tmp_path,
        target_type=family,
        draft_type=family,
        target_options={
            "sliding_window": 32,
            "layer_types": ["sliding_attention", "full_attention"],
        },
        sliding_window=16,
        layer_types=["full_attention", "sliding_attention"],
    )
    contract = create_token_runtime_contract(config)
    resource = create_token_executor_config(config)
    assert contract.capabilities.state_dependency == "own_kv"
    assert resource.cache_config.sliding_window is None
    assert config.cache_config.sliding_window is None
    assert resource.model_config.hf_text_config.layer_types == [
        "full_attention",
        "sliding_attention",
    ]
    assert resource.model_config.get_sliding_window() == 16
    assert config.model_config.get_sliding_window() == 32


@pytest.mark.parametrize("role", ["target", "draft"])
def test_token_contract_rejects_mixed_windows_ignored_by_native_qwen(tmp_path, role):
    from vllm.v1.spec_decode.remote.runtime.token_schema import (
        create_token_runtime_contract,
    )

    options = {
        "sliding_window": 16,
        "layer_types": ["full_attention", "sliding_attention"],
        "use_sliding_window": True,
        "max_window_layers": 2,
    }
    config = token_config(
        tmp_path,
        target_options=options if role == "target" else None,
        **(options if role == "draft" else {}),
    )
    with pytest.raises(ValueError, match=f"{role} mixed attention requires native"):
        create_token_runtime_contract(config)


@pytest.mark.parametrize(
    "actual_names,actual_windows,error",
    [
        ((1, 0), (16, None), None),
        ((0, 1), (16, None), "layer 0 differs from its contract"),
        ((0, 1), (None, 32), "layer 1 differs from its contract"),
        ((0, 1), (None, None), "layer 1 differs from its contract"),
        ((0,), (None,), "one attention KV per model layer"),
        ((0, 0), (None, 16), "indices must be unique and complete"),
        ((0, 2), (None, 16), "indices must be unique and complete"),
        ((0, None), (None, 16), "KV must identify its model layer"),
    ],
)
def test_loaded_token_kv_matches_attention_by_layer_not_group_counts(
    tmp_path, actual_names, actual_windows, error
):
    import torch

    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
    from vllm.v1.spec_decode.remote.runtime.token_gpu import (
        create_token_executor_config,
        validate_token_model_kv_specs,
    )

    config = create_token_executor_config(
        token_config(
            tmp_path,
            draft_type="mistral",
            sliding_window=16,
            layer_types=["full_attention", "sliding_attention"],
        )
    )
    specs = {}
    for n, (index, window) in enumerate(zip(actual_names, actual_windows)):
        name = "draft.model.attn" if index is None else f"draft.model.layers.{index}"
        # The duplicate-index case has distinct module names within one layer.
        name += f".{'self_attn' if n == 0 else 'other_attn'}.attn"
        geometry = dict(block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32)
        specs[name] = (
            FullAttentionSpec(**geometry)
            if window is None
            else SlidingWindowSpec(**geometry, sliding_window=window)
        )
    if error is None:
        validate_token_model_kv_specs(config, specs)
        assert config.cache_config.sliding_window is None
    else:
        with pytest.raises(ValueError, match=error):
            validate_token_model_kv_specs(config, specs)


@pytest.mark.parametrize(
    "options,error_type,reason",
    [
        (
            {"layer_types": ["full_attention", "sliding_attention"]},
            ValueError,
            "with a declared sliding window",
        ),
        (
            {"layer_types": ["sliding_attention", "sliding_attention"]},
            ValueError,
            "with a declared sliding window",
        ),
        ({"sliding_window": -1}, ValueError, "positive integer sliding window"),
        (
            {"sliding_window": True},
            StrictDataclassFieldValidationError,
            "expected int, got bool",
        ),
        (
            {"layer_types": ["full_attention"]},
            StrictDataclassClassValidationError,
            "must be equal to the number of `layer_types`",
        ),
        (
            {"attention_chunk_size": 16},
            ValueError,
            "does not support chunked attention",
        ),
        ({"is_causal": False}, ValueError, "native decoder-only causal"),
    ],
)
def test_token_contract_rejects_invalid_attention(
    tmp_path, options, error_type, reason
):
    from vllm.v1.spec_decode.remote.runtime.token_schema import (
        create_token_runtime_contract,
    )

    with pytest.raises(error_type, match=reason):
        config = token_config(tmp_path, draft_type="mistral", **options)
        create_token_runtime_contract(config)


def test_contract_uses_draft_checkpoint_assets_not_tokenizer_alias(tmp_path):
    from pathlib import Path

    from vllm.v1.spec_decode.remote.runtime.token_schema import (
        create_token_runtime_contract,
    )

    config = token_config(tmp_path)
    contract = create_token_runtime_contract(config)
    assert contract.schema.slots == (FeatureSlot("token_ids", "int32", ()),)
    (
        Path(config.speculative_config.draft_model_config.model) / "tokenizer.json"
    ).write_text('{"vocab":{"a":1}}')
    with pytest.raises(ValueError, match="same token IDs"):
        create_token_runtime_contract(config)


@pytest.mark.parametrize(
    "field,value", [("tokenizer_mode", "auto"), ("trust_remote_code", True)]
)
def test_contract_requires_identifiable_builtin_hf_tokenizer(tmp_path, field, value):
    from vllm.v1.spec_decode.remote.runtime.token_schema import (
        create_token_runtime_contract,
    )

    config = token_config(tmp_path)
    setattr(config.model_config, field, value)
    with pytest.raises(ValueError, match="built-in HF"):
        create_token_runtime_contract(config)


@pytest.mark.parametrize("previous_k,accepted", [(0, 0), (1, 0), (1, 1)])
def test_worker_physical_query_can_exceed_previous_logical_valid_prefix(
    socket_runtime, previous_k, accepted
):
    state, oracle, _, session = socket_runtime
    worker = worker_for(session)
    context = target_context(
        [("a", [1, 2], [0, 1], True, 2)], sampled=[1], rejected=[0], recovery=[3]
    )
    previous = step(worker, context, previous_k)
    offered = previous.token_ids[0, :previous_k].tolist()
    physical = [3, *offered, *([-1] * (3 - previous_k))]
    update = target_context(
        [("a", physical, [2, 3, 4, 5], False, 2)],
        sampled=[accepted + 1],
        rejected=[3 - accepted],
        recovery=[4],
    )
    current = step(worker, update, 3)
    assert current.valid_lengths.tolist() == [3]
    key = worker._requests["a"].key
    progress = state.sequences[key]
    expected = [1, 2, 3, *offered[:accepted], 4]
    assert progress.committed == progress.materialized == len(expected)
    assert [oracle.kv[progress.req_index][i] for i in range(len(expected))] == expected
    worker.close_requests(["a"])


def test_worker_refuses_accepted_physical_padding_beyond_logical_valid(socket_runtime):
    _, _, _, session = socket_runtime
    worker = worker_for(session)
    context = target_context(
        [("a", [1, 2], [0, 1], True, 2)], sampled=[1], rejected=[0], recovery=[3]
    )
    step(worker, context, 0)
    update = target_context(
        [("a", [3, 0, 0, 0], [2, 3, 4, 5], False, 2)],
        sampled=[2],
        rejected=[2],
        recovery=[4],
    )
    with pytest.raises(RuntimeError, match="outside the previous valid proposal"):
        step(worker, update, 3)
    assert session.records[worker._requests["a"].key].terminal
    worker.close_requests(["a"])


def resolved_tokenizer_revision(resolved, initial, *, repo_id, repo_type="model"):
    from huggingface_hub import ResolvedRevision

    revision = ResolvedRevision(resolved, initial=initial)
    # Cached-service responses include repository provenance on newer Hub
    # releases. Populate that external metadata on older supported releases.
    revision._repo_id, revision._repo_type = repo_id, repo_type
    return revision


@pytest.mark.parametrize("resolved_branch", [False, True])
def test_cached_tokenizer_assets_prove_immutable_snapshot_revision(
    tmp_path, monkeypatch, resolved_branch
):
    import vllm.transformers_utils.repo_utils as repo_utils

    revision = "a" * 40
    snapshot = tmp_path / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text('{"vocab":{"a":0}}')
    (snapshot / "tokenizer_config.json").write_text("{}")
    lookups = []

    def cached(source, name, revision):
        lookups.append((source, revision))
        return snapshot / name if (snapshot / name).is_file() else None

    monkeypatch.setattr(repo_utils, "try_get_local_file", cached)
    pinned = (
        resolved_tokenizer_revision(revision, initial="main", repo_id="Qwen/Qwen3-8B")
        if resolved_branch
        else revision
    )
    assert tokenizer_artifact_fingerprint(
        "Qwen/Qwen3-8B", pinned
    ) == tokenizer_artifact_fingerprint(str(snapshot), None)
    assert lookups and all(
        source == "Qwen/Qwen3-8B" and rev == revision for source, rev in lookups
    )
    with pytest.raises(ValueError, match="immutable revision"):
        tokenizer_artifact_fingerprint("Qwen/Qwen3-8B", "b" * 40)


@pytest.mark.parametrize(
    "repo_id,repo_type", [("Other/Tokenizer", "model"), ("Qwen/Qwen3-8B", "dataset")]
)
def test_tokenizer_revision_must_belong_to_its_model_repository(
    monkeypatch, repo_id, repo_type
):
    import vllm.transformers_utils.repo_utils as repo_utils

    lookups = []
    monkeypatch.setattr(
        repo_utils, "try_get_local_file", lambda *args, **kwargs: lookups.append(args)
    )
    revision = resolved_tokenizer_revision(
        "a" * 40, initial="a" * 40, repo_id=repo_id, repo_type=repo_type
    )
    with pytest.raises(ValueError, match="different repository"):
        tokenizer_artifact_fingerprint("Qwen/Qwen3-8B", revision)
    assert not lookups


def test_custom_tokenizer_implementation_rejected_even_with_matching_vocab(tmp_path):
    (tmp_path / "tokenizer.json").write_text('{"vocab":{"a":0}}')
    (tmp_path / "tokenizer_config.json").write_text(
        '{"auto_map":{"AutoTokenizer":["custom.Code",null]}}'
    )
    with pytest.raises(ValueError, match="custom tokenizer"):
        tokenizer_artifact_fingerprint(str(tmp_path), None)


@pytest.mark.parametrize("prompt_len,computed", [(1, 0), (4, 2), (63, 62)])
def test_worker_padded_final_prefill_sends_only_authoritative_prompt(
    socket_runtime, prompt_len, computed
):
    state, oracle, _, session = socket_runtime
    worker = worker_for(session)
    initial_rows = [("b", [3, 4], [0, 1], True, 2)]
    sampled, rejected, recovery = [1], [0], [7]
    if computed:
        initial_rows.insert(
            0, ("a", [1] * computed, list(range(computed)), True, prompt_len)
        )
        sampled.insert(0, 0)
        rejected.insert(0, 0)
        recovery.insert(0, 0)
    previous = step(
        worker,
        target_context(
            initial_rows, sampled=sampled, rejected=rejected, recovery=recovery
        ),
        3,
    )
    b_drafts = previous.token_ids[-1].tolist()
    logical = prompt_len - computed
    padded = [2] * logical + [-1] * 3
    context = target_context(
        [
            (
                "a",
                padded,
                list(range(computed, computed + len(padded))),
                True,
                prompt_len,
            ),
            ("b", [7, *b_drafts], [2, 3, 4, 5], False, 2),
        ],
        sampled=[1, 4],
        rejected=[3, 0],
        recovery=[8, 9],
    )
    result = step(worker, context, 3)
    assert result.valid_lengths.tolist() == [min(3, 64 - prompt_len - 1), 3]
    request = worker._requests["a"]
    progress = state.sequences[request.key]
    assert request.target_computed == prompt_len
    expected = [1] * computed + [2] * logical + [8]
    assert progress.committed == progress.materialized == len(expected)
    assert [oracle.kv[progress.req_index][i] for i in range(len(expected))] == expected
    assert all(-1 not in call[2] for call in oracle.calls)
    followup = target_context(
        [("b", [9, *result.token_ids[1].tolist()], [6, 7, 8, 9], False, 2)],
        sampled=[1],
        rejected=[3],
        recovery=[10],
    )
    assert step(worker, followup, 1).valid_lengths.tolist() == [1]
    worker.close_requests(["a", "b"])
    assert not state.sequences


@pytest.mark.parametrize("sampled,rejected", [(0, 3), (2, 2), (1, 0), (1, 2), (1, 4)])
def test_worker_padded_prefill_refuses_inconsistent_sampling_counts(
    socket_runtime, sampled, rejected
):
    _, _, _, session = socket_runtime
    worker = worker_for(session)
    context = target_context(
        [("a", [2, -1, -1, -1], [0, 1, 2, 3], True, 1)],
        sampled=[sampled],
        rejected=[rejected],
        recovery=[3],
    )
    with pytest.raises(RuntimeError, match="sampling counts in token prefill"):
        step(worker, context, 3)
    assert session.records[worker._requests["a"].key].terminal
    worker.close_requests(["a"])


@pytest.mark.parametrize("policy", ["target_only", "error"])
@pytest.mark.parametrize("stage", ["registration", "tokens"])
def test_unregistered_token_allocation_failure_stays_terminal(
    socket_runtime, monkeypatch, policy, stage
):
    import torch

    from vllm.v1.worker.gpu.spec_decode import token_remote

    _, _, _, session = socket_runtime
    session.failure_policy = policy
    worker = worker_for(session)
    context = target_context(
        [("a", [1, 2], [0, 1], True, 2)],
        sampled=[1],
        rejected=[0],
        recovery=[3],
    )
    original = torch.Tensor.tolist

    def fail_registration(*args, **kwargs):
        raise MemoryError("new request allocation failed")

    def fail_tokens(tensor):
        if tensor.data_ptr() == context.input_batch.input_ids.data_ptr():
            raise MemoryError("new request allocation failed")
        return original(tensor)

    with monkeypatch.context() as patch:
        if stage == "registration":
            patch.setattr(token_remote, "_Request", fail_registration)
        else:
            patch.setattr(torch.Tensor, "tolist", fail_tokens)
        if policy == "error":
            with pytest.raises(RuntimeError, match="new request allocation failed"):
                step(worker, context, 3)
            assert worker._pending is None
            return
        assert step(worker, context, 3).valid_lengths.tolist() == [0]
    assert not session.records
    next_context = target_context(
        [("a", [3], [2], False, 2)],
        sampled=[1],
        rejected=[0],
        recovery=[4],
    )
    assert step(worker, next_context, 3).valid_lengths.tolist() == [0]
    worker.close_requests(["a"])
    assert not worker._terminal_requests
    assert step(worker, context, 3).valid_lengths.tolist() == [3]
    worker.close()
