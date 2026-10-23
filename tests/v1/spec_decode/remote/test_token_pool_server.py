# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pool admission and cross-client execution through real token sockets."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from tests.v1.spec_decode.remote.test_token_runtime import KVOracle
from vllm.v1.spec_decode.remote.capabilities import SpeculatorPlacementCapabilities
from vllm.v1.spec_decode.remote.protocol import (
    FeatureSlot,
    RemoteServerLimits,
    TargetFeatureSchema,
)
from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
from vllm.v1.spec_decode.remote.runtime.schema import RuntimeContract
from vllm.v1.spec_decode.remote.runtime.token_adapter import TokenOnlyDraftAdapter
from vllm.v1.spec_decode.remote.runtime.token_state import (
    TokenAdvanceRow,
    TokenStateMachine,
)
from vllm.v1.spec_decode.remote.runtime.work_queue import DraftWork, DraftWorkQueue
from vllm.v1.spec_decode.remote.server import AdapterQueueFull, ServerShutdownTimeout
from vllm.v1.spec_decode.remote.token_protocol import TokenHello
from vllm.v1.spec_decode.remote.token_server import TokenDraftServer
from vllm.v1.spec_decode.remote.token_session import TokenOnlyRemoteSession


def until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline
        threading.Event().wait(0.001)


class BatchOracle(KVOracle):
    def __init__(self):
        super().__init__()
        self.threads = set()
        self.batches = []
        self.blocked = threading.Event()
        self.release = threading.Event()

    def warmup(self):
        self.threads.add(threading.get_ident())
        super().warmup()

    def forward(self, key, req_index, start, tokens, blocks, *, sample):
        self.threads.add(threading.get_ident())
        if tokens == (99,):
            self.blocked.set()
            assert self.release.wait(5)
        return super().forward(key, req_index, start, tokens, blocks, sample=sample)

    def forward_batch(self, rows):
        self.batches.append(tuple(row.key for row in rows))
        return tuple(
            self.forward(
                row.key,
                row.req_index,
                row.start,
                row.tokens,
                row.blocks,
                sample=row.sample,
            )
            for row in rows
        )


@contextmanager
def runtime(tmp_path, **kwargs):
    oracle = BatchOracle()
    state = TokenStateMachine(
        oracle,
        DraftBlockPool(65, 4),
        max_num_reqs=8,
        max_model_len=64,
        vocab_size=128,
        max_k=3,
    )
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
        limits=RemoteServerLimits(8, 64, 8, 64),
        **kwargs,
    )
    endpoint = server.start(f"unix://{tmp_path}/p.sock")
    clients = []

    def client(identity):
        hello = TokenHello(identity, "target", "draft", "tokenizer", 3, 128, 8, 64, 64)
        session = TokenOnlyRemoteSession(endpoint, hello, request_timeout_ms=3000)
        session.connect()
        clients.append(session)
        return session

    try:
        yield state, oracle, server, client
    finally:
        oracle.release.set()
        for session in clients:
            session.close()
        server.stop()
        if server.healthy:
            assert not state.sequences
            assert not len(server.registry)
            assert state.block_pool.num_free_blocks == state.block_pool.num_blocks - 1


def prepared(client, sequence_id=0, token=1):
    key = client.open_sequence(sequence_id, deadline=client.deadline())
    client.prefill(key, (token,), is_final=True, deadline=client.deadline())
    return key


def proposal(client, key, *, token=2, k=3, batch_id=0):
    return client.collect(
        client.dispatch(
            batch_id,
            (TokenAdvanceRow(key, 1, None, 0, 0, (token,)),),
            k,
            deadline=client.deadline(),
        )
    )


@pytest.mark.parametrize("ka_budget,kb_budget", [(0, 3), (1, 2), (3, 3)])
def test_cross_client_batch_preserves_namespaces_and_has_one_gpu_owner(
    tmp_path, ka_budget, kb_budget
):
    with runtime(tmp_path, batch_wait_ms=0) as (state, oracle, server, client):
        a, b, blocker = (client(name) for name in ("a", "b", "blocker"))
        ka, kb = prepared(a, token=1), prepared(b, token=3)
        kc = blocker.open_sequence(0, deadline=blocker.deadline())
        with ThreadPoolExecutor(max_workers=3) as executor:
            held = executor.submit(
                blocker.prefill, kc, (99,), is_final=True, deadline=blocker.deadline()
            )
            assert oracle.blocked.wait(3)
            pa = executor.submit(proposal, a, ka, token=2, k=ka_budget, batch_id=7)
            pb = executor.submit(proposal, b, kb, token=4, k=kb_budget, batch_id=19)
            until(lambda: server.work_queue.depth == 3)
            oracle.release.set()
            held.result(3)
            ra, rb = pa.result(3), pb.result(3)
        assert ra.valid_lengths == (ka_budget,)
        assert rb.valid_lengths == (kb_budget,)
        if ka_budget:
            assert ra.token_ids[0][0] == KVOracle.predict([1, 2])
        assert rb.token_ids[0][0] == KVOracle.predict([3, 4])
        assert state.sequences[ka].proposal_id == 7
        assert state.sequences[kb].proposal_id == 19
        assert any(set(batch) == {ka, kb} for batch in oracle.batches)
        assert len(oracle.threads) == 1
        assert threading.get_ident() not in oracle.threads
        assert state.sequences[ka].committed == state.sequences[kb].committed == 2
        a.close_sequences((ka,))
        assert kb in state.sequences


def test_per_verifier_sequence_credit_leaves_capacity_for_other_clients(tmp_path):
    with runtime(tmp_path, max_sequences_per_verifier=1) as (state, _, _, client):
        a, b = client("a"), client("b")
        ka = prepared(a)
        a.failure_policy = "target_only"
        refused = a.open_sequence(1, deadline=a.deadline())
        assert not a.active(refused)
        kb = prepared(b)
        assert proposal(b, kb).valid_lengths == (3,)
        assert set(state.sequences) == {ka, kb}


def test_queue_fairness_and_byte_credits_include_running_work():
    entered, release = threading.Event(), threading.Event()
    order = []
    queue = DraftWorkQueue(
        lambda _: None,
        max_jobs=4,
        max_bytes=16,
        max_jobs_per_owner=3,
        max_bytes_per_owner=12,
        max_cleanup_jobs=2,
        max_batch_rows=4,
        max_batch_tokens=16,
        batch_wait_s=0,
    )
    queue.start(lambda: None)

    def hold():
        entered.set()
        assert release.wait(3)
        order.append("held")

    try:
        held = queue.submit(DraftWork("a", hold, num_bytes=4))
        assert entered.wait(3)
        first = queue.submit(DraftWork("a", lambda: order.append("a1"), num_bytes=4))
        second = queue.submit(DraftWork("a", lambda: order.append("a2"), num_bytes=4))
        with pytest.raises(AdapterQueueFull):
            queue.submit(DraftWork("a", lambda: None, num_bytes=1))
        other = queue.submit(DraftWork("b", lambda: order.append("b"), num_bytes=4))
        assert queue.pending_bytes == 16
        with pytest.raises(AdapterQueueFull):
            queue.submit(DraftWork("c", lambda: None, num_bytes=1))
        release.set()
        for work in (held, first, second, other):
            work.wait()
        assert order == ["held", "a1", "b", "a2"]
        assert queue.pending_bytes == queue.depth == 0
        assert queue.call("a", lambda: 7, num_bytes=12) == 7
    finally:
        release.set()
        queue.stop(time.monotonic() + 3)


def test_queue_shutdown_retains_running_work_and_its_credits():
    entered, release = threading.Event(), threading.Event()
    queue = DraftWorkQueue(
        lambda _: None,
        max_jobs=1,
        max_bytes=8,
        max_jobs_per_owner=1,
        max_bytes_per_owner=8,
        max_cleanup_jobs=1,
        max_batch_rows=1,
        max_batch_tokens=8,
    )
    queue.start(lambda: None)

    def hold():
        entered.set()
        assert release.wait(3)

    try:
        work = queue.submit(DraftWork("a", hold, num_bytes=8))
        assert entered.wait(3)
        with pytest.raises(ServerShutdownTimeout, match="still running"):
            queue.stop(time.monotonic())
        assert queue.depth == 1 and queue.pending_bytes == 8
        with pytest.raises(ServerShutdownTimeout):
            queue.submit(DraftWork("b", lambda: None))
        release.set()
        work.wait()
    finally:
        release.set()
        queue.stop(time.monotonic() + 3)


def test_expired_queued_client_skips_forward_and_healthy_client_progresses(tmp_path):
    with runtime(tmp_path, batch_wait_ms=0) as (state, oracle, server, client):
        a, b, blocker = (client(name) for name in ("a", "b", "blocker"))
        ka, kb = prepared(a), prepared(b, token=3)
        kc = blocker.open_sequence(0, deadline=blocker.deadline())
        a.failure_policy = "target_only"
        a.request_timeout_s = 0.03
        with ThreadPoolExecutor(max_workers=3) as executor:
            held = executor.submit(
                blocker.prefill, kc, (99,), is_final=True, deadline=blocker.deadline()
            )
            assert oracle.blocked.wait(3)
            pa = executor.submit(proposal, a, ka)
            until(lambda: server.work_queue.depth == 2)
            # The relative server budget starts no later than queue admission.
            expiry = time.monotonic() + a.request_timeout_s
            pb = executor.submit(proposal, b, kb, token=4)
            until(lambda: server.work_queue.depth == 3)
            assert pa.result(3).valid_lengths == (0,)
            threading.Event().wait(max(0, expiry - time.monotonic()))
            oracle.release.set()
            held.result(3)
            result = pb.result(3)
        assert result.valid_lengths == (3,)
        until(lambda: server.work_queue.depth == 0)
        assert ka not in state.sequences
        assert not any(call[0] == ka and call[3] for call in oracle.calls)
        assert b.active(kb) and server.healthy


def test_pending_credit_refusal_does_not_disconnect_another_verifier(tmp_path):
    from vllm.v1.spec_decode.remote.session import RemoteDraftError

    with runtime(tmp_path, max_jobs_per_verifier=1) as (_, oracle, server, client):
        first, same, other = (client(name) for name in ("a", "a", "b"))
        kfirst = first.open_sequence(0, deadline=first.deadline())
        ksame, kother = prepared(same, sequence_id=1), prepared(other)
        with ThreadPoolExecutor(max_workers=3) as executor:
            held = executor.submit(
                first.prefill, kfirst, (99,), is_final=True, deadline=first.deadline()
            )
            assert oracle.blocked.wait(3)
            refused = executor.submit(proposal, same, ksame)
            with pytest.raises(RemoteDraftError):
                refused.result(3)
            healthy = executor.submit(proposal, other, kother)
            until(lambda: server.work_queue.depth >= 2)
            oracle.release.set()
            held.result(3)
            assert healthy.result(3).valid_lengths == (3,)
        assert first.active(kfirst) and other.active(kother)
        assert server.healthy


def test_server_shutdown_does_not_free_active_gpu_owner_state(tmp_path):
    from vllm.v1.spec_decode.remote.session import RemoteDraftError

    with runtime(tmp_path, shutdown_timeout_s=0.03) as (state, oracle, server, client):
        session = client("a")
        key = session.open_sequence(0, deadline=session.deadline())
        with ThreadPoolExecutor(max_workers=1) as executor:
            held = executor.submit(
                session.prefill, key, (99,), is_final=True, deadline=session.deadline()
            )
            assert oracle.blocked.wait(3)
            free = state.block_pool.num_free_blocks
            started = time.monotonic()
            try:
                with pytest.raises(ServerShutdownTimeout):
                    server.stop()
                assert time.monotonic() - started < 1
                assert not server.healthy
                assert key in state.sequences
                assert state.block_pool.num_free_blocks == free
                assert server.work_queue.depth == 1
            finally:
                oracle.release.set()
                # Join the synthetic CPU owner only after proving the first stop
                # retained its live state. Real fenced GPU processes are replaced.
                server.work_queue.stop(time.monotonic() + 3)
                if server._acceptor is not None:
                    server._acceptor.join(3)
                for worker in server._workers:
                    worker.join(3)
            with pytest.raises(RemoteDraftError):
                held.result(3)
        server.stop()
        # A real fenced device exits. The completed CPU oracle can release its pool.
        state.close((key,))
