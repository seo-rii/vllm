# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real v4 socket routing with CPU snapshots and a bounded external executor."""

import socket
import threading
import time
from contextlib import ExitStack
from dataclasses import replace

import pytest
import torch
from msgspec.structs import replace as replace_message

from tests.v1.spec_decode.remote.test_target_kv_wire import SnapshotOracle
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.session import RemoteDraftError
from vllm.v1.spec_decode.remote.target_kv import (
    SnapshotBudget,
    SnapshotIdentity,
    SnapshotLayer,
    TargetKVSnapshot,
)
from vllm.v1.spec_decode.remote.target_kv_pool import TargetKVDraftPool
from vllm.v1.spec_decode.remote.target_kv_protocol import (
    TargetKVHello,
    TargetKVHelloAck,
    decode_target_kv_message,
    encode_target_kv_message,
    snapshot_wire_plan,
)
from vllm.v1.spec_decode.remote.target_kv_server import TargetKVDraftServer
from vllm.v1.spec_decode.remote.transport import FramedConnection


@pytest.fixture
def pool_runtime(request):
    quotas = getattr(request, "param", {})
    specs = {
        "full": FullAttentionSpec(
            block_size=4, num_kv_heads=1, head_size=4, dtype=torch.bfloat16
        ),
        "window": SlidingWindowSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.bfloat16,
            sliding_window=5,
        ),
    }
    budget = SnapshotBudget(128, 1 << 20, 64)
    servers, oracles = [], []
    with ExitStack() as cleanup:
        for _ in range(2):
            oracle = SnapshotOracle(specs, budget)
            server = TargetKVDraftServer(
                oracle,
                contract_fingerprint="contract",
                hidden_size=8,
                vocab_size=128,
                expected_specs=specs,
                snapshot_budget=budget,
                receiver_budget=budget,
                max_num_seqs=4,
                max_connections=4,
                max_sequences=quotas.get("max_sequences", 16),
                max_sequences_per_verifier=quotas.get("max_sequences_per_verifier", 4),
            )
            server.start("tcp://127.0.0.1:0")
            cleanup.callback(server.stop)
            servers.append(server)
            oracles.append(oracle)
        hello = TargetKVHello(
            "verifier", "contract", 8, 128, 64, 128, 1 << 20, max_num_seqs=4
        )

        def client(identity="verifier", policy="error", timeout_ms=1000):
            endpoints = []
            for server in servers:
                assert server.endpoint is not None
                endpoints.append(server.endpoint)
            pool = TargetKVDraftPool(
                tuple(endpoints),
                replace_message(hello, verifier_instance_id=identity),
                expected_specs=specs,
                failure_policy=policy,
                request_timeout_ms=timeout_ms,
            )
            cleanup.callback(pool.close)
            pool.connect()
            return pool

        try:
            yield client, servers, oracles
        finally:
            for oracle in oracles:
                oracle.release.set()


def snapshots(pool, keys, *, round_id=0):
    lengths = (5, 9, 13, 17)
    return tuple(
        TargetKVSnapshot(
            SnapshotIdentity(key, pool.epoch_for(key), round_id),
            lengths[i],
            tuple(
                SnapshotLayer(
                    plan.name,
                    plan.spec,
                    plan.logical_pages,
                    torch.full(plan.shape, i + 1, dtype=torch.bfloat16),
                )
                for plan in snapshot_wire_plan(
                    pool.expected_specs, lengths[i], pool.budget
                )
            ),
        )
        for i, key in enumerate(keys)
    )


def dispatch(pool, keys, *, round_id=0, k=1, deadline=None):
    return pool.dispatch_batch(
        snapshots(pool, keys, round_id=round_id),
        tuple(key.sequence_id + 10 for key in keys),
        torch.stack(
            [torch.full((8,), key.sequence_id, dtype=torch.bfloat16) for key in keys]
        ),
        k=k,
        deadline=pool.deadline() if deadline is None else deadline,
    )


def test_two_verifiers_reorder_ragged_snapshots_without_route_or_namespace_mixups(
    pool_runtime,
):
    client, servers, oracles = pool_runtime
    first, second = client(), client("other-verifier")
    for pool in (first, second):
        keys = pool.open_sequences((0, 1, 2, 3), deadline=pool.deadline())
        keys = (keys[3], keys[0], keys[2], keys[1])
        results = pool.collect_batch(dispatch(pool, keys))
        assert tuple(result.token_ids for result in results) == (
            (37,),
            (10,),
            (28,),
            (19,),
        )
        assert all(result.valid_length == 1 for result in results)
        before = tuple(len(oracle.calls) for oracle in oracles)
        assert all(
            result.valid_length == 0
            for result in pool.collect_batch(dispatch(pool, keys, round_id=1, k=0))
        )
        assert before == tuple(len(oracle.calls) for oracle in oracles)
        assert all(
            result.valid_length == 1
            for result in pool.collect_batch(dispatch(pool, keys, round_id=2))
        )
    assert all(len(server.registry) == 4 for server in servers)
    for pool in (first, second):
        pool.close_sequences(tuple(pool.records))
    assert all(len(server.registry) == 0 for server in servers)


def test_parallel_drivers_and_deferred_unrelated_close_preserve_pending_batch(
    pool_runtime,
):
    client, servers, oracles = pool_runtime
    pool = client()
    keys = pool.open_sequences((0, 1, 2), deadline=pool.deadline())
    for oracle in oracles:
        oracle.release.clear()
    handle = dispatch(pool, keys[:2])
    try:
        assert all(oracle.entered.wait(2) for oracle in oracles)
        pool.close_sequences(())
        pool.close_sequences((SequenceKey("unknown", 99, 0),))
        pool.close_sequences((keys[2],))
        assert not pool.active(keys[2])
        assert sum(len(server.registry) for server in servers) == 3
    finally:
        for oracle in oracles:
            oracle.release.set()
    assert tuple(result.token_ids for result in pool.collect_batch(handle)) == (
        (10,),
        (19,),
    )
    assert keys[2] not in pool.records
    assert sum(len(server.registry) for server in servers) == 2
    with pytest.raises(RuntimeError, match="stale"):
        pool.collect_batch(handle)


@pytest.mark.parametrize("policy", ["target_only", "error"])
def test_one_worker_loss_preserves_healthy_results_and_drains_strict_jobs(
    pool_runtime, policy
):
    client, servers, _ = pool_runtime
    pool = client(policy=policy)
    keys = pool.open_sequences((0, 1), deadline=pool.deadline())
    servers[0].stop()
    handle = dispatch(pool, keys)
    if policy == "error":
        with pytest.raises(RemoteDraftError):
            pool.collect_batch(handle)
    else:
        assert tuple(result.valid_length for result in pool.collect_batch(handle)) == (
            0,
            1,
        )
    assert all(job.done() for _, _, job in handle.jobs)
    assert not pool.active(keys[0]) and pool.active(keys[1])
    fresh = pool.open_sequences((2,), deadline=pool.deadline())[0]
    assert pool.active(fresh)
    assert all(
        result.valid_length == 1
        for result in pool.collect_batch(dispatch(pool, (fresh, keys[1]), round_id=1))
    )
    assert len(servers[1].registry) == 2
    pool.close_sequences(keys + (fresh,))


def test_two_blocked_workers_share_one_collect_deadline(pool_runtime):
    client, _, oracles = pool_runtime
    pool = client(policy="target_only")
    keys = pool.open_sequences((0, 1), deadline=pool.deadline())
    for oracle in oracles:
        oracle.release.clear()
    started = time.monotonic()
    handle = dispatch(pool, keys, deadline=started + 0.3)
    try:
        assert all(oracle.entered.wait(1) for oracle in oracles)
        assert all(result.valid_length == 0 for result in pool.collect_batch(handle))
        assert time.monotonic() - started < 0.55
        assert all(job.done() for _, _, job in handle.jobs)
        assert not any(pool.active(key) for key in keys)
    finally:
        for oracle in oracles:
            oracle.release.set()


def test_many_open_requests_share_one_deadline_even_when_peers_never_ack():
    release = threading.Event()
    entered = [threading.Event(), threading.Event()]
    errors = []
    threads = []
    endpoints = []
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=4, dtype=torch.bfloat16
    )
    with ExitStack() as cleanup:
        for index in range(2):
            listener = socket.socket()
            cleanup.callback(listener.close)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            listener.settimeout(2)
            endpoints.append(f"tcp://127.0.0.1:{listener.getsockname()[1]}")

            def peer(listener=listener, index=index):
                conn = None
                try:
                    sock, _ = listener.accept()
                    conn = FramedConnection(sock)
                    envelope, _ = decode_target_kv_message(conn.recv(timeout=2).body)
                    sid = f"peer-{index}"
                    conn.send_control(
                        encode_target_kv_message(
                            TargetKVHelloAck(sid, 1, "contract", 128, 1 << 20, 4),
                            session_id=sid,
                            request_id=envelope.request_id,
                        )
                    )
                    conn.recv(timeout=2)
                    entered[index].set()
                    assert release.wait(3)
                except Exception as error:
                    errors.append(error)
                finally:
                    if conn is not None:
                        conn.close()

            thread = threading.Thread(target=peer)
            thread.start()
            threads.append(thread)
        pool = TargetKVDraftPool(
            tuple(endpoints),
            TargetKVHello("v", "contract", 8, 128, 64, 128, 1 << 20, max_num_seqs=4),
            expected_specs={"full": spec},
            failure_policy="target_only",
        )
        try:
            pool.connect()
            start = time.monotonic()
            keys = pool.open_sequences((0, 1, 2, 3), deadline=start + 0.3)
            assert all(event.is_set() for event in entered)
            assert time.monotonic() - start < 0.55
            assert len(keys) == 4 and not any(pool.active(key) for key in keys)
        finally:
            release.set()
            pool.close()
            for thread in threads:
                thread.join(3)
                assert not thread.is_alive()
        assert not errors


@pytest.mark.parametrize("finish", ["cancel", "close"])
def test_cancel_and_participating_close_drain_all_jobs_and_retire_deferred_rows(
    pool_runtime, finish
):
    client, servers, _ = pool_runtime
    pool = client()
    first, unrelated = pool.open_sequences((0, 1), deadline=pool.deadline())
    handle = dispatch(pool, (first,))
    pool.close_sequences((unrelated,))
    if finish == "cancel":
        pool.cancel(handle)
    else:
        pool.close_sequences((first,))
    assert unrelated not in pool.records
    assert len(servers[1].registry) == 0
    assert not pool.active(first)
    assert all(job.done() for _, _, job in handle.jobs)
    with pytest.raises(RuntimeError, match="stale"):
        pool.collect_batch(handle)


def test_sequence_reuse_changes_generation_even_when_worker_changes(pool_runtime):
    client, _, _ = pool_runtime
    pool = client()
    first = pool.open_sequences((0,), deadline=pool.deadline())[0]
    pool.close_sequences((first,))
    second = pool.open_sequences((0,), deadline=pool.deadline())[0]
    assert first != second and first.generation != second.generation
    pool.close_sequences((first,))
    assert pool.active(second)
    assert pool.collect_batch(dispatch(pool, (second,)))[0].valid_length == 1


def test_reconnected_worker_admits_new_identity_without_reviving_terminal_rows(
    pool_runtime,
):
    client, servers, oracles = pool_runtime
    pool = client(policy="target_only")
    old, healthy = pool.open_sequences((0, 1), deadline=pool.deadline())
    endpoint = servers[0].endpoint
    servers[0].stop()
    assert tuple(
        result.valid_length
        for result in pool.collect_batch(dispatch(pool, (old, healthy)))
    ) == (0, 1)
    replacement = TargetKVDraftServer(
        oracles[0],
        contract_fingerprint="contract",
        hidden_size=8,
        vocab_size=128,
        expected_specs=pool.expected_specs,
        snapshot_budget=pool.budget,
        receiver_budget=pool.budget,
        max_num_seqs=4,
    )
    replacement.start(endpoint)
    try:
        pool.connect(deadline=pool.deadline())
        fresh = pool.open_sequences((2,), deadline=pool.deadline())[0]
        assert pool.active(fresh) and not pool.active(old)
        assert len(replacement.registry) == 1
        assert tuple(
            result.valid_length
            for result in pool.collect_batch(
                dispatch(pool, (old, fresh, healthy), round_id=1)
            )
        ) == (0, 1, 1)
        pool.close_sequences((old, fresh, healthy))
        assert len(replacement.registry) == 0
    finally:
        replacement.stop()


@pytest.mark.parametrize("policy", ["target_only", "error"])
def test_expired_open_cleans_unreturned_strict_keys_and_bounds_terminal_records(
    pool_runtime, policy
):
    client, servers, _ = pool_runtime
    pool = client(policy=policy)
    deadline = time.monotonic() - 1
    if policy == "error":
        with pytest.raises(RemoteDraftError, match="open"):
            pool.open_sequences((0, 1, 2, 3), deadline=deadline)
        assert not pool.records
        assert all(not session.records for session in pool._sessions)
        assert not pool._routes and pool._pending is None
    else:
        keys = pool.open_sequences((0, 1, 2, 3), deadline=deadline)
        assert len(pool.records) == 4
        assert not any(pool.active(key) for key in keys)
        with pytest.raises(ValueError, match="capacity"):
            pool.open_sequences((4,), deadline=pool.deadline())
        pool.close_sequences(keys)
        assert not pool.records
    assert all(len(server.registry) == 0 for server in servers)


def test_invalid_batch_and_foreign_handles_leave_current_proposal_usable(pool_runtime):
    client, _, _ = pool_runtime
    pool, foreign = client(), client("other")
    keys = pool.open_sequences((0, 1), deadline=pool.deadline())
    with pytest.raises(ValueError, match="capacity"):
        pool.open_sequences((2, 3, 4), deadline=pool.deadline())
    with pytest.raises(ValueError, match="identity"):
        pool.dispatch_batch(
            tuple(
                replace(s, identity=replace(s.identity, session_epoch=99999))
                for s in snapshots(pool, keys)
            ),
            (10, 11),
            torch.zeros((2, 8), dtype=torch.bfloat16),
            deadline=pool.deadline(),
        )
    with pytest.raises(ValueError, match="caller to CPU"):
        pool.dispatch_batch(
            snapshots(pool, keys),
            (10, 11),
            torch.empty((2, 8), dtype=torch.bfloat16, device="meta"),
            deadline=pool.deadline(),
        )
    handle = dispatch(pool, keys)
    with pytest.raises(RuntimeError, match="busy"):
        dispatch(pool, keys, round_id=1)
    with pytest.raises(RuntimeError, match="foreign"):
        foreign.collect_batch(handle)
    assert all(result.valid_length == 1 for result in pool.collect_batch(handle))
    pool.close()
    assert not pool.records and not pool._routes and not pool._epochs
    assert not pool._probes and pool._pending is None


@pytest.mark.parametrize("policy", ["error", "target_only"])
@pytest.mark.parametrize(
    "pool_runtime",
    [{"max_sequences_per_verifier": 1}, {"max_sequences": 4}],
    indirect=True,
    ids=["verifier-credit", "global-credit"],
)
def test_open_credit_rejection_preserves_established_sibling_proposals(
    pool_runtime, policy
):
    client, servers, _ = pool_runtime
    pool = client(policy=policy)
    established = pool.open_sequences((0, 1), deadline=pool.deadline())
    if servers[0].limits.max_sequences == 4:
        # Fill the shared worker capacity with independent verifier identities.
        second, third = client("second"), client("third")
        second.open_sequences((10, 11, 12, 13), deadline=second.deadline())
        third.open_sequences((20, 21), deadline=third.deadline())
        assert all(len(server.registry) == 4 for server in servers)
    if policy == "error":
        with pytest.raises(RemoteDraftError, match="QUEUE_FULL: sequence credit"):
            pool.open_sequences((2, 3), deadline=pool.deadline())
        assert set(pool.records) == set(established)
    else:
        denied = pool.open_sequences((2, 3), deadline=pool.deadline())
        assert not any(pool.active(key) for key in denied)
    assert all(pool.active(key) for key in established)
    assert tuple(
        result.token_ids for result in pool.collect_batch(dispatch(pool, established))
    ) == ((10,), (19,))
