# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Own-KV pool routing through real sockets, codecs, and request KV state."""

import threading
import time
from contextlib import ExitStack
from dataclasses import replace

import pytest
import torch
from msgspec.structs import replace as replace_struct

from tests.v1.spec_decode.remote.runtime_fixtures import (
    K,
    KVDigestExecutor,
    digest_row,
    make_planner,
)
from tests.v1.spec_decode.remote.test_runtime import CAPABILITIES, IDENTITY, SCHEMA
from vllm.v1.spec_decode.remote.conditioned_pool import ConditionedDraftPool
from vllm.v1.spec_decode.remote.protocol import (
    FeatureSlot,
    RemoteServerLimits,
    SpeculatorStatusCode,
)
from vllm.v1.spec_decode.remote.runtime.adapter import TargetConditionedDraftAdapter
from vllm.v1.spec_decode.remote.server import RemoteDraftServer
from vllm.v1.spec_decode.remote.session import RemoteDraftError
from vllm.v1.spec_decode.remote.state import (
    FeatureBatch,
    RemoteProposalBatch,
    SequenceState,
)

POOL_SCHEMA = replace_struct(
    SCHEMA,
    slots=(*SCHEMA.slots, FeatureSlot("last_hidden_states", "float32", (1,))),
)


class GatedExecutor(KVDigestExecutor):
    """External model boundary with deterministic completion and feature checks."""

    def __init__(self, pool):
        super().__init__(pool)
        self.entered = threading.Event()
        self.completed = threading.Event()
        self.release = None

    def execute(self, plan, features, *, num_speculative_tokens=None):
        self.entered.set()
        if self.release is not None:
            assert self.release.wait(5), "test must release the external executor"
        torch.testing.assert_close(features[2][:, 0], features[0].float() / 4)
        result = super().execute(
            plan, features, num_speculative_tokens=num_speculative_tokens
        )
        self.completed.set()
        return result


@pytest.fixture
def conditioned_pool_runtime():
    servers, planners, executors = [], [], []
    with ExitStack() as cleanup:
        for _ in range(2):
            planner = make_planner(num_blocks=64, max_num_reqs=8)
            executor = GatedExecutor(planner.block_pool)
            adapter = TargetConditionedDraftAdapter(
                executor,
                planner,
                capabilities=replace_struct(
                    CAPABILITIES,
                    required_features=(
                        *CAPABILITIES.required_features,
                        "last_hidden_states",
                    ),
                ),
                feature_schema=POOL_SCHEMA,
                methods={"mtp"},
            )
            server = RemoteDraftServer(adapter, limits=RemoteServerLimits(4, 64, 8, 64))
            server.start("tcp://127.0.0.1:0")
            cleanup.callback(server.stop)
            servers.append(server)
            planners.append(planner)
            executors.append(executor)

        def client(identity="verifier-a", policy="error"):
            pool = ConditionedDraftPool(
                tuple(server.endpoint for server in servers),
                replace(
                    IDENTITY,
                    verifier_instance_id=identity,
                    provided_features=(
                        *IDENTITY.provided_features,
                        "last_hidden_states",
                    ),
                ),
                failure_policy=policy,
                request_timeout_ms=5000,
            )
            cleanup.callback(pool.close)
            pool.connect()
            return pool

        try:
            yield client, servers, planners, executors
        finally:
            for executor in executors:
                if executor.release is not None:
                    executor.release.set()


def features(tokens, lookahead=None):
    values = torch.tensor(tokens, dtype=torch.int32)
    return FeatureBatch(
        POOL_SCHEMA.schema_id,
        (
            values,
            None
            if lookahead is None
            else torch.tensor(lookahead, dtype=torch.int32).reshape(-1, 1),
            values.float().reshape(-1, 1) / 4,
        ),
    )


def proposal(batch_id, keys, query_lens, tokens, accepted, recovery, k=K):
    return RemoteProposalBatch(
        batch_id,
        tuple(keys),
        torch.tensor(accepted, dtype=torch.int32),
        torch.tensor(recovery, dtype=torch.int64),
        features(tokens, [17 + i for i in range(len(keys))]),
        torch.tensor(query_lens, dtype=torch.int32),
        k,
    )


def run(pool, batch, chunks=()):
    return pool.collect(
        pool.dispatch_prepared(
            batch, tuple(chunks), deadline=pool.preparation_deadline()
        )
    )


def start_prompts(pool, ids=(0, 1), prompts=((1, 2), (7, 8)), *, batch_id=0):
    keys = pool.reserve_sequences(ids, deadline=pool.preparation_deadline())
    output = run(
        pool,
        proposal(
            batch_id, keys, [0] * len(keys), [], [0] * len(keys), [30] * len(keys)
        ),
        [(key, features(prompt), True) for key, prompt in zip(keys, prompts)],
    )
    assert output.output.valid_lengths.tolist() == [K] * len(keys)
    return keys, output.output.token_ids.tolist()


def test_ragged_reorder_and_final_prefill_keep_token_and_sequence_rows(
    conditioned_pool_runtime,
):
    client, _, planners, _ = conditioned_pool_runtime
    pool = client()
    (a, b), (ad, bd) = start_prompts(pool)
    (c,) = pool.reserve_sequences((2,), deadline=pool.preparation_deadline())
    assert pool.sequence_state(c) is SequenceState.OPENING
    result = run(
        pool,
        proposal(
            1, [b, c, a], [4, 0, 2], [30, *bd, 30, ad[0]], [1, 0, 0], [40, 50, 60]
        ),
        ((c, features([11, 12, 13]), True),),
    )
    assert result.row_statuses == (SpeculatorStatusCode.OK,) * 3
    assert result.output.token_ids.tolist() == [
        digest_row([7, 8, 30, bd[0]], 40),
        digest_row([11, 12, 13], 50),
        digest_row([1, 2, 30], 60),
    ]
    pool.close_sequences((a, b, c))
    assert all(planner.block_pool.num_free_blocks == 63 for planner in planners)


def test_two_verifiers_using_same_ids_keep_separate_kv(conditioned_pool_runtime):
    client, _, planners, _ = conditioned_pool_runtime
    first, second = client(), client("verifier-b")
    a, ad = start_prompts(first)
    b, bd = start_prompts(second, prompts=((11, 12), (17, 18)))
    assert set(a).isdisjoint(b)
    for pool, keys, drafts, prompts in (
        (first, a, ad, ((1, 2), (7, 8))),
        (second, b, bd, ((11, 12), (17, 18))),
    ):
        result = run(
            pool,
            proposal(
                1,
                keys[::-1],
                [4, 4],
                [30, *drafts[1], 30, *drafts[0]],
                [0, 3],
                [40, 50],
            ),
        )
        assert result.output.token_ids.tolist() == [
            digest_row([*prompts[1], 30], 40),
            digest_row([*prompts[0], 30, *drafts[0]], 50),
        ]
        pool.close_sequences(keys)
    assert all(planner.block_pool.num_free_blocks == 63 for planner in planners)


def test_prefill_only_and_zero_k_preserve_context_for_later_drafts(
    conditioned_pool_runtime,
):
    client, _, _, _ = conditioned_pool_runtime
    pool = client()
    keys = pool.reserve_sequences((0, 1), deadline=pool.preparation_deadline())
    assert all(pool.sequence_state(key) is SequenceState.OPENING for key in keys)
    result = run(
        pool,
        proposal(0, [], [], [], [], [], k=0),
        ((keys[0], features([1, 2], [3]), False), (keys[1], features([7], [8]), False)),
    )
    assert result.output.token_ids.shape == (0, K)
    assert all(pool.sequence_state(key) is SequenceState.PREFILLING for key in keys)
    result = run(
        pool,
        proposal(1, keys, [0, 0], [], [0, 0], [30, 40], k=0),
        ((keys[0], features([3]), True), (keys[1], features([8, 9]), True)),
    )
    assert result.output.valid_lengths.tolist() == [0, 0]
    result = run(pool, proposal(2, keys[::-1], [1, 1], [40, 30], [0, 0], [50, 60], k=1))
    assert result.output.valid_lengths.tolist() == [1, 1]
    assert result.output.token_ids[:, 0].tolist() == [
        digest_row([7, 8, 9, 40], 50)[0],
        digest_row([1, 2, 3, 30], 60)[0],
    ]


def test_worker_loss_only_falls_back_its_sticky_rows(conditioned_pool_runtime):
    client, servers, _, _ = conditioned_pool_runtime
    pool = client(policy="target_only")
    keys, drafts = start_prompts(pool)
    servers[0].stop()
    result = run(
        pool,
        proposal(1, keys, [4, 4], [30, *drafts[0], 30, *drafts[1]], [0, 0], [40, 50]),
    )
    assert result.output.valid_lengths.tolist() == [0, K]
    assert result.output.token_ids[1].tolist() == digest_row([7, 8, 30], 50)
    assert pool.sequence_state(keys[0]) in (
        SequenceState.INVALID,
        SequenceState.TARGET_ONLY,
    )
    assert pool.sequence_state(keys[1]) is SequenceState.READY
    fresh, _ = start_prompts(pool, (2,), ((11,),), batch_id=2)
    assert pool.sequence_state(fresh[0]) is SequenceState.READY
    assert pool.sequence_state(keys[0]) in (
        SequenceState.INVALID,
        SequenceState.TARGET_ONLY,
    )


def test_unrelated_close_does_not_cancel_pending_proposal(conditioned_pool_runtime):
    client, _, _, executors = conditioned_pool_runtime
    pool = client()
    keys, drafts = start_prompts(pool, (0, 1, 2), ((1,), (2,), (3,)))
    release = threading.Event()
    for executor in executors:
        executor.entered.clear()
        executor.release = release
    handle = pool.dispatch_prepared(
        proposal(
            1, keys[:2], [4, 4], [30, *drafts[0], 30, *drafts[1]], [0, 0], [40, 50]
        ),
        (),
        deadline=pool.preparation_deadline(),
    )
    try:
        assert all(executor.entered.wait(3) for executor in executors)
        pool.close_sequences(())
        pool.close_sequences((keys[2],))
    finally:
        release.set()
    assert pool.collect(handle).output.valid_lengths.tolist() == [K, K]
    assert pool.sequence_state(keys[2]) is None
    with pytest.raises(RuntimeError, match="stale|collected"):
        pool.collect(handle)


def test_reused_sequence_id_on_other_worker_has_a_new_generation(
    conditioned_pool_runtime,
):
    client, _, _, _ = conditioned_pool_runtime
    pool = client()
    (old,), _ = start_prompts(pool, (0,), ((1,),))
    pool.close_sequences((old,))
    (new,), _ = start_prompts(pool, (0,), ((7,),), batch_id=1)
    assert old != new
    assert (old.verifier_instance_id, old.generation) != (
        new.verifier_instance_id,
        new.generation,
    )
    pool.close_sequences((old,))
    assert pool.sequence_state(new) is SequenceState.READY


@pytest.mark.parametrize("policy", ["error", "target_only"])
def test_blocked_prefill_cannot_prevent_healthy_endpoint_from_proposing(
    conditioned_pool_runtime, policy
):
    client, _, _, executors = conditioned_pool_runtime
    pool = client(policy=policy)
    a, b = pool.reserve_sequences((0, 1), deadline=pool.preparation_deadline())
    release = threading.Event()
    executors[0].release = release
    started = time.monotonic()
    handle = pool.dispatch_prepared(
        proposal(0, (b,), [0], [], [0], [30]),
        ((a, features([1, 2], [3]), False), (b, features([7, 8]), True)),
        deadline=started + 1,
    )
    try:
        assert executors[0].entered.wait(3)
        assert executors[1].completed.wait(3)
        if policy == "target_only":
            result = pool.collect(handle)
            assert result.output.valid_lengths.tolist() == [K]
            assert result.output.token_ids[0].tolist() == digest_row([7, 8], 30)
        else:
            with pytest.raises(RemoteDraftError, match="prefill|TIMEOUT|deadline"):
                pool.collect(handle)
        assert time.monotonic() - started < 2
        assert pool.sequence_state(b) is SequenceState.READY
        assert pool.sequence_state(a) in (
            SequenceState.TARGET_ONLY,
            SequenceState.INVALID,
        )
    finally:
        release.set()
