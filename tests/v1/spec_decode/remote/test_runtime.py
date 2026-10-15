# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone own-KV runtime: draft KV pool, step planning, and the
target-conditioned adapter driven end to end through the server."""

from collections import defaultdict

import numpy as np
import pytest
import torch

from vllm.v1.spec_decode.remote.capabilities import (
    SpeculatorPlacementCapabilities,
    TargetFeatureKind,
)
from vllm.v1.spec_decode.remote.protocol import (
    FeatureSlot,
    SequenceKey,
    SpeculatorStatusCode,
    TargetFeatureSchema,
)
from vllm.v1.spec_decode.remote.runtime.adapter import (
    DraftStepExecutor,
    TargetConditionedDraftAdapter,
)
from vllm.v1.spec_decode.remote.runtime.kv import (
    NULL_BLOCK_ID,
    DraftBlockPool,
    DraftKVExhausted,
)
from vllm.v1.spec_decode.remote.runtime.step import (
    BlockTableUpdate,
    DraftStepPlan,
    DraftStepPlanner,
    DraftStepRow,
)
from vllm.v1.spec_decode.remote.server import RemoteDraftServer
from vllm.v1.spec_decode.remote.session import RemoteDraftSession, VerifierIdentity
from vllm.v1.spec_decode.remote.state import (
    FeatureBatch,
    RemoteProposalBatch,
    SequenceState,
)

OK = SpeculatorStatusCode.OK
K = 3
VOCAB = 1000
BLOCK_SIZE = 2
STRIDE = 1 << 20


def key(sequence_id: int) -> SequenceKey:
    return SequenceKey("verifier", sequence_id, 0)


def make_planner(num_blocks=64, max_num_reqs=4, max_model_len=64):
    return DraftStepPlanner(
        DraftBlockPool(num_blocks, BLOCK_SIZE),
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        num_speculative_tokens=K,
    )


# ----------------------------------------------------------------------
# Block pool
# ----------------------------------------------------------------------


def test_pool_never_hands_out_the_null_block():
    pool = DraftBlockPool(num_blocks=4, block_size=2)
    assert pool.ensure(0, 6) == [1, 2, 3]
    assert NULL_BLOCK_ID not in pool.blocks(0)
    assert pool.num_free_blocks == 0


def test_pool_growth_is_all_or_nothing():
    pool = DraftBlockPool(num_blocks=4, block_size=2)
    assert pool.ensure(0, 3) == [1, 2]
    with pytest.raises(DraftKVExhausted):
        pool.ensure(1, 4)
    assert pool.blocks(1) == ()
    assert pool.ensure(0, 4) == []
    assert pool.ensure(0, 5) == [3]


def test_pool_release_returns_blocks_and_slots_cross_blocks():
    pool = DraftBlockPool(num_blocks=8, block_size=2)
    pool.ensure(0, 2)
    pool.ensure(1, 4)
    assert pool.slot_mapping(1, np.arange(4)).tolist() == [4, 5, 6, 7]
    pool.release(1)
    assert pool.num_free_blocks == 6
    assert pool.ensure(2, 2) == [2]


# ----------------------------------------------------------------------
# Step planning
# ----------------------------------------------------------------------


def test_plan_mirrors_a_mixed_runner_step():
    planner = make_planner()
    for i in (1, 2):
        assert planner.open(key(i))
    plan, failures = planner.plan(
        [
            DraftStepRow(key(1), 3, num_sampled=0, next_prefill_tokens=(9,)),
            DraftStepRow(key(2), 2, num_sampled=1, last_sampled=7),
        ]
    )
    assert failures == {}
    assert plan.req_indices.tolist() == [0, 1]
    assert plan.query_start_loc.tolist() == [0, 3, 5]
    assert plan.positions.tolist() == [0, 1, 2, 0, 1]
    assert plan.seq_lens.tolist() == [3, 2]
    assert plan.next_prefill_tokens.tolist() == [[9], [0]]
    # Room for the step plus K draft positions, in fresh tables.
    assert plan.block_table_updates == {
        0: BlockTableUpdate(0, [1, 2, 3]),
        1: BlockTableUpdate(0, [4, 5, 6]),
    }
    assert plan.slot_mapping.tolist() == [2, 3, 4, 8, 9]


def test_commit_keeps_only_the_accepted_prefix():
    planner = make_planner()
    planner.open(key(1))
    planner.commit(planner.plan([DraftStepRow(key(1), 4, num_sampled=1)])[0])
    # Verify one recovered + three drafts; two drafts are rejected.
    plan, _ = planner.plan(
        [DraftStepRow(key(1), 4, num_sampled=2, num_rejected=2, last_sampled=5)]
    )
    assert plan.positions.tolist() == [4, 5, 6, 7]
    planner.commit(plan)
    assert planner.num_computed(key(1)) == 6
    plan, _ = planner.plan([DraftStepRow(key(1), 4, num_sampled=4)])
    assert plan.positions.tolist() == [6, 7, 8, 9]
    assert plan.block_table_updates[0].start == 6


def test_rows_that_do_not_fit_fail_alone():
    planner = make_planner(num_blocks=5, max_model_len=8)
    for i in (1, 2, 3):
        planner.open(key(i))
    plan, failures = planner.plan(
        [
            DraftStepRow(key(1), 2, num_sampled=1),
            DraftStepRow(key(2), 9, num_sampled=1),
            DraftStepRow(key(3), 4, num_sampled=1),
            DraftStepRow(key(9), 1, num_sampled=1),
        ]
    )
    assert [row.key for row in plan.rows] == [key(1)]
    assert {i: status for i, (status, _) in failures.items()} == {
        1: SpeculatorStatusCode.OUT_OF_MEMORY,
        2: SpeculatorStatusCode.OUT_OF_MEMORY,
        3: SpeculatorStatusCode.SEQUENCE_DESYNC,
    }
    assert "max_model_len" in failures[1][1]


def test_request_slots_are_bounded_and_reused():
    planner = make_planner(max_num_reqs=1)
    assert planner.open(key(1))
    assert not planner.open(key(2))
    planner.commit(planner.plan([DraftStepRow(key(1), 3, num_sampled=1)])[0])
    planner.close(key(1))
    assert planner.open(key(2))
    plan, _ = planner.plan([DraftStepRow(key(2), 1, num_sampled=1)])
    assert plan.req_indices.tolist() == [0]
    assert plan.positions.tolist() == [0]
    assert plan.block_table_updates[0].start == 0


@pytest.mark.parametrize(
    "row",
    [
        DraftStepRow(key(1), 0, num_sampled=1),
        DraftStepRow(key(1), 2, num_sampled=1, num_rejected=2),
        DraftStepRow(key(1), 2, num_sampled=0),
        DraftStepRow(
            key(1), 2, num_sampled=0, num_rejected=1, next_prefill_tokens=(1,)
        ),
    ],
    ids=["empty", "all-rejected", "chunk-without-lookahead", "chunk-rejects"],
)
def test_malformed_rows_are_caller_bugs(row):
    planner = make_planner()
    planner.open(key(1))
    with pytest.raises(ValueError):
        planner.plan([row])


# ----------------------------------------------------------------------
# Adapter end to end
# ----------------------------------------------------------------------


def digest_row(prefix, next_token) -> list[int]:
    digest = sum((token + 1) * (position + 1) for position, token in enumerate(prefix))
    return [(digest + next_token + j) % VOCAB for j in range(K)]


class KVDigestExecutor(DraftStepExecutor):
    """CPU stand-in for a device executor.

    Writes (position, target token) into every token's KV slot and drafts
    from a digest of the committed prefix read back through its own block
    tables, so any position, slot, block table, or committed-length error
    in the runtime changes the drafts.
    """

    def __init__(self, pool: DraftBlockPool):
        self.pool = pool
        self.kv = np.full(pool.num_blocks * pool.block_size, -1, dtype=np.int64)
        self.block_tables: dict[int, list[int]] = defaultdict(list)
        self.plans: list[DraftStepPlan] = []

    def execute(self, plan, features):
        self.plans.append(plan)
        for req_index, update in plan.block_table_updates.items():
            table = self.block_tables[req_index][: update.start]
            self.block_tables[req_index] = table + update.block_ids
        tokens = features[0].numpy().astype(np.int64)
        assert tokens.shape == (plan.num_tokens,)
        self.kv[plan.slot_mapping] = plan.positions * STRIDE + tokens
        drafts = []
        for i in range(plan.num_reqs):
            committed = int(plan.seq_lens[i] - plan.num_rejected[i])
            table = np.array(self.block_tables[int(plan.req_indices[i])])
            positions = np.arange(committed)
            slots = table[positions // BLOCK_SIZE] * BLOCK_SIZE + positions % BLOCK_SIZE
            values = self.kv[slots]
            assert (values // STRIDE == positions).all()
            next_token = (
                plan.last_sampled[i]
                if plan.num_sampled[i]
                else plan.next_prefill_tokens[i][0]
            )
            drafts.append(digest_row((values % STRIDE).tolist(), int(next_token)))
        return torch.tensor(drafts)


SCHEMA = TargetFeatureSchema(
    schema_id=1,
    slots=(
        FeatureSlot(
            kind=TargetFeatureKind.TOKEN_IDS.value, dtype="int32", trailing_shape=()
        ),
        FeatureSlot(
            kind=TargetFeatureKind.NEXT_PREFILL_TOKENS.value,
            dtype="int32",
            trailing_shape=(1,),
            optional=True,
            per_sequence=True,
        ),
    ),
)
CAPABILITIES = SpeculatorPlacementCapabilities(
    state_dependency="own_kv",
    required_features=(
        TargetFeatureKind.TOKEN_IDS.value,
        TargetFeatureKind.NEXT_PREFILL_TOKENS.value,
    ),
    num_prefill_lookahead=1,
    standalone_weights="complete",
)
IDENTITY = VerifierIdentity(
    verifier_instance_id="verifier",
    target_fingerprint="",
    tokenizer_fingerprint="",
    method="mtp",
    num_speculative_tokens=K,
    parallel_drafting=False,
    provided_features=CAPABILITIES.required_features,
    max_num_seqs=4,
    max_num_batched_tokens=64,
    max_model_len=64,
    vocab_size=VOCAB,
)


@pytest.fixture
def runtime():
    planner = make_planner(num_blocks=12)
    executor = KVDigestExecutor(planner.block_pool)
    adapter = TargetConditionedDraftAdapter(
        executor,
        planner,
        capabilities=CAPABILITIES,
        feature_schema=SCHEMA,
        methods={"mtp"},
    )
    server = RemoteDraftServer(adapter)
    server.start("tcp://127.0.0.1:0")
    session = RemoteDraftSession(server.endpoint, IDENTITY, request_timeout_ms=5000)
    session.connect()
    yield session, executor
    session.close()
    server.stop()


def chunk(tokens, lookahead=None):
    return FeatureBatch(
        1,
        (
            torch.tensor(tokens, dtype=torch.int32),
            None
            if lookahead is None
            else torch.tensor([[lookahead]], dtype=torch.int32),
        ),
    )


def round_(session, batch_id, keys, query_lens, tokens, accepted, recovery):
    return session.collect(
        session.dispatch(
            RemoteProposalBatch(
                batch_id=batch_id,
                keys=tuple(keys),
                accepted_counts=torch.tensor(accepted, dtype=torch.int32),
                recovery_tokens=torch.tensor(recovery, dtype=torch.int64),
                features=chunk(tokens),
                query_lens=torch.tensor(query_lens, dtype=torch.int32),
            )
        )
    )


def test_adapter_drafts_from_the_verified_prefix(runtime):
    session, executor = runtime
    a = session.open_sequence(1)
    session.prefill(a, chunk([11, 12, 13], lookahead=14), is_final=False)
    b = session.open_sequence(2)
    session.prefill_many([(a, chunk([14, 15]), True), (b, chunk([21, 22]), True)])

    # The first round only delivers the tokens sampled from the prompts.
    result = round_(session, 0, [a, b], [0, 0], [], [0, 0], [30, 40])
    a_drafts = digest_row([11, 12, 13, 14, 15], 30)
    b_drafts = digest_row([21, 22], 40)
    assert result.row_statuses == (OK, OK)
    assert result.output.token_ids.tolist() == [a_drafts, b_drafts]

    # a accepts two drafts, b none; rejected positions must not leak into
    # the next prefix.
    result = round_(
        session,
        1,
        [a, b],
        [4, 4],
        [30, *a_drafts, 40, *b_drafts],
        [2, 0],
        [31, 41],
    )
    a_prefix = [11, 12, 13, 14, 15, 30, *a_drafts[:2]]
    assert result.output.token_ids.tolist() == [
        digest_row(a_prefix, 31),
        digest_row([21, 22, 40], 41),
    ]
    a_drafts = result.output.token_ids[0].tolist()

    # b finishes; its request slot and blocks go to a new sequence c.
    session.close_sequences((b,))
    c = session.open_sequence(3)
    session.prefill(c, chunk([7]), is_final=True)
    result = round_(session, 2, [a, c], [4, 0], [31, *a_drafts], [3, 0], [32, 50])
    assert result.output.token_ids.tolist() == [
        digest_row([*a_prefix, 31, *a_drafts], 32),
        digest_row([7], 50),
    ]
    assert result.output.valid_lengths.tolist() == [K, K]
    c_update = executor.plans[-1].block_table_updates[1]
    assert c_update.start == 0


def test_round_must_not_resend_a_buffered_prompt_chunk(runtime):
    session, _ = runtime
    a = session.open_sequence(1)
    session.prefill(a, chunk([11, 12]), is_final=True)
    result = round_(session, 0, [a], [2], [11, 12], [0], [30])
    assert result.row_statuses == (SpeculatorStatusCode.SEQUENCE_DESYNC,)
    assert session.sequence_state(a) is SequenceState.DESYNCED


def test_prompt_chunk_without_lookahead_is_refused(runtime):
    session, _ = runtime
    a = session.open_sequence(1)
    session.prefill(a, chunk([11, 12]), is_final=False)
    assert session.sequence_state(a) is SequenceState.TARGET_ONLY


def test_exhausted_draft_kv_degrades_only_that_request(runtime):
    session, _ = runtime
    a = session.open_sequence(1)
    # 11 usable blocks of 2 slots; this prompt plus K lookahead needs 12.
    session.prefill(a, chunk(list(range(20)), lookahead=99), is_final=False)
    assert session.sequence_state(a) is SequenceState.TARGET_ONLY
    b = session.open_sequence(2)
    session.prefill(b, chunk([5]), is_final=True)
    result = round_(session, 0, [b], [0], [], [0], [6])
    assert result.row_statuses == (OK,)


def test_request_slots_beyond_the_runtime_are_queue_full(runtime):
    session, _ = runtime
    keys = [session.open_sequence(i) for i in range(5)]
    assert [session.sequence_state(k) for k in keys] == [
        SequenceState.PREFILLING
    ] * 4 + [SequenceState.TARGET_ONLY]
