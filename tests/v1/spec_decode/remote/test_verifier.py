# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 request/chunk translation against a real socket and stateful CPU KV.

These tests exercise the public propose/lifecycle boundary without CUDA. The
KV digest makes missing, duplicated and rejected context observable in drafts.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.spec_decode.remote.test_runtime import (
    VOCAB,
    K,
    KVDigestExecutor,
    digest_row,
    make_planner,
)
from vllm.config.speculative import RemoteDraftConfig
from vllm.v1.spec_decode.remote.capabilities import SpeculatorPlacementCapabilities
from vllm.v1.spec_decode.remote.protocol import (
    FeatureSlot,
    SpeculatorStatusCode,
    TargetFeatureSchema,
)
from vllm.v1.spec_decode.remote.runtime.adapter import TargetConditionedDraftAdapter
from vllm.v1.spec_decode.remote.runtime.schema import RuntimeContract
from vllm.v1.spec_decode.remote.server import AdapterError, RemoteDraftServer
from vllm.v1.worker.gpu.spec_decode import remote

SCHEMA = TargetFeatureSchema(
    1,
    (
        FeatureSlot("token_ids", "int32", ()),
        FeatureSlot("positions", "int64", ()),
        FeatureSlot("last_hidden_states", "float32", (4,)),
        FeatureSlot(
            "next_prefill_tokens", "int64", (1,), optional=True, per_sequence=True
        ),
    ),
)
CONTRACT = RuntimeContract(
    SCHEMA,
    SpeculatorPlacementCapabilities(
        state_dependency="own_kv",
        required_features=("token_ids", "positions", "last_hidden_states"),
        num_prefill_lookahead=1,
        standalone_weights="complete",
    ),
    "target",
    "tokenizer",
    "draft",
)


def test_ipc_features_do_not_materialize_hidden_states_on_host():
    instance = object.__new__(remote.RemoteSpeculator)
    instance.device = torch.device("meta")
    instance._contract = CONTRACT
    instance._session = SimpleNamespace(selected_transport="cuda_ipc")
    batch = inputs(["a"], [[1, 2]], [[0, 1]], [2])
    hidden = torch.empty(2, 4, device="meta")
    # A meta tensor cannot be copied to CPU. This catches an accidental host
    # materialization without requiring a GPU in the protocol test suite.
    features = instance._features(batch, hidden, None, torch.tensor([[3]]))
    assert features["last_hidden_states"].device.type == "meta"
    assert features["token_ids"].tolist() == [1, 2]
    assert features["positions"].tolist() == [0, 1]
    assert features["next_prefill_tokens"].tolist() == [[3]]


class TPGroup:
    def __init__(self):
        self.rank_in_group = 0
        self.payload = None

    def broadcast_object(self, value, src):
        if self.rank_in_group == src:
            self.payload = value
        return self.payload

    def broadcast(self, value, src):
        pytest.fail("remote drafts must not initialize a GPU tensor collective")


def config(endpoint):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            async_scheduling=False, max_num_seqs=4, max_num_batched_tokens=64
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        model_config=SimpleNamespace(
            is_multimodal_model=False, max_model_len=64, get_vocab_size=lambda: VOCAB
        ),
        lora_config=None,
        watermark_config=None,
        speculative_config=SimpleNamespace(
            method="eagle3",
            num_speculative_tokens=K,
            enable_adaptive_verification=False,
            remote_draft=RemoteDraftConfig(
                endpoint=endpoint, failure_policy="error", request_timeout_ms=5000
            ),
        ),
    )


@pytest.fixture
def verifier(monkeypatch, request):
    planner = make_planner()
    executor = KVDigestExecutor(planner.block_pool)
    adapter = TargetConditionedDraftAdapter(
        executor,
        planner,
        capabilities=CONTRACT.capabilities,
        feature_schema=SCHEMA,
        methods={"eagle3"},
        draft_checkpoint_fingerprint=CONTRACT.draft_fingerprint,
    )
    server = RemoteDraftServer(
        adapter,
        target_fingerprint=CONTRACT.target_fingerprint,
        tokenizer_fingerprint=CONTRACT.tokenizer_fingerprint,
    )
    server.start("tcp://127.0.0.1:0")
    group = TPGroup()
    monkeypatch.setattr(remote, "get_tp_group", lambda: group)
    monkeypatch.setattr(remote, "create_runtime_contract", lambda _: CONTRACT)
    settings = config(server.endpoint)
    settings.speculative_config.remote_draft.failure_policy = getattr(
        request, "param", "error"
    )
    instance = remote.RemoteSpeculator(settings, torch.device("cpu"))
    yield instance, executor, settings, group
    instance.close()
    server.stop()


def inputs(ids, tokens, positions, prefill, slots=None):
    count = len(ids)
    slots = slots if slots is not None else list(range(count))
    lens = [len(row) for row in tokens]
    return SimpleNamespace(
        req_ids=ids,
        num_reqs=count,
        num_tokens=sum(lens),
        idx_mapping=torch.tensor(slots),
        query_start_loc_np=np.cumsum([0, *lens]),
        input_ids=torch.tensor(sum(tokens, []), dtype=torch.int32),
        positions=torch.tensor(sum(positions, []), dtype=torch.int64),
        is_prefilling_np=np.array([length > 0 for length in prefill]),
        prefill_len_np=np.array(prefill),
    )


def propose(verifier, batch, sampled, rejected, recovery, lookahead=None, **kwargs):
    last_sampled = torch.zeros(4, 1, dtype=torch.int64)
    last_sampled[batch.idx_mapping, 0] = torch.tensor(recovery)
    next_prefill = torch.zeros(1, 4, dtype=torch.int32)
    if lookahead is not None:
        next_prefill[0, batch.idx_mapping] = torch.tensor(lookahead, dtype=torch.int32)
    return verifier.propose(
        batch,
        {},
        {},
        torch.zeros(batch.num_tokens, 4),
        None,
        torch.tensor(sampled),
        torch.tensor(rejected),
        last_sampled,
        next_prefill,
        kwargs.pop("temperature", torch.zeros(4)),
        torch.zeros(4, dtype=torch.int64),
        **kwargs,
    )


def test_chunks_rejections_and_request_reordering_preserve_verified_context(verifier):
    instance, executor, _, _ = verifier
    batch = inputs(["a", "b"], [[11, 12], [21, 22]], [[0, 1], [0, 1]], [4, 2], [3, 1])
    drafts = propose(instance, batch, [0, 1], [0, 0], [0, 30], [13, 0])
    b_drafts = digest_row([21, 22], 30)
    assert drafts.tolist() == [[-1] * K, b_drafts]
    assert executor.plans[0].next_prefill_tokens.tolist() == [[13]]

    # Finish a's prompt while b rejects its last two proposals. The final
    # prompt must be buffered then consumed once by a zero-query round.
    batch = inputs(
        ["b", "a"],
        [[30, *b_drafts], [13, 14]],
        [[2, 3, 4, 5], [2, 3]],
        [0, 4],
        [1, 3],
    )
    drafts = propose(instance, batch, [2, 1], [2, 0], [31, 40])
    b_prefix = [21, 22, 30, b_drafts[0]]
    assert drafts.tolist() == [
        digest_row(b_prefix, 31),
        digest_row([11, 12, 13, 14], 40),
    ]
    assert executor.plans[-1].num_computed_tokens.tolist() == [2, 2]
    assert executor.plans[-1].num_rejected.tolist() == [2, 0]

    batch = inputs(["b"], [[31, *drafts[0].tolist()]], [[4, 5, 6, 7]], [0], [1])
    assert propose(instance, batch, [1], [3], [32]).tolist() == [
        digest_row([*b_prefix, 31], 32)
    ]


def test_close_releases_kv_and_reused_request_id_starts_from_zero(verifier):
    instance, executor, _, _ = verifier
    batch = inputs(["request"], [[1, 2]], [[0, 1]], [2], [3])
    propose(instance, batch, [1], [0], [3])
    old_key = executor.plans[-1].rows[0].key
    instance.close_requests({"request", "already-gone"})
    batch = inputs(["request"], [[7]], [[0]], [1], [3])
    assert propose(instance, batch, [1], [0], [8]).tolist() == [digest_row([7], 8)]
    plan = executor.plans[-1]
    assert plan.rows[0].key != old_key
    assert plan.num_computed_tokens.tolist() == [0]
    assert list(plan.block_table_updates.values())[0].start == 0


@pytest.mark.parametrize("flag", ["dummy_run", "is_profile"])
def test_dummy_and_profile_do_not_contact_or_advance_server(
    verifier, monkeypatch, flag
):
    instance, executor, _, _ = verifier

    def reject_network(*args, **kwargs):
        pytest.fail("dummy/profile run contacted remote session")

    monkeypatch.setattr(instance._session, "open_sequence", reject_network)
    monkeypatch.setattr(instance._session, "prefill_many", reject_network)
    monkeypatch.setattr(instance._session, "dispatch", reject_network)
    batch = inputs(["a"], [[1]], [[0]], [1])
    assert propose(instance, batch, [1], [0], [2], **{flag: True}).tolist() == [
        [-1] * K
    ]
    assert executor.plans == []


def test_nonzero_tp_rank_receives_cpu_tokens_without_gpu_collective(verifier):
    instance, executor, settings, group = verifier
    group.rank_in_group = 1
    peer = remote.RemoteSpeculator(settings, torch.device("cpu"))
    assert peer._session is None
    group.rank_in_group = 0
    batch = inputs(["a"], [[1]], [[0]], [1])
    expected = propose(instance, batch, [1], [0], [2])
    assert group.payload == (None, expected.tolist())
    plan_count = len(executor.plans)
    group.rank_in_group = 1
    actual = propose(peer, batch, [1], [0], [2])
    assert torch.equal(actual, expected)
    assert len(executor.plans) == plan_count
    group.rank_in_group = 0


def test_error_is_shared_instead_of_tokens_with_tp_peer(verifier):
    instance, _, settings, group = verifier
    group.rank_in_group = 1
    peer = remote.RemoteSpeculator(settings, torch.device("cpu"))
    group.rank_in_group = 0
    batch = inputs(["a"], [[1]], [[0]], [1])
    with pytest.raises(RuntimeError, match="greedy target sampling"):
        propose(instance, batch, [1], [0], [2], temperature=torch.ones(4))
    group.rank_in_group = 1
    with pytest.raises(RuntimeError, match="greedy target sampling"):
        propose(peer, batch, [1], [0], [2])
    assert group.payload[1] is None
    group.rank_in_group = 0


def test_request_close_error_is_shared_with_tp_peer_without_session(
    verifier, monkeypatch
):
    instance, _, settings, group = verifier
    group.rank_in_group = 1
    peer = remote.RemoteSpeculator(settings, torch.device("cpu"))
    group.rank_in_group = 0
    batch = inputs(["a"], [[1]], [[0]], [1])
    propose(instance, batch, [1], [0], [2])

    def fail(keys):
        raise RuntimeError("close failed")

    monkeypatch.setattr(instance._session, "close_sequences", fail)
    with pytest.raises(RuntimeError, match="request cleanup failed.*close failed"):
        instance.close_requests({"a"})
    group.rank_in_group = 1
    with pytest.raises(RuntimeError, match="request cleanup failed.*close failed"):
        peer.close_requests({"a"})
    group.rank_in_group = 0


def test_shutdown_error_does_not_mask_an_engine_error(verifier, monkeypatch):
    instance, _, _, _ = verifier
    session = instance._session
    close = session.close

    def fail():
        close()
        raise OSError("socket close failed")

    monkeypatch.setattr(session, "close", fail)
    instance.close()
    instance.close()
    assert instance._session is None


@pytest.mark.parametrize("existing,computed", [(True, 0), (False, 4)])
def test_runner_rejects_streaming_and_missing_prefix_before_removing_state(
    existing, computed
):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(use_remote_draft=lambda: True),
        req_states=SimpleNamespace(req_id_to_index={"a": 0} if existing else {}),
    )
    scheduler = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="a", prefill_token_ids=[1, 2], num_computed_tokens=computed
            )
        ]
    )
    with pytest.raises(ValueError, match="streaming input updates"):
        GPUModelRunner.add_requests(runner, scheduler)


def test_invalid_verification_counts_fail_before_advancing_context(verifier):
    instance, executor, _, _ = verifier
    batch = inputs(["a"], [[1]], [[0]], [1])
    drafts = propose(instance, batch, [1], [0], [2]).tolist()[0]
    plan_count = len(executor.plans)
    batch = inputs(["a"], [[2, *drafts]], [[1, 2, 3, 4]], [0])
    with pytest.raises(RuntimeError, match="counts do not cover"):
        propose(instance, batch, [2], [0], [3])
    assert len(executor.plans) == plan_count


@pytest.mark.parametrize("verifier", ["target_only"], indirect=True)
def test_failed_remote_row_stays_target_only_without_reusing_stale_kv(
    verifier, monkeypatch
):
    instance, executor, _, _ = verifier
    calls = []

    def fail(plan, features):
        calls.append(plan)
        raise AdapterError(SpeculatorStatusCode.OUT_OF_MEMORY, "draft KV exhausted")

    monkeypatch.setattr(executor, "execute", fail)
    batch = inputs(["a"], [[1]], [[0]], [1])
    assert propose(instance, batch, [1], [0], [2]).tolist() == [[-1] * K]
    batch = inputs(["a"], [[2]], [[1]], [0])
    assert propose(instance, batch, [1], [0], [3]).tolist() == [[-1] * K]
    assert len(calls) == 1


@pytest.mark.parametrize("option", ["prefix", "async", "tp4", "pp2", "embeddings"])
def test_unsupported_placement_fails_before_connecting(option, monkeypatch):
    settings = config("tcp://127.0.0.1:1")
    if option == "prefix":
        settings.cache_config.enable_prefix_caching = True
    elif option == "async":
        settings.scheduler_config.async_scheduling = True
    elif option == "tp4":
        settings.parallel_config.tensor_parallel_size = 4
    elif option == "pp2":
        settings.parallel_config.pipeline_parallel_size = 2
    else:
        settings.model_config.enable_prompt_embeds = True
    with pytest.raises(ValueError, match="remote drafter"):
        remote.RemoteSpeculator(settings, torch.device("cpu"))
