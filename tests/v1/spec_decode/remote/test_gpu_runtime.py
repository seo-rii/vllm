# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the standalone host's wire, batch and partial-weight contracts."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.runtime.adapter import TargetConditionedDraftAdapter
from vllm.v1.spec_decode.remote.runtime.gpu import (
    GPUDraftStepExecutor,
    make_input_batch,
)
from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
from vllm.v1.spec_decode.remote.runtime.schema import create_runtime_contract
from vllm.v1.spec_decode.remote.runtime.step import DraftStepPlanner, DraftStepRow
from vllm.v1.spec_decode.remote.runtime.weights import selected_checkpoint_weights
from vllm.v1.spec_decode.remote.server import RemoteDraftServer
from vllm.v1.worker.gpu.input_batch import InputBuffers


class _Config(SimpleNamespace):
    def to_dict(self):
        return vars(self).copy()


def test_sampling_buffers_match_mrv2_prefill_kernel_token_dtypes():
    executor = object.__new__(GPUDraftStepExecutor)
    executor.device = torch.device("cpu")
    executor._init_sampling_buffers(3)
    assert executor.last_sampled.dtype == torch.int64
    assert executor.lookahead.dtype == torch.int32
    assert executor.lookahead.shape == (1, 3)
    assert executor.temperature.dtype == torch.float32
    assert executor.seeds.dtype == torch.int64


class _Model(SimpleNamespace):
    def get_hidden_size(self):
        return self.hf_config.hidden_size

    def get_vocab_size(self):
        return self.hf_config.vocab_size


def _config(*, use_aux=False):
    target = _Model(
        model="target",
        revision="target-sha",
        dtype=torch.bfloat16,
        quantization="mxfp4",
        tokenizer="target",
        tokenizer_revision="target-sha",
        tokenizer_mode="auto",
        hf_config=_Config(
            model_type="gpt_oss",
            hidden_size=16,
            vocab_size=32,
            eos_token_id=2,
            architectures=["GptOssForCausalLM"],
        ),
    )
    draft = _Model(
        model="draft",
        revision="draft-sha",
        dtype=torch.bfloat16,
        quantization=None,
        hf_config=_Config(
            model_type="llama",
            hidden_size=16,
            vocab_size=32,
            architectures=["Eagle3LlamaForCausalLM"],
            eagle_config={"use_aux_hidden_state": use_aux},
        ),
    )
    return SimpleNamespace(
        model_config=target,
        speculative_config=SimpleNamespace(
            draft_model_config=draft,
            method="eagle3",
            num_speculative_tokens=3,
            parallel_drafting=False,
            draft_sample_method="greedy",
        ),
    )


@pytest.mark.parametrize(
    ("method", "aux", "max_model_len", "num_blocks", "block_size"),
    [
        ("eagle3", False, 32, 16, 4),
        ("eagle3", True, 32, 16, 4),
        ("mtp", False, 32, 16, 4),
        ("mtp", False, 1, 2, 2),
    ],
)
@pytest.mark.parametrize("failure", [None, "decode", "synchronize"])
def test_gpu_warmup_completes_before_listening_without_live_request_state(
    method, aux, max_model_len, num_blocks, block_size, failure, monkeypatch
):
    import vllm.v1.spec_decode.remote.runtime.gpu as gpu

    config = _config(use_aux=aux)
    config.model_config.max_model_len = max_model_len
    if method == "mtp":
        config.speculative_config.method = "mtp"
        config.speculative_config.num_speculative_tokens = 1
        config.speculative_config.draft_model_config.hf_config.architectures = [
            "MiMoMTP"
        ]
    num_drafts = config.speculative_config.num_speculative_tokens
    executor = object.__new__(GPUDraftStepExecutor)
    executor.vllm_config = config
    executor.contract = create_runtime_contract(config)
    executor.device = torch.device("cpu")
    executor.num_blocks = num_blocks
    executor.block_size = block_size
    executor.input_buffers = InputBuffers(2, 16, executor.device)
    executor._init_sampling_buffers(2)
    planner = DraftStepPlanner(
        DraftBlockPool(executor.num_blocks, executor.block_size),
        max_num_reqs=2,
        max_model_len=max_model_len,
        num_speculative_tokens=num_drafts,
    )
    adapter = TargetConditionedDraftAdapter(
        executor,
        planner,
        capabilities=executor.contract.capabilities,
        feature_schema=executor.contract.schema,
        methods={method},
    )
    server = RemoteDraftServer(adapter)
    calls = []
    warmup_planners = []

    def warmup_planner(*args, **kwargs):
        result = DraftStepPlanner(*args, **kwargs)
        warmup_planners.append(result)
        return result

    def execute(plan, features):
        assert server.endpoint is None
        assert planner.num_open == 0
        batch = make_input_batch(plan, features[0], executor.input_buffers)
        phase = "prefill" if batch.has_prefill else "decode"
        calls.append(phase)
        assert bool(plan.num_computed_tokens[0]) is (
            phase == "decode" and max_model_len > 1
        )
        for slot, tensor in zip(executor.contract.schema.slots, features):
            count = plan.num_reqs if slot.per_sequence else plan.num_tokens
            assert tensor.shape == (count, *slot.trailing_shape)
            assert tensor.dtype == getattr(torch, slot.dtype)
        assert torch.equal(features[1], torch.from_numpy(plan.positions))
        executor.last_sampled.fill_(17)
        executor.lookahead.fill_(19)
        if phase == failure:
            raise RuntimeError("warmup failed")
        return torch.zeros(plan.num_reqs, num_drafts, dtype=torch.int64)

    def synchronize(device):
        assert server.endpoint is None
        assert device == executor.device
        calls.append("synchronize")
        if failure == "synchronize":
            raise RuntimeError("warmup failed")

    monkeypatch.setattr(gpu, "DraftStepPlanner", warmup_planner)
    monkeypatch.setattr(executor, "execute", execute)
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)
    try:
        if failure:
            with pytest.raises(RuntimeError, match="warmup failed"):
                server.start("tcp://127.0.0.1:0")
            assert server.endpoint is None
        else:
            server.start("tcp://127.0.0.1:0")
            assert server.endpoint is not None
        assert calls == ["prefill", "decode", "synchronize"]
        assert len(warmup_planners) == 1
        assert warmup_planners[0].num_open == 0
        assert warmup_planners[0].block_pool.num_free_blocks == num_blocks - 1
        assert planner.num_open == 0
        assert planner.block_pool.num_free_blocks == num_blocks - 1
        assert not executor.last_sampled.any()
        assert not executor.lookahead.any()
    finally:
        server.stop()


@pytest.mark.parametrize("aux", [False, True])
def test_contract_transports_only_the_checkpoint_hidden_state_kind(aux):
    contract = create_runtime_contract(_config(use_aux=aux))
    hidden = contract.schema.slots[2]
    assert hidden.kind == ("aux_hidden_states" if aux else "last_hidden_states")
    assert hidden.trailing_shape == (48 if aux else 16,)
    assert hidden.dtype == "bfloat16"
    assert len(contract.schema.slots) == 4
    lookahead = contract.schema.slots[-1]
    assert lookahead.optional and lookahead.per_sequence
    assert lookahead.trailing_shape == (1,)


def test_contract_fingerprint_changes_with_weights_precision_and_feature_policy():
    config = _config()
    baseline = create_runtime_contract(config)
    config.speculative_config.draft_model_config.revision = "other-sha"
    changed_draft = create_runtime_contract(config)
    assert changed_draft.draft_fingerprint != baseline.draft_fingerprint
    assert changed_draft.target_fingerprint == baseline.target_fingerprint
    config.speculative_config.draft_model_config.revision = "draft-sha"
    config.speculative_config.num_speculative_tokens = 5
    assert (
        create_runtime_contract(config).draft_fingerprint != baseline.draft_fingerprint
    )
    config.model_config.dtype = torch.float16
    assert (
        create_runtime_contract(config).target_fingerprint
        != baseline.target_fingerprint
    )
    assert create_runtime_contract(_config(use_aux=True)).schema != baseline.schema


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", "draft_model"),
        ("parallel_drafting", True),
        ("draft_sample_method", "probabilistic"),
    ],
)
def test_contract_rejects_unimplemented_drafting_modes(field, value):
    config = _config()
    setattr(config.speculative_config, field, value)
    with pytest.raises(ValueError):
        create_runtime_contract(config)


def test_partial_weights_download_only_shards_containing_shared_tensors(
    tmp_path, monkeypatch
):
    import vllm.v1.spec_decode.remote.runtime.weights as weights

    embedding = torch.randn(8, 4)
    head = torch.randn(8, 4)
    save_file(
        {"model.embed_tokens.weight": embedding}, tmp_path / "embedding.safetensors"
    )
    save_file({"lm_head.weight": head}, tmp_path / "head.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "embedding.safetensors",
                    "lm_head.weight": "head.safetensors",
                    "model.layers.0.weight": "never-download.safetensors",
                }
            }
        )
    )
    fetched = []

    def download(repo, filename, *, revision):
        assert repo == "test/target" and revision == "immutable-sha"
        fetched.append(filename)
        return str(tmp_path / filename)

    monkeypatch.setattr(
        weights, "hf_api", lambda: SimpleNamespace(hf_hub_download=download)
    )
    actual = selected_checkpoint_weights(
        "test/target", "immutable-sha", ("model.embed_tokens.weight", "lm_head.weight")
    )
    assert set(fetched) == {
        "model.safetensors.index.json",
        "embedding.safetensors",
        "head.safetensors",
    }
    assert torch.equal(actual["model.embed_tokens.weight"], embedding)
    assert torch.equal(actual["lm_head.weight"], head)


def test_partial_weight_auth_errors_do_not_fall_back_to_full_checkpoint(monkeypatch):
    import vllm.v1.spec_decode.remote.runtime.weights as weights

    def denied(*args, **kwargs):
        raise PermissionError("access denied")

    monkeypatch.setattr(
        weights, "hf_api", lambda: SimpleNamespace(hf_hub_download=denied)
    )
    with pytest.raises(PermissionError, match="access denied"):
        selected_checkpoint_weights("test/target", "sha", ("lm_head.weight",))


def test_input_batch_preserves_reordered_request_slots_and_rejected_tail():
    planner = DraftStepPlanner(
        DraftBlockPool(32, 4),
        max_num_reqs=3,
        max_model_len=32,
        num_speculative_tokens=3,
    )
    a, b = SequenceKey("v", 1, 0), SequenceKey("v", 2, 0)
    planner.open(a)
    planner.open(b)
    prompt, failures = planner.plan([DraftStepRow(a, 5, 1), DraftStepRow(b, 3, 1)])
    assert not failures
    planner.commit(prompt)
    plan, failures = planner.plan(
        [
            DraftStepRow(b, 4, 2, num_rejected=2, last_sampled=15),
            DraftStepRow(a, 2, 2, last_sampled=19),
        ]
    )
    assert not failures
    batch = make_input_batch(
        plan, torch.tensor([1, 2, 3, 4, 5, 6]), InputBuffers(3, 16, torch.device("cpu"))
    )
    assert batch.idx_mapping.tolist() == [1, 0]
    assert batch.query_start_loc.tolist() == [0, 4, 6]
    assert batch.seq_lens.tolist() == [7, 7]
    assert batch.positions.tolist() == [3, 4, 5, 6, 5, 6]
    assert batch.num_computed_tokens_np.tolist() == [3, 5]
    assert batch.input_ids.tolist() == [1, 2, 3, 4, 5, 6]
    assert not batch.is_padding.any()
    planner.commit(plan)
    assert planner.num_computed(b) == 5
    assert planner.num_computed(a) == 7


def test_final_prompt_chunk_keeps_prefill_metadata_after_an_earlier_chunk():
    planner = DraftStepPlanner(
        DraftBlockPool(32, 4),
        max_num_reqs=1,
        max_model_len=32,
        num_speculative_tokens=3,
    )
    key = SequenceKey("v", 1, 0)
    planner.open(key)
    first, _ = planner.plan(
        [
            DraftStepRow(key, 4, 0, next_prefill_tokens=(5,)),
        ]
    )
    planner.commit(first)
    final, _ = planner.plan([DraftStepRow(key, 3, 1, last_sampled=8)])
    batch = make_input_batch(
        final,
        torch.tensor([5, 6, 7]),
        InputBuffers(1, 16, torch.device("cpu")),
    )
    assert batch.has_prefill
    assert batch.is_prefilling_np.tolist() == [True]
    assert batch.prefill_len_np.tolist() == [7]
    assert batch.positions.tolist() == [4, 5, 6]
