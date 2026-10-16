# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the standalone host's wire, batch and partial-weight contracts."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.runtime.gpu import (
    GPUDraftStepExecutor,
    make_input_batch,
)
from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
from vllm.v1.spec_decode.remote.runtime.schema import create_runtime_contract
from vllm.v1.spec_decode.remote.runtime.step import DraftStepPlanner, DraftStepRow
from vllm.v1.spec_decode.remote.runtime.weights import selected_checkpoint_weights
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
