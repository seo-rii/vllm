# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU admission and shared-weight checks for standalone MTP model families."""

import copy
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, PreTrainedConfig, Qwen3Config

from tests.v1.spec_decode.remote.runtime_fixtures import (
    cpu_parallel_state as cpu_parallel_state,
)
from tests.v1.spec_decode.remote.runtime_fixtures import make_config
from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.config.speculative import RemoteDraftConfig, SpeculativeExecutionConfig
from vllm.v1.spec_decode.remote.runtime.schema import create_runtime_contract
from vllm.v1.spec_decode.remote.runtime.weights import materialize_target_shell

MODELS = (
    ("mimo", "MiMoForCausalLM", "MiMoMTPModel", 1),
    ("ernie4_5_moe", "Ernie4_5_MoeForCausalLM", "ErnieMTPModel", 1),
    ("qwen3_next", "Qwen3NextForCausalLM", "Qwen3NextMTP", 3),
    ("glm4_moe", "Glm4MoeForCausalLM", "Glm4MoeMTPModel", 3),
    ("glm4_moe_lite", "Glm4MoeLiteForCausalLM", "Glm4MoeLiteMTPModel", 3),
    ("exaone_moe", "ExaoneMoeForCausalLM", "ExaoneMoeMTP", 3),
    ("nemotron_h", "NemotronHForCausalLM", "NemotronHMTPModel", 3),
    ("hy_v3", "HYV3ForCausalLM", "HYV3MTPModel", 3),
)
NEMOTRON_EMBEDDING_NAMES = (
    "model.embed_tokens.weight",
    "model.embeddings.weight",
    "backbone.embed_tokens.weight",
    "backbone.embeddings.weight",
)

pytestmark = pytest.mark.usefixtures("cpu_parallel_state")


@cache
def _template(
    model_type,
    architecture,
    k,
    num_modules=1,
    module_layers=None,
    module_windows=None,
    num_target_layers=2,
    target_windows=None,
):
    kwargs = dict(
        architectures=[architecture],
        hidden_size=16,
        vocab_size=32,
        num_hidden_layers=num_target_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_nextn_predict_layers=num_modules,
        max_position_embeddings=64,
    )
    if model_type == "nemotron_h":
        kwargs["mtp_hybrid_override_pattern"] = "*E"
    if model_type == "exaone_moe":
        kwargs.update(
            layer_types=["full_attention"] * num_target_layers,
            mtp_layer_types=list(module_layers)
            if module_layers is not None
            else ["full_attention"] * num_modules,
            sliding_window=16,
            mtp_sliding_windows=module_windows,
            sliding_windows=target_windows,
        )
    hf_config = (
        PreTrainedConfig(model_type=model_type, **kwargs)
        if model_type == "mimo"
        else AutoConfig.for_model(model_type, **kwargs)
    )
    directory = TemporaryDirectory(prefix="vllm-mtp-contract-")
    root = Path(directory.name)
    Qwen3Config(**kwargs).save_pretrained(root)
    target = ModelConfig(
        model=str(root),
        dtype="bfloat16",
        max_model_len=64,
        skip_tokenizer_init=True,
        hf_overrides=lambda _: copy.deepcopy(hf_config),
    )
    config = make_config()
    config.model_config = target
    draft = ModelConfig(
        model=str(root),
        runner="draft",
        dtype="bfloat16",
        max_model_len=64,
        skip_tokenizer_init=True,
        hf_overrides=lambda _: SpeculativeConfig.hf_config_override(
            copy.deepcopy(hf_config)
        ),
    )
    config.speculative_config.method = "mtp"
    config.speculative_config.num_speculative_tokens = k
    config.speculative_config.draft_model_config = draft
    # Schema tests vary checkpoint identities independently from parsed metadata.
    target.model = target.tokenizer = "test/target"
    target.revision = target.tokenizer_revision = "a" * 40
    draft = config.speculative_config.draft_model_config
    draft.model, draft.revision = "test/target", "a" * 40
    return directory, config


def _config(
    model_type="ernie4_5_moe",
    architecture="Ernie4_5_MoeForCausalLM",
    k=1,
    num_modules=1,
    module_layers=None,
    module_windows=None,
    num_target_layers=2,
    target_windows=None,
):
    return copy.deepcopy(
        _template(
            model_type,
            architecture,
            k,
            num_modules,
            module_layers,
            module_windows,
            num_target_layers,
            target_windows,
        )[1]
    )


@pytest.mark.parametrize("model_type,target_arch,draft_arch,k", MODELS)
def test_native_mtp_config_override_resolves_own_kv_contract(
    model_type, target_arch, draft_arch, k
):
    config = _config(model_type, target_arch, k)
    assert config.speculative_config.draft_model_config.hf_config.architectures == [
        draft_arch
    ]
    contract = create_runtime_contract(config)
    assert contract.capabilities.state_dependency == "own_kv"
    assert contract.capabilities.num_prefill_lookahead == 1
    assert not contract.capabilities.supports_multi_module
    assert contract.schema.slots[1].trailing_shape == ()
    assert contract.schema.slots[2].kind == "last_hidden_states"
    assert contract.schema.slots[2].trailing_shape == (16,)


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_exaone_checkpoint_depth_determines_prefill_lookahead(depth):
    config = _config("exaone_moe", "ExaoneMoeForCausalLM", depth, num_modules=depth)
    contract = create_runtime_contract(config)
    assert contract.capabilities.num_prefill_lookahead == depth
    assert contract.capabilities.supports_multi_module
    lookahead = next(
        slot for slot in contract.schema.slots if slot.kind == "next_prefill_tokens"
    )
    assert lookahead.per_sequence and lookahead.trailing_shape == (depth,)
    assert contract.schema.slots[2].trailing_shape == (16,)
    config.speculative_config.remote_draft = RemoteDraftConfig(
        endpoint="unix:///tmp/unused-multi-module.sock"
    )
    assert config.speculative_config.use_remote_draft()
    assert config.speculative_config.use_multi_module_mtp()
    assert config.num_prefill_lookahead_tokens == depth


@pytest.mark.parametrize("transport", ["auto", "inline"])
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_exaone_module_chain_accepts_single_inline_endpoint(transport, depth):
    config = _config("exaone_moe", "ExaoneMoeForCausalLM", depth, num_modules=depth)
    config.speculative_config.execution = SpeculativeExecutionConfig(
        placement="remote",
        endpoints=("unix:///tmp/unused-multi-module.sock",),
        transport=transport,
    )
    contract = create_runtime_contract(config)
    assert contract.capabilities.supports_multi_module
    assert contract.capabilities.num_prefill_lookahead == depth


@pytest.mark.parametrize("depth", [2, 3])
def test_exaone_module_chain_accepts_single_ipc_endpoint(depth):
    config = _config("exaone_moe", "ExaoneMoeForCausalLM", depth, num_modules=depth)
    native = create_runtime_contract(config)
    config.speculative_config.execution = SpeculativeExecutionConfig(
        placement="remote",
        endpoints=("unix:///tmp/unused-multi-module.sock",),
        transport="cuda_ipc",
    )
    contract = create_runtime_contract(config)
    assert contract == native
    assert contract.capabilities.supports_multi_module
    assert contract.capabilities.num_prefill_lookahead == depth


@pytest.mark.parametrize("transport", ["inline", "cuda_ipc"])
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_exaone_module_chain_rejects_target_tensor_parallelism(transport, depth):
    config = _config("exaone_moe", "ExaoneMoeForCausalLM", depth, num_modules=depth)
    config.speculative_config.execution = SpeculativeExecutionConfig(
        placement="remote",
        endpoints=("unix:///tmp/unused-multi-module.sock",),
        transport=transport,
    )
    config.parallel_config = ParallelConfig(tensor_parallel_size=2)
    with pytest.raises(ValueError, match="target TP1"):
        create_runtime_contract(config)


@pytest.mark.parametrize("transport", ["inline", "cuda_ipc"])
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_exaone_module_chain_rejects_pooling(transport, depth):
    config = _config("exaone_moe", "ExaoneMoeForCausalLM", depth, num_modules=depth)
    config.speculative_config.execution = SpeculativeExecutionConfig(
        placement="remote",
        endpoints=(
            "unix:///tmp/unused-worker-a.sock",
            "unix:///tmp/unused-worker-b.sock",
        ),
        transport=transport,
    )
    with pytest.raises(ValueError, match="one (?:inline )?endpoint"):
        create_runtime_contract(config)


def test_exaone_module_count_changes_state_and_feature_contract_identity():
    single = create_runtime_contract(_config("exaone_moe", "ExaoneMoeForCausalLM", 2))
    multi = create_runtime_contract(
        _config("exaone_moe", "ExaoneMoeForCausalLM", 2, num_modules=2)
    )
    assert single.target_fingerprint != multi.target_fingerprint
    assert single.draft_fingerprint != multi.draft_fingerprint
    assert single.tokenizer_fingerprint == multi.tokenizer_fingerprint


@pytest.mark.parametrize("depth", [2, 3, 4])
@pytest.mark.parametrize("difference", [-1, 1])
def test_exaone_module_chain_requires_configured_depth(depth, difference):
    config = _config(
        "exaone_moe", "ExaoneMoeForCausalLM", depth + difference, num_modules=depth
    )
    with pytest.raises(ValueError, match="configured K"):
        create_runtime_contract(config)


@pytest.mark.parametrize(
    "module_layers,module_windows",
    [
        (("sliding_attention", "sliding_attention"), None),
        (("full_attention", "sliding_attention"), None),
        (("sliding_attention", "full_attention"), None),
        (("full_attention", "sliding_attention", "sliding_attention"), (0, 16, 32)),
        (("full_attention", "sliding_attention") * 2, (0, 16, 0, 32)),
    ],
)
def test_exaone_module_chain_accepts_dense_full_and_sliding_groups(
    module_layers, module_windows
):
    depth = len(module_layers)
    config = _config(
        "exaone_moe",
        "ExaoneMoeForCausalLM",
        depth,
        num_modules=depth,
        module_layers=module_layers,
        module_windows=module_windows,
    )
    contract = create_runtime_contract(config)
    assert contract.capabilities.num_prefill_lookahead == depth
    assert contract.capabilities.supports_multi_module


@pytest.mark.parametrize("module_windows", [None, ()])
@pytest.mark.parametrize(
    "module_layers,target_windows",
    [
        (
            ("full_attention", "sliding_attention", "sliding_attention"),
            (0, 16, 32, 64, 128),
        ),
        (("sliding_attention", "full_attention", "full_attention"), (16,)),
        (("full_attention",) * 3, ()),
    ],
)
def test_exaone_module_windows_inherit_only_used_target_indexes(
    module_windows, module_layers, target_windows
):
    config = _config(
        "exaone_moe",
        "ExaoneMoeForCausalLM",
        3,
        num_modules=3,
        module_layers=module_layers,
        module_windows=module_windows,
        num_target_layers=5,
        target_windows=target_windows,
    )
    assert config.model_config.hf_config.num_hidden_layers == 5
    draft = config.speculative_config.draft_model_config.hf_config
    resolved = draft.mtp_sliding_windows or draft.sliding_windows
    assert resolved == target_windows
    contract = create_runtime_contract(config)
    assert contract.capabilities.num_prefill_lookahead == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("mtp_layer_types", ["full_attention"]),
        (
            "mtp_layer_types",
            ["full_attention", "compressed_attention", "full_attention"],
        ),
        ("mtp_sliding_windows", [0, 16]),
        ("mtp_sliding_windows", [0, 0, 32]),
        ("mtp_sliding_windows", [0, -1, 32]),
        ("mtp_sliding_windows", [0, True, 32]),
    ],
)
def test_exaone_module_chain_rejects_incomplete_or_invalid_layout(field, value):
    config = _config(
        "exaone_moe",
        "ExaoneMoeForCausalLM",
        3,
        num_modules=3,
        module_layers=("full_attention", "sliding_attention", "sliding_attention"),
        module_windows=(0, 16, 32),
    )
    setattr(config.speculative_config.draft_model_config.hf_config, field, value)
    with pytest.raises(ValueError, match="MTP.*(layer|window)"):
        create_runtime_contract(config)


@pytest.mark.parametrize("model_type,target_arch,draft_arch,k", MODELS[:2])
def test_non_recurrent_mtp_models_keep_k_one(model_type, target_arch, draft_arch, k):
    config = _config(model_type, target_arch, 2)
    with pytest.raises(ValueError, match="exactly one speculative token"):
        create_runtime_contract(config)


@pytest.mark.parametrize("model_type,target_arch,draft_arch,k", MODELS)
def test_mtp_contract_rejects_missing_or_multiple_checkpoint_modules(
    model_type, target_arch, draft_arch, k
):
    for count in (0,) if model_type == "exaone_moe" else (0, 2):
        config = _config(model_type, target_arch, k, num_modules=count)
        if model_type == "nemotron_h" and count == 0:
            # The native override does not select an MTP architecture without
            # checkpoint MTP layers, so admission rejects that architecture.
            assert (
                config.speculative_config.draft_model_config.hf_config.architectures
                == [target_arch]
            )
            message = "unsupported standalone MTP architecture"
        else:
            message = "checkpoint MTP module"
        with pytest.raises(ValueError, match=message):
            create_runtime_contract(config)


@pytest.mark.parametrize("kind", ["architecture", "hidden", "vocab"])
def test_mtp_contract_does_not_accept_unrelated_target(kind):
    config = _config()
    target = config.model_config.hf_config
    if kind == "architecture":
        target.architectures = ["Qwen3ForCausalLM"]
    elif kind == "hidden":
        target.hidden_size *= 2
    else:
        target.vocab_size += 1
    config.model_config.model_arch_config = config.model_config.get_model_arch_config()
    with pytest.raises(ValueError, match="matching target"):
        create_runtime_contract(config)


@pytest.mark.parametrize(
    "architecture",
    ["MiMoMTP", "MiMoV2MTPModel", "Gemma4MTPModel", "Qwen3_5MTP", "DeepSeekV4MTPModel"],
)
def test_unsupported_feature_and_state_families_remain_fail_closed(architecture):
    config = _config()
    config.speculative_config.draft_model_config.hf_config.architectures = [
        architecture
    ]
    with pytest.raises(ValueError, match="unsupported standalone MTP architecture"):
        create_runtime_contract(config)


@pytest.mark.parametrize("field,value", [("hc_mult", 4), ("text_config", {})])
def test_target_feature_contract_rejects_nested_or_multistream_state(field, value):
    config = _config()
    setattr(config.model_config.hf_config, field, value)
    with pytest.raises(ValueError, match="scalar positions and text features"):
        create_runtime_contract(config)


def test_mrope_is_not_silently_encoded_as_scalar_positions():
    config = _config()
    config.model_config.hf_config.rope_scaling = {"mrope_section": [2, 2, 4]}
    with pytest.raises(ValueError, match="scalar positions and text features"):
        create_runtime_contract(config)


@pytest.mark.parametrize(
    "layer_types",
    [[], ["moe"], ["linear_attention", "full_attention", "moe"], ["mlp", "moe"]],
)
def test_nemotron_mtp_does_not_accept_target_mamba_state(layer_types):
    config = _config("nemotron_h", "NemotronHForCausalLM", 3)
    draft = config.speculative_config.draft_model_config.hf_config
    draft.mtp_layers_block_type = layer_types
    with pytest.raises(ValueError, match="attention/MoE"):
        create_runtime_contract(config)


def test_glm_dense_mla_contract_rejects_sparse_indexer_variant():
    config = _config("glm4_moe_lite", "Glm4MoeLiteForCausalLM", 3)
    config.speculative_config.draft_model_config.hf_config.index_topk = 2048
    with pytest.raises(ValueError, match="sparse indexer"):
        create_runtime_contract(config)


@pytest.mark.parametrize("embedding_name", NEMOTRON_EMBEDDING_NAMES)
def test_nemotron_shared_embedding_uses_checkpoint_mapper_and_pinned_revision(
    monkeypatch, tmp_path, embedding_name
):
    import vllm.v1.spec_decode.remote.runtime.weights as weights

    config = _config("nemotron_h", "NemotronHForCausalLM", 3).model_config
    config.hf_config._commit_hash = "c" * 40
    embedding = torch.arange(512, dtype=torch.bfloat16).reshape(32, 16)
    head = embedding + 16
    calls = []

    save_file(
        {embedding_name: embedding, "lm_head.weight": head},
        tmp_path / "model.safetensors",
    )

    class HubService:
        def hf_hub_download(self, source, filename, *, revision):
            calls.append((source, revision))
            return str(tmp_path / filename)

    monkeypatch.setattr(weights, "hf_api", HubService)
    shell = materialize_target_shell(config, torch.device("cpu"))
    assert calls == [("test/target", "a" * 40)] * 2
    torch.testing.assert_close(shell.model.embed_tokens.weight[:32], embedding)
    torch.testing.assert_close(shell.lm_head.weight[:32], head)
    assert list(dict(shell.named_parameters())) == [
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]


@pytest.mark.parametrize("embedding_name", NEMOTRON_EMBEDDING_NAMES)
def test_tied_nemotron_checkpoint_needs_no_separate_head(tmp_path, embedding_name):
    config = _config("nemotron_h", "NemotronHForCausalLM", 3).model_config
    config.model = str(tmp_path)
    config.hf_config.tie_word_embeddings = True
    embedding = torch.randn(32, 16, dtype=torch.bfloat16)
    save_file({embedding_name: embedding}, tmp_path / "model.safetensors")
    shell = materialize_target_shell(config, torch.device("cpu"))
    assert shell.model.embed_tokens.weight is shell.lm_head.weight
    torch.testing.assert_close(shell.model.embed_tokens.weight[:32], embedding)
    torch.testing.assert_close(shell.lm_head.weight[:32], embedding)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "tied,include_head", [(True, False), (True, True), (False, True)]
)
def test_shared_weight_storage_preserves_tying(tmp_path, device, tied, include_head):
    import vllm.model_executor.layers.vocab_parallel_embedding as embeddings
    import vllm.v1.spec_decode.remote.runtime.weights as weights

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    config = _config().model_config
    config.model = str(tmp_path)
    config.hf_config.tie_word_embeddings = tied
    embedding = torch.arange(512, dtype=config.dtype).reshape(32, 16)
    checkpoint = {"model.embed_tokens.weight": embedding}
    if include_head:
        checkpoint["lm_head.weight"] = embedding.clone() if tied else embedding + 16
    save_file(checkpoint, tmp_path / "model.safetensors")
    shell = materialize_target_shell(config, torch.device(device))
    embed = shell.model.embed_tokens.weight
    head = shell.lm_head.weight
    assert (embed is head) is tied
    torch.testing.assert_close(embed[:32].cpu(), embedding)
    torch.testing.assert_close(
        head[:32].cpu(), embedding if tied else checkpoint["lm_head.weight"]
    )
    storages = {
        p.untyped_storage().data_ptr(): p.untyped_storage().nbytes()
        for p in shell.parameters()
    }
    matrix_bytes = embeddings.pad_vocab_size(32) * 16 * embedding.element_size()
    assert sum(storages.values()) == matrix_bytes * (1 if tied else 2)
    shell.register_parameter(
        "draft_weight",
        torch.nn.Parameter(torch.empty(3, dtype=config.dtype, device=device)),
    )
    assert weights.model_weight_bytes(shell) == (
        3 * embedding.element_size(),
        matrix_bytes,
        0 if tied else matrix_bytes,
    )


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize(
    "embedding_names",
    [
        ("model.embed_tokens.weight", "backbone.embeddings.weight"),
        ("model.embed_tokens.weight", "model.embeddings.weight"),
        ("backbone.embeddings.weight", "backbone.embed_tokens.weight"),
    ],
)
def test_conflicting_nemotron_embedding_aliases_are_rejected(
    tmp_path, tied, embedding_names
):
    config = _config("nemotron_h", "NemotronHForCausalLM", 3).model_config
    config.model = str(tmp_path)
    config.hf_config.tie_word_embeddings = tied
    # Even identical values cannot disambiguate two checkpoint names that map
    # to the same native embedding parameter.
    checkpoint = {name: torch.zeros(32, 16) for name in embedding_names}
    if not tied:
        checkpoint["lm_head.weight"] = torch.ones(32, 16)
    save_file(checkpoint, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="exactly one embedding weight"):
        materialize_target_shell(config, torch.device("cpu"))


@pytest.mark.parametrize(
    "weight",
    [
        torch.zeros(32, 15),
        torch.zeros(32, 16, dtype=torch.int8),
        torch.zeros(32, 16, dtype=torch.float8_e4m3fn),
    ],
)
def test_shared_weights_cannot_silently_reinterpret_shape_or_quantization(
    tmp_path, weight
):
    config = _config().model_config
    config.model = str(tmp_path)
    config.hf_config.tie_word_embeddings = True
    save_file({"model.embed_tokens.weight": weight}, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="requires an unquantized"):
        materialize_target_shell(config, torch.device("cpu"))


@pytest.mark.parametrize(
    "tied,changed_name",
    [
        (False, "model.embed_tokens.weight"),
        (False, "lm_head.weight"),
        (True, "model.embed_tokens.weight"),
    ],
)
def test_alternate_shared_weights_must_match_the_target_before_loading(
    tmp_path, tied, changed_name
):
    config = _config().model_config
    target, alternate = tmp_path / "target", tmp_path / "shared"
    target.mkdir()
    alternate.mkdir()
    config.model = str(target)
    config.hf_config.tie_word_embeddings = tied
    tensors = {"model.embed_tokens.weight": torch.ones(32, 16)}
    if not tied:
        tensors["lm_head.weight"] = torch.zeros(32, 16)
    save_file(tensors, target / "model.safetensors")
    save_file(tensors, alternate / "model.safetensors")
    shell = materialize_target_shell(
        config, torch.device("cpu"), shared_weights_source=str(alternate)
    )
    assert torch.equal(
        shell.model.embed_tokens.weight[:32], tensors["model.embed_tokens.weight"]
    )
    tensors[changed_name][0, 0] = 2
    save_file(tensors, alternate / "model.safetensors")
    with pytest.raises(ValueError, match="differ from the target"):
        materialize_target_shell(
            config, torch.device("cpu"), shared_weights_source=str(alternate)
        )
    with pytest.raises(ValueError, match="unpinned alternate repositories"):
        materialize_target_shell(
            config, torch.device("cpu"), shared_weights_source="other/repo"
        )
