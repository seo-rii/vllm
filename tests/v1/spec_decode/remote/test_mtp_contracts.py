# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU admission and shared-weight checks for standalone MTP model families."""

import copy
import json
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from huggingface_hub.errors import StrictDataclassFieldValidationError
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    DeepseekV3Config,
    PreTrainedConfig,
    Qwen3Config,
    Step3p7TextConfig,
    TokenizersBackend,
)

from tests.v1.e2e.spec_decode.remote.test_remote_mtp import _separate_mtp_draft
from tests.v1.e2e.spec_decode.remote.test_remote_mtp import (
    tiny_ernie_mtp_model as tiny_ernie_mtp_model,
)
from tests.v1.e2e.spec_decode.remote.test_remote_mtp import (
    tiny_hyv3_mtp_model as tiny_hyv3_mtp_model,
)
from tests.v1.spec_decode.remote.runtime_fixtures import (
    cpu_parallel_state as cpu_parallel_state,
)
from tests.v1.spec_decode.remote.runtime_fixtures import make_config
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.config.speculative import RemoteDraftConfig, SpeculativeExecutionConfig
from vllm.v1.spec_decode.remote.runtime.schema import create_runtime_contract
from vllm.v1.spec_decode.remote.runtime.weights import (
    materialize_target_shell,
    model_weight_bytes,
)

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


def _mtp_context_config(
    source, destination, *, target_length, draft_length=16, draft_max_length=None, k=1
):
    from vllm.engine.arg_utils import EngineArgs

    draft_path = _separate_mtp_draft(source, destination)
    config_path = Path(draft_path) / "config.json"
    draft_config = json.loads(config_path.read_text())
    draft_config["max_position_embeddings"] = draft_length
    config_path.write_text(json.dumps(draft_config))
    speculative_config = {
        "method": "mtp",
        "model": draft_path,
        "num_speculative_tokens": k,
    }
    if draft_max_length is not None:
        speculative_config["max_model_len"] = draft_max_length
    return EngineArgs(
        model=source,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=target_length,
        enforce_eager=True,
        speculative_config=speculative_config,
    ).create_engine_config()


@pytest.mark.parametrize("family,k", [("ernie", 1), ("hyv3", 3)])
@pytest.mark.parametrize("draft_length,draft_max_length", [(16, None), (128, 16)])
def test_mtp_public_contract_rejects_short_draft_context(
    request, tmp_path, family, k, draft_length, draft_max_length
):
    """Reject a short native draft before prefill, decode or rollout can run."""
    source = request.getfixturevalue(f"tiny_{family}_mtp_model")
    config = _mtp_context_config(
        source,
        tmp_path / "short-draft",
        target_length=64,
        draft_length=draft_length,
        draft_max_length=draft_max_length,
        k=k,
    )
    assert config.speculative_config.draft_model_config.max_model_len == 16
    with pytest.raises(ValueError, match="standalone MTP.*draft.*max_model_len"):
        create_runtime_contract(config)


@pytest.mark.parametrize("prefill_length", [15, 16, 17])
def test_mtp_admitted_context_keeps_planned_positions_inside_native_rope(
    request, tmp_path, prefill_length
):
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.spec_decode.remote.protocol import SequenceKey
    from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
    from vllm.v1.spec_decode.remote.runtime.step import DraftStepPlanner, DraftStepRow

    config = _mtp_context_config(
        request.getfixturevalue("tiny_ernie_mtp_model"),
        tmp_path / "short-draft",
        target_length=16,
    )
    create_runtime_contract(config)
    draft = config.speculative_config.draft_model_config
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cpu"),
    ):
        model = get_model_loader(config.load_config).create_model(config, draft)
    rope = next(iter(model.model.layers.values())).mtp_block.self_attn.rotary_emb
    assert rope.cos_sin_cache.shape[0] == 16
    planner = DraftStepPlanner(
        DraftBlockPool(32, 2),
        max_num_reqs=1,
        max_model_len=config.model_config.max_model_len,
        num_speculative_tokens=1,
    )
    key = SequenceKey("context-boundary", 0, 1)
    assert planner.open(key)
    plan, failures = planner.plan([DraftStepRow(key, prefill_length, 1)])
    if prefill_length > 16:
        assert not plan.rows
        assert "max_model_len=16" in failures[0][1]
    else:
        assert not failures
        positions = torch.from_numpy(plan.positions)
        query = torch.zeros(prefill_length, draft.get_hidden_size())
        actual, _ = rope.forward_native(positions, query, query.clone())
        assert torch.equal(actual, query)
        assert positions.tolist() == list(range(prefill_length))


@pytest.mark.parametrize("mtp_family", ["hyv3", "ernie"])
def test_native_mtp_uses_separate_draft_checkpoint_settings(
    request, tmp_path, mtp_family
):
    """Admitted draft metadata must determine the loaded native head's math."""
    from vllm.config import set_current_vllm_config
    from vllm.distributed import destroy_model_parallel, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.utils.torch_utils import set_default_torch_dtype

    source = request.getfixturevalue(f"tiny_{mtp_family}_mtp_model")
    draft_path = _separate_mtp_draft(source, tmp_path / "separate-draft")
    config = EngineArgs(
        model=source,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=128,
        enforce_eager=True,
        speculative_config={
            "method": "mtp",
            "model": draft_path,
            "num_speculative_tokens": 1,
        },
    ).create_engine_config()
    target_config = config.model_config.hf_config.to_dict()
    assert config.speculative_config is not None
    draft = config.speculative_config.draft_model_config
    create_runtime_contract(config)
    loader = get_model_loader(config.load_config)
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cpu"),
    ):
        destroy_model_parallel()
        initialize_model_parallel(backend="gloo")
        model = loader.create_model(config, draft)
        loader.load_weights(model, draft)
        layer = next(iter(model.model.layers.values()))
        norm = layer.enorm if mtp_family == "hyv3" else layer.mtp_emb_norm
        inputs = torch.full((1, draft.get_hidden_size()), 0.01, dtype=torch.float32)
        expected = inputs * torch.rsqrt(
            inputs.square().mean(-1, keepdim=True) + draft.hf_config.rms_norm_eps
        )
        expected = (expected.to(norm.weight.dtype) * norm.weight).to(inputs.dtype)
        assert torch.equal(norm.forward_native(inputs), expected)
        assert norm.variance_epsilon == draft.hf_config.rms_norm_eps
        assert (
            layer.mtp_block.self_attn.rotary_emb.base
            == draft.hf_config.rope_parameters["rope_theta"]
        )
    assert config.model_config.hf_config.to_dict() == target_config


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


def _deepseek_public_config(
    tmp_path,
    *,
    k=3,
    routed=False,
    target_options=None,
    draft_options=None,
    wrong_target=False,
    dtype="bfloat16",
    target_tp=1,
    draft_tp=1,
    transport="inline",
    endpoints=None,
    cache_dtype="auto",
    spec_options=None,
    model_impl="auto",
):
    options = dict(
        architectures=["DeepseekV3ForCausalLM"],
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=64,
        kv_lora_rank=512,
        q_lora_rank=128,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_routed_experts=4 if routed else 0,
        n_shared_experts=1 if routed else 0,
        first_k_dense_replace=1 if routed else 3,
        num_experts_per_tok=2 if routed else 1,
        n_group=1,
        topk_group=1,
        num_nextn_predict_layers=1,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    )

    def checkpoint(path, hf):
        hf.save_pretrained(path)
        TokenizersBackend(
            tokenizer_object=Tokenizer(WordLevel({f"token{i}": i for i in range(32)})),
            bos_token="token0",
            eos_token="token1",
            pad_token="token2",
        ).save_pretrained(path)
        if getattr(hf, "quantization_config", None) is None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                model = AutoModelForCausalLM.from_config(hf, dtype=torch.bfloat16)
            embedding = model.get_input_embeddings().weight.detach()
            save_file(
                {"model.embed_tokens.weight": embedding},
                path / "model.safetensors",
            )

    target_path = draft_path = tmp_path
    if target_options or draft_options or wrong_target:
        target_path, draft_path = tmp_path / "target", tmp_path / "draft"
    target_hf = (
        Qwen3Config(
            architectures=["Qwen3ForCausalLM"],
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=32,
            max_position_embeddings=64,
        )
        if wrong_target
        else DeepseekV3Config(**(options | (target_options or {})))
    )
    checkpoint(target_path, target_hf)
    if draft_path != target_path:
        checkpoint(draft_path, DeepseekV3Config(**(options | (draft_options or {}))))
    target = ModelConfig(
        model=str(target_path),
        dtype=dtype,
        max_model_len=64,
        skip_tokenizer_init=True,
        model_impl=model_impl,
    )
    parallel = ParallelConfig(tensor_parallel_size=target_tp)
    placement = (
        dict(
            execution=SpeculativeExecutionConfig(
                placement="remote", endpoints=endpoints, transport=transport
            )
        )
        if endpoints
        else dict(
            remote_draft=RemoteDraftConfig(
                endpoint="unix:///tmp/deepseek-contract.sock", transport=transport
            )
        )
    )
    spec = SpeculativeConfig(
        model=str(draft_path),
        method="mtp",
        num_speculative_tokens=k,
        draft_tensor_parallel_size=draft_tp,
        target_model_config=target,
        target_parallel_config=parallel,
        **placement,
        **(spec_options or {}),
    )
    return VllmConfig(
        model_config=target,
        parallel_config=parallel,
        speculative_config=spec,
        cache_config=CacheConfig(enable_prefix_caching=False, cache_dtype=cache_dtype),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=64,
            max_model_len=64,
            async_scheduling=False,
            is_encoder_decoder=False,
        ),
        device_config=DeviceConfig("cpu"),
    )


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("routed", [False, True], ids=["dense", "moe"])
def test_deepseek_v3_public_override_resolves_single_native_mtp_head(
    tmp_path, k, routed
):
    config = _deepseek_public_config(tmp_path, k=k, routed=routed)
    target = config.model_config
    spec = config.speculative_config
    assert spec is not None and spec.method == "mtp"
    draft = spec.draft_model_config
    assert draft is not None
    assert target.hf_config.architectures == ["DeepseekV3ForCausalLM"]
    assert draft.hf_config.architectures == ["DeepSeekMTPModel"]
    assert draft.hf_config.n_predict == draft.hf_config.num_nextn_predict_layers == 1
    assert target.dtype == draft.dtype == torch.bfloat16
    assert target.quantization is None and draft.quantization is None
    assert all(not hasattr(model.hf_config, "index_topk") for model in (target, draft))
    assert all(
        model.hf_config.first_k_dense_replace == (1 if routed else 3)
        and model.hf_config.n_routed_experts == (4 if routed else 0)
        and model.hf_config.n_shared_experts == (1 if routed else 0)
        for model in (target, draft)
    )
    contract = create_runtime_contract(config)
    assert contract.capabilities.state_dependency == "own_kv"
    assert contract.capabilities.num_prefill_lookahead == 1
    assert not contract.capabilities.supports_multi_module
    assert contract.schema.slots[1].trailing_shape == ()
    assert contract.schema.slots[2].kind == "last_hidden_states"
    assert contract.schema.slots[2].trailing_shape == (16,)


@pytest.mark.parametrize("role", ["target", "draft"])
@pytest.mark.parametrize("index_topk", [0, 4])
def test_deepseek_public_config_rejects_any_indexer_attribute(
    tmp_path, role, index_topk
):
    config = _deepseek_public_config(
        tmp_path, **{f"{role}_options": {"index_topk": index_topk}}
    )
    with pytest.raises(ValueError, match="sparse indexer"):
        create_runtime_contract(config)


def test_deepseek_public_config_requires_one_checkpoint_head(tmp_path):
    config = _deepseek_public_config(
        tmp_path, k=2, draft_options={"num_nextn_predict_layers": 2}
    )
    with pytest.raises(ValueError, match="one checkpoint"):
        create_runtime_contract(config)


@pytest.mark.parametrize(
    "target_options", [{"num_hidden_layers": 3}, {"num_nextn_predict_layers": 2}]
)
def test_deepseek_mtp_rejects_different_native_layer_layout(tmp_path, target_options):
    """The native head's layer indices and count come from the target config."""
    config = _deepseek_public_config(tmp_path, target_options=target_options)
    with pytest.raises(ValueError, match="matching native configs"):
        create_runtime_contract(config)


@pytest.mark.parametrize("role", ["target", "draft"])
def test_deepseek_public_config_rejects_multi_channel_features(tmp_path, role):
    config = _deepseek_public_config(tmp_path, **{f"{role}_options": {"hc_mult": 2}})
    with pytest.raises(ValueError, match="scalar positions"):
        create_runtime_contract(config)


def test_deepseek_public_config_requires_its_actual_target_family(tmp_path):
    config = _deepseek_public_config(tmp_path, wrong_target=True)
    with pytest.raises(ValueError, match="matching target architecture"):
        create_runtime_contract(config)


def test_deepseek_public_config_rejects_float32_execution(tmp_path):
    config = _deepseek_public_config(tmp_path, dtype="float32")
    with pytest.raises(ValueError, match="unquantized bfloat16"):
        create_runtime_contract(config)


@pytest.mark.parametrize("role", ["target", "draft"])
def test_deepseek_public_config_preserves_rejected_fp8_metadata(tmp_path, role):
    quantization = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    with pytest.raises(ValueError, match="quant|fp8|bfloat16"):
        config = _deepseek_public_config(
            tmp_path, **{f"{role}_options": {"quantization_config": quantization}}
        )
        create_runtime_contract(config)
    saved = json.loads((tmp_path / role / "config.json").read_text())
    assert saved["quantization_config"] == quantization


@pytest.mark.parametrize("target_tp,draft_tp", [(2, 1), (1, 2), (2, 2)])
def test_deepseek_public_config_rejects_tensor_parallel_drafting(
    tmp_path, target_tp, draft_tp
):
    with pytest.raises(ValueError, match="TP1|draft_tensor_parallel_size"):
        config = _deepseek_public_config(
            tmp_path, target_tp=target_tp, draft_tp=draft_tp
        )
        create_runtime_contract(config)


def test_deepseek_public_config_rejects_ipc(tmp_path):
    config = _deepseek_public_config(tmp_path, transport="cuda_ipc")
    with pytest.raises(ValueError, match="one inline endpoint"):
        create_runtime_contract(config)


def test_deepseek_public_config_rejects_multiple_endpoints(tmp_path):
    config = _deepseek_public_config(
        tmp_path,
        endpoints=("unix:///tmp/deepseek-a.sock", "unix:///tmp/deepseek-b.sock"),
    )
    with pytest.raises(ValueError, match="one inline endpoint"):
        create_runtime_contract(config)


def _step35_public_config(
    tmp_path,
    *,
    k=3,
    options=None,
    target_tp=1,
    draft_tp=1,
    transport="inline",
    dtype="bfloat16",
):
    from transformers.models.step3p7.modeling_step3p7 import Step3p7TextModel

    hf = Step3p7TextConfig(
        **(
            dict(
                architectures=["Step3p5ForCausalLM"],
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                vocab_size=32,
                max_position_embeddings=64,
                num_nextn_predict_layers=1,
                layer_types=["full_attention"] * 3,
                mlp_layer_types=["dense", "sparse", "dense"],
                n_routed_experts=4,
                num_experts_per_tok=2,
                moe_intermediate_size=64,
                share_expert_dim=128,
                rope_theta=10000.0,
                zero_centered=True,
                use_qk_norm=True,
                use_head_wise_attn_gate=True,
                per_layer_config=None,
                bos_token_id=0,
                eos_token_id=1,
                pad_token_id=2,
            )
            | (options or {})
        )
    )
    hf.save_pretrained(tmp_path)
    if not hf.is_heterogeneous:
        # HF serialization omits None; preserve the checkpoint's public choice.
        path = tmp_path / "config.json"
        saved = json.loads(path.read_text())
        saved["per_layer_config"] = None
        path.write_text(json.dumps(saved))
    TokenizersBackend(
        tokenizer_object=Tokenizer(WordLevel({f"token{i}": i for i in range(32)})),
        bos_token="token0",
        eos_token="token1",
        pad_token="token2",
    ).save_pretrained(tmp_path)
    if getattr(hf, "quantization_config", None) is None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = Step3p7TextModel(hf).to(torch.bfloat16)
            head = torch.nn.Linear(128, 32, bias=False).to(torch.bfloat16)
        save_file(
            {
                "model.embed_tokens.weight": model.embed_tokens.weight.detach(),
                "lm_head.weight": head.weight.detach(),
            },
            tmp_path / "model.safetensors",
        )
    target = ModelConfig(
        model=str(tmp_path),
        dtype=dtype,
        max_model_len=64,
        skip_tokenizer_init=True,
    )
    parallel = ParallelConfig(tensor_parallel_size=target_tp)
    spec = SpeculativeConfig(
        model=str(tmp_path),
        method="mtp",
        num_speculative_tokens=k,
        draft_tensor_parallel_size=draft_tp,
        target_model_config=target,
        target_parallel_config=parallel,
        remote_draft=RemoteDraftConfig(
            endpoint="unix:///tmp/step35-contract.sock", transport=transport
        ),
    )
    return VllmConfig(
        model_config=target,
        parallel_config=parallel,
        speculative_config=spec,
        cache_config=CacheConfig(enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=64,
            max_model_len=64,
            async_scheduling=False,
            is_encoder_decoder=False,
        ),
        device_config=DeviceConfig("cpu"),
    )


@pytest.mark.parametrize("k", [1, 3])
def test_step35_public_override_preserves_scalar_owned_head_contract(tmp_path, k):
    config = _step35_public_config(tmp_path, k=k)
    target = config.model_config
    spec = config.speculative_config
    assert spec is not None and spec.draft_model_config is not None
    draft = spec.draft_model_config
    assert isinstance(target.hf_config, Step3p7TextConfig)
    assert isinstance(draft.hf_config, Step3p7TextConfig)
    assert target.hf_config.architectures == ["Step3p5ForCausalLM"]
    assert draft.hf_config.architectures == ["Step3p5MTP"]
    assert draft.hf_config.layer_types == ["full_attention"] * 2
    assert draft.hf_config.mtp_layer_types == ["full_attention"]
    assert draft.hf_config.mtp_mlp_layer_types == ["dense"]
    assert draft.hf_config.n_predict == draft.hf_config.num_nextn_predict_layers == 1
    assert target.dtype == draft.dtype == torch.bfloat16
    assert target.quantization is None and draft.quantization is None
    contract = create_runtime_contract(config)
    assert contract.capabilities.state_dependency == "own_kv"
    assert contract.capabilities.num_prefill_lookahead == 1
    assert not contract.capabilities.supports_multi_module
    assert contract.schema.slots[1].trailing_shape == ()
    assert contract.schema.slots[2].kind == "last_hidden_states"
    assert contract.schema.slots[2].trailing_shape == (128,)
    assert not contract.requires_target_lm_head


def test_step35_public_contract_rejects_full_attention_head_chains(tmp_path):
    config = _step35_public_config(
        tmp_path,
        options={
            "num_nextn_predict_layers": 3,
            "layer_types": ["full_attention"] * 5,
            "mlp_layer_types": ["dense", "sparse", "dense", "dense", "dense"],
        },
    )
    with pytest.raises(ValueError, match="multiple checkpoint heads"):
        create_runtime_contract(config)


@pytest.mark.parametrize(
    "options,detail",
    [
        (
            {"layer_types": ["full_attention"] * 2, "mtp_layer_types": []},
            "homogeneous full/SWA dense MTP layers",
        ),
        ({"per_layer_config": {}}, "explicit MTP head geometry"),
        (
            {
                "num_nextn_predict_layers": 2,
                "layer_types": ["full_attention"] * 3 + ["sliding_attention"],
                "mlp_layer_types": ["dense", "sparse", "dense", "dense"],
            },
            "homogeneous full/SWA dense MTP layers",
        ),
    ],
)
def test_step35_public_contract_requires_explicit_mtp_tail_geometry(
    tmp_path, options, detail
):
    config = _step35_public_config(
        tmp_path, k=options.get("num_nextn_predict_layers", 3), options=options
    )
    with pytest.raises(ValueError, match=detail):
        create_runtime_contract(config)


@pytest.mark.parametrize("k", [1, 3])
def test_step35_public_three_swa_heads_use_only_one_configured_step(tmp_path, k):
    config = _step35_public_config(
        tmp_path,
        k=k,
        options={
            "num_nextn_predict_layers": 3,
            "layer_types": ["full_attention"] + ["sliding_attention"] * 4,
            "mlp_layer_types": ["dense", "sparse", "dense", "dense", "dense"],
            "sliding_window": 16,
            "num_sliding_attention_heads": 4,
            "per_layer_config": {},
        },
    )
    assert config.speculative_config.draft_model_config.hf_config.n_predict == 3
    if k == 3:
        assert config.speculative_config.use_multi_module_mtp()
        assert config.num_prefill_lookahead_tokens == 3
        with pytest.raises(ValueError, match="multiple checkpoint heads"):
            create_runtime_contract(config)
    else:
        assert not config.speculative_config.use_multi_module_mtp()
        assert config.num_prefill_lookahead_tokens == 1
        contract = create_runtime_contract(config)
        assert not contract.requires_target_lm_head
        assert not contract.capabilities.supports_multi_module
        assert contract.capabilities.num_prefill_lookahead == 1


@pytest.mark.parametrize(
    "options,detail",
    [
        ({"target_tp": 2}, "TP1"),
        ({"draft_tp": 2}, "TP1|draft_tensor_parallel_size"),
        ({"transport": "cuda_ipc"}, "one inline endpoint"),
        ({"dtype": "float32"}, "unquantized bfloat16"),
    ],
)
def test_step35_public_contract_keeps_bounded_bf16_placement(tmp_path, options, detail):
    with pytest.raises(ValueError, match=detail):
        config = _step35_public_config(tmp_path, **options)
        create_runtime_contract(config)


def test_step35_public_contract_requires_rope_for_the_actual_mtp_type(tmp_path):
    config = _step35_public_config(
        tmp_path,
        options={
            "layer_types": ["sliding_attention"] * 2 + ["full_attention"],
            "sliding_window": 16,
            "num_sliding_attention_heads": 4,
        },
    )
    assert "full_attention" not in config.model_config.hf_config.rope_parameters
    with pytest.raises(ValueError, match="requires MTP attention RoPE parameters"):
        create_runtime_contract(config)


def test_step35_public_config_preserves_rejected_fp8_metadata(tmp_path):
    quantization = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    with pytest.raises(ValueError, match="quant|fp8|bfloat16"):
        config = _step35_public_config(
            tmp_path, options={"quantization_config": quantization}
        )
        create_runtime_contract(config)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["quantization_config"] == quantization


def test_owned_head_materialization_requires_only_the_consumed_embedding(tmp_path):
    config = _step35_public_config(tmp_path)
    path = tmp_path / "model.safetensors"
    embedding = load_file(path)["model.embed_tokens.weight"]
    save_file({"model.embed_tokens.weight": embedding}, path)
    shell = materialize_target_shell(
        config.model_config, torch.device("cpu"), include_lm_head=False
    )
    assert not hasattr(shell, "lm_head")
    assert list(dict(shell.named_parameters())) == ["model.embed_tokens.weight"]
    assert torch.equal(shell.model.embed_tokens.weight[:32], embedding)
    with pytest.raises(ValueError, match="no lm_head.weight"):
        materialize_target_shell(config.model_config, torch.device("cpu"))


def test_owned_head_alternate_weights_still_verify_the_consumed_embedding(tmp_path):
    config = _step35_public_config(tmp_path / "target")
    tensors = load_file(tmp_path / "target/model.safetensors")
    alternate = tmp_path / "shared"
    alternate.mkdir()
    tensors["lm_head.weight"].add_(1)
    save_file(tensors, alternate / "model.safetensors")
    shell = materialize_target_shell(
        config.model_config,
        torch.device("cpu"),
        shared_weights_source=str(alternate),
        include_lm_head=False,
    )
    assert not hasattr(shell, "lm_head")
    assert torch.equal(
        shell.model.embed_tokens.weight[:32], tensors["model.embed_tokens.weight"]
    )
    tensors["model.embed_tokens.weight"][0, 0] += 1
    save_file(tensors, alternate / "model.safetensors")
    with pytest.raises(ValueError, match="differ from the target"):
        materialize_target_shell(
            config.model_config,
            torch.device("cpu"),
            shared_weights_source=str(alternate),
            include_lm_head=False,
        )


def test_owned_head_bytes_count_distinct_nested_heads_once(tmp_path):
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    config = _step35_public_config(tmp_path)
    shell = materialize_target_shell(config.model_config, torch.device("cpu"))
    owned_head = ParallelLMHead(32, 128, params_dtype=torch.bfloat16)
    shell.model.add_module(
        "layers",
        torch.nn.ModuleDict(
            {
                "2": torch.nn.ModuleDict({"shared_head": owned_head}),
                "3": torch.nn.ModuleDict({"shared_head": shell.lm_head}),
                "4": torch.nn.ModuleDict({"shared_head": owned_head}),
            }
        ),
    )
    shell.register_parameter(
        "draft_weight", torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
    )
    embedding_bytes = shell.model.embed_tokens.weight.untyped_storage().nbytes()
    head_bytes = (
        shell.lm_head.weight.untyped_storage().nbytes()
        + owned_head.weight.untyped_storage().nbytes()
    )
    assert model_weight_bytes(shell) == (6, embedding_bytes, head_bytes)


@pytest.mark.parametrize("profile", ["one_full", "three_swa"])
@pytest.mark.parametrize(
    "layout,omitted",
    [("split", None), ("split", "q"), ("split", "k"), ("split", "v"), ("fused", None)],
)
def test_step35_native_load_requires_each_packed_attention_source(
    tmp_path, profile, layout, omitted
):
    """A loaded sibling projection cannot hide an uninitialized Q/K/V slice."""
    from tests.v1.e2e.spec_decode.remote.test_remote_mtp import _save_step3p5_mtp_model
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.utils.torch_utils import set_default_torch_dtype

    source, _ = _save_step3p5_mtp_model(tmp_path, profile)
    path = Path(source) / "model.safetensors"
    weights = load_file(path)
    expected = {}
    heads = 3 if profile == "three_swa" else 1
    for index in range(2, 2 + heads):
        prefix = f"model.layers.{index}.self_attn."
        parts = [prefix + part + "_proj.weight" for part in ("q", "k", "v")]
        expected[index] = torch.cat([weights[name] for name in parts])
        if layout == "fused":
            weights[prefix + "qkv_proj.weight"] = expected[index]
            for name in parts:
                del weights[name]
    if omitted is not None:
        del weights[f"model.layers.{heads + 1}.self_attn.{omitted}_proj.weight"]
    save_file(weights, path, metadata={"format": "pt"})
    config = EngineArgs(
        model=str(source),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=256 if heads == 3 else 128,
        enforce_eager=True,
        model_loader_extra_config={"enable_weights_track": True},
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
    ).create_engine_config()
    assert config.speculative_config is not None
    draft = config.speculative_config.draft_model_config
    loader = get_model_loader(config.load_config)
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cpu"),
    ):
        model = loader.create_model(config, draft)
        if omitted is not None:
            with pytest.raises(ValueError, match=rf"{omitted}_proj\.weight"):
                loader.load_weights(model, draft)
        else:
            loader.load_weights(model, draft)
            assert torch.equal(
                model.model.embed_tokens.weight, weights["model.embed_tokens.weight"]
            )
            for index, qkv in expected.items():
                layer = model.model.layers[str(index)]
                assert torch.equal(layer.mtp_block.self_attn.qkv_proj.weight, qkv)
                assert torch.equal(
                    layer.shared_head.head.weight,
                    weights[
                        f"model.layers.{index}.transformer.shared_head.output.weight"
                    ],
                )


@pytest.mark.parametrize("draft_dtype", [None, "auto", "fp8"])
def test_deepseek_public_config_rejects_fp8_target_or_draft_kv(tmp_path, draft_dtype):
    options = {} if draft_dtype is None else {"kv_cache_dtype": draft_dtype}
    config = _deepseek_public_config(
        tmp_path,
        cache_dtype="auto" if draft_dtype == "fp8" else "fp8",
        spec_options=options,
    )
    with pytest.raises(ValueError, match="bfloat16 KV"):
        create_runtime_contract(config)


def _mimo_flash_public_config(
    tmp_path,
    *,
    k=3,
    target_options=None,
    draft_options=None,
    wrong_target=False,
    identity=True,
    transport="inline",
):
    options = dict(
        architectures=["MiMoV2FlashForCausalLM"],
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=192,
        v_head_dim=128,
        swa_num_attention_heads=2,
        swa_num_key_value_heads=2,
        swa_head_dim=192,
        swa_v_head_dim=128,
        hybrid_layer_pattern=[0, 1],
        sliding_window=128,
        sliding_window_size=128,
        attention_chunk_size=128,
        attention_bias=False,
        add_swa_attention_sink_bias=True,
        add_full_attention_sink_bias=False,
        attention_value_scale=0.707,
        partial_rotary_factor=0.334,
        rope_theta=5000000,
        swa_rope_theta=10000,
        layernorm_epsilon=1e-5,
        max_position_embeddings=256,
        hidden_act="silu",
        moe_layer_freq=[0, 1],
        moe_intermediate_size=128,
        n_routed_experts=4,
        n_shared_experts=None,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        n_group=1,
        topk_group=1,
        topk_method="noaux_tc",
        routed_scaling_factor=None,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )

    def checkpoint(path, hf):
        hf.save_pretrained(path)
        TokenizersBackend(
            tokenizer_object=Tokenizer(WordLevel({f"token{i}": i for i in range(128)})),
            bos_token="token1",
            eos_token="token2",
            pad_token="token0",
        ).save_pretrained(path)
        if identity:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                model = AutoModelForCausalLM.from_config(hf, dtype=torch.bfloat16)
            embedding = model.get_input_embeddings().weight.detach()
            save_file(
                {"model.embed_tokens.weight": embedding}, path / "model.safetensors"
            )

    target_path = draft_path = tmp_path
    if target_options or draft_options or wrong_target:
        target_path, draft_path = tmp_path / "target", tmp_path / "draft"
    target_hf = (
        Qwen3Config(
            architectures=["Qwen3ForCausalLM"],
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            vocab_size=128,
            max_position_embeddings=256,
        )
        if wrong_target
        else AutoConfig.for_model("mimo_v2_flash", **(options | (target_options or {})))
    )
    checkpoint(target_path, target_hf)
    if draft_path != target_path:
        checkpoint(
            draft_path,
            AutoConfig.for_model("mimo_v2_flash", **(options | (draft_options or {}))),
        )
    target = ModelConfig(
        model=str(target_path),
        dtype="bfloat16",
        max_model_len=128,
        skip_tokenizer_init=True,
    )
    parallel = ParallelConfig()
    spec = SpeculativeConfig(
        model=str(draft_path),
        method="mtp",
        num_speculative_tokens=k,
        draft_tensor_parallel_size=1,
        target_model_config=target,
        target_parallel_config=parallel,
        remote_draft=RemoteDraftConfig(
            endpoint="unix:///tmp/mimo-flash-contract.sock", transport=transport
        ),
    )
    return VllmConfig(
        model_config=target,
        parallel_config=parallel,
        speculative_config=spec,
        cache_config=CacheConfig(enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=128,
            max_model_len=128,
            async_scheduling=False,
            is_encoder_decoder=False,
        ),
        device_config=DeviceConfig("cpu"),
    )


@pytest.mark.parametrize("k", [1, 3])
def test_mimo_flash_public_override_resolves_native_diffkv_mtp(tmp_path, k):
    config = _mimo_flash_public_config(tmp_path, k=k)
    target = config.model_config.hf_config
    spec = config.speculative_config
    assert spec is not None and spec.method == "mtp"
    draft = spec.draft_model_config
    assert draft is not None
    assert type(target).__module__.startswith("transformers.models.mimo_v2_flash")
    assert not getattr(target, "auto_map", None)
    assert target.architectures == ["MiMoV2FlashForCausalLM"]
    assert draft.hf_config.architectures == ["MiMoV2MTPModel"]
    assert draft.hf_config.n_predict == draft.hf_config.num_nextn_predict_layers == 1
    assert target.hybrid_layer_pattern == [0, 1]
    assert (target.num_key_value_heads, target.swa_num_key_value_heads) == (1, 2)
    assert (target.head_dim, target.v_head_dim) == (192, 128)
    assert (target.swa_head_dim, target.swa_v_head_dim) == (192, 128)
    assert target.add_swa_attention_sink_bias and target.n_routed_experts == 4
    contract = create_runtime_contract(config)
    assert contract.capabilities.state_dependency == "own_kv"
    assert contract.capabilities.num_prefill_lookahead == 1
    assert not contract.capabilities.supports_multi_module
    assert contract.schema.slots[2].kind == "last_hidden_states"
    assert contract.schema.slots[2].trailing_shape == (64,)


@pytest.mark.parametrize(
    "role,field,width",
    [
        ("target", "head_dim", 128),
        ("draft", "v_head_dim", 64),
        ("draft", "swa_head_dim", 128),
        ("target", "swa_v_head_dim", 64),
    ],
)
def test_mimo_flash_public_config_rejects_incompatible_diffkv_widths(
    tmp_path, role, field, width
):
    config = _mimo_flash_public_config(
        tmp_path, identity=False, **{f"{role}_options": {field: width}}
    )
    with pytest.raises(ValueError, match="QK192/V128 DiffKV"):
        create_runtime_contract(config)


@pytest.mark.parametrize(
    "role,heads",
    [
        ("target", {"num_attention_heads": 0}),
        ("draft", {"swa_num_attention_heads": 3, "swa_num_key_value_heads": 2}),
    ],
)
def test_mimo_flash_public_config_requires_valid_global_and_swa_heads(
    tmp_path, role, heads
):
    config = _mimo_flash_public_config(
        tmp_path, identity=False, **{f"{role}_options": heads}
    )
    with pytest.raises(ValueError, match="divisible Q/KV heads"):
        create_runtime_contract(config)


@pytest.mark.parametrize("window", [-1, 0])
def test_mimo_flash_public_config_rejects_nonpositive_swa_window(tmp_path, window):
    with pytest.raises(
        (ValueError, StrictDataclassFieldValidationError),
        match="positive SWA window|sliding_window",
    ):
        config = _mimo_flash_public_config(
            tmp_path,
            identity=False,
            draft_options={"sliding_window": window, "sliding_window_size": window},
        )
        create_runtime_contract(config)


def test_mimo_flash_runtime_contract_rejects_nonpositive_swa_window(tmp_path):
    """A zero-sized SWA history survives HF's query-inclusive window parsing."""
    config = _mimo_flash_public_config(
        tmp_path,
        identity=False,
        draft_options={"sliding_window": 1, "sliding_window_size": 0},
    )
    assert (
        config.speculative_config.draft_model_config.hf_config.sliding_window_size == 0
    )
    with pytest.raises(ValueError, match="positive SWA window"):
        create_runtime_contract(config)


@pytest.mark.parametrize(
    "draft_options",
    [
        {"sliding_window": 64, "sliding_window_size": 64},
        {"swa_rope_theta": 20000},
        {"swa_num_attention_heads": 4},
        {"layernorm_epsilon": 1e-4},
    ],
)
def test_mimo_flash_rejects_draft_settings_ignored_by_native_constructor(
    tmp_path, draft_options
):
    """Equal hidden widths cannot validate a head built from target metadata."""
    config = _mimo_flash_public_config(tmp_path, draft_options=draft_options)
    with pytest.raises(ValueError, match="matching native configs"):
        create_runtime_contract(config)


@pytest.mark.parametrize("family", ["deepseek", "mimo_flash"])
def test_native_mtp_accepts_independent_checkpoint_metadata(tmp_path, family):
    factory = (
        _deepseek_public_config if family == "deepseek" else _mimo_flash_public_config
    )
    options = {"checkpoint_notes": "separately published MTP head"}
    if family == "deepseek":
        # DeepSeek constructs the block's norms from the draft config itself.
        options["rms_norm_eps"] = 1e-4
    config = factory(tmp_path, draft_options=options)
    assert (
        config.model_config.model != config.speculative_config.draft_model_config.model
    )
    contract = create_runtime_contract(config)
    assert contract.capabilities.required_features == (
        "token_ids",
        "positions",
        "last_hidden_states",
    )


@pytest.mark.parametrize("pattern", [[0], [0, 2]])
def test_mimo_flash_public_config_requires_a_complete_layer_pattern(tmp_path, pattern):
    with pytest.raises(ValueError, match="layer pattern|layer_types|length"):
        config = _mimo_flash_public_config(
            tmp_path, identity=False, target_options={"hybrid_layer_pattern": pattern}
        )
        create_runtime_contract(config)


def test_mimo_flash_public_config_requires_its_actual_target_family(tmp_path):
    config = _mimo_flash_public_config(tmp_path, identity=False, wrong_target=True)
    with pytest.raises(ValueError, match="matching target architecture"):
        create_runtime_contract(config)


def test_mimo_flash_public_override_preserves_rejected_fp8_metadata(tmp_path):
    quantization = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    with pytest.raises(ValueError, match="quant|fp8|bfloat16"):
        config = _mimo_flash_public_config(
            tmp_path,
            identity=False,
            draft_options={"quantization_config": quantization},
        )
        create_runtime_contract(config)
    saved = json.loads((tmp_path / "draft/config.json").read_text())
    assert saved["quantization_config"] == quantization


def test_mimo_flash_public_config_rejects_ipc_placement(tmp_path):
    config = _mimo_flash_public_config(tmp_path, identity=False, transport="cuda_ipc")
    with pytest.raises(ValueError, match="one inline endpoint"):
        create_runtime_contract(config)


@pytest.mark.parametrize("draft_limit", [16, 64])
def test_mtp_public_config_requires_draft_context_to_cover_target(draft_limit):
    """A public draft config must cover every context admitted by the target."""
    directory, _ = _template("ernie4_5_moe", "Ernie4_5_MoeForCausalLM", 1)
    config = _config()
    save_file(
        {"identity_only": torch.zeros(1)}, Path(directory.name) / "model.safetensors"
    )
    hf_config = copy.deepcopy(config.model_config.hf_config)
    hf_config.max_position_embeddings = draft_limit
    draft = ModelConfig(
        model=directory.name,
        runner="draft",
        dtype="bfloat16",
        max_model_len=draft_limit,
        skip_tokenizer_init=True,
        hf_overrides=lambda _: SpeculativeConfig.hf_config_override(hf_config),
    )
    assert draft.max_model_len == draft_limit
    config.speculative_config.draft_model_config = draft
    if draft_limit < config.model_config.max_model_len:
        with pytest.raises(ValueError, match="standalone MTP.*draft.*max_model_len"):
            create_runtime_contract(config)
    else:
        assert create_runtime_contract(config).capabilities.state_dependency == "own_kv"


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
    assert contract.requires_target_lm_head


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


@pytest.mark.parametrize(
    "target_windows,draft_windows",
    [((32,), (16,)), ((None,), (16,)), ((32, 32), (None, 16))],
)
def test_exaone_native_mtp_uses_separate_draft_attention_policy(
    tmp_path, target_windows, draft_windows
):
    from tests.v1.e2e.spec_decode.remote.test_remote_mtp import (
        _tiny_exaone_mtp_model,
    )
    from vllm.config import set_current_vllm_config
    from vllm.distributed import destroy_model_parallel, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.utils.torch_utils import set_default_torch_dtype

    target_path = _tiny_exaone_mtp_model(tmp_path / "target", target_windows)
    draft_path = _tiny_exaone_mtp_model(tmp_path / "draft", draft_windows)
    config = EngineArgs(
        model=target_path,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=128,
        enforce_eager=True,
        async_scheduling=False,
        speculative_config={
            "method": "mtp",
            "model": draft_path,
            "num_speculative_tokens": len(draft_windows),
        },
    ).create_engine_config()
    assert config.speculative_config is not None
    draft = config.speculative_config.draft_model_config
    assert draft is not None
    contract = create_runtime_contract(config)
    assert contract.capabilities.num_prefill_lookahead == len(draft_windows)
    loader = get_model_loader(config.load_config)
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cpu"),
    ):
        destroy_model_parallel()
        initialize_model_parallel(backend="gloo")
        model = loader.create_model(config, draft)
        loader.load_weights(model, draft)
        assert (
            tuple(layer.self_attn.sliding_window_size for layer in model.model.layers)
            == draft_windows
        )
        assert config.model_config.hf_config.mtp_layer_types == [
            "full_attention" if window is None else "sliding_attention"
            for window in target_windows
        ]
        assert config.model_config.hf_config.mtp_sliding_windows == [
            window or 32 for window in target_windows
        ]


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
    ["MiMoMTP", "Gemma4MTPModel", "Qwen3_5MTP", "DeepSeekV4MTPModel"],
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


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize(
    "omitted,tracking,packed",
    [
        (None, None, False),
        (None, None, True),
        ("eh_proj", None, False),
        ("final_layernorm", None, False),
        ("self_attn.q_proj", None, False),
        ("self_attn.qkv_proj", None, True),
        ("mlp.shared_mlp.gate_proj", None, False),
        ("mlp.experts.0.gate_proj", None, False),
        ("mlp.experts.0.down_proj", None, False),
        ("mlp.experts", None, False),
        ("eh_proj", False, False),
    ],
)
def test_hyv3_native_load_rejects_uninitialized_mtp_parameters(
    request, omitted, tracking, packed, tied
):
    """Whole missing MTP tensors must not bypass native loader completeness."""
    from vllm.config import set_current_vllm_config
    from vllm.distributed import destroy_model_parallel, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.utils.torch_utils import set_default_torch_dtype

    source = Path(request.getfixturevalue("tiny_hyv3_mtp_model"))
    config_path = source / "config.json"
    hf_config = json.loads(config_path.read_text())
    hf_config["tie_word_embeddings"] = tied
    config_path.write_text(json.dumps(hf_config))
    path = source / "model.safetensors"
    weights = load_file(path)
    if tied:
        del weights["lm_head.weight"]
    reference = weights.copy()
    prefix = "model.layers.2."
    if packed:
        q, k, v = (
            weights.pop(prefix + f"self_attn.{part}_proj.weight")
            for part in ("q", "k", "v")
        )
        weights[prefix + "self_attn.qkv_proj.weight"] = torch.cat(
            (
                q.reshape(2, 2, 32, 128),
                k.reshape(2, 1, 32, 128),
                v.reshape(2, 1, 32, 128),
            ),
            dim=1,
        ).reshape(-1, 128)
        weights[prefix + "mlp.shared_mlp.gate_and_up_proj.weight"] = torch.cat(
            [
                weights.pop(prefix + f"mlp.shared_mlp.{part}_proj.weight")
                for part in ("up", "gate")
            ]
        )
    if omitted is not None:
        missing_weights = [
            name for name in weights if name.startswith(prefix + omitted + ".")
        ]
        assert missing_weights
        for name in missing_weights:
            del weights[name]
        if omitted == "self_attn.q_proj":
            # A duplicate K write must not count as the absent Q component.
            name = prefix + "self_attn.k_proj.weight"
            save_file({name: weights[name]}, source / "duplicate.safetensors")
    save_file(weights, path, metadata={"format": "pt"})
    config = EngineArgs(
        model=str(source),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=128,
        enforce_eager=True,
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
        model_loader_extra_config={"enable_weights_track": tracking},
    ).create_engine_config()
    assert config.speculative_config is not None
    draft = config.speculative_config.draft_model_config
    assert draft is not None
    loader = get_model_loader(config.load_config)
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cpu"),
    ):
        destroy_model_parallel()
        initialize_model_parallel(backend="gloo")
        model = loader.create_model(config, draft)
        if omitted is not None and tracking is not False:
            missing = {
                "self_attn.q_proj": "qkv_proj",
                "mlp.shared_mlp.gate_proj": "gate_up_proj",
                "mlp.experts.0.gate_proj": "w13_weight",
                "mlp.experts.0.down_proj": "w2_weight",
            }.get(omitted, omitted)
            with pytest.raises(
                ValueError, match=rf"not initialized.*checkpoint.*{missing}"
            ):
                loader.load_weights(model, draft)
        else:
            if omitted is not None:
                with torch.no_grad():
                    model.get_parameter(f"model.layers.2.{omitted}.weight").fill_(-2)
            loader.load_weights(model, draft)
            for name in ("eh_proj", "final_layernorm", "enorm", "hnorm"):
                actual = model.get_parameter(f"model.layers.2.{name}.weight")
                if name == omitted:
                    assert torch.equal(actual, torch.full_like(actual, -2))
                    continue
                assert torch.equal(actual, weights[f"model.layers.2.{name}.weight"])
            assert torch.equal(
                model.get_parameter("model.layers.2.shared_head.head.weight"),
                weights["model.embed_tokens.weight" if tied else "lm_head.weight"],
            )
            for native, checkpoint_parts in (
                (
                    "self_attn.qkv_proj.weight",
                    (
                        "self_attn.q_proj.weight",
                        "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight",
                    ),
                ),
                (
                    "mlp.shared_mlp.gate_up_proj.weight",
                    (
                        "mlp.shared_mlp.gate_proj.weight",
                        "mlp.shared_mlp.up_proj.weight",
                    ),
                ),
            ):
                assert torch.equal(
                    model.get_parameter(prefix + "mtp_block." + native),
                    torch.cat([reference[prefix + name] for name in checkpoint_parts]),
                )
            for native, parts in (
                ("w13_weight", ("gate", "up")),
                ("w2_weight", ("down",)),
            ):
                assert torch.equal(
                    model.get_parameter(
                        prefix + "mtp_block.mlp.experts.routed_experts." + native
                    ),
                    torch.stack(
                        [
                            torch.cat(
                                [
                                    reference[
                                        prefix
                                        + f"mlp.experts.{expert}.{part}_proj.weight"
                                    ]
                                    for part in parts
                                ]
                            )
                            for expert in range(4)
                        ]
                    ),
                )


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("mode", ["target_shared", "standalone", "missing_fusion"])
def test_hyv3_draft_only_checkpoint_preserves_target_shared_parameters(
    tmp_path, request, tied, mode
):
    from vllm.config import set_current_vllm_config
    from vllm.distributed import destroy_model_parallel, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.spec_decode.remote.runtime.weights import materialize_target_shell

    target = Path(request.getfixturevalue("tiny_hyv3_mtp_model"))
    metadata = json.loads((target / "config.json").read_text())
    metadata["tie_word_embeddings"] = tied
    (target / "config.json").write_text(json.dumps(metadata))
    draft_path = tmp_path / "draft-only"
    draft_path.mkdir()
    (draft_path / "config.json").write_text(json.dumps(metadata))
    weights = load_file(target / "model.safetensors")
    if tied:
        del weights["lm_head.weight"]
        save_file(weights, target / "model.safetensors", metadata={"format": "pt"})
    draft_weights = {
        name: weight
        for name, weight in weights.items()
        if name.startswith("model.layers.2.")
    }
    if mode == "missing_fusion":
        del draft_weights["model.layers.2.eh_proj.weight"]
    save_file(
        draft_weights, draft_path / "model.safetensors", metadata={"format": "pt"}
    )
    config = EngineArgs(
        model=str(target),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=128,
        enforce_eager=True,
        speculative_config={
            "method": "mtp",
            "model": str(draft_path),
            "num_speculative_tokens": 1,
        },
    ).create_engine_config()
    assert config.speculative_config is not None
    draft = config.speculative_config.draft_model_config
    assert draft is not None
    if mode == "standalone":
        config.speculative_config = None
    loader = get_model_loader(config.load_config)
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cpu"),
    ):
        destroy_model_parallel()
        initialize_model_parallel(backend="gloo")
        model = loader.create_model(config, draft)
        if mode != "target_shared":
            missing = "embed_tokens" if mode == "standalone" else "eh_proj"
            with pytest.raises(
                ValueError, match=rf"not initialized.*checkpoint.*{missing}"
            ):
                loader.load_weights(model, draft)
        else:
            loader.load_weights(model, draft)
            assert torch.equal(
                model.get_parameter("model.layers.2.eh_proj.weight"),
                weights["model.layers.2.eh_proj.weight"],
            )
            target_shell = materialize_target_shell(
                config.model_config, torch.device("cpu")
            )
            assert torch.equal(
                target_shell.model.embed_tokens.weight[:128],
                weights["model.embed_tokens.weight"],
            )
            assert torch.equal(
                target_shell.lm_head.weight[:128],
                weights["model.embed_tokens.weight" if tied else "lm_head.weight"],
            )
