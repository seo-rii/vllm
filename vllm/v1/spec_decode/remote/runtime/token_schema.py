# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint and actual tokenizer-asset identity for token-only placement."""

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from vllm.v1.spec_decode.remote.capabilities import SpeculatorPlacementCapabilities
from vllm.v1.spec_decode.remote.protocol import FeatureSlot, TargetFeatureSchema
from vllm.v1.spec_decode.remote.runtime.schema import (
    RuntimeContract,
    _fingerprint,
    _model_identity,
    checkpoint_revision,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_STATEFUL_TOKEN_TARGETS = (
    "Qwen3NextForCausalLM",
    "NemotronHForCausalLM",
    "Glm4MoeLiteForCausalLM",
)

_ASSETS = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
)


def tokenizer_artifact_fingerprint(source: str, revision: str | None) -> str:
    """Compare actual trained token IDs, not the configured target-tokenizer alias.

    Remote assets must already exist in the immutable HF cache. Model staging is
    separate from worker startup; missing identities fail closed.
    """
    artifacts: dict[str, object] = {}
    local = Path(source).is_dir()
    revision = checkpoint_revision(source, revision)
    for name in _ASSETS:
        if local:
            path = Path(source) / name
            if not path.is_file():
                continue
        else:
            from vllm.transformers_utils.repo_utils import try_get_local_file

            cached = try_get_local_file(source, name, revision=revision)
            if not isinstance(cached, Path):
                continue
            path = cached
            if path.parent.name != revision or path.parent.parent.name != "snapshots":
                raise ValueError(
                    "cached tokenizer artifact does not match its immutable revision"
                )
        if name.endswith(".json"):
            value = json.loads(path.read_text())
            if name == "tokenizer_config.json" and value.get("auto_map"):
                raise ValueError("custom tokenizer implementations are unsupported")
            artifacts[name] = value
        else:
            artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not any(
        name in artifacts
        for name in ("tokenizer.json", "tokenizer.model", "spiece.model")
    ):
        raise ValueError(
            f"token-only placement requires staged tokenizer artifacts: {source}"
        )
    if "tokenizer_config.json" not in artifacts:
        raise ValueError(
            f"token-only placement requires tokenizer_config.json: {source}"
        )
    return _fingerprint(artifacts)


def _validate_token_inputs(model, role: str) -> None:
    """Require native text-only causal tokens in both execution roles."""
    if (
        model.runner_type not in ("generate", "draft")
        or model.using_transformers_backend()
        or model.is_encoder_decoder
        or model.is_multimodal_model
        or model.is_diffusion
        or model.is_attention_free
        or model.has_noops
        or model.uses_mrope
        or not getattr(model.hf_text_config, "is_causal", True)
    ):
        raise ValueError(
            f"token-only {role} requires a native decoder-only causal attention model"
        )
    if model.attention_chunk_size is not None:
        raise ValueError(f"token-only {role} does not support chunked attention")
    if model.get_total_num_hidden_layers() < 1:
        raise ValueError(f"token-only {role} requires at least one attention layer")


def _validate_token_target(model) -> None:
    if model.architecture not in _STATEFUL_TOKEN_TARGETS:
        _validate_token_model(model, "target")
        return
    _validate_token_inputs(model, "target")
    # The verifier owns all target recurrence/MLA state. The remote model
    # receives only authoritative token deltas and must still pass the dense
    # draft contract below.
    if model.architecture == "Glm4MoeLiteForCausalLM":
        if (
            not model.use_mla
            or model.is_hybrid
            or model.has_inner_state
            or hasattr(model.hf_text_config, "index_topk")
        ):
            raise ValueError("token-only GLM4MoELite target requires dense MLA")
    elif not model.is_hybrid or not model.has_inner_state or model.use_mla:
        raise ValueError("token-only hybrid target requires native recurrent state")


def _validate_token_model(model, role: str) -> tuple[int | None, ...]:
    """Validate the dense KV contract and each layer's attention window."""
    _validate_token_inputs(model, role)
    if role == "draft" and model.architecture == "NemotronHForCausalLM":
        hf = model.hf_text_config
        pattern = getattr(hf, "hybrid_override_pattern", None)
        if (
            hf.model_type != "nemotron_h"
            or hf.architectures != ["NemotronHForCausalLM"]
            or not model.is_hybrid
            or not model.has_inner_state
            or model.use_mla
            or getattr(hf, "num_nextn_predict_layers", 0) != 0
            or not isinstance(pattern, str)
            or len(pattern) != model.get_total_num_hidden_layers()
            or "M" not in pattern
            or "*" not in pattern
            or any(layer not in "M*-" for layer in pattern)
            or model.get_sliding_window() is not None
        ):
            raise ValueError(
                "token-only recurrent draft requires native Mamba2 and full "
                "attention without MTP or experts"
            )
        if model.quantization is not None or str(model.dtype) != "torch.bfloat16":
            raise ValueError("token-only recurrent draft requires unquantized bfloat16")
        return (None,) * model.get_total_num_hidden_layers()
    if role == "draft" and model.architecture == "DeepseekV2ForCausalLM":
        hf = model.hf_text_config
        layers = getattr(hf, "layer_types", None)
        if (
            hf.model_type != "deepseek_v2"
            or hf.architectures != ["DeepseekV2ForCausalLM"]
            or not model.use_mla
            or model.is_hybrid
            or model.has_inner_state
            or hasattr(hf, "index_topk")
            or model.get_sliding_window() is not None
            or (
                layers is not None
                and (
                    len(layers) != model.get_total_num_hidden_layers()
                    or any(layer != "full_attention" for layer in layers)
                )
            )
        ):
            raise ValueError("token-only draft requires dense MLA without an indexer")
        if model.quantization is not None or str(model.dtype) != "torch.bfloat16":
            raise ValueError("token-only MLA draft requires unquantized bfloat16")
        return (None,) * model.get_total_num_hidden_layers()
    if model.is_hybrid or model.has_inner_state or model.use_mla:
        raise ValueError(
            f"token-only {role} requires a native decoder-only causal attention model"
        )
    window = model.get_sliding_window()
    if window is not None and (type(window) is not int or window <= 0):
        raise ValueError(
            f"token-only {role} requires a positive integer sliding window"
        )
    layers = getattr(model.hf_text_config, "layer_types", None)
    num_layers = model.get_total_num_hidden_layers()
    if layers is None:
        return (window,) * num_layers
    if not isinstance(layers, (list, tuple)) or len(layers) != num_layers:
        raise ValueError(f"token-only {role} requires one attention type per layer")
    if all(layer in ("attention", "full_attention") for layer in layers):
        # Match EngineArgs: an unused window must not override full layers.
        return (None,) * num_layers
    if window is None or any(
        layer not in ("attention", "full_attention", "sliding_attention")
        for layer in layers
    ):
        raise ValueError(
            f"token-only {role} requires full or sliding-window attention "
            "with a declared sliding window"
        )
    if not all(layer == "sliding_attention" for layer in layers) and (
        model.architecture not in ("LlamaForCausalLM", "MistralForCausalLM")
    ):
        # Other native models may ignore layer_types or need extra state.
        raise ValueError(
            f"token-only {role} mixed attention requires native Llama or Mistral"
        )
    return tuple(window if layer == "sliding_attention" else None for layer in layers)


def create_token_runtime_contract(config: "VllmConfig") -> RuntimeContract:
    spec = config.speculative_config
    if spec is None or spec.method != "draft_model" or spec.draft_model_config is None:
        raise ValueError("token-only runtime requires an independent draft_model")
    if (
        spec.parallel_drafting
        or spec.draft_sample_method != "greedy"
        or spec.use_heterogeneous_vocab
    ):
        raise ValueError(
            "token-only runtime requires sequential same-tokenizer greedy drafting"
        )
    target, draft = config.model_config, spec.draft_model_config
    if (
        target.tokenizer_mode != "hf"
        or getattr(target, "trust_remote_code", False)
        or getattr(draft, "trust_remote_code", False)
    ):
        raise ValueError(
            "token-only runtime requires the built-in HF tokenizer implementation"
        )
    _validate_token_target(target)
    if (
        target.architecture in _STATEFUL_TOKEN_TARGETS
        and len(spec.get_remote_draft_endpoints()) > 1
    ):
        raise ValueError("token-only stateful target requires a single remote endpoint")
    _validate_token_model(draft, "draft")
    recurrent = draft.architecture == "NemotronHForCausalLM"
    if recurrent:
        from vllm.config.mamba import MambaBackendEnum

        if (
            target.architecture != "Qwen3ForCausalLM"
            or _validate_token_model(target, "target")
            != (None,) * target.get_total_num_hidden_layers()
            or config.cache_config.sliding_window is not None
        ):
            raise ValueError("token-only recurrent draft requires a dense Qwen3 target")
        if target.quantization is not None or str(target.dtype) != "torch.bfloat16":
            raise ValueError("token-only recurrent draft requires unquantized bfloat16")
        if (
            config.parallel_config.tensor_parallel_size != 1
            or spec.draft_parallel_config.tensor_parallel_size != 1
        ):
            raise ValueError(
                "token-only recurrent draft requires target TP1 and draft TP1"
            )
        remote = spec.get_remote_draft_config()
        if remote is not None and (
            remote.transport not in ("auto", "inline")
            or len(spec.get_remote_draft_endpoints()) != 1
        ):
            raise ValueError("token-only recurrent draft requires one inline endpoint")
        cache = config.cache_config
        if (
            (spec.kv_cache_dtype or cache.cache_dtype) not in ("auto", "bfloat16")
            or cache.mamba_cache_dtype not in ("auto", "bfloat16")
            or cache.mamba_ssm_cache_dtype not in ("auto", "float32")
        ):
            raise ValueError(
                "token-only recurrent draft requires bfloat16 KV/conv and float32 SSM"
            )
        if (
            cache.mamba_cache_mode != "none"
            or cache.use_replayssm
            or cache.use_kda_recoverssm
            or config.mamba_config.backend != MambaBackendEnum.TRITON
            or config.mamba_config.enable_stochastic_rounding
        ):
            raise ValueError(
                "token-only recurrent draft requires ordinary Mamba state "
                "with deterministic Triton"
            )
    if draft.use_mla:
        if (
            target.architecture != "Qwen3ForCausalLM"
            or _validate_token_model(target, "target")
            != (None,) * target.get_total_num_hidden_layers()
            or config.cache_config.sliding_window is not None
        ):
            raise ValueError("token-only MLA draft requires a dense Qwen3 target")
        if target.quantization is not None or str(target.dtype) != "torch.bfloat16":
            raise ValueError("token-only MLA draft requires unquantized bfloat16")
        if (
            config.parallel_config.tensor_parallel_size != 1
            or spec.draft_parallel_config.tensor_parallel_size != 1
        ):
            raise ValueError("token-only MLA draft requires target TP1 and draft TP1")
        remote = spec.get_remote_draft_config()
        if remote is not None and (
            remote.transport not in ("auto", "inline")
            or len(spec.get_remote_draft_endpoints()) != 1
        ):
            raise ValueError("token-only MLA draft requires one inline endpoint")
        if (spec.kv_cache_dtype or config.cache_config.cache_dtype) not in (
            "auto",
            "bfloat16",
        ):
            raise ValueError("token-only MLA draft requires bfloat16 KV")
    if target.get_vocab_size() != draft.get_vocab_size():
        raise ValueError("token-only runtime requires equal model vocabulary sizes")
    if target.max_model_len > draft.max_model_len:
        raise ValueError("token-only target context exceeds the draft context")
    target_tokenizer = tokenizer_artifact_fingerprint(
        target.tokenizer, target.tokenizer_revision
    )
    draft_tokenizer = tokenizer_artifact_fingerprint(draft.model, draft.revision)
    if target_tokenizer != draft_tokenizer:
        raise ValueError(
            "target and draft tokenizer artifacts do not define the same token IDs"
        )
    target_special = {
        key: getattr(target.hf_config, key, None)
        for key in ("bos_token_id", "eos_token_id", "pad_token_id")
    }
    draft_special = {key: getattr(draft.hf_config, key, None) for key in target_special}
    if target_special != draft_special:
        raise ValueError("target and draft special token IDs differ")
    draft_identity = {
        "model": _model_identity(draft),
        "method": "draft_model",
        "k": spec.num_speculative_tokens,
    }
    if recurrent:
        draft_identity["state_contract"] = {
            "policy": "computed_prefix_replay_v1",
            "conv_dtype": "bfloat16",
            "ssm_dtype": "float32",
        }
    return RuntimeContract(
        schema=TargetFeatureSchema(3, (FeatureSlot("token_ids", "int32", ()),)),
        capabilities=SpeculatorPlacementCapabilities(
            state_dependency="own_kv",
            required_features=("token_ids",),
            standalone_weights="complete",
            num_prefill_lookahead=0,
        ),
        target_fingerprint=_fingerprint(_model_identity(target)),
        tokenizer_fingerprint=_fingerprint(
            {
                "artifacts": target_tokenizer,
                "mode": target.tokenizer_mode,
                "special_ids": target_special,
            }
        ),
        draft_fingerprint=_fingerprint(draft_identity),
    )
