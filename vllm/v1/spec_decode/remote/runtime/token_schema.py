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


def _validate_token_model(model, role: str) -> None:
    """Require the native causal token/KV contract consumed by this executor."""
    if (
        model.runner_type not in ("generate", "draft")
        or model.using_transformers_backend()
        or model.is_encoder_decoder
        or model.is_multimodal_model
        or model.is_diffusion
        or model.is_attention_free
        or model.is_hybrid
        or model.has_inner_state
        or model.has_noops
        or model.uses_mrope
        or model.use_mla
        or not getattr(model.hf_text_config, "is_causal", True)
    ):
        raise ValueError(
            f"token-only {role} requires a native decoder-only causal attention model"
        )
    layers = getattr(model.hf_text_config, "layer_types", None)
    if (
        model.get_sliding_window() is not None
        or model.attention_chunk_size is not None
        or (
            layers is not None
            and any(layer not in ("attention", "full_attention") for layer in layers)
        )
    ):
        raise ValueError(f"token-only {role} requires homogeneous full attention")


def supports_token_prefix_replay(config: "VllmConfig") -> bool:
    """Whether cached target tokens can bootstrap an ordinary draft prefix."""
    spec = config.speculative_config
    if (
        spec is None
        or spec.method != "draft_model"
        or not spec.use_remote_draft()
        or spec.draft_model_config is None
        or spec.draft_parallel_config is None
        or config.parallel_config.tensor_parallel_size != 1
        or spec.draft_parallel_config.tensor_parallel_size != 1
        or (spec.execution is not None and len(spec.execution.endpoints) != 1)
        or config.cache_config.sliding_window is not None
    ):
        return False
    remote = spec.get_remote_draft_config()
    if remote is None or remote.transport not in ("auto", "inline"):
        return False
    for model, role in (
        (config.model_config, "target"),
        (spec.draft_model_config, "draft"),
    ):
        if model.is_hybrid or model.has_inner_state or model.use_mla:
            return False
        try:
            _validate_token_model(model, role)
        except ValueError:
            return False
    return True


def supports_token_async_scheduling(config: "VllmConfig") -> bool:
    """Whether the verifier can use async scheduling with same-step token RPCs."""
    return (
        supports_token_prefix_replay(config)
        and config.use_v2_model_runner
        and config.model_config.enforce_eager
    )


def _validate_token_draft_topology(config: "VllmConfig") -> int:
    spec = config.speculative_config
    assert spec is not None and spec.draft_model_config is not None
    parallel = spec.draft_parallel_config
    assert parallel is not None
    tp = parallel.tensor_parallel_size
    if tp == 1:
        return tp
    if tp != 2:
        raise ValueError("token-only draft supports tensor parallel size 1 or 2")
    if config.parallel_config.tensor_parallel_size != 1 or any(
        getattr(parallel, name) != 1
        for name in (
            "pipeline_parallel_size",
            "data_parallel_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        )
    ):
        raise ValueError("tensor-parallel draft requires target TP1 and draft TP2 only")
    if (
        config.scheduler_config.async_scheduling
        or config.cache_config.enable_prefix_caching
        or not config.model_config.enforce_eager
    ):
        raise ValueError(
            "tensor-parallel draft requires eager synchronous execution "
            "without prefix caching"
        )
    remote = spec.get_remote_draft_config()
    if (
        remote is None
        or remote.transport not in ("auto", "inline")
        or spec.execution is None
        or len(spec.execution.endpoints) != 1
    ):
        raise ValueError("tensor-parallel draft requires one inline endpoint")
    for model, role in (
        (config.model_config, "target"),
        (spec.draft_model_config, "draft"),
    ):
        if model.is_moe or model.is_hybrid or model.has_inner_state or model.use_mla:
            raise ValueError(
                "tensor-parallel draft requires dense full-attention target "
                "and draft models"
            )
        try:
            _validate_token_model(model, role)
        except ValueError as error:
            raise ValueError(
                "tensor-parallel draft requires dense full-attention target "
                "and draft models"
            ) from error
        if model.quantization is not None or str(model.dtype) != "torch.bfloat16":
            raise ValueError("tensor-parallel draft requires unquantized bfloat16")
    if config.cache_config.sliding_window is not None:
        raise ValueError("tensor-parallel draft requires full-attention KV")
    if (spec.kv_cache_dtype or config.cache_config.cache_dtype) not in (
        "auto",
        "bfloat16",
    ):
        raise ValueError("tensor-parallel draft requires bfloat16 KV")
    draft = spec.draft_model_config
    draft.verify_with_parallel_config(parallel)
    for layer in range(draft.get_total_num_hidden_layers()):
        geometry = draft.model_arch_config[layer]
        if geometry.total_num_attention_heads % tp:
            raise ValueError(
                f"draft attention heads in layer {layer} must be divisible by TP"
            )
        kv_heads = geometry.total_num_kv_heads
        if (
            type(kv_heads) is not int
            or kv_heads < 1
            or (kv_heads >= tp and kv_heads % tp)
            or (kv_heads < tp and tp % kv_heads)
        ):
            raise ValueError(
                "draft KV heads must shard or replicate evenly across TP ranks"
            )
    if draft.get_hidden_size() % tp:
        raise ValueError("draft hidden size must be divisible by tensor parallel size")
    intermediate = getattr(draft.hf_text_config, "intermediate_size", None)
    if isinstance(intermediate, int) and intermediate % tp:
        raise ValueError(
            "draft intermediate size must be divisible by tensor parallel size"
        )
    return tp


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
    if config.cache_config.enable_prefix_caching and not supports_token_prefix_replay(
        config
    ):
        raise ValueError(
            "remote prefix caching requires full-attention token-only target and "
            "draft models, TP1 and one inline endpoint"
        )
    if config.scheduler_config.async_scheduling and not supports_token_async_scheduling(
        config
    ):
        raise ValueError(
            "remote async scheduling requires eager MRV2, full-attention token-only "
            "target and draft models, TP1 and one inline endpoint"
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
    draft_tp = _validate_token_draft_topology(config)
    _validate_token_model(target, "target")
    _validate_token_model(draft, "draft")
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
        draft_fingerprint=_fingerprint(
            {
                "model": _model_identity(draft),
                "method": "draft_model",
                "k": spec.num_speculative_tokens,
                **({"tensor_parallel_size": draft_tp} if draft_tp != 1 else {}),
            }
        ),
    )
