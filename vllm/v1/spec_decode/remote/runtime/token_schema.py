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
            }
        ),
    )
