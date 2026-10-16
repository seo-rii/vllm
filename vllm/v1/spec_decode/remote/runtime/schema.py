# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-derived contract shared by the verifier and standalone runtime."""

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.v1.spec_decode.remote.capabilities import (
    SpeculatorPlacementCapabilities,
    TargetFeatureKind,
)
from vllm.v1.spec_decode.remote.protocol import FeatureSlot, TargetFeatureSchema

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class RuntimeContract:
    schema: TargetFeatureSchema
    capabilities: SpeculatorPlacementCapabilities
    target_fingerprint: str
    tokenizer_fingerprint: str
    draft_fingerprint: str


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _model_identity(model: Any) -> dict[str, Any]:
    config = model.hf_config.to_dict()
    for field in ("_name_or_path", "transformers_version"):
        config.pop(field, None)
    return {
        "model": model.model,
        "revision": model.revision,
        "resolved_revision": getattr(model.hf_config, "_commit_hash", None),
        "config": config,
        "dtype": str(model.dtype),
        "quantization": model.quantization,
    }


def create_runtime_contract(vllm_config: "VllmConfig") -> RuntimeContract:
    """Resolve the supported feature layout without loading any model weights."""
    spec = vllm_config.speculative_config
    if spec is None or spec.draft_model_config is None:
        raise ValueError("standalone drafting requires a resolved speculative config")
    target = vllm_config.model_config
    draft = spec.draft_model_config
    target_type = target.hf_config.model_type
    if spec.method == "eagle3":
        if target_type not in ("gpt_oss", "qwen3"):
            raise ValueError("standalone EAGLE3 currently supports GPT-OSS and Qwen3")
    elif spec.method == "mtp":
        if "MiMoMTP" not in draft.hf_config.architectures:
            raise ValueError("standalone MTP currently supports MiMoMTP only")
        if spec.num_speculative_tokens != 1:
            raise ValueError("MiMoMTP supports exactly one speculative token")
    else:
        raise ValueError(f"unsupported standalone method: {spec.method}")
    if spec.parallel_drafting or spec.draft_sample_method != "greedy":
        raise ValueError("standalone runtime requires sequential greedy drafting")

    hidden_size = target.get_hidden_size()
    dtype = str(target.dtype).removeprefix("torch.")
    eagle = getattr(draft.hf_config, "eagle_config", None) or {}
    use_aux = spec.method == "eagle3" and eagle.get("use_aux_hidden_state", True)
    slots = [
        FeatureSlot(TargetFeatureKind.TOKEN_IDS.value, "int32", ()),
        FeatureSlot(TargetFeatureKind.POSITIONS.value, "int64", ()),
    ]
    if use_aux:
        count = getattr(draft.hf_config, "num_aux_hidden_states", None)
        if count is None:
            layer_ids = (
                getattr(draft.hf_config, "eagle_aux_hidden_state_layer_ids", None)
                or eagle.get("layer_ids")
                or eagle.get("eagle_aux_hidden_state_layer_ids")
            )
            count = len(layer_ids) if layer_ids else 3
        slots.append(
            FeatureSlot(
                TargetFeatureKind.AUX_HIDDEN_STATES.value, dtype, (hidden_size * count,)
            )
        )
    else:
        slots.append(
            FeatureSlot(
                TargetFeatureKind.LAST_HIDDEN_STATES.value, dtype, (hidden_size,)
            )
        )
    slots.append(
        FeatureSlot(
            TargetFeatureKind.NEXT_PREFILL_TOKENS.value,
            "int64",
            (1,),
            optional=True,
            per_sequence=True,
        )
    )
    schema = TargetFeatureSchema(schema_id=1, slots=tuple(slots))
    layout = [(s.kind, s.dtype, s.trailing_shape) for s in slots]
    target_identity = _model_identity(target)
    return RuntimeContract(
        schema=schema,
        capabilities=SpeculatorPlacementCapabilities(
            state_dependency="own_kv",
            required_features=tuple(s.kind for s in slots if not s.optional),
            num_prefill_lookahead=1,
            standalone_weights="materializable",
        ),
        target_fingerprint=_fingerprint({"target": target_identity, "layout": layout}),
        tokenizer_fingerprint=_fingerprint(
            {
                "tokenizer": target.tokenizer,
                "revision": target.tokenizer_revision,
                "mode": target.tokenizer_mode,
                "vocab_size": target.get_vocab_size(),
                "special_tokens": {
                    k: target_identity["config"].get(k)
                    for k in ("bos_token_id", "eos_token_id", "pad_token_id")
                },
            }
        ),
        draft_fingerprint=_fingerprint(
            {
                "draft": _model_identity(draft),
                "layout": layout,
                "method": spec.method,
                "num_speculative_tokens": spec.num_speculative_tokens,
            }
        ),
    )
