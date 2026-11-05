# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-derived Gemma4 contract for the bounded Q-only remote runtime."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec, SlidingWindowSpec
from vllm.v1.spec_decode.remote.runtime.schema import (
    _fingerprint,
    _model_identity,
    _tokenizer_identity,
)
from vllm.v1.spec_decode.remote.target_kv import (
    AttentionSpec,
    SnapshotBudget,
    _table_width,
    gemma4_source_layers,
)
from vllm.v1.spec_decode.remote.target_kv_protocol import (
    MAX_SNAPSHOT_BATCH_SIZE,
    snapshot_batch_wire_plan,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class TargetKVRuntimeContract:
    fingerprint: str
    expected_specs: dict[str, AttentionSpec]
    source_indices: dict[str, int]
    draft_source_names: tuple[str, ...]
    hidden_size: int
    vocab_size: int
    draft_num_attention_heads: int
    snapshot_budget: SnapshotBudget
    receiver_budget: SnapshotBudget
    max_num_seqs: int = 1


def create_target_kv_contract(config: "VllmConfig") -> TargetKVRuntimeContract:
    """Resolve the exact native page geometry before either peer loads weights."""
    from vllm.v1.core.kv_cache_planning import unify_kv_cache_spec_page_size

    target = config.model_config
    spec = config.speculative_config
    if spec is None or spec.draft_model_config is None:
        raise ValueError("target-KV runtime requires an assistant checkpoint")
    draft = spec.draft_model_config
    errors = []
    if (
        spec.method != "mtp"
        or spec.num_speculative_tokens != 1
        or spec.draft_sample_method != "greedy"
        or spec.parallel_drafting
        or spec.enable_adaptive_verification
    ):
        errors.append("sequential greedy MTP K1 is required")
    if (
        any(
            getattr(config.parallel_config, name) != 1
            for name in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
            )
        )
        or spec.draft_tensor_parallel_size != 1
    ):
        errors.append("target and draft require TP1/PP1/DP1/DCP1/PCP1")
    if (
        not 1 <= config.scheduler_config.max_num_seqs <= MAX_SNAPSHOT_BATCH_SIZE
        or config.scheduler_config.async_scheduling
        or config.scheduler_config.disable_hybrid_kv_cache_manager
    ):
        errors.append("1..16 sequences, synchronous scheduling and hybrid KV required")
    backend = config.attention_config.backend
    if getattr(backend, "name", backend) != "TRITON_ATTN":
        errors.append("TRITON_ATTN is required")
    if not target.enforce_eager:
        errors.append("enforce_eager is required")
    if target.dtype != torch.bfloat16 or draft.dtype != torch.bfloat16:
        errors.append("BF16 target and assistant are required")
    if target.quantization is not None or draft.quantization is not None:
        errors.append("quantized models are unsupported")
    cache = config.cache_config
    if (
        cache.block_size != 16
        or cache.cache_dtype not in ("auto", "bfloat16")
        or cache.enable_prefix_caching
        or getattr(cache, "calculate_kv_scales", False)
    ):
        errors.append(
            "block_size=16 and unscaled BF16 KV without prefix reuse required"
        )
    if (
        config.lora_config is not None
        or getattr(config, "watermark_config", None) is not None
        or target.enable_prompt_embeds
    ):
        errors.append("LoRA, watermarking and prompt embeddings are unsupported")
    if target.is_multimodal_model and not (
        target.multimodal_config is not None
        and target.multimodal_config.language_model_only
    ):
        errors.append("multimodal Gemma requires language_model_only=True")
    if not 1 <= target.max_model_len <= 8192:
        errors.append("max_model_len must be between 1 and 8192")
    if (
        target.hf_config.model_type not in ("gemma4", "gemma4_text")
        or draft.hf_config.model_type != "gemma4_mtp"
        or draft.hf_config.architectures != ["Gemma4MTPModel"]
    ):
        errors.append("native Gemma4 target and Gemma4MTPModel assistant required")
    if errors:
        raise ValueError("remote target-KV: " + "; ".join(errors))

    text = target.hf_config.get_text_config()
    assistant = draft.hf_config.get_text_config()
    if (
        min(
            text.hidden_size,
            assistant.hidden_size,
            text.vocab_size,
            text.num_hidden_layers,
            assistant.num_hidden_layers,
            text.num_attention_heads,
            assistant.num_attention_heads,
        )
        < 1
        or assistant.num_kv_shared_layers != 0
        or draft.hf_config.backbone_hidden_size != text.hidden_size
        or assistant.vocab_size != text.vocab_size
        or getattr(text, "use_bidirectional_attention", None) == "all"
        or getattr(assistant, "use_bidirectional_attention", None) == "all"
        or not getattr(text, "is_causal", True)
        or not getattr(assistant, "is_causal", True)
        or getattr(assistant, "enable_moe_block", False)
        or getattr(assistant, "hidden_size_per_layer_input", 0) != 0
        or getattr(assistant, "vocab_size_per_layer_input", 0) != 0
        or getattr(assistant, "use_double_wide_mlp", False)
        or getattr(target, "rswa_window", None) is not None
        or target.max_model_len > assistant.max_position_embeddings
    ):
        raise ValueError("remote target-KV requires compatible causal Gemma4 configs")
    if (
        len(text.layer_types) != text.num_hidden_layers
        or len(assistant.layer_types) != assistant.num_hidden_layers
    ):
        raise ValueError("Gemma4 layer types must cover every layer")
    owners = gemma4_source_layers(
        text.layer_types,
        assistant.layer_types,
        num_kv_shared_layers=text.num_kv_shared_layers,
    )
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    specs: dict[str, KVCacheSpec] = {}
    layer_spec: AttentionSpec
    for i, kind in enumerate(
        text.layer_types[: text.num_hidden_layers - text.num_kv_shared_layers]
    ):
        layer = text.per_layer_config[i]
        if (
            not TritonAttentionBackend.supports_head_size(layer.head_dim)
            or layer.head_dim % 2
            or layer.num_key_value_heads < 1
            or text.num_attention_heads % layer.num_key_value_heads
        ):
            raise ValueError("unsupported Gemma4 target attention geometry")
        kwargs = dict(
            block_size=16,
            num_kv_heads=layer.num_key_value_heads,
            head_size=layer.head_dim,
            dtype=torch.bfloat16,
        )
        if kind == "sliding_attention":
            if text.sliding_window < 1:
                raise ValueError("Gemma4 sliding window must be positive")
            layer_spec = SlidingWindowSpec(**kwargs, sliding_window=text.sliding_window)
        else:
            layer_spec = FullAttentionSpec(**kwargs)
        specs[f"target.layers.{i}"] = layer_spec
    unified = unify_kv_cache_spec_page_size(specs)
    source_names = tuple(f"target.layers.{i}" for i in owners)
    expected: dict[str, AttentionSpec] = {}
    for name in dict.fromkeys(source_names):
        unified_spec = unified[name]
        assert isinstance(unified_spec, (FullAttentionSpec, SlidingWindowSpec))
        expected[name] = unified_spec
    for i, name in enumerate(source_names):
        layer = assistant.per_layer_config[i]
        owner = expected[name]
        if (
            layer.head_dim != owner.head_size
            or layer.num_key_value_heads != owner.num_kv_heads
            or assistant.num_attention_heads % owner.num_kv_heads
            or (
                isinstance(owner, SlidingWindowSpec)
                and assistant.sliding_window != owner.sliding_window
            )
        ):
            raise ValueError("assistant attention must match its target KV owner")
    budget = SnapshotBudget(1024, 128 * 1024**2, target.max_model_len)
    # Native page unification may add physical padding that snapshots cannot
    # represent. Validate that contract, and maximum-context capacity, before
    # either process loads weights or the target allocates its KV cache.
    snapshot_batch_wire_plan(
        expected,
        (target.max_model_len,) * config.scheduler_config.max_num_seqs,
        budget,
        budget,
    )
    pages = nbytes = table_bytes = 0
    for layer in expected.values():
        count = (target.max_model_len + layer.block_size - 1) // layer.block_size
        table_bytes += _table_width(layer, count) * 4
        if layer.sliding_window is not None:
            count = min(
                count,
                (layer.sliding_window + 2 * layer.block_size - 2) // layer.block_size,
            )
        pages += count
        nbytes += count * layer.unpadded_page_size_bytes
    size = config.scheduler_config.max_num_seqs
    budget.check(target.max_model_len, size * pages, size * nbytes)
    budget.check(
        target.max_model_len,
        size * pages + len(expected),
        size * (nbytes + table_bytes)
        + sum(layer.unpadded_page_size_bytes for layer in expected.values()),
    )
    identity = {
        "target": _model_identity(target),
        "draft": _model_identity(draft),
        "draft_suppress_tokens": sorted(
            set(draft.try_get_generation_config().get("suppress_tokens") or ())
        ),
        "tokenizer": _tokenizer_identity(target),
        "tokenizer_mode": target.tokenizer_mode,
        "sources": source_names,
        "specs": {name: vars(value) for name, value in expected.items()},
        "max_model_len": target.max_model_len,
        "contract": "gemma4-q-only-bf16-k1-v2",
    }
    return TargetKVRuntimeContract(
        fingerprint=_fingerprint(identity),
        expected_specs=expected,
        source_indices={name: int(name.rsplit(".", 1)[1]) for name in expected},
        draft_source_names=source_names,
        hidden_size=text.hidden_size,
        vocab_size=text.vocab_size,
        draft_num_attention_heads=assistant.num_attention_heads,
        snapshot_budget=budget,
        receiver_budget=budget,
        max_num_seqs=config.scheduler_config.max_num_seqs,
    )
