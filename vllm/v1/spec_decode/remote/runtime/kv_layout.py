# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validated own-KV layouts for the standalone draft planner.

Sliding-window attention keeps a full-context block table here. The attention
backend applies its window; the planner does not evict or recycle old positions.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    KVQuantMode,
    MLAAttentionSpec,
    SlidingWindowSpec,
    SparseCacheRole,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout


@dataclass(frozen=True)
class SparseMLAKVContract:
    """A single native DeepSeek V3.2 MTP cache and its FP8 indexer record."""

    topk_tokens: int
    block_size: int = 64

    def __post_init__(self) -> None:
        if type(self.topk_tokens) is not int or self.topk_tokens not in (
            512,
            1024,
            2048,
        ):
            raise ValueError("standalone sparse MLA requires top-k 512, 1024 or 2048")
        if type(self.block_size) is not int or self.block_size != 64:
            raise ValueError("standalone sparse MLA requires 64-token blocks")

    def validate_specs(self, specs: Mapping[str, KVCacheSpec]) -> None:
        if len(specs) != 2 or any(
            type(spec) is not MLAAttentionSpec for spec in specs.values()
        ):
            raise ValueError("standalone sparse MLA requires one MLA/indexer pair")
        attention = [
            name
            for name, spec in specs.items()
            if isinstance(spec, MLAAttentionSpec)
            and spec.cache_role == SparseCacheRole.SPARSE
        ]
        if len(attention) != 1 or not attention[0].endswith(".self_attn.attn"):
            raise ValueError("standalone sparse MLA requires a native attention name")
        attention_name = attention[0]
        indexer_name = attention_name.removesuffix(".attn") + ".indexer.k_cache"
        if indexer_name not in specs:
            raise ValueError("standalone sparse MLA requires its own layer indexer")
        for name, spec in specs.items():
            assert isinstance(spec, MLAAttentionSpec)
            is_indexer = name == indexer_name
            if (
                spec.block_size != self.block_size
                or spec.num_kv_heads != 1
                or spec.max_tp_shards != 1
                or spec.head_size != (132 if is_indexer else 576)
                or spec.dtype != (torch.uint8 if is_indexer else torch.bfloat16)
                or spec.cache_role
                != (SparseCacheRole.INDEXER if is_indexer else SparseCacheRole.SPARSE)
                or spec.is_index_group_leader != (not is_indexer)
                or spec.topk_buffer_width != (None if is_indexer else self.topk_tokens)
                or spec.head_size_v != 0
                or spec.tokens_per_state != 1
                or spec.num_head_slots is not None
                or spec.state_content_bytes is not None
                or spec.kv_quant_mode != KVQuantMode.NONE
                or spec.page_size_padded is not None
                or spec.alignment is not None
                or spec.model_version is not None
                or spec.non_causal
                or spec.non_causal_multi_token_decode
                or spec.sliding_window is not None
                or spec.attention_chunk_size is not None
                or spec.cache_dtype_str
                not in ((None,) if is_indexer else (None, "auto", "bfloat16"))
            ):
                raise ValueError(
                    "standalone sparse MLA requires native BF16/byte pages"
                )


def _validate_standalone_kv_spec(spec: KVCacheSpec) -> AttentionSpec:
    # Subclasses include sparse/compressed caches and target-owned state with
    # different position or allocation contracts; accepting their base is unsafe.
    if type(spec) not in (FullAttentionSpec, MLAAttentionSpec, SlidingWindowSpec):
        raise ValueError(
            "standalone runtime requires full-attention, dense MLA, or "
            "sliding-window KV; " + type(spec).__name__ + " is unsupported"
        )
    assert isinstance(spec, AttentionSpec)
    if (
        spec.tokens_per_state != 1
        or spec.num_head_slots is not None
        or spec.state_content_bytes is not None
        or spec.kv_quant_mode != KVQuantMode.NONE
        or spec.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or spec.page_size_padded is not None
    ):
        raise ValueError("standalone runtime requires dense unquantized KV pages")
    if isinstance(spec, FullAttentionSpec) and (
        spec.non_causal or spec.attention_chunk_size is not None
    ):
        raise ValueError("standalone runtime requires causal token-addressed KV")
    if isinstance(spec, MLAAttentionSpec) and (
        spec.num_kv_heads != 1
        or spec.head_size_v != 0
        or spec.model_version is not None
        or spec.topk_buffer_width is not None
        or spec.is_index_group_leader
        or spec.cache_role != SparseCacheRole.SPARSE
        or spec.non_causal_multi_token_decode
        or spec.sliding_window is not None
    ):
        raise ValueError("standalone runtime supports dense MLA without an indexer")
    return spec


def select_standalone_kv_spec(specs: Mapping[str, KVCacheSpec]) -> AttentionSpec:
    """Validate the homogeneous dense cache consumed by the own-KV planner."""
    if not specs:
        raise ValueError("standalone drafter has no attention KV layers")
    first = next(iter(specs.values()))
    if any(type(spec) is not type(first) or spec != first for spec in specs.values()):
        raise ValueError("standalone runtime requires one uniform KV group")
    return _validate_standalone_kv_spec(first)


def group_standalone_kv_specs(
    specs: Mapping[str, KVCacheSpec],
    *,
    sparse_mla: SparseMLAKVContract | None = None,
) -> list[KVCacheGroupSpec]:
    """Group validated layers while sharing one logical block ID namespace."""
    if not specs:
        raise ValueError("standalone drafter has no attention KV layers")
    if sparse_mla is not None:
        sparse_mla.validate_specs(specs)
    groups: list[KVCacheGroupSpec] = []
    first: AttentionSpec | None = None
    for name, unvalidated in specs.items():
        spec: AttentionSpec
        if sparse_mla is not None:
            assert isinstance(unvalidated, MLAAttentionSpec)
            spec = unvalidated
        else:
            spec = _validate_standalone_kv_spec(unvalidated)
        if first is None:
            first = spec
        elif sparse_mla is None and (
            spec.block_size != first.block_size or spec.dtype != first.dtype
        ):
            raise ValueError(
                "standalone KV groups require a common block size and dtype"
            )
        elif isinstance(spec, MLAAttentionSpec) != isinstance(first, MLAAttentionSpec):
            raise ValueError("standalone runtime cannot mix MLA and dense attention")
        for group in groups:
            if spec == group.kv_cache_spec:
                group.layer_names.append(name)
                break
        else:
            groups.append(KVCacheGroupSpec([name], spec, is_eagle_group=True))
    return groups


def make_standalone_kv_config(
    spec: AttentionSpec,
    layer_names: list[str],
    kv_cache_bytes: int,
    layout: KVCacheLayout,
) -> KVCacheConfig:
    """Describe disjoint layer ranges for the shared KV allocator."""
    if not layer_names or len(set(layer_names)) != len(layer_names):
        raise ValueError("standalone KV layer names must be nonempty and unique")
    return make_grouped_standalone_kv_config(
        dict.fromkeys(layer_names, spec), kv_cache_bytes, layout
    )


def make_grouped_standalone_kv_config(
    specs: Mapping[str, KVCacheSpec],
    kv_cache_bytes: int,
    layout: KVCacheLayout,
    *,
    sparse_mla: SparseMLAKVContract | None = None,
) -> KVCacheConfig:
    """Reserve every logical block in every layer, without cross-group aliasing."""
    groups = group_standalone_kv_specs(specs, sparse_mla=sparse_mla)
    if not layout.is_layer_compact:
        raise ValueError("standalone runtime requires layer-compact KV layout")
    page_bytes = sum(spec.page_size_bytes for spec in specs.values())
    num_blocks = kv_cache_bytes // page_bytes
    if num_blocks < 2:
        raise ValueError("draft KV budget cannot hold null block plus one block")
    tensors = []
    offset = 0
    for group in groups:
        spec = group.kv_cache_spec
        layer_stride, block_stride, *_ = compute_layout_strides(
            spec, num_blocks, len(group.layer_names), layout
        )
        if block_stride % spec.get_block_stride_alignment():
            raise ValueError("standalone KV layout does not satisfy block alignment")
        # Native allocation uses one backing buffer for all descriptors. Each
        # group owns a disjoint range even when it uses the same logical block.
        tensors.append(
            KVCacheTensor(
                size=page_bytes * num_blocks,
                layers=list(group.layer_names),
                layer_stride=layer_stride,
                block_stride=block_stride,
                offset=offset,
            )
        )
        offset += spec.page_size_bytes * num_blocks * len(group.layer_names)
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        kv_cache_layout=layout.name,
    )
