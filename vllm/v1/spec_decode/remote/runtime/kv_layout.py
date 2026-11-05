# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense own-KV layouts for the standalone draft planner.

Sliding-window attention keeps a full-context block table here. The attention
backend applies its window; the planner does not evict or recycle old positions.
"""

from collections.abc import Mapping

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
) -> list[KVCacheGroupSpec]:
    """Group dense layers while sharing one logical block ID namespace."""
    if not specs:
        raise ValueError("standalone drafter has no attention KV layers")
    groups: list[KVCacheGroupSpec] = []
    first: AttentionSpec | None = None
    for name, unvalidated in specs.items():
        spec = _validate_standalone_kv_spec(unvalidated)
        if first is None:
            first = spec
        elif spec.block_size != first.block_size or spec.dtype != first.dtype:
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
) -> KVCacheConfig:
    """Reserve every logical block in every layer, without cross-group aliasing."""
    groups = group_standalone_kv_specs(specs)
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
