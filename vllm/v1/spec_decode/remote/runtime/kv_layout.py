# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense own-KV layouts for the standalone draft planner.

Sliding-window attention keeps a full-context block table here. The attention
backend applies its window; the planner does not evict or recycle old positions.
"""

from collections.abc import Mapping

import torch

from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroup,
    KVCacheLayerConfig,
    KVCacheTensor,
    KVQuantMode,
    MLAAttentionSpec,
    SparseCacheRole,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout


def select_standalone_kv_layer_config(
    layer_configs: Mapping[str, KVCacheLayerConfig],
) -> KVCacheLayerConfig:
    """Validate the homogeneous dense cache consumed by the own-KV planner."""
    if not layer_configs:
        raise ValueError("standalone drafter has no attention KV layers")
    first_layer = next(iter(layer_configs.values()))
    first = first_layer.spec
    # Subclasses include sparse/compressed caches and target-owned state with
    # different position or allocation contracts; accepting their base is unsafe.
    if type(first) not in (AttentionSpec, MLAAttentionSpec):
        raise ValueError(
            "standalone runtime requires full-attention, dense MLA, or "
            "sliding-window KV; " + type(first).__name__ + " is unsupported"
        )
    if any(
        type(layer.spec) is not type(first)
        or layer.spec != first
        or layer.manager_config != first_layer.manager_config
        for layer in layer_configs.values()
    ):
        raise ValueError("standalone runtime requires one uniform KV group")
    assert isinstance(first, AttentionSpec)
    if (
        first.tokens_per_state != 1
        or first.num_head_slots is not None
        or first.state_content_bytes is not None
        or first.kv_quant_mode != KVQuantMode.NONE
        or first.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or first.page_size_padded is not None
    ):
        raise ValueError("standalone runtime requires dense unquantized KV pages")
    if isinstance(first, MLAAttentionSpec) and (
        first.num_kv_heads != 1
        or first.head_size_v != 0
        or first.model_version is not None
        or first.is_index_group_leader
        or first.cache_role != SparseCacheRole.SPARSE
        or first.topk_buffer_width is not None
        or first.alignment is not None
        or first_layer.manager_cls is not FullAttentionManager
    ):
        raise ValueError("standalone runtime supports dense MLA without an indexer")
    if first_layer.manager_cls not in (FullAttentionManager, SlidingWindowManager):
        raise ValueError("standalone runtime requires causal token-addressed KV")
    if first_layer.manager_cls is SlidingWindowManager:
        params = first_layer.manager_config.params
        assert isinstance(params, SlidingWindowManager.Params)
        if params.bounded_replay:
            raise ValueError("standalone runtime requires causal token-addressed KV")
    return first_layer


def group_standalone_kv_specs(
    layer_configs: Mapping[str, KVCacheLayerConfig],
) -> list[KVCacheGroup]:
    """Group dense layers while sharing one logical block ID namespace."""
    if not layer_configs:
        raise ValueError("standalone drafter has no attention KV layers")
    grouped: dict[KVCacheLayerConfig, dict[str, KVCacheLayerConfig]] = {}
    first: AttentionSpec | None = None
    for name, layer in layer_configs.items():
        select_standalone_kv_layer_config({name: layer})
        spec = layer.spec
        assert isinstance(spec, AttentionSpec)
        if first is None:
            first = spec
        elif spec.block_size != first.block_size or spec.dtype != first.dtype:
            raise ValueError(
                "standalone KV groups require a common block size and dtype"
            )
        elif isinstance(spec, MLAAttentionSpec) != isinstance(first, MLAAttentionSpec):
            raise ValueError("standalone runtime cannot mix MLA and dense attention")
        grouped.setdefault(layer, {})[name] = layer
    return [
        KVCacheGroup.from_layers(layers, is_eagle_group=True)
        for layers in grouped.values()
    ]


def make_standalone_kv_config(
    layer_config: KVCacheLayerConfig,
    layer_names: list[str],
    kv_cache_bytes: int,
    layout: KVCacheLayout,
) -> KVCacheConfig:
    """Describe disjoint layer ranges for the shared KV allocator."""
    if not layer_names or len(set(layer_names)) != len(layer_names):
        raise ValueError("standalone KV layer names must be nonempty and unique")
    return make_grouped_standalone_kv_config(
        dict.fromkeys(layer_names, layer_config), kv_cache_bytes, layout
    )


def make_grouped_standalone_kv_config(
    layer_configs: Mapping[str, KVCacheLayerConfig],
    kv_cache_bytes: int,
    layout: KVCacheLayout,
) -> KVCacheConfig:
    """Reserve every logical block in every layer, without cross-group aliasing."""
    groups = group_standalone_kv_specs(layer_configs)
    if not layout.is_layer_compact:
        raise ValueError("standalone runtime requires layer-compact KV layout")
    page_bytes = sum(layer.spec.page_size_bytes for layer in layer_configs.values())
    num_blocks = kv_cache_bytes // page_bytes
    if num_blocks < 2:
        raise ValueError("draft KV budget cannot hold null block plus one block")
    tensors = []
    offset = 0
    for group in groups:
        (spec,) = group.distinct_specs
        layer_stride, block_stride, *_ = compute_layout_strides(
            spec, num_blocks, len(group.layer_names), layout
        )
        if block_stride % spec.get_block_stride_alignment():
            raise ValueError("standalone KV layout does not satisfy block alignment")
        # Every descriptor refers to the same backing buffer, with disjoint ranges.
        tensors.append(
            KVCacheTensor(
                size=page_bytes * num_blocks,
                layers=group.layer_names,
                layer_stride=layer_stride,
                block_stride=block_stride,
                offset=offset,
            )
        )
        offset += group.page_size_bytes * num_blocks
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        kv_cache_layout=layout.name,
    )
