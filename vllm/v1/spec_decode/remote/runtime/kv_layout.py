# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense own-KV layouts for the standalone draft planner.

Sliding-window attention keeps a full-context block table here. The attention
backend applies its window; the planner does not evict or recycle old positions.
"""

from collections.abc import Mapping
from dataclasses import dataclass

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

    def validate_specs(self, specs: Mapping[str, KVCacheLayerConfig]) -> None:
        if len(specs) != 2 or any(
            type(layer.spec) is not MLAAttentionSpec for layer in specs.values()
        ):
            raise ValueError("standalone sparse MLA requires one MLA/indexer pair")
        attention = [
            name
            for name, layer in specs.items()
            if isinstance(layer.spec, MLAAttentionSpec)
            and layer.spec.cache_role == SparseCacheRole.SPARSE
        ]
        if len(attention) != 1 or not attention[0].endswith(".self_attn.attn"):
            raise ValueError("standalone sparse MLA requires a native attention name")
        attention_name = attention[0]
        indexer_name = attention_name.removesuffix(".attn") + ".indexer.k_cache"
        if indexer_name not in specs:
            raise ValueError("standalone sparse MLA requires its own layer indexer")
        for name, layer in specs.items():
            spec = layer.spec
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
                or layer.manager_cls is not FullAttentionManager
                or spec.cache_dtype_str
                not in ((None,) if is_indexer else (None, "auto", "bfloat16"))
            ):
                raise ValueError(
                    "standalone sparse MLA requires native BF16/byte pages"
                )


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
    *,
    sparse_mla: SparseMLAKVContract | None = None,
) -> list[KVCacheGroup]:
    """Group dense layers while sharing one logical block ID namespace."""
    if not layer_configs:
        raise ValueError("standalone drafter has no attention KV layers")
    if sparse_mla is not None:
        sparse_mla.validate_specs(layer_configs)
    grouped: dict[KVCacheLayerConfig, dict[str, KVCacheLayerConfig]] = {}
    first: AttentionSpec | None = None
    for name, layer in layer_configs.items():
        if sparse_mla is None:
            select_standalone_kv_layer_config({name: layer})
        spec = layer.spec
        assert isinstance(spec, AttentionSpec)
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
    *,
    sparse_mla: SparseMLAKVContract | None = None,
) -> KVCacheConfig:
    """Reserve every logical block in every layer, without cross-group aliasing."""
    groups = group_standalone_kv_specs(layer_configs, sparse_mla=sparse_mla)
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
