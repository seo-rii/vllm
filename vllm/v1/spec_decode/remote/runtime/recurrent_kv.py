# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token pages and fixed recurrent slots in one native backing allocation."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.spec_decode.remote.runtime.kv_layout import group_standalone_kv_specs

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(kw_only=True)
class RecurrentKVCacheConfig(KVCacheConfig):
    """Keep native attention-page capacity separate from running state slots."""

    recurrent_layers: frozenset[str]
    recurrent_num_blocks: int

    def num_blocks_of(self, tensor: KVCacheTensor) -> int:
        known = {name for group in self.kv_cache_groups for name in group.layer_names}
        if (
            tensor.host_resident
            or not tensor.layers
            or any(name not in known for name in tensor.layers)
        ):
            raise ValueError(
                "recurrent KV requires known device-only layer descriptors"
            )
        kinds = {name in self.recurrent_layers for name in tensor.layers}
        if len(kinds) != 1:
            raise ValueError("one KV descriptor cannot mix token pages and state slots")
        if True in kinds:
            return self.recurrent_num_blocks
        return super().num_blocks_of(tensor)


def make_recurrent_kv_config(
    config: "VllmConfig",
    specs: Mapping[str, KVCacheSpec],
    kv_cache_bytes: int,
    max_num_reqs: int,
    layout: KVCacheLayout,
) -> RecurrentKVCacheConfig:
    """Reserve one running state per slot before budgeting attention token pages."""
    model = config.model_config
    pattern = getattr(model.hf_text_config, "hybrid_override_pattern", "")
    if (
        model.architecture != "NemotronHForCausalLM"
        or model.quantization is not None
        or model.dtype != torch.bfloat16
        or not isinstance(pattern, str)
        or len(pattern) != model.get_total_num_hidden_layers()
        or any(kind not in "M*-" for kind in pattern)
        or getattr(model.hf_text_config, "num_nextn_predict_layers", 0) != 0
        or config.speculative_config is not None
        or config.parallel_config.tensor_parallel_size != 1
        or config.cache_config.mamba_cache_mode != "none"
        or config.cache_config.use_replayssm
        or config.cache_config.use_kda_recoverssm
        or type(kv_cache_bytes) is not int
        or kv_cache_bytes <= 0
        or max_num_reqs != config.scheduler_config.max_num_seqs
        or max_num_reqs <= 0
        or not layout.is_layer_compact
    ):
        raise ValueError("recurrent KV requires an ordinary native NemotronH config")
    attention: dict[str, KVCacheSpec] = {}
    recurrent: dict[str, MambaSpec] = {}
    indices: set[int] = set()
    for name, spec in specs.items():
        index = extract_layer_index(name)
        if index in indices or not 0 <= index < len(pattern):
            raise ValueError("recurrent KV layer indices do not match the model")
        indices.add(index)
        if type(spec) is FullAttentionSpec and pattern[index] == "*":
            if (
                spec.dtype != torch.bfloat16
                or spec.block_size != config.cache_config.block_size
                or spec.num_kv_heads != model.get_num_kv_heads(config.parallel_config)
                or spec.head_size != model.get_head_size()
            ):
                raise ValueError("loaded recurrent draft attention geometry differs")
            attention[name] = spec
        elif type(spec) is MambaSpec and pattern[index] == "M":
            if (
                spec.mamba_type != MambaAttentionBackendEnum.MAMBA2
                or spec.mamba_cache_mode != "none"
                or spec.num_speculative_blocks != 0
                or spec.tp_replicated
                or spec.block_size != config.cache_config.mamba_block_size
                or spec.num_heads != 1
                or spec.get_num_kernel_states(spec.block_size) != 1
                or spec.dtypes != (torch.bfloat16, torch.float32)
                or len(spec.shapes) != 2
                or any(
                    not shape or any(type(dim) is not int or dim <= 0 for dim in shape)
                    for shape in spec.shapes
                )
            ):
                raise ValueError(
                    "loaded recurrent draft requires ordinary Mamba2 state"
                )
            recurrent[name] = spec
        else:
            raise ValueError("loaded recurrent draft has an unsupported cache layer")
    expected = {i for i, kind in enumerate(pattern) if kind in "M*"}
    if not attention or not recurrent or indices != expected:
        raise ValueError("loaded recurrent draft is missing attention or Mamba layers")
    groups = group_standalone_kv_specs(attention)
    for name, spec in recurrent.items():
        for group in groups:
            if group.kv_cache_spec == spec:
                group.layer_names.append(name)
                break
        else:
            groups.append(KVCacheGroupSpec([name], spec))
    recurrent_blocks = max_num_reqs + 1
    recurrent_bytes = recurrent_blocks * sum(
        s.page_size_bytes for s in recurrent.values()
    )
    attention_page_bytes = sum(s.page_size_bytes for s in attention.values())
    blocks = (kv_cache_bytes - recurrent_bytes) // attention_page_bytes
    if blocks < 2:
        raise ValueError(
            "draft KV budget cannot hold recurrent slots and two token pages"
        )
    total_bytes = recurrent_bytes + blocks * attention_page_bytes
    tensors = []
    offset = 0
    for group in groups:
        spec = group.kv_cache_spec
        capacity = recurrent_blocks if isinstance(spec, MambaSpec) else blocks
        layer_stride, block_stride, *_ = compute_layout_strides(
            spec, capacity, len(group.layer_names), layout
        )
        if block_stride % spec.get_block_stride_alignment():
            raise ValueError("recurrent KV layout does not satisfy block alignment")
        tensors.append(
            KVCacheTensor(
                size=total_bytes,
                layers=list(group.layer_names),
                offset=offset,
                layer_stride=layer_stride,
                block_stride=block_stride,
            )
        )
        offset += spec.page_size_bytes * capacity * len(group.layer_names)
    return RecurrentKVCacheConfig(
        num_blocks=blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        kv_cache_layout=layout.name,
        recurrent_layers=frozenset(recurrent),
        recurrent_num_blocks=recurrent_blocks,
    )
