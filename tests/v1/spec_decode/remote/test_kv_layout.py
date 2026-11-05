# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise standalone KV descriptors through the real CPU storage allocator."""

from dataclasses import replace

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    SparseCacheRole,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
from vllm.v1.spec_decode.remote.runtime.kv_layout import (
    group_standalone_kv_specs,
    make_grouped_standalone_kv_config,
    make_standalone_kv_config,
    select_standalone_kv_spec,
)
from vllm.v1.spec_decode.remote.runtime.step import DraftStepPlanner, DraftStepRow
from vllm.v1.worker.utils import allocate_kv_cache


def _spec(kind):
    kwargs = dict(
        block_size=4,
        num_kv_heads=2,
        head_size=3,
        dtype=torch.float32 if kind == "full" else torch.bfloat16,
    )
    if kind == "mla":
        return MLAAttentionSpec(**(kwargs | {"num_kv_heads": 1, "head_size": 5}))
    if kind == "swa":
        return SlidingWindowSpec(**kwargs, head_size_v=4, sliding_window=5)
    return FullAttentionSpec(**kwargs)


@pytest.mark.parametrize("kind", ["full", "mla", "swa"])
@pytest.mark.parametrize("layout_name", ["LBHNC", "LBNHC", "LHBNC"])
def test_dense_own_kv_layers_and_blocks_never_alias(kind, layout_name):
    spec = _spec(kind)
    names = ["draft.0", "draft.1", "draft.2"]
    assert select_standalone_kv_spec(dict.fromkeys(names, spec)) is spec
    page_group_bytes = len(names) * spec.page_size_bytes
    budget = page_group_bytes * 6 - 1
    layout = KVCacheLayout[layout_name]
    config = make_standalone_kv_config(spec, names, budget, layout)
    caches = allocate_kv_cache(config, torch.device("cpu"), layout)
    assert config.num_blocks == 5
    assert config.kv_cache_groups[0].kv_cache_spec is spec
    for layer, cache in enumerate(caches.values()):
        assert cache.shape == (
            5,
            spec.num_kv_heads,
            4,
            spec.head_size + spec.head_size_v,
        )
        assert cache.untyped_storage().nbytes() == page_group_bytes * 5
        for block in range(1, 5):
            cache[block].fill_(10 * layer + block)
    for layer, cache in enumerate(caches.values()):
        assert not cache[0].count_nonzero().item()
        for block in range(1, 5):
            assert torch.all(cache[block] == 10 * layer + block)
    assert len({cache.untyped_storage().data_ptr() for cache in caches.values()}) == 1
    assert page_group_bytes * 5 <= budget


@pytest.mark.parametrize("layout_name", ["LBHNC", "LBNHC", "LHBNC"])
@pytest.mark.parametrize("split_kernel_blocks", [False, True])
def test_mixed_attention_groups_keep_every_layer_and_logical_block_independent(
    layout_name, split_kernel_blocks
):
    full = replace(_spec("full"), dtype=torch.bfloat16)
    sliding = _spec("swa")
    wider = replace(sliding, sliding_window=9, num_kv_heads=1, head_size=7)
    specs = {"a": full, "b": sliding, "c": full, "d": wider, "e": sliding}
    page_bytes = sum(spec.page_size_bytes for spec in specs.values())
    budget = page_bytes * 6 - 1
    layout = KVCacheLayout[layout_name]
    config = make_grouped_standalone_kv_config(specs, budget, layout)
    assert [group.layer_names for group in config.kv_cache_groups] == [
        ["a", "c"],
        ["b", "e"],
        ["d"],
    ]
    assert [group.kv_cache_spec for group in config.kv_cache_groups] == [
        full,
        sliding,
        wider,
    ]
    assert config.num_blocks == 5
    kernel_sizes = [4, 2, 1] if split_kernel_blocks else [4, 4, 4]
    if split_kernel_blocks and layout_name == "LHBNC":
        # Native head-outermost storage cannot subdivide a multi-head page.
        with pytest.raises(ValueError, match="does not store blocks as dense"):
            allocate_kv_cache(config, torch.device("cpu"), layout, kernel_sizes)
        kernel_sizes[1] = 4
    caches = allocate_kv_cache(config, torch.device("cpu"), layout, kernel_sizes)
    for group, kernel_size in zip(config.kv_cache_groups, kernel_sizes):
        ratio = full.block_size // kernel_size
        for name in group.layer_names:
            cache = caches[name]
            spec = specs[name]
            assert cache.shape == (
                5 * ratio,
                spec.num_kv_heads,
                kernel_size,
                spec.head_size + spec.head_size_v,
            )
            for block in range(1, 5):
                cache[block * ratio : (block + 1) * ratio].fill_(
                    10 * list(specs).index(name) + block
                )
    for group, kernel_size in zip(config.kv_cache_groups, kernel_sizes):
        ratio = full.block_size // kernel_size
        for name in group.layer_names:
            cache = caches[name]
            assert not cache[:ratio].count_nonzero().item()
            for block in range(1, 5):
                assert torch.all(
                    cache[block * ratio : (block + 1) * ratio]
                    == 10 * list(specs).index(name) + block
                )
    storages = {
        cache.untyped_storage().data_ptr(): cache.untyped_storage().nbytes()
        for cache in caches.values()
    }
    assert list(storages.values()) == [5 * page_bytes]
    assert sum(storages.values()) <= budget
    with pytest.raises(ValueError, match="null block plus one block"):
        make_grouped_standalone_kv_config(specs, 2 * page_bytes - 1, layout)


@pytest.mark.parametrize(
    "other, message",
    [
        (replace(_spec("swa"), block_size=8), "common block size and dtype"),
        (_spec("full"), "common block size and dtype"),
        (_spec("mla"), "cannot mix MLA and dense"),
        (replace(_spec("swa"), tokens_per_state=2), "dense unquantized"),
        (replace(_spec("swa"), page_size_padded=256), "dense unquantized"),
        (
            replace(_spec("full"), dtype=torch.bfloat16, non_causal=True),
            "causal token-addressed",
        ),
    ],
)
def test_mixed_groups_do_not_bypass_per_layer_or_shared_planner_contract(
    other, message
):
    with pytest.raises(ValueError, match=message):
        group_standalone_kv_specs({"a": _spec("swa"), "b": other})


def test_grouped_cache_requires_at_least_one_attention_layer():
    with pytest.raises(ValueError, match="no attention KV layers"):
        make_grouped_standalone_kv_config({}, 4096, KVCacheLayout.LBNHC)


@pytest.mark.parametrize("kind", ["full", "mla", "swa"])
def test_each_layer_budget_reserves_a_null_block(kind):
    spec = _spec(kind)
    names = ["a"] if kind == "full" else ["a", "b"]
    page_group_bytes = len(names) * spec.page_size_bytes
    with pytest.raises(ValueError, match="null block plus one block"):
        make_standalone_kv_config(
            spec, names, 2 * page_group_bytes - 1, KVCacheLayout.LBNHC
        )
    config = make_standalone_kv_config(
        spec, names, 2 * page_group_bytes, KVCacheLayout.LBNHC
    )
    pool = DraftBlockPool(config.num_blocks, spec.block_size)
    assert pool.ensure(0, spec.block_size) == [1]
    assert pool.num_free_blocks == 0


@pytest.mark.parametrize("layout_name", ["LBHNC", "LBNHC", "LHBNC"])
def test_mla_kernel_block_splitting_preserves_latent_vectors(layout_name):
    spec = _spec("mla")
    layout = KVCacheLayout[layout_name]
    config = make_standalone_kv_config(
        spec, ["a", "b"], 6 * spec.page_size_bytes, layout
    )
    caches = allocate_kv_cache(config, torch.device("cpu"), layout, [2])
    for layer, cache in enumerate(caches.values()):
        assert cache.shape == (6, 1, 2, 5)
        for kernel_block in range(2, 6):
            cache[kernel_block].fill_(10 * layer + kernel_block)
    for layer, cache in enumerate(caches.values()):
        assert not cache[:2].count_nonzero().item()
        for kernel_block in range(2, 6):
            assert torch.all(cache[kernel_block] == 10 * layer + kernel_block)


def test_sliding_window_planner_keeps_full_context_for_rejection():
    spec = _spec("swa")
    config = make_standalone_kv_config(
        spec, ["draft"], 8 * spec.page_size_bytes, KVCacheLayout.LBNHC
    )
    pool = DraftBlockPool(config.num_blocks, spec.block_size)
    planner = DraftStepPlanner(
        pool, max_num_reqs=1, max_model_len=24, num_speculative_tokens=3
    )
    key = SequenceKey("verifier", 1, 1)
    assert planner.open(key)
    plan, failures = planner.plan([DraftStepRow(key, 12, 1)])
    assert not failures
    planner.commit(plan)
    before = pool.blocks(planner.req_index(key))
    assert len(before) == 4  # Query plus two lookahead positions, beyond the window.
    plan, failures = planner.plan([DraftStepRow(key, 3, 1, num_rejected=2)])
    assert not failures
    planner.commit(plan)
    assert planner.num_computed(key) == 13
    assert pool.blocks(planner.req_index(key)) == before
    planner.close(key)
    assert pool.num_free_blocks == config.num_blocks - 1


@pytest.mark.parametrize("kind", ["full", "mla", "swa"])
def test_mixed_or_differently_sized_layers_are_rejected(kind):
    spec = _spec(kind)
    with pytest.raises(ValueError, match="one uniform KV group"):
        select_standalone_kv_spec({"a": spec, "b": replace(spec, head_size=7)})
    with pytest.raises(ValueError, match="one uniform KV group"):
        select_standalone_kv_spec(
            {"a": spec, "b": _spec("swa" if kind != "swa" else "mla")}
        )


@pytest.mark.parametrize(
    "spec",
    [
        replace(_spec("mla"), tokens_per_state=2),
        replace(_spec("mla"), state_content_bytes=32),
        replace(_spec("swa"), page_size_padded=256),
        replace(_spec("full"), num_head_slots=3),
        replace(_spec("full"), dtype=torch.uint8),
    ],
)
def test_compressed_packed_padded_and_quantized_pages_are_rejected(spec):
    with pytest.raises(ValueError, match="dense unquantized KV pages"):
        select_standalone_kv_spec({"draft": spec})


@pytest.mark.parametrize(
    "changes",
    [
        {"model_version": "deepseek_v4"},
        {"is_index_group_leader": True},
        {"cache_role": SparseCacheRole.INDEXER},
        {"non_causal_multi_token_decode": True},
        {"sliding_window": 8},
        {"head_size_v": 3},
        {"num_kv_heads": 2},
    ],
)
def test_sparse_and_nonstandard_mla_require_a_separate_contract(changes):
    with pytest.raises(ValueError, match="dense MLA without an indexer"):
        select_standalone_kv_spec({"draft": replace(_spec("mla"), **changes)})


@pytest.mark.parametrize("spec_type", [SlidingWindowMLASpec, HiddenStateCacheSpec])
def test_other_cache_spec_subclasses_are_not_implicitly_admitted(spec_type):
    kwargs = dict(block_size=4, num_kv_heads=1, head_size=5, dtype=torch.bfloat16)
    if spec_type is SlidingWindowMLASpec:
        kwargs["sliding_window"] = 8
    with pytest.raises(ValueError, match="unsupported"):
        select_standalone_kv_spec({"draft": spec_type(**kwargs)})


@pytest.mark.parametrize("changes", [{"non_causal": True}, {"attention_chunk_size": 8}])
def test_noncausal_and_chunked_attention_are_rejected(changes):
    with pytest.raises(ValueError, match="causal token-addressed KV"):
        select_standalone_kv_spec({"draft": replace(_spec("full"), **changes)})


def test_nonempty_unique_layer_names_and_hardware_alignment_are_required():
    spec = _spec("full")
    for names in ([], ["a", "a"]):
        with pytest.raises(ValueError, match="nonempty and unique"):
            make_standalone_kv_config(spec, names, 4096, KVCacheLayout.LBNHC)
    with pytest.raises(ValueError, match="block alignment"):
        make_standalone_kv_config(
            replace(spec, block_stride_alignment=1024), ["a"], 4096, KVCacheLayout.LBNHC
        )


@pytest.mark.parametrize("kind", ["full", "mla", "swa"])
@pytest.mark.parametrize("layout_name", ["BLHNC", "BLNHC", "BHLNC"])
def test_all_cache_types_require_layer_compact_layout(kind, layout_name):
    with pytest.raises(ValueError, match="layer-compact"):
        make_standalone_kv_config(
            _spec(kind), ["draft"], 4096, KVCacheLayout[layout_name]
        )
