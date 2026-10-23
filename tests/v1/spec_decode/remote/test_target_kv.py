# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Snapshot ownership and absolute page mapping, without loading a model.

The CPU oracle resolves each attended token through independently constructed
source and receiver block tables. The optional CUDA case checks copy completion
and cross-stream allocation lifetime, not model or IPC integration.
"""

from dataclasses import replace

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    KVQuantMode,
    MLAAttentionSpec,
    SlidingWindowSpec,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.spec_decode.remote import target_kv
from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.target_kv import (
    DraftPosition,
    SnapshotBudget,
    SnapshotIdentity,
    TargetKVSource,
    gemma4_source_layers,
    import_target_kv,
    import_target_kv_batch,
    select_draft_position,
    snapshot_target_kv,
)
from vllm.v1.worker.block_table import get_block_table_width

IDENTITY = SnapshotIdentity(SequenceKey("verifier", 8, 3), 2, 7)
BUDGET = SnapshotBudget(max_pages=100, max_bytes=1 << 20, max_model_len=128)


def _source(
    *,
    window=None,
    block_size=4,
    num_kv_heads=2,
    head_size=2,
    name="model.layers.2.self_attn.attn",
    layout=KVCacheLayout.LBHNC,
    device="cpu",
):
    args = dict(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
    )
    spec = (
        FullAttentionSpec(**args)
        if window is None
        else SlidingWindowSpec(**args, sliding_window=window)
    )
    cache = torch.full(
        (6, num_kv_heads, block_size, 2 * head_size),
        -99,
        dtype=spec.dtype,
        device=device,
    )
    table = torch.tensor([3, 1, 5, 2, 4], dtype=torch.int32, device=device)
    for logical, physical in enumerate(table.tolist()):
        for token in range(block_size):
            cache[physical, :, token] = logical * block_size + token + 1
    if layout == KVCacheLayout.LBNHC:
        cache = cache.transpose(1, 2).contiguous().transpose(1, 2)
    return TargetKVSource(name, spec, cache, table, layout)


def _import(snapshot, source, **kwargs):
    return import_target_kv(
        snapshot,
        expected_identity=IDENTITY,
        expected_materialized_length=kwargs.pop(
            "expected_materialized_length", snapshot.materialized_length
        ),
        expected_specs={source.layer_name: source.spec},
        budget=kwargs.pop("budget", BUDGET),
        device=kwargs.pop("device", torch.device("cpu")),
        **kwargs,
    )


def _token(cache, table, position, block_size):
    return cache[int(table[position // block_size]), :, position % block_size]


@pytest.mark.parametrize("lengths", [(17, 3), (9, 17, 3, 12)])
def test_batch_import_preserves_ragged_rows_and_disjoint_page_ownership(lengths):
    sources = []
    snapshots = []
    for row, length in enumerate(lengths):
        pair = (_source(name="full"), _source(name="window", window=5))
        for source in pair:
            source.cache.add_(100 * row)
        identity = SnapshotIdentity(SequenceKey("batch", row, 1), 2, 7)
        snapshots.append(snapshot_target_kv(identity, length, pair, budget=BUDGET))
        sources.append(pair)
    imported = import_target_kv_batch(
        snapshots,
        expected_identities=tuple(snapshot.identity for snapshot in snapshots),
        expected_materialized_lengths=lengths,
        expected_specs={source.layer_name: source.spec for source in sources[0]},
        budget=BUDGET,
        device=torch.device("cpu"),
    )
    assert imported.materialized_lengths == lengths
    for layer_index, layer in enumerate(imported.layers):
        assert layer.block_table.shape[0] == len(lengths)
        assert not layer.cache[0].count_nonzero()
        owned: set[int] = set()
        for row, length in enumerate(lengths):
            source = sources[row][layer_index]
            start = max(0, length - source.spec.sliding_window) if layer_index else 0
            table = layer.block_table[row]
            live = set(table[table > 0].tolist())
            assert not owned.intersection(live)
            owned.update(live)
            for position in range(length):
                actual = _token(layer.cache, table, position, source.spec.block_size)
                if position < start:
                    assert not actual.count_nonzero()
                else:
                    torch.testing.assert_close(
                        actual,
                        _token(
                            source.cache,
                            source.block_table,
                            position,
                            source.spec.block_size,
                        ),
                        rtol=0,
                        atol=0,
                    )
            last_page = layer.cache[int(table[(length - 1) // source.spec.block_size])]
            tail = length % source.spec.block_size
            if tail:
                assert not last_page[:, tail:].count_nonzero()
        assert owned == set(range(1, layer.cache.shape[0]))
        expected = layer.cache.clone()
        with torch.inference_mode():
            for snapshot in snapshots:
                snapshot.layers[layer_index].pages.fill_(-1)
        torch.testing.assert_close(layer.cache, expected, rtol=0, atol=0)


def test_batch_import_budget_includes_all_rows_and_padded_block_tables():
    source = _source()
    identities = tuple(
        SnapshotIdentity(SequenceKey("batch", row, 1), 2, 7) for row in range(2)
    )
    lengths = (1, 17)
    snapshots = tuple(
        snapshot_target_kv(identity, length, (source,), budget=BUDGET)
        for identity, length in zip(identities, lengths)
    )
    kwargs = dict(
        expected_identities=identities,
        expected_materialized_lengths=lengths,
        expected_specs={source.layer_name: source.spec},
        device=torch.device("cpu"),
    )
    imported = import_target_kv_batch(snapshots, budget=BUDGET, **kwargs)
    assert imported.num_pages == 1 + 1 + 5
    assert imported.nbytes == (
        imported.num_pages * source.spec.unpadded_page_size_bytes
        + 2 * get_block_table_width(5, source.spec.block_size) * 4
    )
    exact = SnapshotBudget(imported.num_pages, imported.nbytes, 128)
    assert (
        import_target_kv_batch(snapshots, budget=exact, **kwargs).nbytes
        == exact.max_bytes
    )
    for budget in (
        replace(exact, max_pages=exact.max_pages - 1),
        replace(exact, max_bytes=exact.max_bytes - 1),
    ):
        with pytest.raises(ValueError, match="budget"):
            import_target_kv_batch(snapshots, budget=budget, **kwargs)


def test_batch_import_rejects_duplicate_identity_or_stale_sibling():
    source = _source()
    identity = SnapshotIdentity(SequenceKey("batch", 1, 1), 2, 7)
    snapshot = snapshot_target_kv(identity, 7, (source,), budget=BUDGET)
    kwargs = dict(
        expected_materialized_lengths=(7, 7),
        expected_specs={source.layer_name: source.spec},
        budget=BUDGET,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="distinct"):
        import_target_kv_batch(
            (snapshot, snapshot), expected_identities=(identity, identity), **kwargs
        )
    sibling = SnapshotIdentity(SequenceKey("batch", 2, 1), 2, 7)
    with pytest.raises(ValueError, match="identity"):
        import_target_kv_batch(
            (snapshot, snapshot), expected_identities=(identity, sibling), **kwargs
        )


def test_gemma_mapping_uses_last_non_shared_owner_of_each_type():
    assert gemma4_source_layers(
        ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        ["full_attention", "sliding_attention", "full_attention"],
        num_kv_shared_layers=1,
    ) == (1, 2, 1)


@pytest.mark.parametrize(
    "target,draft,shared",
    [
        (["sliding_attention"], ["full_attention"], 0),
        (["full_attention"], ["full_attention"], 1),
        (["linear_attention"], ["full_attention"], 0),
        (["full_attention"], [], 0),
    ],
)
def test_gemma_mapping_rejects_missing_or_unsupported_owners(target, draft, shared):
    with pytest.raises(ValueError):
        gemma4_source_layers(target, draft, num_kv_shared_layers=shared)


@pytest.mark.parametrize("rejected", [0, 1, 2])
def test_selected_query_excludes_rejected_target_positions(rejected):
    assert select_draft_position(
        query_start=5,
        query_length=3,
        sequence_length=11,
        num_rejected=rejected,
        num_sampled=3 - rejected,
    ) == DraftPosition(7 - rejected, 10 - rejected, 11 - rejected)


def test_unfinished_prefill_does_not_create_a_snapshot_query():
    assert (
        select_draft_position(
            query_start=0,
            query_length=4,
            sequence_length=8,
            num_rejected=0,
            num_sampled=0,
        )
        is None
    )
    with pytest.raises(ValueError):
        select_draft_position(
            query_start=0,
            query_length=4,
            sequence_length=8,
            num_rejected=1,
            num_sampled=0,
        )


@pytest.mark.parametrize("layout", [KVCacheLayout.LBHNC, KVCacheLayout.LBNHC])
@pytest.mark.parametrize("window", [None, 1, 3, 5])
@pytest.mark.parametrize("block_size", [2, 4])
def test_full_and_window_snapshots_match_independent_per_token_oracle(
    layout, window, block_size
):
    source = _source(window=window, block_size=block_size, layout=layout)
    for length in range(1, 5 * block_size + 1):
        snapshot = snapshot_target_kv(IDENTITY, length, [source], budget=BUDGET)
        imported = _import(snapshot, source)
        actual = imported.layers[0]
        start = 0 if window is None else max(0, length - window)
        logical_pages = tuple(
            range(start // block_size, (length - 1) // block_size + 1)
        )
        assert snapshot.layers[0].logical_pages == logical_pages
        assert snapshot.num_pages == len(logical_pages)
        assert imported.num_pages == snapshot.num_pages + 1
        assert torch.count_nonzero(actual.cache[0]) == 0
        assert torch.count_nonzero(actual.block_table[: start // block_size]) == 0
        for position in range(start, length):
            torch.testing.assert_close(
                _token(actual.cache, actual.block_table, position, block_size),
                _token(source.cache, source.block_table, position, block_size),
            )
        # Never retain rejected/padding tails or the unused prefix of a window page.
        for i, logical in enumerate(logical_pages):
            for offset in range(block_size):
                if not start <= logical * block_size + offset < length:
                    assert (
                        torch.count_nonzero(snapshot.layers[0].pages[i, :, offset]) == 0
                    )
                    assert torch.count_nonzero(actual.cache[i + 1, :, offset]) == 0


def test_sliding_snapshot_ignores_evicted_pages_but_preserves_absolute_indices():
    source = _source(window=3)
    source.block_table[:2] = 0
    snapshot = snapshot_target_kv(IDENTITY, 12, [source], budget=BUDGET)
    assert snapshot.layers[0].logical_pages == (2,)
    imported = _import(snapshot, source)
    assert imported.layers[0].block_table[:3].tolist() == [0, 0, 1]
    assert torch.count_nonzero(imported.layers[0].block_table[3:]) == 0
    with pytest.raises(ValueError, match="live target-KV page"):
        snapshot_target_kv(IDENTITY, 10, [source], budget=BUDGET)


def test_target_reuse_snapshot_mutation_and_sibling_writes_are_isolated():
    source = _source()
    snapshot = snapshot_target_kv(IDENTITY, 7, [source], budget=BUDGET)
    saved = snapshot.layers[0].pages.clone()
    source.cache.fill_(123)
    source.block_table.zero_()
    torch.testing.assert_close(snapshot.layers[0].pages, saved)
    first = _import(snapshot, source)
    second = _import(snapshot, source)
    saved_receiver = second.layers[0].cache.clone()
    with torch.inference_mode():
        snapshot.layers[0].pages.fill_(456)
        first.layers[0].cache.fill_(789)
        first.layers[0].block_table.zero_()
    torch.testing.assert_close(second.layers[0].cache, saved_receiver)
    assert second.layers[0].block_table[:2].tolist() == [1, 2]


@pytest.mark.parametrize(
    "identity",
    [
        replace(IDENTITY, session_epoch=3),
        replace(IDENTITY, round_id=8),
        replace(IDENTITY, key=SequenceKey("verifier", 8, 4)),
        replace(IDENTITY, key=SequenceKey("verifier", 9, 3)),
        replace(IDENTITY, key=SequenceKey("another-verifier", 8, 3)),
    ],
)
def test_import_fences_exact_sequence_generation_epoch_and_round(identity):
    source = _source()
    snapshot = snapshot_target_kv(identity, 1, [source], budget=BUDGET)
    with pytest.raises(ValueError, match="identity"):
        _import(snapshot, source)


def test_import_rejects_changed_prefix_length_even_with_identical_page_count(
    monkeypatch,
):
    source = _source()
    snapshot = snapshot_target_kv(IDENTITY, 7, [source], budget=BUDGET)
    snapshot = replace(snapshot, materialized_length=6)
    monkeypatch.setattr(
        torch,
        "zeros",
        lambda *a, **k: pytest.fail("allocated before length validation"),
    )
    with pytest.raises(ValueError, match="expected query"):
        _import(snapshot, source, expected_materialized_length=7)


def test_snapshot_and_receiver_budgets_include_different_owned_allocations():
    source = _source(window=3)
    snapshot = snapshot_target_kv(IDENTITY, 12, [source], budget=BUDGET)
    assert snapshot.nbytes == source.spec.unpadded_page_size_bytes
    exact = SnapshotBudget(1, snapshot.nbytes, 12)
    snapshot_target_kv(IDENTITY, 12, [source], budget=exact)
    with pytest.raises(ValueError, match="budget"):
        snapshot_target_kv(
            IDENTITY, 12, [source], budget=replace(exact, max_bytes=exact.max_bytes - 1)
        )
    imported = _import(snapshot, source)
    assert imported.layers[0].block_table.numel() == 32  # 128 tokens / block size 4.
    assert imported.nbytes == snapshot.nbytes * 2 + 32 * 4
    _import(snapshot, source, budget=SnapshotBudget(2, imported.nbytes, 12))
    with pytest.raises(ValueError, match="budget"):
        _import(snapshot, source, budget=exact)
    with pytest.raises(ValueError, match="length limit"):
        snapshot_target_kv(
            IDENTITY, 12, [source], budget=replace(BUDGET, max_model_len=11)
        )


def test_all_sources_are_validated_before_copy_allocation(monkeypatch):
    full, window = _source(), _source(window=3, name="window")
    window.block_table[1] = 0
    monkeypatch.setattr(
        torch, "zeros", lambda *a, **k: pytest.fail("allocated before validation")
    )
    with pytest.raises(ValueError, match="live target-KV page"):
        snapshot_target_kv(IDENTITY, 7, [full, window], budget=BUDGET)


def test_exact_receiver_budget_does_not_allocate_an_unbudgeted_index_tensor(
    monkeypatch,
):
    source = _source(window=3)
    snapshot = snapshot_target_kv(IDENTITY, 12, [source], budget=BUDGET)
    expected_bytes = snapshot.nbytes * 2 + 32 * 4
    original_zeros = torch.zeros
    allocated: list[int] = []

    def record_allocation(*args, **kwargs):
        tensor = original_zeros(*args, **kwargs)
        allocated.append(tensor.numel() * tensor.element_size())
        return tensor

    monkeypatch.setattr(torch, "zeros", record_allocation)
    monkeypatch.setattr(
        torch, "arange", lambda *a, **k: pytest.fail("unbudgeted index tensor")
    )
    imported = _import(snapshot, source, budget=SnapshotBudget(2, expected_bytes, 12))
    assert sum(allocated) == imported.nbytes == expected_bytes
    assert imported.layers[0].block_table[:3].tolist() == [0, 0, 1]
    assert torch.count_nonzero(imported.layers[0].block_table[3:]) == 0


@pytest.mark.parametrize("layout", [KVCacheLayout.LBHNC, KVCacheLayout.LBNHC])
@pytest.mark.parametrize("window", [None, 32])
def test_native_allocator_nonzero_layer_view_is_snapshotted_without_other_layers(
    layout, window
):
    from vllm.v1.worker.utils import allocate_kv_cache

    source = _source(block_size=16, head_size=256, window=window, layout=layout)
    names = ["owner.0", "owner.1"]
    spec = source.spec
    layer_stride, block_stride, *_ = compute_layout_strides(spec, 6, 2, layout)
    config = KVCacheConfig(
        num_blocks=6,
        kv_cache_tensors=[
            KVCacheTensor(
                size=12 * spec.page_size_bytes,
                layers=names,
                layer_stride=layer_stride,
                block_stride=block_stride,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(names, spec)],
        kv_cache_layout=layout.name,
    )
    caches = allocate_kv_cache(config, torch.device("cpu"), layout)
    caches["owner.0"].fill_(-11)
    caches["owner.1"].copy_(source.cache)
    assert caches["owner.1"].storage_offset() > 0
    source = replace(source, layer_name="owner.1", cache=caches["owner.1"])
    snapshot = snapshot_target_kv(IDENTITY, 35, [source], budget=BUDGET)
    imported = _import(snapshot, source)
    start = 0 if window is None else 35 - window
    for position in range(start, 35):
        receiver = imported.layers[0]
        torch.testing.assert_close(
            _token(receiver.cache, receiver.block_table, position, 16),
            _token(source.cache, source.block_table, position, 16),
        )
    caches["owner.0"].fill_(22)
    caches["owner.1"].fill_(33)
    assert not torch.any(snapshot.layers[0].pages == -11)
    assert (
        imported.layers[0].cache.untyped_storage().data_ptr()
        != caches["owner.1"].untyped_storage().data_ptr()
    )


def _copy_operation(phase, sources, snapshot):
    if phase == "snapshot":
        return snapshot_target_kv(IDENTITY, 7, sources, budget=BUDGET)
    return import_target_kv(
        snapshot,
        expected_identity=IDENTITY,
        expected_materialized_length=7,
        expected_specs={s.layer_name: s.spec for s in sources},
        budget=BUDGET,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("phase", ["snapshot", "import"])
def test_later_allocation_oom_cannot_leave_reads_of_source_state(phase, monkeypatch):
    sources = [_source(), _source(window=3, name="window")]
    snapshot = snapshot_target_kv(IDENTITY, 7, sources, budget=BUDGET)
    original_zeros = torch.zeros
    calls = 0

    def fail_later_allocation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == (2 if phase == "snapshot" else 3):
            raise torch.OutOfMemoryError("injected later destination allocation")
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", fail_later_allocation)
    monkeypatch.setattr(
        target_kv,
        "_copy_live_tokens",
        lambda *a, **k: pytest.fail("source read before all allocations completed"),
    )
    with pytest.raises(torch.OutOfMemoryError):
        _copy_operation(phase, sources, snapshot)


@pytest.mark.parametrize("phase", ["snapshot", "import"])
@pytest.mark.parametrize("fatal", [False, True])
def test_copy_failure_fences_submitted_work_unless_cuda_context_is_fatal(
    phase, fatal, monkeypatch
):
    sources = [_source(), _source(window=3, name="window")]
    snapshot = snapshot_target_kv(IDENTITY, 7, sources, budget=BUDGET)
    copied: list[bool] = []
    completed: list[torch.device] = []
    original_copy = target_kv._copy_live_tokens
    error = (
        RuntimeError("CUDA error: an illegal memory access was encountered")
        if fatal
        else torch.OutOfMemoryError("injected copy allocation failure")
    )

    def fail_second_copy(*args, **kwargs):
        if copied:
            raise error
        original_copy(*args, **kwargs)
        copied.append(True)

    monkeypatch.setattr(target_kv, "_copy_live_tokens", fail_second_copy)
    monkeypatch.setattr(
        target_kv, "_complete_copies", lambda device: completed.append(device)
    )
    with pytest.raises(type(error)) as caught:
        _copy_operation(phase, sources, snapshot)
    assert caught.value is error
    assert len(copied) == 1
    assert len(completed) == (0 if fatal else 1)


def test_multilayer_budget_is_total_and_import_requires_exact_layer_contract():
    full, window = (
        _source(),
        _source(window=3, name="window", num_kv_heads=1, head_size=4),
    )
    snapshot = snapshot_target_kv(IDENTITY, 7, [full, window], budget=BUDGET)
    assert snapshot.num_pages == 3
    with pytest.raises(ValueError, match="budget"):
        snapshot_target_kv(
            IDENTITY, 7, [full, window], budget=replace(BUDGET, max_pages=2)
        )
    with pytest.raises(ValueError, match="layers"):
        _import(snapshot, full)
    imported = import_target_kv(
        snapshot,
        expected_identity=IDENTITY,
        expected_materialized_length=7,
        expected_specs={s.layer_name: s.spec for s in (full, window)},
        budget=BUDGET,
        device=torch.device("cpu"),
    )
    assert imported.num_pages == 5
    assert imported.layers[0].cache.shape[1:] == (2, 4, 4)
    assert imported.layers[1].cache.shape[1:] == (1, 4, 8)


def test_receiver_scrubs_unattended_snapshot_padding_even_if_payload_is_modified():
    source = _source(window=3)
    snapshot = snapshot_target_kv(IDENTITY, 6, [source], budget=BUDGET)
    # Only positions [3, 6) are readable; poison the unused portions of both pages.
    with torch.inference_mode():
        snapshot.layers[0].pages[0, :, :3].fill_(999)
        snapshot.layers[0].pages[1, :, 2:].fill_(999)
    imported = _import(snapshot, source)
    assert torch.count_nonzero(imported.layers[0].cache[1, :, :3]) == 0
    assert torch.count_nonzero(imported.layers[0].cache[2, :, 2:]) == 0


@pytest.mark.parametrize(
    "mutation", ["dtype", "shape", "strides", "layout", "null", "alias", "range"]
)
def test_source_contract_mismatches_fail_closed(mutation):
    source = _source()
    if mutation == "dtype":
        source = replace(source, cache=source.cache.float())
    elif mutation == "shape":
        source = replace(source, cache=source.cache[..., :2])
    elif mutation == "strides":
        source = replace(source, layout=KVCacheLayout.LBNHC)
    elif mutation == "layout":
        source = replace(source, layout=KVCacheLayout.LHBNC)
    elif mutation == "null":
        source.block_table[1] = 0
    elif mutation == "alias":
        source.block_table[1] = source.block_table[0]
    else:
        source.block_table[1] = 99
    with pytest.raises(ValueError):
        snapshot_target_kv(IDENTITY, 7, [source], budget=BUDGET)


@pytest.mark.parametrize(
    "changes",
    [
        dict(dtype=torch.float32),
        dict(kv_quant_mode=KVQuantMode.FP8_PER_TENSOR),
        dict(head_size_v=4),
        dict(tokens_per_state=2),
        dict(state_content_bytes=16),
        dict(page_size_padded=1024),
        dict(non_causal=True),
        dict(attention_chunk_size=4),
    ],
)
def test_unsupported_attention_specs_are_not_treated_as_dense_bf16(changes):
    source = _source()
    with pytest.raises(ValueError):
        snapshot_target_kv(
            IDENTITY,
            1,
            [replace(source, spec=replace(source.spec, **changes))],
            budget=BUDGET,
        )


def test_mla_subclass_is_not_accepted_as_full_attention():
    source = _source()
    mla = MLAAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=4, dtype=torch.bfloat16
    )
    with pytest.raises(ValueError, match="plain full or sliding"):
        snapshot_target_kv(IDENTITY, 1, [replace(source, spec=mla)], budget=BUDGET)


@pytest.mark.parametrize("mutation", ["indices", "shape", "dtype", "spec"])
def test_import_validates_payload_before_allocating(monkeypatch, mutation):
    source = _source(window=3)
    snapshot = snapshot_target_kv(IDENTITY, 12, [source], budget=BUDGET)
    layer = snapshot.layers[0]
    if mutation == "indices":
        layer = replace(layer, logical_pages=(0,))
    elif mutation == "shape":
        layer = replace(layer, pages=layer.pages[..., :2])
    elif mutation == "dtype":
        layer = replace(layer, pages=layer.pages.float())
    else:
        layer = replace(layer, spec=replace(layer.spec, sliding_window=4))
    snapshot = replace(snapshot, layers=(layer,))
    monkeypatch.setattr(
        torch, "zeros", lambda *a, **k: pytest.fail("allocated before validation")
    )
    with pytest.raises(ValueError):
        _import(snapshot, source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_copies_complete_before_return_and_receiver_survives_other_stream_use():
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(producer):
        source = _source(window=5, device="cuda")
        snapshot = snapshot_target_kv(IDENTITY, 9, [source], budget=BUDGET)
        expected = snapshot.layers[0].pages.cpu()
        source.cache.fill_(77)
    with pytest.raises(ValueError, match="same CUDA device"):
        _import(snapshot, source, device=torch.device("cpu"))
    # snapshot_target_kv completed its copy even though source reuse is pending.
    with torch.cuda.stream(consumer):
        imported = _import(snapshot, source, device=torch.device("cuda"))
        imported.record_stream(consumer)
        result = imported.layers[0].cache[1:].clone()
        del imported
    consumer.synchronize()
    torch.testing.assert_close(result.cpu(), expected)
    producer.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("window,head_size", [(None, 64), (17, 64), (1024, 256)])
@pytest.mark.parametrize("rejected", [0, 1, 2])
def test_cuda_triton_q_only_attention_matches_native_query_after_rejection(
    window, head_size, rejected
):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    source = _source(window=window, block_size=16, head_size=head_size, device="cuda")
    table_width = get_block_table_width(
        source.block_table.numel(),
        source.spec.block_size,
        token_alignment=source.spec.block_table_token_alignment,
    )
    native_table = torch.zeros(table_width, dtype=torch.int32, device="cuda")
    native_table[: source.block_table.numel()].copy_(source.block_table)
    source = replace(source, block_table=native_table)
    source.cache.copy_(
        torch.arange(source.cache.numel(), device="cuda")
        .float()
        .sin()
        .view_as(source.cache)
    )
    queries = torch.linspace(
        -1, 1, 3 * 2 * head_size, device="cuda", dtype=torch.bfloat16
    )
    queries = queries.reshape(3, 2, head_size)
    selection = select_draft_position(
        query_start=0,
        query_length=3,
        sequence_length=35,
        num_rejected=rejected,
        num_sampled=3 - rejected,
    )
    assert selection is not None
    snapshot = snapshot_target_kv(
        IDENTITY, selection.materialized_length, [source], budget=BUDGET
    )
    imported = _import(snapshot, source, device=torch.device("cuda"))

    def attend(query, cache, table, length):
        keys, values = cache.transpose(1, 2).split(head_size, dim=-1)
        output = torch.empty_like(query)
        unified_attention(
            q=query,
            k=keys,
            v=values,
            out=output,
            cu_seqlens_q=torch.tensor(
                [0, query.shape[0]], dtype=torch.int32, device="cuda"
            ),
            max_seqlen_q=query.shape[0],
            seqused_k=torch.tensor([length], dtype=torch.int32, device="cuda"),
            max_seqlen_k=length,
            softmax_scale=head_size**-0.5,
            causal=True,
            window_size=(-1, -1) if window is None else (window - 1, 0),
            block_table=table.unsqueeze(0),
            softcap=0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
        )
        return output

    native = attend(queries, source.cache, source.block_table, 35)
    receiver = imported.layers[0]
    copied = attend(
        queries[selection.hidden_index : selection.hidden_index + 1],
        receiver.cache,
        receiver.block_table,
        selection.materialized_length,
    )
    torch.testing.assert_close(
        copied[0], native[selection.hidden_index], rtol=1e-2, atol=1.5e-2
    )
