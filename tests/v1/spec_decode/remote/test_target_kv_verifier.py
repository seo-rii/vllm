# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify native query selection, profiling and terminal fallback without CUDA."""

import errno
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config.speculative import RemoteDraftConfig, SpeculativeConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.spec_decode.contracts import DraftStep, LocalExecutionContext
from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.session import RemoteDraftError
from vllm.v1.spec_decode.remote.target_kv import (
    SnapshotBudget,
    SnapshotCompletionError,
    TargetKVSnapshot,
)
from vllm.v1.spec_decode.remote.target_kv_session import (
    TargetKVProposalHandle,
    TargetKVResult,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode import target_kv_remote as remote


class Session:
    def __init__(self, *args, failure_policy, **kwargs):
        self.policy = failure_policy
        self.records = {}
        self.epoch = 1
        self.calls = []
        self.generation = 0
        self.pending = None
        self.batch_calls = []
        self.open_calls = []

    def connect(self):
        pass

    def deadline(self):
        return 1000

    def open_sequence(self, sequence_id, **kwargs):
        key = SequenceKey("v", sequence_id, self.generation)
        self.generation += 1
        self.records[key] = True
        return key

    def open_sequences(self, sequence_ids, **kwargs):
        self.open_calls.append((sequence_ids, kwargs))
        return tuple(self.open_sequence(sequence_id) for sequence_id in sequence_ids)

    def epoch_for(self, key):
        return self.epoch

    def active(self, key):
        return self.records.get(key, False)

    def dispatch(self, snapshot, token, hidden, **kwargs):
        self.calls.append((snapshot, token, hidden.clone()))
        self.pending = object()
        return self.pending

    def dispatch_batch(self, snapshots, tokens, hidden, **kwargs):
        self.batch_calls.append((snapshots, tokens, hidden.clone(), kwargs))
        self.calls.extend(
            (snapshot, token, hidden[row : row + 1].clone())
            for row, (snapshot, token) in enumerate(zip(snapshots, tokens, strict=True))
        )
        self.pending = TargetKVProposalHandle(
            self,
            self.epoch,
            1,
            tuple(snapshot.identity for snapshot in snapshots),
            tuple(snapshot.materialized_length for snapshot in snapshots),
            kwargs["deadline"],
            kwargs["k"],
        )
        self.pending_results = tuple(
            TargetKVResult((23 + snapshot.identity.key.sequence_id,), 1)
            for snapshot in snapshots
        )
        return self.pending

    def collect_batch(self, handle):
        assert self.pending is handle
        self.pending = None
        return self.pending_results

    def collect(self, handle):
        assert self.pending is handle
        self.pending = None
        return NS(token_ids=(23,), valid_length=1)

    def fail(self, detail):
        self.records = dict.fromkeys(self.records, False)
        if self.policy == "error":
            raise RemoteDraftError(detail)

    def cancel(self, handle):
        assert self.pending is handle
        self.pending = None
        self.records = dict.fromkeys(self.records, False)

    def close_sequences(self, keys):
        for key in keys:
            self.records.pop(key, None)

    def close(self):
        self.records.clear()


@pytest.fixture
def verifier(monkeypatch):
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=2, dtype=torch.bfloat16
    )
    budget = SnapshotBudget(32, 4096, 32)
    contract = NS(
        expected_specs={"target.layers.0": spec},
        source_indices={"target.layers.0": 0},
        hidden_size=3,
        vocab_size=100,
        fingerprint="test",
        snapshot_budget=budget,
        max_num_seqs=4,
    )
    owner = NS(
        kv_cache=torch.arange(9 * 1 * 4 * 4).reshape(9, 1, 4, 4).to(torch.bfloat16),
        get_kv_cache_spec=lambda config: spec,
    )
    cfg = NS(
        speculative_config=NS(
            get_remote_draft_config=lambda: RemoteDraftConfig(
                endpoint="unix:///unused", failure_policy="target_only"
            ),
            get_remote_draft_endpoints=lambda: ("unix:///unused",),
        ),
        compilation_config=NS(
            static_forward_context={"model.layers.0.self_attn.attn": owner}
        ),
        cache_config=NS(get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC),
    )
    monkeypatch.setattr(remote, "create_target_kv_contract", lambda cfg: contract)
    monkeypatch.setattr(remote, "TargetKVRemoteSession", Session)
    instance = remote.TargetKVRemoteSpeculator(cfg, torch.device("cpu"))
    batch = replace(
        InputBatch.make_dummy(1, 2, InputBuffers(4, 8, torch.device("cpu"))),
        req_ids=["a"],
        positions=torch.tensor([4, 5]),
        seq_lens=torch.tensor([6]),
    )
    metadata = remote.TritonAttentionMetadata(
        2,
        2,
        torch.tensor([0, 2]),
        6,
        torch.tensor([6]),
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int32),
        torch.tensor([4, 5]),
        0,
        1,
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        True,
    )
    context = LocalExecutionContext(
        batch,
        {"model.layers.0.self_attn.attn": metadata},
        {},
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.bfloat16),
        None,
        torch.tensor([1]),
        torch.tensor([1]),
        torch.tensor([7]),
        torch.tensor([0]),
        torch.tensor([0.0]),
        torch.tensor([1]),
    )
    yield instance, context
    instance.close()


@pytest.mark.parametrize("rejected", [0, 1])
def test_snapshot_uses_native_selected_hidden_and_recovery_token(verifier, rejected):
    instance, context = verifier
    context = replace(
        context,
        num_rejected=torch.tensor([rejected]),
        num_sampled=torch.tensor([2 - rejected]),
    )
    step = DraftStep.from_batch(context.input_batch, 1)
    output = instance.collect(instance.dispatch(step, context))
    snapshot, token, hidden = instance._session.calls[0]
    assert snapshot.materialized_length == 6 - rejected
    assert token == 7
    torch.testing.assert_close(
        hidden, context.last_hidden_states[1 - rejected : 2 - rejected]
    )
    # Rejected target KV is zeroed rather than sent as committed context.
    tail = (6 - rejected) % 4
    assert not snapshot.layers[0].pages[-1, :, tail:].count_nonzero()
    assert output.to_legacy_tokens().tolist() == [[23]]


def _mixed_context(context):
    batch = replace(
        InputBatch.make_dummy(3, 6, InputBuffers(4, 8, torch.device("cpu"))),
        req_ids=["decode-b", "prefill", "decode-a"],
        query_start_loc_np=np.array([0, 2, 5, 6]),
        query_start_loc=torch.tensor([0, 2, 5, 6], dtype=torch.int32),
        is_prefilling_np=np.array([False, True, False]),
        idx_mapping=torch.tensor([2, 0, 1]),
        idx_mapping_np=np.array([2, 0, 1]),
        positions=torch.tensor([4, 5, 0, 1, 2, 8]),
        seq_lens=torch.tensor([6, 3, 9]),
    )
    metadata = replace(
        context.attn_metadata["model.layers.0.self_attn.attn"],
        block_table=torch.tensor([[1, 2, 3], [3, 4, 5], [6, 7, 8]], dtype=torch.int32),
    )
    return replace(
        context,
        input_batch=batch,
        attn_metadata={"model.layers.0.self_attn.attn": metadata},
        last_hidden_states=torch.arange(18).reshape(6, 3).to(torch.bfloat16),
        num_rejected=torch.tensor([1, 0, 0]),
        num_sampled=torch.tensor([1, 0, 1]),
        last_sampled=torch.tensor([17, 37, 47]),
        temperature=torch.zeros(3),
    )


def test_mixed_batch_uses_row_owned_kv_and_restores_unsampled_rows(verifier):
    """Packed queries, request-state indices and physical KV rows are distinct."""
    instance, context = verifier
    context = _mixed_context(context)
    output = instance.collect(
        instance.dispatch(DraftStep.from_batch(context.input_batch, 1), context)
    )
    assert output.to_legacy_tokens().tolist() == [[23], [-1], [25]]
    snapshots, tokens, hidden, kwargs = instance._session.batch_calls[0]
    assert [snapshot.materialized_length for snapshot in snapshots] == [5, 9]
    assert tokens == (47, 37)
    torch.testing.assert_close(hidden, context.last_hidden_states[[0, 5]])
    cache = instance.config.compilation_config.static_forward_context[
        "model.layers.0.self_attn.attn"
    ].kv_cache
    torch.testing.assert_close(snapshots[0].layers[0].pages[0], cache[1])
    torch.testing.assert_close(snapshots[1].layers[0].pages[0], cache[6])
    assert not snapshots[0].layers[0].pages[-1, :, 1:].count_nonzero()
    assert not snapshots[1].layers[0].pages[-1, :, 1:].count_nonzero()
    assert instance._session.open_calls == [((0, 1, 2), {"deadline": 1000})]
    assert kwargs == {"k": 1, "deadline": 1000}


def test_mixed_batch_excludes_terminal_rows_without_changing_other_positions(verifier):
    instance, context = verifier
    context = _mixed_context(context)
    zero = DraftStep.from_batch(context.input_batch, 0)
    assert instance.collect(
        instance.dispatch(zero, context)
    ).valid_lengths.tolist() == [
        0,
        0,
        0,
    ]
    assert not instance._session.batch_calls
    instance._session.records[instance._requests["decode-b"]] = False
    result = instance.collect(
        instance.dispatch(DraftStep.from_batch(context.input_batch, 1), context)
    )
    assert result.to_legacy_tokens().tolist() == [[-1], [-1], [25]]
    snapshots, tokens, hidden, _ = instance._session.batch_calls[0]
    assert len(snapshots) == 1 and snapshots[0].materialized_length == 9
    assert tokens == (37,)
    torch.testing.assert_close(hidden, context.last_hidden_states[5:6])


def test_batch_snapshot_budget_is_checked_before_any_export(verifier, monkeypatch):
    instance, context = verifier
    context = _mixed_context(context)
    # Both rows fit separately (2 and 3 pages), but cannot coexist in four pages.
    instance.contract.snapshot_budget = SnapshotBudget(4, 4096, 32)
    export = Mock(side_effect=AssertionError("export preceded aggregate admission"))
    monkeypatch.setattr(remote, "snapshot_target_kv", export)
    handle = instance.dispatch(DraftStep.from_batch(context.input_batch, 1), context)
    with pytest.raises(ValueError, match="page or byte budget"):
        instance.collect(handle)
    assert not export.called and not instance._session.batch_calls


def test_partial_batch_export_oom_fences_every_request(verifier, monkeypatch):
    instance, context = verifier
    context = _mixed_context(context)
    original = remote.snapshot_target_kv
    exports: list[TargetKVSnapshot] = []

    def export(*args, **kwargs):
        if exports:
            raise torch.OutOfMemoryError("second snapshot allocation")
        snapshot = original(*args, **kwargs)
        exports.append(snapshot)
        return snapshot

    monkeypatch.setattr(remote, "snapshot_target_kv", export)
    step = DraftStep.from_batch(context.input_batch, 1)
    for _ in range(2):
        assert instance.collect(
            instance.dispatch(step, context)
        ).valid_lengths.tolist() == [
            0,
            0,
            0,
        ]
    assert len(exports) == 1 and not instance._session.batch_calls
    assert not any(instance._session.records.values())
    assert instance._pending is None and instance._session.pending is None


@pytest.mark.parametrize("k,sampled", [(0, 1), (1, 0)])
def test_no_query_round_keeps_request_reusable_without_snapshot(
    verifier, monkeypatch, k, sampled
):
    instance, context = verifier
    context = replace(
        context, num_sampled=torch.tensor([sampled]), num_rejected=torch.tensor([0])
    )
    monkeypatch.setattr(
        remote, "snapshot_target_kv", Mock(side_effect=AssertionError("snapshot"))
    )
    out = instance.collect(
        instance.dispatch(DraftStep.from_batch(context.input_batch, k), context)
    )
    assert out.valid_lengths.tolist() == [0]
    assert not instance._session.calls
    assert all(instance._session.records.values())


def test_profile_reserves_full_snapshot_peak_without_request_or_rpc(
    verifier, monkeypatch
):
    instance, context = verifier
    allocations = []
    selected = []
    ordinary = torch.zeros
    select = torch.Tensor.index_select

    def allocate(*args, **kwargs):
        result = ordinary(*args, **kwargs)
        allocations.append(result.numel() * result.element_size())
        return result

    def select_rows(tensor, *args, **kwargs):
        result = select(tensor, *args, **kwargs)
        selected.append(result)
        return result

    monkeypatch.setattr(torch, "zeros", allocate)
    monkeypatch.setattr(torch.Tensor, "index_select", select_rows)
    context = replace(context, is_profile=True)
    out = instance.collect(
        instance.dispatch(DraftStep.from_batch(context.input_batch, 1), context)
    )
    assert allocations == [8 * 1 * 4 * 4 * 2] * 4
    assert len(selected) == 1 and selected[0].shape == (4, 3)
    assert selected[0].data_ptr() != context.last_hidden_states.data_ptr()
    assert not instance._requests and not instance._session.calls
    assert out.valid_lengths.tolist() == [0]


@pytest.mark.parametrize("policy", ["target_only", "error"])
@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("export allocation"),
        MemoryError("export allocation"),
        OSError(errno.ENOMEM, "export allocation"),
        RuntimeError(
            "DefaultCPUAllocator: can't allocate memory: you tried to allocate "
            "4096 bytes. Error code 12 (Cannot allocate memory)"
        ),
    ],
)
def test_snapshot_oom_is_terminal_and_obeys_failure_policy(
    verifier, monkeypatch, policy, error
):
    instance, context = verifier
    instance._session.policy = policy
    prepare = Mock(side_effect=error)
    monkeypatch.setattr(remote, "snapshot_target_kv", prepare)
    step = DraftStep.from_batch(context.input_batch, 1)
    handle = instance.dispatch(step, context)
    if policy == "error":
        with pytest.raises(RemoteDraftError, match="allocation failed"):
            instance.collect(handle)
    else:
        assert instance.collect(handle).valid_lengths.tolist() == [0]
        assert instance.collect(
            instance.dispatch(step, context)
        ).valid_lengths.tolist() == [0]
    assert prepare.call_count == 1
    assert not any(instance._session.records.values())
    assert instance._pending is None and instance._session.pending is None


@pytest.mark.parametrize(
    "error",
    [torch.OutOfMemoryError("profile allocation"), MemoryError("profile allocation")],
)
def test_profile_allocation_failure_is_not_hidden_by_fallback(
    verifier, monkeypatch, error
):
    instance, context = verifier
    monkeypatch.setattr(instance, "_profile_snapshot", Mock(side_effect=error))
    context = replace(context, is_profile=True)
    handle = instance.dispatch(DraftStep.from_batch(context.input_batch, 1), context)
    with pytest.raises(type(error)) as caught:
        instance.collect(handle)
    assert caught.value is error
    assert instance._pending is None and instance._session.pending is None
    assert not instance._session.calls and instance.stats["fallback"] == 0


@pytest.mark.parametrize(
    "kind,wrapper",
    [
        ("unknown", None),
        ("fatal", None),
        ("ordinary", None),
        ("unknown", MemoryError),
        ("unknown", torch.OutOfMemoryError),
        ("fatal", MemoryError),
        ("fatal", torch.OutOfMemoryError),
    ],
)
def test_unknown_completion_never_falls_back(verifier, monkeypatch, kind, wrapper):
    instance, context = verifier
    error = {
        "unknown": SnapshotCompletionError("unknown completion"),
        "fatal": RuntimeError("CUDA illegal memory access"),
        "ordinary": RuntimeError("unrelated snapshot failure"),
    }[kind]
    if wrapper is not None:
        wrapped = wrapper("snapshot allocation after unsafe completion")
        if wrapper is MemoryError:
            wrapped.__cause__ = error
        else:
            wrapped.__context__ = error
        error = wrapped
    monkeypatch.setattr(remote, "snapshot_target_kv", Mock(side_effect=error))
    handle = instance.dispatch(DraftStep.from_batch(context.input_batch, 1), context)
    with pytest.raises(type(error), match=str(error)):
        instance.collect(handle)
    assert instance._pending is None and instance._session.pending is None


def test_cancel_close_reuse_fences_old_handles_and_generations(verifier):
    instance, context = verifier
    step = DraftStep.from_batch(context.input_batch, 1)
    handle = instance.dispatch(step, context)
    old_key = instance._requests["a"]
    instance.close_requests(["a"])
    with pytest.raises(RuntimeError, match="consumed"):
        instance.collect(handle)
    new = instance.dispatch(step, context)
    assert instance._requests["a"].generation > old_key.generation
    instance.collect(new)
    with pytest.raises(RuntimeError, match="consumed"):
        instance.collect(new)


@pytest.mark.parametrize(
    "override",
    [
        {},
        {"num_speculative_tokens": 2},
        {"parallel_drafting": True},
        {"draft_tensor_parallel_size": 2},
        {"draft_sample_method": "probabilistic"},
    ],
)
def test_gemma_config_permits_only_the_dedicated_snapshot_contract(override):
    spec = object.__new__(SpeculativeConfig)
    spec.method = "mtp"
    spec.draft_model_config = NS(hf_config=NS(model_type="gemma4_mtp"))
    spec.remote_draft = RemoteDraftConfig(endpoint="unix:///unused")
    spec.execution = None
    spec.num_speculative_tokens = 1
    spec.draft_tensor_parallel_size = 1
    spec.draft_sample_method = "greedy"
    spec.parallel_drafting = False
    spec.use_heterogeneous_vocab = False
    for key, value in override.items():
        setattr(spec, key, value)
    if override:
        with pytest.raises(ValueError, match="remote Gemma4"):
            spec._validate_remote_draft()
    else:
        spec._validate_remote_draft()
        spec.remote_draft.transport = "cuda_ipc"
        with pytest.raises(ValueError, match="inline"):
            spec._validate_remote_draft()


def test_oom_in_first_preparation_allocation_is_terminal(verifier, monkeypatch):
    instance, context = verifier
    check = Mock(side_effect=torch.OutOfMemoryError("temperature guard"))
    monkeypatch.setattr(torch, "any", check)
    step = DraftStep.from_batch(context.input_batch, 1)
    for _ in range(2):
        assert instance.collect(
            instance.dispatch(step, context)
        ).valid_lengths.tolist() == [0]
    assert check.call_count == 1
    assert not any(instance._session.records.values())
