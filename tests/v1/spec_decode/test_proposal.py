# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SpeculatorOutput structural invariants."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.spec_decode.contracts import DraftStep, LocalExecutionContext
from vllm.v1.spec_decode.proposal import SpeculatorOutput
from vllm.v1.worker.gpu.spec_decode.local_backend import LocalDraftBackend


def test_batch_size_mismatch_rejected():
    with pytest.raises(ValueError, match="batch size mismatch"):
        SpeculatorOutput(
            token_ids=torch.zeros(2, 3, dtype=torch.long),
            valid_lengths=torch.zeros(3, dtype=torch.int32),
        )


def test_wrong_rank_rejected():
    with pytest.raises(ValueError, match="2-D"):
        SpeculatorOutput(
            token_ids=torch.zeros(6, dtype=torch.long),
            valid_lengths=torch.zeros(6, dtype=torch.int32),
        )


def test_non_integer_dtype_rejected():
    with pytest.raises(ValueError, match="integer dtype"):
        SpeculatorOutput(
            token_ids=torch.zeros(2, 3),
            valid_lengths=torch.zeros(2, dtype=torch.int32),
        )


@pytest.mark.parametrize("dtype", [torch.uint8, torch.bool, torch.int8, torch.int16])
@pytest.mark.parametrize("field", ["token_ids", "valid_lengths"])
def test_result_rejects_unsigned_boolean_and_narrow_integer_dtypes(dtype, field):
    tensors = {
        "token_ids": torch.ones(1, 2, dtype=torch.int64),
        "valid_lengths": torch.ones(1, dtype=torch.int32),
    }
    tensors[field] = tensors[field].to(dtype)
    with pytest.raises(ValueError, match=f"{field} must have an int32 or int64"):
        SpeculatorOutput(**tensors)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("batch_size", [0, 1])
def test_signed_result_preserves_negative_legacy_suffix(dtype, batch_size):
    output = SpeculatorOutput(
        torch.ones(batch_size, 2, dtype=dtype),
        torch.ones(batch_size, dtype=dtype),
    )
    legacy = output.to_legacy_tokens()
    assert legacy.dtype == dtype and legacy.shape == (batch_size, 2)
    assert legacy.tolist() == [[1, -1]] * batch_size


def local_context():
    batch = SimpleNamespace(
        req_ids=["a", "b"],
        query_start_loc_np=[0, 2, 3],
        is_prefilling_np=[True, False],
        num_reqs=2,
    )
    return LocalExecutionContext(
        batch,
        {},
        {},
        torch.zeros(3, 4),
        None,
        torch.tensor([1, 1]),
        torch.tensor([0, 0]),
        torch.tensor([4, 5]),
        torch.tensor([[6, 7]]),
        torch.zeros(2),
        torch.zeros(2, dtype=torch.long),
    )


def test_step_snapshots_metadata_without_cloning_borrowed_inputs():
    context = local_context()
    step = DraftStep.from_batch(context.input_batch, 1)
    context.input_batch.req_ids.reverse()
    context.input_batch.query_start_loc_np[1] = 1
    assert step.request_ids == ("a", "b")
    assert step.query_lens == (2, 1)


@pytest.mark.parametrize("k,expected", [(0, [0, 0]), (1, [1, 1]), (3, [2, 1])])
def test_local_backend_keeps_native_instance_and_caps_valid_prefix(k, expected):
    native = Mock(num_speculative_steps=3)
    tokens = torch.tensor([[0, 2, -1], [5, -1, 9]])
    native.propose.return_value = tokens
    backend = LocalDraftBackend(native)
    context = local_context()
    step = DraftStep.from_batch(context.input_batch, k)
    handle = backend.dispatch(step, context)
    assert backend.speculator is native
    assert native.propose.call_args.kwargs["input_batch"] is context.input_batch
    assert (
        native.propose.call_args.kwargs["last_hidden_states"]
        is context.last_hidden_states
    )
    assert native.propose.call_args.kwargs["num_speculative_tokens"] == k
    with pytest.raises(RuntimeError, match="in flight"):
        backend.dispatch(step, context)
    result = backend.collect(handle)
    assert result.token_ids is tokens
    assert result.valid_lengths.tolist() == expected
    with pytest.raises(RuntimeError, match="stale|collected"):
        backend.collect(handle)


def test_local_backend_cancel_and_request_reuse_cannot_collect_old_result():
    native = Mock(num_speculative_steps=3)
    native.propose.return_value = torch.tensor([[1, 2, 3], [4, 5, 6]])
    backend = LocalDraftBackend(native)
    other = LocalDraftBackend(native)
    context = local_context()
    step = DraftStep.from_batch(context.input_batch, 3)
    old = backend.dispatch(step, context)
    with pytest.raises(RuntimeError, match="owner|stale"):
        other.collect(old)
    backend.close_requests({"a"})
    current = backend.dispatch(step, context)
    with pytest.raises(RuntimeError, match="stale|cancelled"):
        backend.collect(old)
    assert backend.collect(current).valid_lengths.tolist() == [3, 3]
    native.close_requests.assert_called_once_with({"a"})
    backend.close()
    native.close.assert_called_once()


def test_local_close_discards_uncollected_result_and_rejects_new_dispatch():
    native = Mock(num_speculative_steps=3)
    native.propose.return_value = torch.tensor([[1, 2, 3], [4, 5, 6]])
    backend = LocalDraftBackend(native)
    context = local_context()
    step = DraftStep.from_batch(context.input_batch, 1)
    handle = backend.dispatch(step, context)
    backend.close()
    with pytest.raises(RuntimeError, match="stale|cancelled"):
        backend.collect(handle)
    with pytest.raises(RuntimeError, match="closed"):
        backend.dispatch(step, context)


def test_native_factory_is_unchanged_until_local_execution_opt_in(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode import init_draft_backend, init_speculator
    from vllm.v1.worker.gpu.spec_decode.mtp import speculator as mtp

    native = Mock()
    monkeypatch.setattr(mtp, "MTPSpeculator", lambda *args: native)
    spec = SimpleNamespace(
        use_remote_draft=lambda: False,
        use_gemma4_mtp=lambda: False,
        use_multi_module_mtp=lambda: False,
        method="mtp",
        execution=None,
    )
    config = SimpleNamespace(speculative_config=spec)
    assert init_speculator(config, torch.device("cpu"), None) is native
    assert init_draft_backend(config, native) is None
    native.propose.assert_not_called()
    spec.execution = SimpleNamespace(placement="local")
    backend = init_draft_backend(config, native)
    assert isinstance(backend, LocalDraftBackend)
    assert backend.speculator is native


def test_local_cancel_discards_only_the_pending_result():
    native = Mock(num_speculative_steps=3)
    native.propose.return_value = torch.tensor([[1, 2, 3], [4, 5, 6]])
    backend = LocalDraftBackend(native)
    context = local_context()
    step = DraftStep.from_batch(context.input_batch, 1)
    old = backend.dispatch(step, context)
    backend.cancel(old, "verification step was cancelled")
    with pytest.raises(RuntimeError, match="stale|cancelled"):
        backend.collect(old)
    with pytest.raises(RuntimeError, match="stale|cancelled"):
        backend.cancel(old, "duplicate cancellation")
    current = backend.dispatch(step, context)
    assert backend.collect(current).valid_lengths.tolist() == [1, 1]
    native.close_requests.assert_not_called()
