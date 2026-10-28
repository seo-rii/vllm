# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native entrypoint/metadata CPU controls; model, attention and CUDA are doubles."""

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.runtime.token_gpu import TokenGPUStepExecutor
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator import (
    StandaloneARSpeculator,
)


@pytest.mark.parametrize("kind", ["features", "custom_state"])
def test_token_native_input_contract_rejects_unsupported_dependencies(tmp_path, kind):
    from tests.v1.spec_decode.remote.test_token_runtime import token_config
    from vllm.v1.spec_decode.remote.runtime.token_gpu import validate_token_model_inputs
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    class TokenInput(torch.nn.Module):
        def forward(self, input_ids, positions):
            return input_ids

    class FeatureInput(torch.nn.Module):
        def forward(self, input_ids, positions, hidden_states):
            return hidden_states

    class StatefulInput(TokenInput):
        def get_model_state_cls(self):
            return MambaHybridModelState

    config = token_config(tmp_path)
    model = FeatureInput() if kind == "features" else StatefulInput()
    with pytest.raises(ValueError, match="tokens and positions|default decoder"):
        validate_token_model_inputs(config, model)


def test_native_independent_loader_never_needs_target_weights(monkeypatch):
    import vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator as native

    @dataclass
    class Config:
        speculative_config: object
        cache_config: object
        parallel_config: object
        model_config: object
        quant_config: object

    @dataclass
    class Parallel:
        rank: int = 0

    draft = object()
    config = Config(
        SimpleNamespace(kv_cache_dtype=None, draft_parallel_config=Parallel()),
        object(),
        Parallel(),
        object(),
        object(),
    )
    speculator = object.__new__(StandaloneARSpeculator)
    speculator.speculative_config = config.speculative_config
    speculator.vllm_config = config
    speculator.draft_model_config = draft
    calls = []
    model = object()

    def get_model(**kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setattr(native, "get_model", get_model)
    monkeypatch.setattr(native, "set_model_tag", lambda tag: nullcontext())
    finalized: list[set[str]] = []
    monkeypatch.setattr(speculator, "_finalize_model_load", finalized.append)
    speculator.load_standalone_model()
    assert speculator.model is model and finalized == [set()]
    assert len(calls) == 1
    assert calls[0]["prefix"] == "draft_model"
    assert calls[0]["vllm_config"].model_config is draft
    assert calls[0]["vllm_config"].quant_config is None


@pytest.fixture
def token_executor(monkeypatch):
    import vllm.v1.spec_decode.remote.runtime.token_gpu as runtime
    import vllm.v1.worker.gpu.buffer_utils as buffers
    import vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator as native
    from vllm.v1.worker.gpu.block_table import BlockTables

    class CpuPool:
        def copy_to_uva(self, values):
            return torch.tensor(values, dtype=torch.int32)

    class Kernel:
        def __getitem__(self, grid):
            def apply(output, stride, indices, starts, contents, ends, *a, **kw):
                destinations = set()
                offset = 0
                for row, start, end in zip(indices.tolist(), starts, ends):
                    for col, value in enumerate(contents[offset:end], int(start)):
                        assert (row, col) not in destinations
                        destinations.add((row, col))
                        output[row, col] = value
                    offset = int(end)

            return apply

    monkeypatch.setattr(buffers, "UvaBufferPool", lambda *a, **kw: CpuPool())
    monkeypatch.setattr(
        buffers, "async_tensor_h2d", lambda values, **kw: torch.tensor(values, **kw)
    )
    monkeypatch.setattr(buffers, "_apply_write_kernel", Kernel())

    executor = object.__new__(TokenGPUStepExecutor)
    executor.device = torch.device("cpu")
    executor.block_size = 4
    executor.num_blocks = 9
    executor.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=16, get_vocab_size=lambda: 128),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
    )
    executor.input_buffers = InputBuffers(4, 16, executor.device)
    executor.temperature = torch.zeros(4)
    executor.seeds = torch.zeros(4, dtype=torch.int64)
    executor.attn_groups = []
    executor.kv_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["draft"])]
    )
    calls, synced, metadata = [], [], []
    tables = executor.block_tables = object.__new__(BlockTables)
    staged = buffers.StagedWriteTensor((4, 16), torch.int32, executor.device)
    tables.block_tables, tables.num_kv_cache_groups = [staged], 1
    tables.num_blocks = SimpleNamespace(
        np=np.zeros((1, 4), dtype=np.int32), copy_to_uva=lambda: None
    )

    def slots(indices, qsl, positions, n):
        values: list[int] = []
        for row, index in enumerate(indices.tolist()):
            blocks = staged.gpu[index].tolist()
            block_size = executor.block_size
            values.extend(
                blocks[int(p) // block_size] * block_size + int(p) % block_size
                for p in positions[int(qsl[row]) : int(qsl[row + 1])]
            )
        return torch.tensor([values], dtype=torch.int64)

    monkeypatch.setattr(tables, "gather_block_tables", lambda indices, n: ())
    monkeypatch.setattr(tables, "compute_slot_mappings", slots)

    def prepare(batch, *args):
        metadata.append(
            (
                batch.positions.tolist(),
                batch.seq_lens.tolist(),
                batch.idx_mapping.tolist(),
            )
        )
        return {"draft": "metadata"}

    executor.model_state = SimpleNamespace(prepare_attn=prepare)

    class Model(torch.nn.Module):
        def forward(self, input_ids, positions):
            calls.append((input_ids.tolist(), positions.tolist()))
            return input_ids.to(torch.float32).reshape(-1, 1)

        def compute_logits(self, hidden):
            return torch.cat((hidden, hidden + 1), dim=1)

    speculator = object.__new__(StandaloneARSpeculator)
    speculator.model = Model()
    speculator.vllm_config = executor.vllm_config
    speculator.current_draft_step = torch.tensor(0)
    speculator.use_local_argmax_reduction = False
    speculator.acceptance_estimator = None
    executor.speculator = speculator
    monkeypatch.setattr(
        runtime, "set_current_vllm_config", lambda config: nullcontext()
    )
    monkeypatch.setattr(
        native, "set_forward_context", lambda *args, **kwargs: nullcontext()
    )
    monkeypatch.setattr(torch.accelerator, "set_device_index", lambda device: None)
    monkeypatch.setattr(
        torch.accelerator, "synchronize", lambda device: synced.append(True)
    )
    return executor, calls, synced, metadata


def test_token_executor_calls_native_forward_at_real_positions(token_executor):
    executor, calls, synced, metadata = token_executor
    key = SequenceKey("v", 0, 0)
    assert executor.forward(key, 3, 0, (1, 2), (2,), sample=False) is None
    assert executor.forward(key, 3, 2, (3, 4), (2,), sample=True) == 1
    assert calls == [([1, 2], [0, 1]), ([3, 4], [2, 3])]
    assert metadata == [([0, 1], [2], [3]), ([2, 3], [4], [3])]
    assert len(synced) == 1


def test_token_executor_packs_ragged_rows_into_one_native_forward(token_executor):
    from vllm.v1.spec_decode.remote.runtime.token_state import TokenForwardRow

    executor, calls, synced, metadata = token_executor
    keys = tuple(SequenceKey("packed", i, 0) for i in range(3))
    result = executor.forward_batch(
        (
            TokenForwardRow(keys[0], 3, 0, (1, 2), (2,), True),
            TokenForwardRow(keys[1], 1, 2, (5,), (3,), False),
            TokenForwardRow(keys[2], 2, 0, (6, 7, 8), (4,), True),
        )
    )
    assert result == (1, None, 1)
    assert calls == [([1, 2, 5, 6, 7, 8], [0, 1, 2, 0, 1, 2])]
    assert metadata == [([0, 1, 2, 0, 1, 2], [2, 3, 3], [3, 1, 2])]
    assert not synced  # The single batched CPU result copy supplies completion.


@pytest.mark.parametrize("failure", [None, "stage", "upload"])
def test_token_forward_discards_failed_writes_before_request_slot_reuse(
    token_executor, monkeypatch, failure
):
    import vllm.v1.worker.gpu.buffer_utils as buffers
    from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
    from vllm.v1.spec_decode.remote.runtime.token_state import TokenStateMachine

    executor, calls, _, _ = token_executor
    executor.block_size = 1
    pool = DraftBlockPool(4, 1)
    state = TokenStateMachine(
        executor, pool, max_num_reqs=2, max_model_len=4, vocab_size=128, max_k=1
    )
    old, new, sibling = [SequenceKey("tables", i, 0) for i in range(3)]
    state.open(old)
    old_slot = state.sequences[old].req_index
    staged = executor.block_tables.block_tables[0]

    def fail_upload(*args, **kwargs):
        raise torch.OutOfMemoryError("injected upload failure")

    with monkeypatch.context() as patch:
        if failure == "stage":

            class PartialContents(list):
                def extend(self, values):
                    super().extend(values)
                    raise MemoryError("injected staging failure")

            patch.setattr(staged, "_staged_write_contents", PartialContents())
        elif failure == "upload":
            patch.setattr(buffers, "async_tensor_h2d", fail_upload)
        if failure is None:
            state.prefill(old, 0, (1, 2), is_final=True)
        else:
            with pytest.raises((MemoryError, torch.OutOfMemoryError)):
                state.prefill(old, 0, (1, 2), is_final=True)
            assert not calls
            assert not staged._staged_write_indices
    old_blocks = pool.blocks(old_slot)
    state.close((old,))
    assert pool.num_free_blocks == 3
    state.open(new)
    state.open(sibling)
    assert state.sequences[new].req_index == old_slot
    # Reserve the old first block for a sibling before the reused slot writes.
    sibling_slot = state.sequences[sibling].req_index
    pool.ensure(sibling_slot, 1)
    assert pool.blocks(sibling_slot)[0] == old_blocks[0]
    state.prefill(new, 0, (3,), is_final=True)
    assert pool.blocks(old_slot)[0] != old_blocks[0]
    assert staged.gpu[old_slot, 0].item() == pool.blocks(old_slot)[0]
    assert calls[-1] == ([3], [0])
    assert state.sequences[new].materialized == 1


@pytest.mark.parametrize(
    "capacity,k", [(1, 3), (4, 0), (4, 1), (4, 3), (4, 4), (16, 16)]
)
def test_token_warmup_uses_real_state_and_cleans_buffers(token_executor, capacity, k):
    executor, calls, synced, _ = token_executor
    executor.vllm_config.model_config.max_model_len = capacity
    executor.vllm_config.speculative_config.num_speculative_tokens = k
    executor.num_blocks = 2
    executor.block_size = max(4, capacity)
    executor.warmup()
    assert calls[0] == ([0], [0])
    assert len(calls) >= (2 if capacity >= 2 else 1)
    assert synced
    assert torch.count_nonzero(executor.input_buffers.input_ids).item() == 0
    assert torch.count_nonzero(executor.input_buffers.positions).item() == 0


@pytest.mark.parametrize("failed_forward", [1, 2])
@pytest.mark.parametrize("completion_unknown", [False, True])
def test_token_warmup_unusable_error_skips_gpu_cleanup(
    token_executor, monkeypatch, failed_forward, completion_unknown
):
    from vllm.v1.spec_decode.remote.errors import CompletionUnknownError
    from vllm.v1.spec_decode.remote.runtime.token_state import TokenStateMachine

    executor, _, _, _ = token_executor
    forward, close = executor.forward_batch, TokenStateMachine.close
    failure = (
        CompletionUnknownError("warmup completion is unknown")
        if completion_unknown
        else RuntimeError("CUDA error: device-side assert triggered")
    )
    calls, cpu_closes, after_failure = [], [], []

    def failing_forward(*args, **kwargs):
        calls.append(True)
        if len(calls) == failed_forward:
            raise failure
        return forward(*args, **kwargs)

    def record_close(state, keys):
        cpu_closes.append(keys)
        close(state, keys)
        assert not state.sequences

    zero = torch.Tensor.zero_

    def record_zero(tensor):
        if len(calls) >= failed_forward:
            after_failure.append("zero")
        return zero(tensor)

    def record_sync(device):
        if len(calls) >= failed_forward:
            after_failure.append("sync")

    monkeypatch.setattr(executor, "forward_batch", failing_forward)
    monkeypatch.setattr(TokenStateMachine, "close", record_close)
    monkeypatch.setattr(torch.Tensor, "zero_", record_zero)
    monkeypatch.setattr(torch.accelerator, "synchronize", record_sync)
    with pytest.raises(RuntimeError) as raised:
        executor.warmup()
    assert raised.value is failure
    assert cpu_closes and not after_failure


def test_token_warmup_preserves_initial_error_when_cleanup_fails(
    token_executor, monkeypatch
):
    executor, _, _, _ = token_executor
    failure = torch.OutOfMemoryError("injected recoverable warmup allocation failure")

    def fail_forward(*args, **kwargs):
        raise failure

    def fail_cleanup(tensor):
        raise RuntimeError("injected cleanup failure")

    monkeypatch.setattr(executor, "forward", fail_forward)
    monkeypatch.setattr(torch.Tensor, "zero_", fail_cleanup)
    with pytest.raises(torch.OutOfMemoryError) as raised:
        executor.warmup()
    assert raised.value is failure
    assert any("cleanup failed" in note for note in failure.__notes__)


@pytest.mark.parametrize("cleanup_phase", ["input_ids", "positions", "sync"])
@pytest.mark.parametrize("cuda_error_code", [700, None])
def test_token_server_fences_unusable_warmup_cleanup(
    token_executor, monkeypatch, tmp_path, cleanup_phase, cuda_error_code
):
    from tests.v1.spec_decode.remote.test_token_runtime import token_oracle_contract
    from vllm.v1.spec_decode.remote.errors import (
        CompletionUnknownError,
        CudaRuntimeError,
        runtime_unusable,
    )
    from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
    from vllm.v1.spec_decode.remote.runtime.token_adapter import TokenOnlyDraftAdapter
    from vllm.v1.spec_decode.remote.runtime.token_state import TokenStateMachine
    from vllm.v1.spec_decode.remote.token_server import TokenDraftServer

    executor, _, _, _ = token_executor
    primary = torch.OutOfMemoryError("recoverable warmup allocation failure")
    cleanup = (
        CompletionUnknownError("warmup completion is unknown")
        if cuda_error_code is None
        else CudaRuntimeError("warmup cleanup failed", cuda_error_code=cuda_error_code)
    )
    calls = []

    def fail_forward(*args, **kwargs):
        raise primary

    def clean(phase):
        calls.append(phase)
        if phase == cleanup_phase:
            raise cleanup

    def zero(tensor):
        clean(
            "input_ids" if tensor is executor.input_buffers.input_ids else "positions"
        )
        return tensor

    monkeypatch.setattr(executor, "forward", fail_forward)
    monkeypatch.setattr(torch.Tensor, "zero_", zero)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda device: clean("sync"))
    state = TokenStateMachine(
        executor,
        DraftBlockPool(executor.num_blocks, executor.block_size),
        max_num_reqs=1,
        max_model_len=16,
        vocab_size=128,
        max_k=3,
    )
    adapter = TokenOnlyDraftAdapter(state, token_oracle_contract())
    server = TokenDraftServer(adapter)
    with pytest.raises(type(cleanup)) as raised:
        server.start(f"unix://{tmp_path}/not-listening.sock")
    assert raised.value is cleanup
    assert raised.value.__context__ is primary
    assert runtime_unusable(raised.value)
    assert not server.healthy
    assert server.endpoint is None
    assert calls == ["input_ids", "positions", "sync"][: calls.index(cleanup_phase) + 1]
    with pytest.raises(RuntimeError, match="unhealthy"):
        server.start(f"unix://{tmp_path}/not-listening.sock")


@pytest.mark.parametrize("branch_slots, token_budget", [(0, 64), (2, 64), (1, 4)])
def test_token_cache_executor_reserves_branch_slots_without_changing_live_budgets(
    tmp_path, branch_slots, token_budget
):
    """Resident branch slots must fit without enlarging foreground admission."""
    from tests.v1.spec_decode.remote.test_token_runtime import token_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.spec_decode.remote.runtime.token_gpu import (
        create_token_executor_config,
    )

    assets = token_config(tmp_path)
    assert assets.speculative_config is not None
    assert assets.speculative_config.draft_model_config is not None
    config = EngineArgs(
        model=assets.model_config.model,
        tokenizer_mode="hf",
        dtype="float32",
        skip_tokenizer_init=True,
        max_model_len=64,
        max_num_seqs=4,
        max_num_batched_tokens=token_budget,
        enable_chunked_prefill=True,
        async_scheduling=False,
        enforce_eager=True,
        scheduling_policy="priority",
        max_num_active_seqs=2,
        speculative_config={
            "method": "draft_model",
            "model": assets.speculative_config.draft_model_config.model,
            "num_speculative_tokens": 3,
        },
    ).create_engine_config()
    scheduler = config.scheduler_config
    resource = create_token_executor_config(config, branch_slots=branch_slots)

    assert resource.scheduler_config.max_num_seqs == 4 + branch_slots
    executor_token_budget = max(token_budget, 4 + branch_slots)
    assert resource.scheduler_config.max_num_batched_tokens == executor_token_budget
    assert (
        resource.scheduler_config.max_num_encoder_input_tokens == executor_token_budget
    )
    assert resource.scheduler_config.encoder_cache_size == executor_token_budget
    assert resource.scheduler_config.policy == "priority"
    assert (
        resource.scheduler_config.max_num_active_seqs == scheduler.max_num_active_seqs
    )
    assert resource.scheduler_config.async_scheduling is False
    assert config.speculative_config is not None
    assert resource.model_config is config.speculative_config.draft_model_config
    assert config.scheduler_config is scheduler
    assert scheduler.max_num_seqs == 4
    assert scheduler.max_num_batched_tokens == token_budget
    assert scheduler.max_num_scheduled_tokens == token_budget
