# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native CPU model/metadata and isolated execution-boundary controls."""

import inspect
import os
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
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


def test_recurrent_executor_config_uses_native_ordinary_state_without_lookahead(
    tmp_path, monkeypatch
):
    from tests.v1.spec_decode.remote.test_token_runtime import token_config
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
    from vllm.v1.spec_decode.remote.runtime.recurrent_gpu import (
        create_recurrent_executor_config,
    )

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    config = token_config(
        tmp_path,
        draft_type="nemotron_h",
        dtype="bfloat16",
        num_nextn_predict_layers=0,
        cache_options={"mamba_ssm_cache_dtype": "float32"},
        spec_options={"attention_backend": "TRITON_ATTN"},
    )
    draft = config.speculative_config.draft_model_config
    resource = create_recurrent_executor_config(config)
    assert resource.speculative_config is None and resource.num_speculative_tokens == 0
    assert resource.model_config.model == draft.model
    assert resource.model_config.architecture == "NemotronHForCausalLM"
    assert resource.model_config.hf_config.num_nextn_predict_layers == 0
    assert resource.model_config.dtype is torch.bfloat16
    assert resource.parallel_config.tensor_parallel_size == 1
    assert (
        resource.scheduler_config.max_num_seqs == config.scheduler_config.max_num_seqs
    )
    assert resource.scheduler_config.max_num_batched_tokens == 64
    assert resource.model_config.max_model_len == 64
    assert resource.cache_config.cache_dtype == "bfloat16"
    assert resource.cache_config.mamba_cache_dtype == "bfloat16"
    assert resource.cache_config.mamba_ssm_cache_dtype == "float32"
    assert not resource.cache_config.use_replayssm
    assert resource.cache_config.mamba_cache_mode == "none"
    assert resource.cache_config is not config.cache_config
    assert NemotronHForCausalLM.get_mamba_state_dtype_from_config(resource) == (
        torch.bfloat16,
        torch.float32,
    )
    shapes = NemotronHForCausalLM.get_mamba_state_shape_from_config(resource)
    assert len(shapes) == 2
    assert sorted(shapes[0]) == [3, 48]
    assert shapes[1] == (2, 16, 8)


@pytest.mark.parametrize("explicit_draft", [False, True])
def test_recurrent_executor_honors_selected_checkpoint_loader(tmp_path, explicit_draft):
    """An explicit PT loader must not silently fall back to safetensors."""
    from tests.v1.spec_decode.remote.test_token_runtime import token_config
    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.v1.spec_decode.remote.runtime.recurrent_gpu import (
        create_recurrent_executor_config,
    )

    config = token_config(
        tmp_path,
        draft_type="nemotron_h",
        dtype="bfloat16",
        num_nextn_predict_layers=0,
        cache_options={"mamba_ssm_cache_dtype": "float32"},
    )
    selected = LoadConfig(
        load_format="pt",
        download_dir=str(tmp_path / "weights"),
        use_tqdm_on_load=False,
        model_loader_extra_config={"enable_weights_track": True},
    )
    if explicit_draft:
        config.speculative_config.draft_load_config = selected
    else:
        config.load_config = selected
    resource = create_recurrent_executor_config(config)
    assert (Path(resource.model_config.model) / "model.safetensors").is_file()
    with pytest.raises(RuntimeError, match="Cannot find any model weights"):
        get_model_loader(resource.load_config).download_model(resource.model_config)
    assert resource.load_config == selected


def test_recurrent_executor_rejects_full_prefix_beyond_server_token_budget(
    tmp_path, monkeypatch
):
    from tests.v1.spec_decode.remote.test_token_runtime import token_config
    from vllm.v1.spec_decode.remote.runtime.recurrent_gpu import (
        create_recurrent_executor_config,
    )

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    config = token_config(
        tmp_path,
        draft_type="nemotron_h",
        dtype="bfloat16",
        num_nextn_predict_layers=0,
        cache_options={"mamba_ssm_cache_dtype": "float32"},
        scheduler_options={"max_num_batched_tokens": 32},
    )
    with pytest.raises(
        ValueError, match="full-prefix replay must fit one token budget"
    ):
        create_recurrent_executor_config(config)


def test_recurrent_startup_reads_each_checkpoint_once_and_rechecks_changes(
    tmp_path, monkeypatch
):
    """Each startup fingerprints current bytes once before device admission."""
    import hashlib
    from collections import Counter

    from safetensors.torch import load_file, save_file

    from tests.v1.spec_decode.remote.test_token_runtime import token_config
    from vllm.v1.spec_decode.remote.runtime import schema
    from vllm.v1.spec_decode.remote.runtime.recurrent_gpu import (
        RecurrentGPUStepExecutor,
    )

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    config = token_config(
        tmp_path,
        draft_type="nemotron_h",
        dtype="bfloat16",
        num_nextn_predict_layers=0,
        cache_options={"mamba_ssm_cache_dtype": "float32"},
        spec_options={"attention_backend": "TRITON_ATTN"},
    )
    checkpoints = [
        tmp_path / role / "model.safetensors" for role in ("target", "draft")
    ]
    observations: list[tuple[Path, str]] = []

    def observe_hashes(frame, event, result):
        if (
            event == "return"
            and frame.f_code is schema._local_artifact_digests.__code__
        ):
            observations.extend(
                (frame.f_locals["path"] / name, digest)
                for name, digest in result.items()
            )

    previous_draft_digest = None
    for _ in range(2):
        observations.clear()
        expected = {
            path: hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoints
        }
        unavailable = torch.device("cuda", torch.accelerator.device_count())
        previous_profile = sys.getprofile()
        try:
            sys.setprofile(observe_hashes)
            with pytest.raises(ValueError, match="not a visible CUDA GPU"):
                RecurrentGPUStepExecutor(
                    config, device=unavailable, kv_cache_bytes=1 << 20
                )
        finally:
            sys.setprofile(previous_profile)
        assert Counter(path for path, _ in observations) == Counter(checkpoints)
        assert dict(observations) == expected
        assert expected[checkpoints[1]] != previous_draft_digest
        previous_draft_digest = expected[checkpoints[1]]
        weights = load_file(str(checkpoints[1]))
        weights[sorted(weights)[0]].add_(1)
        save_file(weights, str(checkpoints[1]))


def recurrent_cache_inputs(tmp_path, *, max_num_reqs):
    from tests.v1.spec_decode.remote.test_token_runtime import token_config
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
    from vllm.v1.spec_decode.remote.runtime.recurrent_gpu import (
        create_recurrent_executor_config,
    )

    config = create_recurrent_executor_config(
        token_config(
            tmp_path,
            draft_type="nemotron_h",
            dtype="bfloat16",
            num_nextn_predict_layers=0,
            cache_options={"mamba_ssm_cache_dtype": "float32"},
            scheduler_options={"max_num_seqs": max_num_reqs},
            spec_options={"attention_backend": "TRITON_ATTN"},
        )
    )
    return config, {
        "model.layers.0.mixer": MambaSpec(
            block_size=config.cache_config.mamba_block_size,
            shapes=NemotronHForCausalLM.get_mamba_state_shape_from_config(config),
            dtypes=NemotronHForCausalLM.get_mamba_state_dtype_from_config(config),
        ),
        "model.layers.1.mixer.attn": FullAttentionSpec(
            block_size=config.cache_config.block_size,
            num_kv_heads=config.model_config.get_num_kv_heads(config.parallel_config),
            head_size=config.model_config.get_head_size(),
            dtype=config.model_config.dtype,
        ),
    }


def _run_recurrent_native_cache_allocation(tmp_path):
    """Load native state in a fresh process without changing the test worker."""
    from safetensors import safe_open

    import vllm.ir
    from tests.v1.spec_decode.remote.test_token_runtime import token_config
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
    from vllm.model_executor.model_loader import get_model
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec, MambaSpec
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from vllm.v1.spec_decode.remote.runtime.recurrent_gpu import (
        create_recurrent_executor_config,
    )
    from vllm.v1.spec_decode.remote.runtime.recurrent_kv import make_recurrent_kv_config
    from vllm.v1.worker.utils import allocate_kv_cache

    tmp_path = Path(tmp_path)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["VLLM_TARGET_DEVICE"] == "cpu"
    assert os.environ["VLLM_BATCH_INVARIANT"] == "0"
    config = create_recurrent_executor_config(
        token_config(
            tmp_path,
            draft_type="nemotron_h",
            dtype="bfloat16",
            num_nextn_predict_layers=0,
            cache_options={"mamba_ssm_cache_dtype": "float32"},
            scheduler_options={"max_num_seqs": 2},
            spec_options={"attention_backend": "TRITON_ATTN"},
        )
    )
    with safe_open(tmp_path / "draft/model.safetensors", framework="pt") as weights:
        assert len(weights.keys()) == 17
    assert not torch.cuda.is_initialized()
    assert not torch.distributed.is_initialized()
    initialized = False
    try:
        with (
            set_current_vllm_config(config),
            vllm.ir.enable_torch_wrap(config.compilation_config.ir_enable_torch_wrap),
        ):
            config.kernel_config.ir_op_priority.set_default()
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=(tmp_path / "gloo").as_uri(),
                backend="gloo",
            )
            initialized = True
            initialize_model_parallel(backend="gloo")
            model = get_model(vllm_config=config)
            specs: dict[str, KVCacheSpec] = {}
            for name, layer in model.named_modules():
                if isinstance(layer, (Attention, MambaMixer2)):
                    spec = layer.get_kv_cache_spec(config)
                    assert spec is not None
                    specs[name] = spec
            assert len(specs) == 2
            state_name, state_spec = next(
                (name, spec)
                for name, spec in specs.items()
                if isinstance(spec, MambaSpec)
            )
            attention_name, attention_spec = next(
                (name, spec)
                for name, spec in specs.items()
                if isinstance(spec, FullAttentionSpec)
            )
            assert state_spec.num_prefill_checkpoint_blocks == 1
            assert state_spec.prefill_checkpoint_alignment == 1
            assert state_spec.mamba_cache_mode == "none"
            assert state_spec.num_speculative_blocks == 0
            assert (
                state_spec.max_memory_usage_bytes(config) == state_spec.page_size_bytes
            )
            assert state_spec.max_num_blocks_per_req(config, 64) == 1
            budget = 3 * state_spec.page_size_bytes + 9 * attention_spec.page_size_bytes
            cache = make_recurrent_kv_config(
                config, specs, budget, 2, KVCacheLayout.LBNHC
            )
            assert cache.num_blocks == 9 and cache.recurrent_num_blocks == 3
            allocated = allocate_kv_cache(
                cache, torch.device("cpu"), KVCacheLayout.LBNHC
            )
            assert allocated[attention_name].shape[0] == 9
            assert allocated[state_name].shape == (3, 1, 1, state_spec.page_size_bytes)
            assert all(
                tensor.untyped_storage().nbytes() == budget
                for tensor in allocated.values()
            )
            preserved = next(
                group.kv_cache_spec
                for group in cache.kv_cache_groups
                if isinstance(group.kv_cache_spec, MambaSpec)
            )
            assert preserved == state_spec
    finally:
        if initialized:
            try:
                destroy_model_parallel()
            finally:
                destroy_distributed_environment()
    assert not torch.distributed.is_initialized()
    assert not torch.cuda.is_initialized()


def test_recurrent_allocation_accepts_loaded_native_checkpoint_capability(tmp_path):
    """Native checkpoint export metadata does not reserve align-mode states."""
    before = torch.cuda.is_initialized(), torch.distributed.is_initialized()
    log_path = tmp_path / "native-cache-spec.log"
    try:
        with log_path.open("w") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys; from tests.v1.spec_decode.remote.test_token_gpu "
                    "import _run_recurrent_native_cache_allocation; "
                    "_run_recurrent_native_cache_allocation(sys.argv[1])",
                    str(tmp_path),
                ],
                cwd=Path(__file__).resolve().parents[4],
                env=dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES="",
                    VLLM_TARGET_DEVICE="cpu",
                    VLLM_BATCH_INVARIANT="0",
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
    finally:
        assert (
            torch.cuda.is_initialized(),
            torch.distributed.is_initialized(),
        ) == before
    assert result.returncode == 0, log_path.read_text()[-12000:]


@pytest.mark.parametrize("attention_blocks,recurrent_blocks", [(9, 3), (2, 5)])
def test_recurrent_native_allocation_separates_token_pages_and_live_state_slots(
    tmp_path, monkeypatch, attention_blocks, recurrent_blocks
):
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from vllm.v1.spec_decode.remote.runtime.recurrent_kv import make_recurrent_kv_config
    from vllm.v1.worker.utils import allocate_kv_cache

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    config, specs = recurrent_cache_inputs(tmp_path, max_num_reqs=recurrent_blocks - 1)
    state_name, attention_name = specs
    state_spec, attention_spec = specs.values()
    budget = (
        recurrent_blocks * state_spec.page_size_bytes
        + attention_blocks * attention_spec.page_size_bytes
    )
    cache = make_recurrent_kv_config(
        config, specs, budget, recurrent_blocks - 1, KVCacheLayout.LBNHC
    )
    assert cache.num_blocks == attention_blocks
    assert cache.recurrent_num_blocks == recurrent_blocks
    assert all(tensor.size == budget for tensor in cache.kv_cache_tensors)
    allocated = allocate_kv_cache(cache, torch.device("cpu"), KVCacheLayout.LBNHC)
    state_bytes, attention = allocated[state_name], allocated[attention_name]
    assert attention.shape[0] == attention_blocks
    assert state_bytes.shape == (recurrent_blocks, 1, 1, state_spec.page_size_bytes)
    assert (
        len({tensor.untyped_storage().data_ptr() for tensor in allocated.values()}) == 1
    )
    assert all(
        tensor.untyped_storage().nbytes() == budget for tensor in allocated.values()
    )
    spans = sorted(
        (
            tensor.data_ptr(),
            tensor.data_ptr()
            + tensor.element_size()
            * (
                1
                + sum(
                    (n - 1) * stride for n, stride in zip(tensor.shape, tensor.stride())
                )
            ),
        )
        for tensor in allocated.values()
    )
    assert spans[0][1] <= spans[1][0]
    attention[1:].fill_(3)
    assert not state_bytes.count_nonzero()
    pages = state_bytes.squeeze(dim=(1, 2))
    conv_shape, ssm_shape = state_spec.shapes
    conv_bytes = torch.empty(conv_shape, dtype=torch.bfloat16).numel() * 2
    convolution = (
        pages[:, :conv_bytes].view(torch.bfloat16).view(recurrent_blocks, *conv_shape)
    )
    ssm = pages[:, conv_bytes:].view(torch.float32).view(recurrent_blocks, *ssm_shape)
    convolution[1:].fill_(7)
    assert not ssm.count_nonzero()
    ssm[1:].fill_(11)
    assert torch.all(convolution[1:] == 7) and torch.all(ssm[1:] == 11)
    assert torch.all(attention[1:] == 3)
    assert not attention[0].count_nonzero()
    assert not convolution[0].count_nonzero() and not ssm[0].count_nonzero()


@pytest.mark.parametrize("invalid", ["empty", "mixed", "unknown", "host"])
def test_recurrent_cache_rejects_descriptor_without_one_device_capacity(
    tmp_path, monkeypatch, invalid
):
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from vllm.v1.spec_decode.remote.runtime.recurrent_kv import make_recurrent_kv_config

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    config, specs = recurrent_cache_inputs(tmp_path, max_num_reqs=2)
    state, attention = specs.values()
    budget = 3 * state.page_size_bytes + 9 * attention.page_size_bytes
    cache = make_recurrent_kv_config(config, specs, budget, 2, KVCacheLayout.LBNHC)
    descriptor = cache.kv_cache_tensors[0]
    if invalid == "empty":
        descriptor = replace(descriptor, layers=[])
    elif invalid == "mixed":
        descriptor = replace(descriptor, layers=list(specs))
    elif invalid == "unknown":
        descriptor = replace(descriptor, layers=["unknown.layer"])
    else:
        descriptor = replace(descriptor, host_resident=True)
    with pytest.raises(ValueError, match="known device-only|cannot mix token pages"):
        cache.num_blocks_of(descriptor)


def test_recurrent_cache_budget_keeps_live_states_before_admitting_token_pages(
    tmp_path, monkeypatch
):
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from vllm.v1.spec_decode.remote.runtime.recurrent_kv import make_recurrent_kv_config

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    config, specs = recurrent_cache_inputs(tmp_path, max_num_reqs=2)
    state, attention = specs.values()
    budget = 3 * state.page_size_bytes + attention.page_size_bytes
    with pytest.raises(ValueError, match="recurrent slots and two token pages"):
        make_recurrent_kv_config(config, specs, budget, 2, KVCacheLayout.LBNHC)


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


@pytest.mark.parametrize(
    "backend,expected",
    [
        (None, None),
        ("auto", None),
        ("AUTO", None),
        ("triton_attn", "TRITON_ATTN"),
        ("TRITON_ATTN", "TRITON_ATTN"),
        ("unknown_backend", "invalid"),
    ],
)
def test_token_cli_target_attention_uses_native_normalization(
    monkeypatch, backend, expected
):
    import vllm.engine.arg_utils as engine
    import vllm.v1.spec_decode.remote.runtime.token_cli as cli
    import vllm.v1.spec_decode.remote.runtime.token_gpu as runtime

    captured = {}

    def engine_args(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(create_engine_config=lambda: object())

    monkeypatch.setattr(engine, "EngineArgs", engine_args)
    monkeypatch.setattr(runtime, "make_token_gpu_adapter", lambda *a, **kw: object())
    argv = [
        "--endpoint",
        "unix:///tmp/unused-token-cli.sock",
        "--target-model",
        "test/target",
        "--draft-model",
        "test/draft",
    ]
    if backend is not None:
        argv.extend(["--target-attention-backend", backend])
    args = cli.parser().parse_args(argv)
    if expected == "invalid":
        with pytest.raises(ValueError, match="Unknown attention backend"):
            cli.create_adapter(args)
        assert not captured
        return
    cli.create_adapter(args)
    actual = captured["attention_config"].backend
    assert (actual.name if actual is not None else None) == expected
    assert captured["speculative_config"]["attention_backend"] is None


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
def token_executor(monkeypatch, tmp_path):
    import vllm.v1.spec_decode.remote.runtime.token_gpu as runtime
    import vllm.v1.worker.gpu.buffer_utils as buffers
    import vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator as native
    from vllm.v1.worker.gpu.block_table import BlockTables

    from .test_token_runtime import token_config

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
    executor.vllm_config = token_config(tmp_path)
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
    from vllm.v1.spec_decode.remote.capabilities import (
        SpeculatorPlacementCapabilities,
    )
    from vllm.v1.spec_decode.remote.errors import (
        CompletionUnknownError,
        CudaRuntimeError,
        runtime_unusable,
    )
    from vllm.v1.spec_decode.remote.protocol import FeatureSlot, TargetFeatureSchema
    from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
    from vllm.v1.spec_decode.remote.runtime.schema import RuntimeContract
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
        max_num_reqs=4,
        max_model_len=16,
        vocab_size=128,
        max_k=3,
    )
    contract = RuntimeContract(
        TargetFeatureSchema(3, (FeatureSlot("token_ids", "int32", ()),)),
        SpeculatorPlacementCapabilities(
            state_dependency="own_kv",
            required_features=("token_ids",),
            standalone_weights="complete",
        ),
        "target",
        "tokenizer",
        "draft",
    )
    server = TokenDraftServer(TokenOnlyDraftAdapter(state, contract))
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


@contextmanager
def _fail_token_forward_at(executor, boundary, failure, before_raise=None):
    """Inject host failure around real model/logits/D2H submission boundaries."""
    name, phase = boundary.split("_")
    code = {
        "model": inspect.unwrap(executor.speculator._run_model).__code__,
        "logits": inspect.unwrap(executor.speculator.model.compute_logits).__code__,
        "copy": inspect.unwrap(executor.forward_batch).__code__,
    }.get(name)

    def inject(frame, event, called):
        if name == "copy":
            matches = (
                frame.f_code is code
                and event == f"c_{phase}"
                and getattr(called, "__name__", None) == "cpu"
                and isinstance(getattr(called, "__self__", None), torch.Tensor)
            )
        else:
            matches = frame.f_code is code and event == phase
        if matches:
            if before_raise is not None:
                before_raise()
            raise failure

    previous = sys.getprofile()
    sys.setprofile(inject)
    try:
        yield
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize(
    "boundary",
    [
        "model_call",
        "model_return",
        "logits_call",
        "logits_return",
        "copy_call",
        "copy_return",
    ],
)
@pytest.mark.parametrize("completion_fails", [False, True])
def test_token_forward_failure_waits_before_reusing_request_storage(
    token_executor, monkeypatch, boundary, completion_fails
):
    from vllm.v1.spec_decode.remote.errors import (
        CompletionUnknownError,
        runtime_unusable,
    )
    from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
    from vllm.v1.spec_decode.remote.runtime.token_state import (
        TokenAdvanceRow,
        TokenStateMachine,
    )

    executor, _, synced, _ = token_executor
    pool = DraftBlockPool(9, 4)
    state = TokenStateMachine(
        executor, pool, max_num_reqs=1, max_model_len=16, vocab_size=128, max_k=1
    )
    old, new = SequenceKey("completion", 0, 0), SequenceKey("completion", 0, 1)
    state.open(old)
    state.prefill(old, 0, (1, 2), is_final=True)
    slot = state.sequences[old].req_index
    blocks = pool.blocks(slot)
    synced.clear()
    failure = MemoryError("injected host metadata allocation")
    completion = RuntimeError("injected completion failure")

    def synchronize(device):
        synced.append(True)
        if completion_fails:
            raise completion

    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)
    expected = CompletionUnknownError if completion_fails else MemoryError
    with (
        pytest.raises(expected) as raised,
        _fail_token_forward_at(executor, boundary, failure),
    ):
        state.advance(0, TokenAdvanceRow(old, 2, None, 0, 0, (3,)), 1)
    assert synced == [True]
    assert not state.sequences[old].ready
    if completion_fails:
        assert runtime_unusable(raised.value)
        assert raised.value.__cause__ is completion
        assert completion.__context__ is failure
        assert pool.blocks(slot) == blocks
    else:
        assert raised.value is failure
        assert not runtime_unusable(failure)
        state.close((old,))
        assert pool.num_free_blocks == 8
        state.open(new)
        assert state.sequences[new].req_index == slot
        state.prefill(new, 0, (7, 8), is_final=True)
        assert state.advance(
            1, TokenAdvanceRow(new, 2, None, 0, 0, (9,)), 1
        ).token_ids == (1,)


def test_token_forward_fatal_cuda_failure_does_not_submit_completion_work(
    token_executor,
):
    from vllm.v1.spec_decode.remote.errors import CudaRuntimeError, runtime_unusable

    executor, _, synced, _ = token_executor
    failure = CudaRuntimeError("injected failed CUDA context", cuda_error_code=700)
    with (
        pytest.raises(CudaRuntimeError) as raised,
        _fail_token_forward_at(executor, "model_return", failure),
    ):
        executor.forward(SequenceKey("fatal", 0, 0), 0, 0, (1,), (1,), sample=True)
    assert raised.value is failure and runtime_unusable(raised.value)
    assert not synced


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_token_cuda_forward_failure_completes_before_native_kv_reuse(tmp_path):
    """Exercise all host failure boundaries with native weights, KV and kernels."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import Qwen3Config, Qwen3ForCausalLM, TokenizersBackend

    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.v1.spec_decode.remote.runtime.token_cli import create_adapter, parser
    from vllm.v1.spec_decode.remote.runtime.token_state import TokenAdvanceRow

    path = tmp_path / "qwen3"
    config = Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=128,
        head_dim=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        Qwen3ForCausalLM(config).save_pretrained(path)
    TokenizersBackend(
        tokenizer_object=Tokenizer(WordLevel({f"token{i}": i for i in range(128)})),
        bos_token="token0",
        eos_token="token1",
        pad_token="token2",
    ).save_pretrained(path)
    args = parser().parse_args(
        [
            "--endpoint",
            f"unix://{tmp_path}/unused.sock",
            "--target-model",
            str(path),
            "--draft-model",
            str(path),
            "--device",
            "cuda:0",
            "--dtype",
            "bfloat16",
            "--attention-backend",
            "TRITON_ATTN",
            "--target-attention-backend",
            "TRITON_ATTN",
            "--max-model-len",
            "64",
            "--max-feature-tokens",
            "64",
            "--max-batch-size",
            "1",
            "--num-speculative-tokens",
            "1",
            "--block-size",
            "16",
            "--draft-kv-cache-gib",
            "0.01",
        ]
    )
    try:
        adapter = create_adapter(args)
        adapter.warmup()
        state = adapter.state
        executor = state.executor
        baseline = SequenceKey("cuda-completion", 0, 0)
        state.open(baseline)
        state.prefill(baseline, 0, (1, 2), is_final=True)
        expected = state.advance(0, TokenAdvanceRow(baseline, 2, None, 0, 0, (3,)), 1)
        state.close((baseline,))
        for generation, boundary in enumerate(
            (
                "model_call",
                "model_return",
                "logits_call",
                "logits_return",
                "copy_call",
                "copy_return",
            ),
            start=1,
        ):
            key = SequenceKey("cuda-completion", 0, generation)
            state.open(key)
            state.prefill(key, 0, (1, 2), is_final=True)
            slot = state.sequences[key].req_index
            blocks = state.block_pool.blocks(slot)
            completion = torch.cuda.Event()

            def submit_marker(completion=completion):
                torch.cuda._sleep(100_000_000)
                completion.record()

            failure = MemoryError(f"injected {boundary} failure")
            with (
                pytest.raises(MemoryError) as raised,
                _fail_token_forward_at(executor, boundary, failure, submit_marker),
            ):
                state.advance(0, TokenAdvanceRow(key, 2, None, 0, 0, (3,)), 1)
            assert raised.value is failure
            assert completion.query(), "failed forward returned with GPU work pending"
            state.close((key,))
            assert state.block_pool.num_free_blocks == executor.num_blocks - 1
            replacement = SequenceKey("cuda-reuse", 0, generation)
            state.open(replacement)
            assert state.sequences[replacement].req_index == slot
            state.prefill(replacement, 0, (1, 2), is_final=True)
            assert state.block_pool.blocks(slot) == blocks
            actual = state.advance(
                0, TokenAdvanceRow(replacement, 2, None, 0, 0, (3,)), 1
            )
            assert actual.token_ids == expected.token_ids
            state.close((replacement,))
    finally:
        cleanup_dist_env_and_memory()
