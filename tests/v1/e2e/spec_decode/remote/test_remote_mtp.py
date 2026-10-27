# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real model coverage of process-separated drafting on one NVIDIA GPU."""

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

import pytest

from vllm import SamplingParams
from vllm.platforms import current_platform
from vllm.transformers_utils.utils import maybe_model_redirect

from ..utils import get_spec_decode_metric_value

MODEL = "XiaomiMiMo/MiMo-7B-RL"
REVISION = "6299b5a2c45daf0c429285c92b8e61a5bd011c0d"
GEMMA_MODEL = "google/gemma-4-E2B-it"
GEMMA_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
GEMMA_DRAFT = "google/gemma-4-E2B-it-assistant"
GEMMA_DRAFT_REVISION = "2d874ef7d29f9a30599a1e4b3c1cbc9595f005df"


@contextmanager
def draft_server(
    model: str,
    log_path: Path,
    *,
    target_kv: bool = False,
    token_only_draft: str | None = None,
    max_num_seqs: int = 4,
    revision: str | None = REVISION,
    num_speculative_tokens: int = 1,
    attention_backend: str = "FLASH_ATTN",
    draft_kv_cache_gib: float = 0.25,
    max_feature_tokens: int = 64,
    trust_remote_code: bool = True,
    moe_backend: str | None = None,
    on_ready: Callable[[subprocess.Popen], None] | None = None,
):
    with tempfile.TemporaryDirectory(prefix="vllm-remote-") as directory:
        path = Path(directory) / "draft.sock"
        command = [
            sys.executable,
            "-m",
            "vllm.v1.spec_decode.remote.server",
            "--adapter",
            "gpu",
            "--endpoint",
            f"unix://{path}",
            "--target-model",
            model,
            "--draft-model",
            model,
            "--method",
            "mtp",
            "--num-speculative-tokens",
            str(num_speculative_tokens),
            "--device",
            "cuda:0",
            "--dtype",
            "bfloat16",
            "--max-batch-size",
            "4",
            "--max-model-len",
            "128",
            "--max-feature-tokens",
            str(max_feature_tokens),
            "--draft-kv-cache-gib",
            str(draft_kv_cache_gib),
            "--block-size",
            "16",
            "--attention-backend",
            attention_backend,
            "--target-attention-backend",
            attention_backend,
        ]
        if trust_remote_code:
            command.append("--trust-remote-code")
        if moe_backend is not None:
            command.extend(["--moe-backend", moe_backend])
        if revision is not None:
            command.extend(
                ["--target-revision", revision, "--draft-revision", revision]
            )
        if target_kv:
            command = [
                sys.executable,
                "-m",
                "vllm.v1.spec_decode.remote.runtime.target_kv_cli",
                "--endpoint",
                f"unix://{path}",
                "--target-model",
                model,
                "--draft-model",
                maybe_model_redirect(GEMMA_DRAFT),
                "--target-revision",
                GEMMA_REVISION,
                "--draft-revision",
                GEMMA_DRAFT_REVISION,
                "--device",
                "cuda:0",
                "--max-num-seqs",
                str(max_num_seqs),
                "--max-connections",
                "2",
                "--max-model-len",
                "128",
                "--max-feature-tokens",
                "64",
                "--language-model-only",
            ]
        elif token_only_draft is not None:
            command = [
                sys.executable,
                "-m",
                "vllm.v1.spec_decode.remote.runtime.token_cli",
                "--endpoint",
                f"unix://{path}",
                "--target-model",
                model,
                "--draft-model",
                token_only_draft,
                "--num-speculative-tokens",
                "3",
                "--max-batch-size",
                "4",
                "--max-model-len",
                "128",
                "--max-feature-tokens",
                "64",
                "--draft-kv-cache-gib",
                "0.03125",
                "--attention-backend",
                "TRITON_ATTN",
                "--target-attention-backend",
                "TRITON_ATTN",
            ]
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                deadline = time.monotonic() + 600
                while True:
                    if process.poll() is not None:
                        pytest.fail(
                            f"draft server exited {process.returncode}: {log_path}"
                        )
                    with socket.socket(socket.AF_UNIX) as probe:
                        if probe.connect_ex(str(path)) == 0:
                            break
                    if time.monotonic() >= deadline:
                        pytest.fail(
                            f"draft server did not listen after warmup: {log_path}"
                        )
                    time.sleep(0.1)
                if on_ready is not None:
                    on_ready(process)
                yield f"unix://{path}"
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                        pytest.fail(f"draft server did not shut down: {log_path}")
                assert process.returncode == 0, f"draft server failed: {log_path}"


@contextmanager
def paused_draft_server(process: subprocess.Popen):
    """Stop an owned draft process until the caller finishes its timeout check."""
    try:
        process.send_signal(signal.SIGSTOP)
        deadline = time.monotonic() + 5
        while True:
            stopped = os.waitid(os.P_PID, process.pid, os.WSTOPPED | os.WNOHANG)
            if stopped is not None:
                assert stopped.si_status == signal.SIGSTOP
                break
            assert process.poll() is None, "draft server exited before stopping"
            if time.monotonic() >= deadline:
                pytest.fail("draft server did not stop")
            time.sleep(0.01)
        yield
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGCONT)


@pytest.mark.parametrize(
    "transport,target_kv,num_workers,max_num_seqs",
    [
        ("inline", False, 1, 4),
        ("cuda_ipc", False, 1, 4),
        ("inline", False, 2, 4),
        ("cuda_ipc", False, 2, 4),
        ("inline", True, 1, 4),
        ("inline", True, 2, 4),
        ("inline", True, 1, 1),
    ],
)
def test_remote_mtp_matches_target_and_uses_drafts(
    vllm_runner, monkeypatch, tmp_path, transport, target_kv, num_workers, max_num_seqs
):
    """Exercise real chunked prefill, verification and completed-request reuse."""
    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if max_num_seqs == 1:
        monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    model = maybe_model_redirect(GEMMA_MODEL if target_kv else MODEL)
    revision = GEMMA_REVISION if target_kv else REVISION
    backend = "TRITON_ATTN" if target_kv else "FLASH_ATTN"
    kwargs = dict(
        revision=revision,
        tokenizer_revision=revision,
        trust_remote_code=not target_kv,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=64,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=True,
        block_size=16,
        kv_cache_memory_bytes=1 << 30,
        gpu_memory_utilization=0.6,
        attention_config={"backend": backend},
        disable_log_stats=False,
    )
    if target_kv:
        kwargs["language_model_only"] = True
    prompts = [
        {"prompt_token_ids": [11 + i % 4 for i in range(length)]}
        for length in (15, 16, 17, 65)
    ]
    sampling = SamplingParams(
        temperature=0, max_tokens=16, ignore_eos=True, detokenize=False
    )
    with vllm_runner(model, **kwargs) as baseline:
        expected = baseline.llm.generate(prompts, sampling, use_tqdm=False)
    with ExitStack() as servers:
        endpoints = tuple(
            servers.enter_context(
                draft_server(
                    model,
                    tmp_path / f"draft-{i}.log",
                    target_kv=target_kv,
                    max_num_seqs=max_num_seqs,
                )
            )
            for i in range(num_workers)
        )
        config = dict(
            method="mtp",
            model=maybe_model_redirect(GEMMA_DRAFT) if target_kv else model,
            revision=GEMMA_DRAFT_REVISION if target_kv else REVISION,
            num_speculative_tokens=1,
            draft_tensor_parallel_size=1,
            draft_sample_method="greedy",
            attention_backend=backend,
        )
        remote_options = dict(
            transport=transport,
            failure_policy="error",
            request_timeout_ms=5000,
            startup_timeout_ms=600000,
        )
        if num_workers == 1:
            config["remote_draft"] = dict(endpoint=endpoints[0], **remote_options)
        else:
            config["execution"] = dict(
                placement="remote", endpoints=endpoints, **remote_options
            )
        with vllm_runner(model, speculative_config=config, **kwargs) as remote:
            actual_config = remote.llm.llm_engine.vllm_config.speculative_config
            assert actual_config.use_remote_draft()
            assert actual_config.get_remote_draft_config().transport == transport
            assert actual_config.get_remote_draft_endpoints() == endpoints
            sibling = None
            if target_kv and max_num_seqs == 1:
                from vllm.v1.spec_decode.remote.runtime.target_kv_schema import (
                    create_target_kv_contract,
                )
                from vllm.v1.spec_decode.remote.target_kv_protocol import TargetKVHello
                from vllm.v1.spec_decode.remote.target_kv_session import (
                    TargetKVRemoteSession,
                )

                contract = create_target_kv_contract(remote.llm.llm_engine.vllm_config)
                budget = contract.snapshot_budget
                sibling = servers.enter_context(
                    closing(
                        TargetKVRemoteSession(
                            endpoints[0],
                            TargetKVHello(
                                "completed-request-credit",
                                contract.fingerprint,
                                contract.hidden_size,
                                contract.vocab_size,
                                budget.max_model_len,
                                budget.max_pages,
                                budget.max_bytes,
                                max_num_seqs=contract.max_num_seqs,
                            ),
                            expected_specs=contract.expected_specs,
                            request_timeout_ms=5000,
                            startup_timeout_ms=600000,
                        )
                    )
                )
            for inputs, reference in (
                (prompts, expected),
                (list(reversed(prompts)), list(reversed(expected))),
            ):
                outputs = remote.llm.generate(inputs, sampling, use_tqdm=False)
                assert len(outputs) == len(reference)
                assert [list(x.outputs[0].token_ids) for x in outputs] == [
                    list(x.outputs[0].token_ids) for x in reference
                ]
                assert all(len(x.outputs[0].token_ids) == 16 for x in outputs)
                if sibling is not None:
                    # Drain finished-request cleanup without closing the engine's
                    # draft connection or changing its GPU allocations.
                    remote.llm.sleep(level=0, mode="wait")
                    try:
                        key = sibling.open_sequence(0, deadline=sibling.deadline())
                        assert sibling.active(key)
                        sibling.close_sequences((key,))
                        assert not sibling.records
                    finally:
                        remote.llm.wake_up(tags=["scheduling"])
            remote.llm.llm_engine.do_log_stats()
            metrics = remote.llm.get_metrics()
            for name in (
                "vllm:spec_decode_num_draft_tokens",
                "vllm:spec_decode_num_accepted_tokens",
            ):
                assert get_spec_decode_metric_value(metrics, name) > 0


@pytest.fixture
def sliding_window_models(tmp_path, mixed_attention):
    """Equal real weights with different target/draft attention windows."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import MistralConfig, MistralForCausalLM, TokenizersBackend

    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(WordLevel({f"token{i}": i for i in range(128)})),
        bos_token="token0",
        eos_token="token1",
        pad_token="token2",
    )
    config = MistralConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        sliding_window=32,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = MistralForCausalLM(config)
    paths = (tmp_path / "target", tmp_path / "draft")
    for index, (path, window) in enumerate(zip(paths, (32, 16))):
        model.config.sliding_window = window
        if mixed_attention:
            layers = ["sliding_attention", "full_attention"]
            model.config.layer_types = layers if index == 0 else list(reversed(layers))
        else:
            model.config.layer_types = ["sliding_attention"] * 2
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
    return tuple(str(path) for path in paths)


@pytest.mark.parametrize("mixed_attention", [False, True], ids=["uniform", "mixed"])
def test_remote_sliding_window_draft_matches_native_and_target(
    vllm_runner,
    monkeypatch,
    tmp_path,
    sliding_window_models,
    mixed_attention,
    record_property,
):
    """Retain old KV positions while native attention applies the draft window."""
    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    target, draft = sliding_window_models
    kwargs = dict(
        tokenizer_mode="hf",
        trust_remote_code=False,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=True,
        block_size=16,
        kv_cache_memory_bytes=32 << 20,
        gpu_memory_utilization=0.2,
        attention_config={"backend": "TRITON_ATTN"},
        disable_log_stats=False,
    )
    prompts = [
        {"prompt_token_ids": [3 + (i * 7) % 125 for i in range(length)]}
        for length in (15, 17, 33, 65)
    ]
    batches = (prompts[:1], [prompts[i] for i in (2, 0, 3, 1)])
    sampling = SamplingParams(
        temperature=0, max_tokens=8, ignore_eos=True, detokenize=False
    )

    def generate(runner):
        return [
            [
                list(output.outputs[0].token_ids)
                for output in runner.llm.generate(batch, sampling, use_tqdm=False)
            ]
            for batch in batches
        ]

    def assert_speculation(runner, mode):
        runner.llm.llm_engine.do_log_stats()
        metrics = runner.llm.get_metrics()
        for name in (
            "vllm:spec_decode_num_draft_tokens",
            "vllm:spec_decode_num_accepted_tokens",
        ):
            value = get_spec_decode_metric_value(metrics, name)
            record_property(f"{mode}.{name}", value)
            assert value > 0

    def attention_windows(model):
        return [layer.self_attn.attn.sliding_window for layer in model.model.layers]

    def draft_attention_windows(worker):
        return attention_windows(worker.model_runner.get_draft_model())

    config = dict(
        method="draft_model",
        model=draft,
        num_speculative_tokens=3,
        draft_tensor_parallel_size=1,
        draft_sample_method="greedy",
        attention_backend="TRITON_ATTN",
    )
    with vllm_runner(target, **kwargs) as baseline:
        assert baseline.llm.apply_model(attention_windows) == [
            [32, None] if mixed_attention else [32, 32]
        ]
        expected = generate(baseline)
    assert all(len(tokens) == 8 for batch in expected for tokens in batch)
    with vllm_runner(target, speculative_config=config, **kwargs) as native:
        assert native.llm.collective_rpc(draft_attention_windows) == [
            [None, 16] if mixed_attention else [16, 16]
        ]
        assert generate(native) == expected
        assert_speculation(native, "native")
    for num_workers in (1, 2):
        with ExitStack() as servers:
            endpoints = tuple(
                servers.enter_context(
                    draft_server(
                        target,
                        tmp_path / f"token-{num_workers}-{i}.log",
                        token_only_draft=draft,
                    )
                )
                for i in range(num_workers)
            )
            remote_config = dict(
                config,
                execution=dict(
                    placement="remote",
                    endpoints=endpoints,
                    transport="inline",
                    failure_policy="error",
                    request_timeout_ms=5000,
                    startup_timeout_ms=600000,
                ),
            )
            with vllm_runner(
                target, speculative_config=remote_config, **kwargs
            ) as remote:
                actual_config = remote.llm.llm_engine.vllm_config.speculative_config
                assert actual_config.use_remote_draft()
                assert actual_config.get_remote_draft_endpoints() == endpoints
                assert generate(remote) == expected
                assert_speculation(remote, f"remote-{num_workers}")


def _tiny_exaone_mtp_model(tmp_path, mtp_windows):
    """Initialize distinct real target layers and save them as native MTP modules."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        ExaoneMoeConfig,
        ExaoneMoeForCausalLM,
        TokenizersBackend,
    )
    from transformers.models.exaone_moe.modeling_exaone_moe import ExaoneMoeRMSNorm

    num_modules = len(mtp_windows)
    num_target_layers = max(2, num_modules)
    config = ExaoneMoeConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=num_target_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        layer_types=["full_attention"] * num_target_layers,
        mlp_layer_types=["dense"] * num_target_layers,
        sliding_window=32,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        num_nextn_predict_layers=num_modules,
        mtp_layer_types=[
            "full_attention" if window is None else "sliding_attention"
            for window in mtp_windows
        ],
        mtp_sliding_windows=[window or 32 for window in mtp_windows],
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        tie_word_embeddings=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261024)
        model = ExaoneMoeForCausalLM(config)
        fusion = torch.nn.Linear(256, 128, bias=False)
        norms = torch.nn.ModuleDict(
            {
                name: ExaoneMoeRMSNorm(128, eps=config.rms_norm_eps)
                for name in (
                    "norm",
                    "pre_fc_norm_hidden",
                    "pre_fc_norm_embedding",
                )
            }
        )
    weights = dict(model.state_dict())
    assert 1 <= num_modules <= len(model.model.layers)
    for index in range(num_modules):
        weights.update(
            {
                f"mtp.layers.{index}.{name}": tensor.clone()
                for name, tensor in model.model.layers[index].state_dict().items()
            }
        )
    for left in range(num_modules):
        for right in range(left + 1, num_modules):
            for name in ("self_attn.q_proj.weight", "mlp.gate_proj.weight"):
                assert not torch.equal(
                    weights[f"mtp.layers.{left}.{name}"],
                    weights[f"mtp.layers.{right}.{name}"],
                )
    weights["mtp.fc.weight"] = fusion.weight.detach()
    weights.update(
        {f"mtp.{name}": tensor for name, tensor in norms.state_dict().items()}
    )
    path = tmp_path / "exaone"
    model.save_pretrained(path, state_dict=weights)
    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(WordLevel({f"token{i}": i for i in range(128)})),
        bos_token="token0",
        eos_token="token1",
        pad_token="token2",
    )
    tokenizer.save_pretrained(path)
    return str(path)


@pytest.fixture
def tiny_exaone_mtp_model(tmp_path, mtp_window):
    return _tiny_exaone_mtp_model(tmp_path, (mtp_window,))


@pytest.fixture
def tiny_exaone_two_mtp_model(tmp_path):
    return _tiny_exaone_mtp_model(tmp_path, (None, None))


@pytest.fixture
def tiny_exaone_three_mtp_model(tmp_path):
    return _tiny_exaone_mtp_model(tmp_path, (None, 16, 32))


@pytest.mark.parametrize("mtp_window", [None])
def test_remote_mtp_timeout_falls_back_to_target(
    vllm_runner, monkeypatch, tmp_path, tiny_exaone_mtp_model
):
    """A stalled draft process must preserve complete target-only generation."""
    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    model = tiny_exaone_mtp_model
    kwargs = dict(
        trust_remote_code=False,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=32,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=True,
        block_size=16,
        kv_cache_memory_bytes=32 << 20,
        gpu_memory_utilization=0.2,
        attention_config={"backend": "TRITON_ATTN"},
        disable_log_stats=False,
    )
    prompts = [
        {"prompt_token_ids": [3 + (i * 7) % 125 for i in range(length)]}
        for length in (17, 33)
    ]
    sampling = SamplingParams(
        temperature=0, max_tokens=8, ignore_eos=True, detokenize=False
    )

    def generate(runner):
        outputs = runner.llm.generate(prompts, sampling, use_tqdm=False)
        assert len(outputs) == len(prompts)
        tokens = [list(output.outputs[0].token_ids) for output in outputs]
        assert all(len(row) == 8 for row in tokens)
        return tokens

    def draft_tokens(runner):
        runner.llm.llm_engine.do_log_stats()
        return get_spec_decode_metric_value(
            runner.llm.get_metrics(), "vllm:spec_decode_num_draft_tokens"
        )

    with vllm_runner(model, **kwargs) as baseline:
        expected = generate(baseline)
    processes: list[subprocess.Popen] = []
    with draft_server(
        model,
        tmp_path / "timeout-draft.log",
        revision=None,
        attention_backend="TRITON_ATTN",
        draft_kv_cache_gib=0.03125,
        max_feature_tokens=32,
        trust_remote_code=False,
        on_ready=processes.append,
    ) as endpoint:
        config = dict(
            method="mtp",
            model=model,
            num_speculative_tokens=1,
            draft_tensor_parallel_size=1,
            draft_sample_method="greedy",
            attention_backend="TRITON_ATTN",
            remote_draft=dict(
                endpoint=endpoint,
                transport="inline",
                failure_policy="target_only",
                request_timeout_ms=5000,
                startup_timeout_ms=600000,
            ),
        )
        with vllm_runner(model, speculative_config=config, **kwargs) as remote:
            assert generate(remote) == expected
            drafted = draft_tokens(remote)
            assert drafted > 0
            (process,) = processes
            with paused_draft_server(process):
                assert generate(remote) == expected
                assert draft_tokens(remote) == drafted


def _unpack_mtp_experts(state):
    """Preserve packed HF expert values under native per-expert checkpoint names."""
    import torch

    weights = {}
    for name, tensor in state.items():
        if name.endswith(".experts.gate_up_proj"):
            prefix = name.removesuffix(".gate_up_proj")
            gates, ups = tensor.chunk(2, dim=1)
            for expert in range(tensor.shape[0]):
                weights[f"{prefix}.{expert}.gate_proj.weight"] = gates[expert].clone()
                weights[f"{prefix}.{expert}.up_proj.weight"] = ups[expert].clone()
            reconstructed = torch.stack(
                [
                    torch.cat(
                        (
                            weights[f"{prefix}.{e}.gate_proj.weight"],
                            weights[f"{prefix}.{e}.up_proj.weight"],
                        )
                    )
                    for e in range(tensor.shape[0])
                ]
            )
            assert torch.equal(reconstructed, tensor)
        elif name.endswith((".experts.up_proj", ".experts.down_proj")):
            prefix, projection = name.rsplit(".", 1)
            for expert in range(tensor.shape[0]):
                weights[f"{prefix}.{expert}.{projection}.weight"] = tensor[
                    expert
                ].clone()
            assert torch.equal(
                torch.stack(
                    [
                        weights[f"{prefix}.{e}.{projection}.weight"]
                        for e in range(tensor.shape[0])
                    ]
                ),
                tensor,
            )
        else:
            weights[name] = tensor.clone()
    return weights


def _save_mtp_checkpoint(path, config, weights):
    import torch
    from safetensors.torch import load_file, save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import TokenizersBackend

    for name, tensor in weights.items():
        dtype = (
            torch.float32
            if name.endswith(
                (
                    ".gate.weight",
                    ".e_score_correction_bias",
                    ".expert_bias",
                    ".A_log",
                    ".dt_bias",
                    ".D",
                )
            )
            else torch.bfloat16
        )
        weights[name] = tensor.to(dtype).contiguous()
        assert torch.isfinite(weights[name]).all()
    path.mkdir()
    save_file(weights, path / "model.safetensors", metadata={"format": "pt"})
    reloaded = load_file(path / "model.safetensors")
    assert weights.keys() == reloaded.keys()
    assert all(torch.equal(tensor, reloaded[name]) for name, tensor in weights.items())
    config.save_pretrained(path)
    vocabulary = {token: i for i, token in enumerate(("<pad>", "<s>", "</s>", "<unk>"))}
    vocabulary.update({f"t{i}": i for i in range(4, 128)})
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = TokenizersBackend(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        model_max_length=128,
    )
    tokenizer.save_pretrained(path)
    return str(path)


@pytest.fixture
def tiny_glm4_moe_mtp_model(tmp_path):
    """Keep a real routed target layer and its initialized native MTP counterpart."""
    import torch
    from transformers import Glm4MoeConfig, Glm4MoeForCausalLM
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeRMSNorm

    config = Glm4MoeConfig(
        architectures=["Glm4MoeForCausalLM"],
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        use_qk_norm=True,
        num_mtp_layers=1,
        tie_word_embeddings=False,
        attention_bias=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        dtype="bfloat16",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261024)
        model = Glm4MoeForCausalLM(config)
        state = dict(model.state_dict())
        state.update(
            {
                f"model.layers.2.{name}": tensor.clone()
                for name, tensor in model.model.layers[1].state_dict().items()
            }
        )
        for name in ("enorm", "hnorm", "shared_head.norm"):
            norm = Glm4MoeRMSNorm(128, eps=config.rms_norm_eps)
            state[f"model.layers.2.{name}.weight"] = norm.weight.detach().clone()
        fusion = torch.nn.Linear(256, 128, bias=False)
        state["model.layers.2.eh_proj.weight"] = fusion.weight.detach().clone()

    return _save_mtp_checkpoint(
        tmp_path / "glm4-moe", config, _unpack_mtp_experts(state)
    )


@pytest.fixture
def tiny_glm4_moe_lite_mtp_model(tmp_path):
    """Keep dense MLA dimensions and a routed MTP block in the real checkpoint."""
    import torch
    from transformers import Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
        Glm4MoeLiteRMSNorm,
    )

    config = Glm4MoeLiteConfig(
        architectures=["Glm4MoeLiteForCausalLM"],
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        n_group=1,
        topk_group=1,
        mlp_layer_types=["dense", "sparse"],
        kv_lora_rank=512,
        q_lora_rank=128,
        qk_rope_head_dim=64,
        qk_nope_head_dim=192,
        v_head_dim=256,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        attention_bias=False,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        dtype="bfloat16",
    )
    config.update(
        {
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "num_nextn_predict_layers": 1,
        }
    )
    assert not hasattr(config, "index_topk")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261024)
        model = Glm4MoeLiteForCausalLM(config)
        state = dict(model.state_dict())
        # The native predictor indexes from the target config's layer count,
        # even though the derived draft config reports zero target layers.
        state.update(
            {
                f"model.layers.2.{name}": tensor.clone()
                for name, tensor in model.model.layers[1].state_dict().items()
            }
        )
        for name in ("enorm", "hnorm", "shared_head.norm"):
            norm = Glm4MoeLiteRMSNorm(128, eps=config.rms_norm_eps)
            state[f"model.layers.2.{name}.weight"] = norm.weight.detach().clone()
        fusion = torch.nn.Linear(256, 128, bias=False)
        state["model.layers.2.eh_proj.weight"] = fusion.weight.detach().clone()
        state["model.layers.2.embed_tokens.weight"] = state[
            "model.embed_tokens.weight"
        ].clone()
        state["model.layers.2.shared_head.head.weight"] = state[
            "lm_head.weight"
        ].clone()
    return _save_mtp_checkpoint(
        tmp_path / "glm4-moe-lite", config, _unpack_mtp_experts(state)
    )


@pytest.fixture
def tiny_qwen3_next_mtp_model(tmp_path):
    """Retain real GDN target state while cloning its attention layer for MTP."""
    import torch
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

    config = Qwen3NextConfig(
        architectures=["Qwen3NextForCausalLM"],
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        decoder_sparse_step=1,
        num_nextn_predict_layers=1,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        },
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        dtype="bfloat16",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261024)
        model = Qwen3NextForCausalLM(config)
        state = dict(model.state_dict())
        state.update(
            {
                f"mtp.layers.0.{name}": tensor.clone()
                for name, tensor in model.model.layers[1].state_dict().items()
            }
        )
        for name in ("norm", "pre_fc_norm_hidden", "pre_fc_norm_embedding"):
            norm = Qwen3NextRMSNorm(128, config.rms_norm_eps)
            assert torch.equal(norm.weight, model.model.norm.weight)
            state[f"mtp.{name}.weight"] = norm.weight.detach().clone()
        fusion = torch.nn.Linear(256, 128, bias=False)
        state["mtp.fc.weight"] = fusion.weight.detach().clone()
    return _save_mtp_checkpoint(
        tmp_path / "qwen3-next", config, _unpack_mtp_experts(state)
    )


@pytest.fixture
def tiny_nemotron_h_mtp_model(tmp_path):
    """Exercise the real Mamba/attention/MoE target and attention/MoE MTP."""
    import torch
    from transformers import NemotronHConfig, NemotronHForCausalLM
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHRMSNorm

    config = NemotronHConfig(
        architectures=["NemotronHForCausalLM"],
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        layers_block_type=["linear_attention", "full_attention", "moe"],
        mamba_num_heads=4,
        mamba_head_dim=64,
        ssm_state_size=64,
        n_groups=1,
        conv_kernel=4,
        chunk_size=128,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=128,
        num_nextn_predict_layers=1,
        mtp_layers_block_type=["full_attention", "moe"],
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        dtype="bfloat16",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261024)
        model = NemotronHForCausalLM(config)
        state = dict(model.state_dict())
        for source, destination in ((1, 0), (2, 1)):
            state.update(
                {
                    f"mtp.layers.{destination}.{name}": tensor.clone()
                    for name, tensor in model.model.layers[source].state_dict().items()
                }
            )
        for name in ("layers.0.enorm", "layers.0.hnorm", "layers.1.final_layernorm"):
            norm = NemotronHRMSNorm(128, config.layer_norm_epsilon)
            assert torch.equal(norm.weight, model.model.norm_f.weight)
            state[f"mtp.{name}.weight"] = norm.weight.detach().clone()
        fusion = torch.nn.Linear(256, 128, bias=False)
        state["mtp.layers.0.eh_proj.weight"] = fusion.weight.detach().clone()
    return _save_mtp_checkpoint(
        tmp_path / "nemotron-h", config, _unpack_mtp_experts(state)
    )


@pytest.fixture
def tiny_hyv3_mtp_model(tmp_path):
    """Preserve HYV3's routed layer and distinct post-block MTP normalization."""
    import torch
    from transformers import HYV3Config, HYV3ForCausalLM
    from transformers.models.hy_v3.modeling_hy_v3 import HYV3RMSNorm

    config = HYV3Config(
        architectures=["HYV3ForCausalLM"],
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        num_experts=4,
        num_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        mlp_layer_types=["dense", "sparse"],
        tie_word_embeddings=False,
        attention_bias=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        dtype="bfloat16",
    )
    config.update(
        {
            "first_k_dense_replace": 1,
            "expert_hidden_dim": 128,
            "route_norm": True,
            "qk_norm": True,
            "use_cla": False,
            "num_nextn_predict_layers": 1,
        }
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261024)
        model = HYV3ForCausalLM(config)
        state = dict(model.state_dict())
        state.update(
            {
                f"model.layers.2.{name}": tensor.clone()
                for name, tensor in model.model.layers[1].state_dict().items()
            }
        )
        for name in ("enorm", "hnorm", "final_layernorm"):
            norm = HYV3RMSNorm(128, eps=config.rms_norm_eps)
            state[f"model.layers.2.{name}.weight"] = norm.weight.detach().clone()
        fusion = torch.nn.Linear(256, 128, bias=False)
        state["model.layers.2.eh_proj.weight"] = fusion.weight.detach().clone()
    weights = {}
    for name, tensor in _unpack_mtp_experts(state).items():
        name = name.replace(".mlp.shared_experts.", ".mlp.shared_mlp.")
        name = name.replace(".mlp.e_score_correction_bias", ".mlp.expert_bias")
        assert name not in weights
        weights[name] = tensor
    return _save_mtp_checkpoint(tmp_path / "hyv3", config, weights)


@pytest.mark.parametrize(
    ("omitted", "tied"),
    [
        ("eh_proj", False),
        ("self_attn.q_proj", False),
        ("mlp.experts.0.down_proj", False),
        (None, False),
        (None, True),
    ],
)
def test_hyv3_mtp_loads_only_complete_owned_parameters_through_llm(
    tiny_hyv3_mtp_model, tmp_path, omitted, tied
):
    """Reject missing owned weights while sharing target embeddings and heads."""
    import json

    from safetensors.torch import load_file, save_file

    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    path = Path(tiny_hyv3_mtp_model) / "model.safetensors"
    weights = load_file(path)
    spec = {"method": "mtp", "num_speculative_tokens": 1}
    if omitted is None:
        config_path = path.parent / "config.json"
        config = json.loads(config_path.read_text())
        config["tie_word_embeddings"] = tied
        config_path.write_text(json.dumps(config))
        if tied:
            del weights["lm_head.weight"]
            save_file(weights, path, metadata={"format": "pt"})
        draft_path = tmp_path / "draft-only"
        draft_path.mkdir()
        (draft_path / "config.json").write_text(json.dumps(config))
        save_file(
            {
                name: tensor
                for name, tensor in weights.items()
                if name.startswith("model.layers.2.")
            },
            draft_path / "model.safetensors",
            metadata={"format": "pt"},
        )
        spec["model"] = str(draft_path)
    else:
        del weights[f"model.layers.2.{omitted}.weight"]
        save_file(weights, path, metadata={"format": "pt"})
    missing = {
        "self_attn.q_proj": "qkv_proj",
        "mlp.experts.0.down_proj": "w2_weight",
    }.get(omitted, omitted or "")
    options = dict(
        model=tiny_hyv3_mtp_model,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=32,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        async_scheduling=False,
        attention_config={"backend": "TRITON_ATTN"},
        moe_backend="triton",
        kv_cache_memory_bytes=32 << 20,
        gpu_memory_utilization=0.2,
        speculative_config=spec,
    )
    program = """
import json
import sys
from vllm import LLM, SamplingParams
if sys.argv[2]:
    try:
        LLM(**json.loads(sys.argv[1]))
    except ValueError as error:
        assert "not initialized from checkpoint" in str(error), error
        assert sys.argv[2] in str(error), error
    else:
        raise AssertionError("LLM admitted an incomplete HYV3 MTP checkpoint")
else:
    llm = LLM(**json.loads(sys.argv[1]))

    def shared_parameters(worker):
        target = worker.model_runner.model
        draft = worker.model_runner.get_draft_model()
        return (
            draft.model.embed_tokens is target.model.embed_tokens,
            all(layer.shared_head.head is target.lm_head
                for layer in draft.model.layers.values()),
        )

    assert llm.collective_rpc(shared_parameters) == [(True, True)]
    outputs = llm.generate(
        {"prompt_token_ids": [4, 5, 6]},
        SamplingParams(temperature=0, max_tokens=2, min_tokens=2, ignore_eos=True),
    )
    tokens = outputs[0].outputs[0].token_ids
    assert len(tokens) == 2 and all(0 <= token < 128 for token in tokens), tokens
"""
    log_path = tmp_path / "hyv3-mtp-loading.log"
    with log_path.open("w") as log:
        result = subprocess.run(
            [sys.executable, "-c", program, json.dumps(options), missing],
            env=dict(
                os.environ,
                VLLM_ENABLE_V1_MULTIPROCESSING="0",
                VLLM_USE_V2_MODEL_RUNNER="1",
                VLLM_BATCH_INVARIANT="0",
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
    assert result.returncode == 0, log_path.read_text()[-12000:]


@pytest.fixture
def tiny_ernie_mtp_model(tmp_path):
    """Use a routed ERNIE target and its native dense, single-token MTP block."""
    import torch
    from transformers import Ernie4_5_MoeConfig, Ernie4_5_MoeForCausalLM
    from transformers.models.ernie4_5_moe.modeling_ernie4_5_moe import (
        Ernie4_5_MoeRMSNorm,
    )

    config = Ernie4_5_MoeConfig(
        architectures=["Ernie4_5_MoeForCausalLM"],
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        moe_intermediate_size=128,
        moe_k=2,
        moe_num_experts=4,
        moe_num_shared_experts=1,
        moe_layer_start_index=1,
        moe_layer_end_index=1,
        moe_layer_interval=1,
        num_nextn_predict_layers=1,
        tie_word_embeddings=False,
        use_bias=False,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        dtype="bfloat16",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261024)
        model = Ernie4_5_MoeForCausalLM(config)
        state = dict(model.state_dict())
        state.update(
            {
                f"model.mtp_block.0.{name}": tensor.clone()
                for name, tensor in model.model.layers[0].state_dict().items()
            }
        )
        for name in ("mtp_emb_norm", "mtp_hidden_norm"):
            norm = Ernie4_5_MoeRMSNorm(128, config.rms_norm_eps)
            assert torch.equal(norm.weight, model.model.norm.weight)
            state[f"model.{name}.0.weight"] = norm.weight.detach().clone()
        fusion = torch.nn.Linear(256, 128, bias=False)
        state["model.mtp_linear_proj.0.weight"] = fusion.weight.detach().clone()
    weights = {}
    for name, tensor in _unpack_mtp_experts(state).items():
        name = name.replace(".mlp.gate.moe_statics.", ".mlp.moe_statics.")
        assert name not in weights
        weights[name] = tensor
    return _save_mtp_checkpoint(tmp_path / "ernie", config, weights)


@pytest.mark.parametrize(
    "mtp_family,mtp_window,k",
    [
        pytest.param(family, window, k, id=f"k{k}-{label}")
        for k in (1, 3)
        for family, window, label in (
            ("exaone", None, "exaone-full"),
            ("exaone", 16, "exaone-swa"),
            ("glm4_moe", None, "glm4-moe"),
            ("qwen3_next", None, "qwen3-next"),
            ("nemotron_h", None, "nemotron-h"),
            ("hyv3", None, "hyv3"),
            ("glm4_moe_lite", None, "glm4-moe-lite"),
        )
    ]
    + [
        pytest.param("ernie", None, 1, id="k1-ernie"),
        pytest.param("exaone_two", None, 2, id="k2-exaone-two-full"),
        pytest.param("exaone_three", None, 3, id="k3-exaone-three-mixed"),
    ],
)
def test_remote_native_mtp_matches_native_and_target(
    vllm_runner,
    monkeypatch,
    tmp_path,
    request,
    mtp_family,
    mtp_window,
    k,
    record_property,
):
    """Compare native and standalone MTP across chunk/window boundaries and reuse."""
    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    hybrid_target = mtp_family in ("qwen3_next", "nemotron_h")
    # GDN/Mamba backends do not implement batch-invariant kernels.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0" if hybrid_target else "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    model = request.getfixturevalue(f"tiny_{mtp_family}_mtp_model")
    mtp_windows = {
        "exaone_two": (None, None),
        "exaone_three": (None, 16, 32),
    }.get(mtp_family, (mtp_window,))
    num_modules = len(mtp_windows)
    dense_mla = mtp_family == "glm4_moe_lite"
    attention_backend = "TRITON_MLA" if dense_mla else "TRITON_ATTN"
    attention_config = {"backend": attention_backend}
    if dense_mla:
        attention_config["mla_prefill_backend"] = "FLASH_ATTN"
    kwargs = dict(
        trust_remote_code=False,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=32,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=True,
        block_size=16,
        kv_cache_memory_bytes=32 << 20,
        gpu_memory_utilization=0.2,
        attention_config=attention_config,
        disable_log_stats=False,
    )
    moe_backend = None if mtp_family.startswith("exaone") else "triton"
    if moe_backend is not None:
        kwargs["moe_backend"] = moe_backend
    if hybrid_target:
        kwargs.update(mamba_backend="triton", use_replayssm=False)
    if mtp_family == "qwen3_next":
        kwargs["gdn_prefill_backend"] = "triton"
        monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "triton")
    prompts = [
        {"prompt_token_ids": [3 + (i * 7) % 125 for i in range(length)]}
        for length in (15, 16, 17, 31, 32, 33, 65)
    ]
    batches = (
        prompts[:1],
        [prompts[i] for i in (4, 1, 6, 2)],
        [prompts[i] for i in (5, 3, 2, 0)],
    )
    sampling = SamplingParams(
        temperature=0,
        max_tokens=8,
        ignore_eos=True,
        detokenize=False,
        logprobs=5 if hybrid_target or dense_mla else None,
    )
    # MLA prefill and absorbed decode use different floating-point operations.
    compare_target_logprobs = hybrid_target or dense_mla
    record_property(
        "target_comparison",
        "top5_logprobs" if compare_target_logprobs else "exact_tokens",
    )
    record_property("placement_comparison", "exact_tokens")
    record_property("batch_invariant", not hybrid_target)

    def generate(runner):
        return [
            [
                (
                    list(output.outputs[0].token_ids),
                    output.outputs[0].text,
                    output.outputs[0].logprobs,
                )
                for output in runner.llm.generate(batch, sampling, use_tqdm=False)
            ]
            for batch in batches
        ]

    def compare_outputs(actual, reference, mode):
        from tests.models.utils import check_logprobs_close

        assert len(actual) == len(reference) == len(batches)
        mismatches = 0
        for outputs, expected_outputs, prompts in zip(actual, reference, batches):
            assert len(outputs) == len(expected_outputs) == len(prompts)
            assert all(len(output[0]) == 8 for output in outputs)
            mismatches += sum(
                token != expected_token
                for output, expected_output in zip(outputs, expected_outputs)
                for token, expected_token in zip(output[0], expected_output[0])
            )
            if compare_target_logprobs:
                assert all(
                    output[2] is not None and len(output[2]) == 8
                    for output in (*outputs, *expected_outputs)
                )
                check_logprobs_close(
                    outputs_0_lst=expected_outputs,
                    outputs_1_lst=outputs,
                    name_0="target",
                    name_1=mode,
                )
            else:
                assert [output[0] for output in outputs] == [
                    output[0] for output in expected_outputs
                ]
        record_property(f"{mode}.token_mismatches", mismatches)

    def speculation_counts(runner, mode):
        runner.llm.llm_engine.do_log_stats()
        metrics = runner.llm.get_metrics()
        counts = tuple(
            get_spec_decode_metric_value(metrics, name)
            for name in (
                "vllm:spec_decode_num_drafts",
                "vllm:spec_decode_num_draft_tokens",
                "vllm:spec_decode_num_accepted_tokens",
            )
        )
        for label, value in zip(("proposals", "drafted", "accepted"), counts):
            record_property(f"{mode}.{label}", value)
        assert counts[0] > 0 and counts[1] > 0
        assert 0 <= counts[2] <= counts[1]
        return counts

    def target_windows(model):
        from vllm.model_executor.layers.attention import Attention
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention

        return [
            layer.sliding_window
            for layer in model.modules()
            if isinstance(layer, (Attention, MLAAttention))
        ]

    def mla_layout(model):
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention

        layouts = []
        for layer in model.modules():
            if not isinstance(layer, MLAAttention):
                continue
            assert layer.prefill_backend is not None
            layouts.append(
                {
                    "kv_lora_rank": layer.kv_lora_rank,
                    "qk_nope_head_dim": layer.qk_nope_head_dim,
                    "qk_rope_head_dim": layer.qk_rope_head_dim,
                    "v_head_dim": layer.v_head_dim,
                    "num_heads": layer.num_heads,
                    "head_size": layer.head_size,
                    "cache_shape": tuple(layer.kv_cache.shape[1:]),
                    "decode": layer.get_attn_backend().get_name(),
                    "prefill": layer.prefill_backend.get_name(),
                    "sparse": layer.impl.is_sparse,
                }
            )
        return layouts

    def draft_mla_layout(worker):
        return mla_layout(worker.model_runner.get_draft_model())

    expected_mla_layout = {
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "num_heads": 2,
        "head_size": 576,
        "cache_shape": (16, 576),
        "decode": "TRITON_MLA",
        "prefill": "FLASH_ATTN",
        "sparse": False,
    }

    def moe_backends(model):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )

        backends = []
        for layer in model.modules():
            if isinstance(layer, RoutedExperts):
                method = layer.quant_method
                assert isinstance(method, UnquantizedFusedMoEMethod)
                backends.append(method.unquantized_backend.name)
        return backends

    def draft_moe_backends(worker):
        return moe_backends(worker.model_runner.get_draft_model())

    def recurrent_policy(worker):
        from vllm.model_executor.layers.mamba.abstract import MambaBase
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )
        from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
        from vllm.v1.kv_cache_interface import MambaSpec

        runner = worker.model_runner
        layers = []
        for layer in runner.get_model().modules():
            if not isinstance(layer, MambaBase):
                continue
            spec = layer.get_kv_cache_spec(worker.vllm_config)
            assert isinstance(spec, MambaSpec)
            shapes = tuple(layer.get_state_shape())
            assert spec.shapes == shapes
            assert tuple(tuple(t.shape[1:]) for t in layer.kv_cache) == shapes
            kernels = {}
            if isinstance(layer, QwenGatedDeltaNetAttention):
                kernels = {
                    "prefill": layer.gdn_prefill_backend,
                    "decode": layer.gdn_decode_kernel,
                }
            elif isinstance(layer, MambaMixer2):
                kernels = {"ssu": layer.mamba_config.backend.name}
            layers.append(
                {"type": layer.mamba_type.name, "shapes": shapes, "kernels": kernels}
            )
        return {
            "state": type(runner.model_state).__name__,
            "k": worker.vllm_config.num_speculative_tokens,
            "layers": layers,
        }

    def check_recurrence(runner, mode):
        policies = runner.llm.collective_rpc(recurrent_policy)
        record_property(f"{mode}.recurrent_policy", policies)
        assert len(policies) == 1
        policy = policies[0]
        assert policy["k"] == (0 if mode == "target" else k)
        if hybrid_target:
            assert policy["state"] == "MambaHybridModelState"
            assert len(policy["layers"]) == 1
            layer = policy["layers"][0]
            is_gdn = mtp_family == "qwen3_next"
            assert layer["type"] == ("GDN_ATTN" if is_gdn else "MAMBA2")
            conv_shape, state_shape = layer["shapes"]
            assert sorted(conv_shape) == [3 + policy["k"], 512 if is_gdn else 384]
            assert state_shape == ((2, 128, 128) if is_gdn else (4, 64, 64))
            assert layer["kernels"] == (
                {"prefill": "triton", "decode": "triton"}
                if is_gdn
                else {"ssu": "TRITON"}
            )
        else:
            assert policy["state"] == "DefaultModelState"
            assert policy["layers"] == []
        return policies

    def draft_recurrent_layers(worker):
        from vllm.model_executor.layers.mamba.abstract import MambaBase

        return sum(
            isinstance(layer, MambaBase)
            for layer in worker.model_runner.get_draft_model().modules()
        )

    def draft_module_identity(worker):
        import torch

        from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
            MultiModuleMTPSpeculator,
        )

        runner = worker.model_runner
        assert isinstance(runner.speculator, MultiModuleMTPSpeculator)
        draft = runner.get_draft_model()
        layers = draft.model.layers
        assert len(layers) == draft.model.num_mtp_layers == num_modules
        for left in range(num_modules):
            for right in range(left + 1, num_modules):
                assert not torch.equal(
                    layers[left].self_attn.qkv_proj.weight,
                    layers[right].self_attn.qkv_proj.weight,
                )
                assert not torch.equal(
                    layers[left].mlp.gate_up_proj.weight,
                    layers[right].mlp.gate_up_proj.weight,
                )
        return {
            "speculator": type(runner.speculator).__name__,
            "model": type(draft).__name__,
            "modules": len(layers),
            "distinct_attention_and_mlp_weights": True,
        }

    def draft_kv_policy(worker):
        from vllm.model_executor.layers.attention import Attention
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention

        draft = worker.model_runner.get_draft_model()
        policies = []
        for attention in draft.modules():
            if not isinstance(attention, (Attention, MLAAttention)):
                continue
            spec = attention.get_kv_cache_spec(worker.vllm_config)
            policies.append(
                (
                    attention.sliding_window,
                    type(spec).__name__,
                    getattr(spec, "sliding_window", None),
                )
            )
        return policies

    config = dict(
        method="mtp",
        model=model,
        num_speculative_tokens=k,
        draft_tensor_parallel_size=1,
        draft_sample_method="greedy",
        attention_backend=attention_backend,
    )
    with vllm_runner(model, **kwargs) as baseline:
        windows = baseline.llm.apply_model(target_windows)
        assert windows == [[None] * (1 if hybrid_target else max(2, num_modules))]
        record_property("target.attention_windows", windows)
        if dense_mla:
            layout = baseline.llm.apply_model(mla_layout)
            assert layout == [[expected_mla_layout] * 2]
            record_property("target.mla_layout", layout)
        backends = baseline.llm.apply_model(moe_backends)
        expected_moe_count = 2 if mtp_family == "qwen3_next" else int(bool(moe_backend))
        assert backends == [["TRITON"] * expected_moe_count]
        record_property("target.moe_backends", backends)
        check_recurrence(baseline, "target")
        expected = generate(baseline)
    assert all(len(output[0]) == 8 for batch in expected for output in batch)
    with vllm_runner(model, speculative_config=config, **kwargs) as native:
        policy = native.llm.collective_rpc(draft_kv_policy)
        assert policy == [
            [
                (
                    window,
                    "MLAAttentionSpec"
                    if dense_mla
                    else "FullAttentionSpec"
                    if window is None
                    else "SlidingWindowSpec",
                    window,
                )
                for window in mtp_windows
            ]
        ]
        record_property("native.kv_policy", policy)
        if num_modules > 1:
            identity = native.llm.collective_rpc(draft_module_identity)
            record_property("native.module_identity", identity)
        if dense_mla:
            layout = native.llm.collective_rpc(draft_mla_layout)
            assert layout == [[expected_mla_layout]]
            record_property("native.mla_layout", layout)
        backends = native.llm.collective_rpc(draft_moe_backends)
        draft_has_moe = not mtp_family.startswith("exaone") and mtp_family != "ernie"
        assert backends == [["TRITON"] if draft_has_moe else []]
        record_property("native.moe_backends", backends)
        recurrent_layers = native.llm.collective_rpc(draft_recurrent_layers)
        assert recurrent_layers == [0]
        record_property("native.recurrent_layers", recurrent_layers)
        native_recurrence = check_recurrence(native, "native")
        native_outputs = generate(native)
        compare_outputs(native_outputs, expected, "native")
        native_counts = speculation_counts(native, "native")
    server_log = tmp_path / f"{mtp_family}-draft.log"
    with draft_server(
        model,
        server_log,
        revision=None,
        num_speculative_tokens=k,
        attention_backend=attention_backend,
        draft_kv_cache_gib=0.03125,
        max_feature_tokens=32,
        trust_remote_code=False,
        moe_backend=moe_backend,
    ) as endpoint:
        remote_config = dict(
            config,
            remote_draft=dict(
                endpoint=endpoint,
                transport="inline",
                failure_policy="error",
                request_timeout_ms=5000,
                startup_timeout_ms=600000,
            ),
        )
        with vllm_runner(model, speculative_config=remote_config, **kwargs) as remote:
            actual_config = remote.llm.llm_engine.vllm_config.speculative_config
            assert actual_config.use_remote_draft()
            assert actual_config.get_remote_draft_endpoints() == (endpoint,)
            assert check_recurrence(remote, "remote") == native_recurrence
            remote_outputs = generate(remote)
            compare_outputs(remote_outputs, expected, "remote")
            for actual_batch, native_batch in zip(remote_outputs, native_outputs):
                assert [output[0] for output in actual_batch] == [
                    output[0] for output in native_batch
                ]
                if dense_mla:
                    assert [output[2] for output in actual_batch] == [
                        output[2] for output in native_batch
                    ]
            if dense_mla:
                record_property("placement.logprobs", "exact_returned_top5")
            assert speculation_counts(remote, "remote") == native_counts
    if dense_mla:
        assert any(
            "Using " in line and "TRITON_MLA" in line and "backend." in line
            for line in server_log.read_text().splitlines()
        )
        record_property("remote.decode_backend_from_server_log", "TRITON_MLA")


def _run_multi_module_oracle(model_path: str, result_path: str) -> None:
    """Compare incremental native GPU state with fresh full-prefix executions."""
    import itertools
    import json

    import torch
    from transformers import ExaoneMoeForCausalLM

    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
    from vllm.v1.spec_decode.remote.protocol import SequenceKey, SpeculatorStatusCode
    from vllm.v1.spec_decode.remote.runtime.gpu import (
        GPUDraftStepExecutor,
        make_gpu_adapter,
    )
    from vllm.v1.spec_decode.remote.server import PrefillInputs, StandaloneProposalBatch
    from vllm.v1.spec_decode.remote.session import frame_to_tensor, tensor_to_frame
    from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
        MultiModuleMTPSpeculator,
    )

    torch.set_num_threads(1)
    # Features come from the actual target checkpoint. Controlled verification
    # outcomes below test state alignment, not acceptance quality of random weights.
    target = ExaoneMoeForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    depth = target.config.num_nextn_predict_layers
    assert depth in (2, 3)
    expected_windows = [None, None] if depth == 2 else [None, 16, 32]
    config = EngineArgs(
        model=model_path,
        trust_remote_code=False,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=128,
        # Four incremental rows plus one independent full-prefix reference.
        max_num_seqs=5,
        max_num_batched_tokens=256,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=True,
        block_size=16,
        attention_config={"backend": "TRITON_ATTN"},
        speculative_config={
            "method": "mtp",
            "model": model_path,
            "num_speculative_tokens": depth,
            "draft_tensor_parallel_size": 1,
            "attention_backend": "TRITON_ATTN",
        },
    ).create_engine_config()
    adapter = make_gpu_adapter(
        config, device=torch.device("cuda:0"), kv_cache_bytes=32 << 20
    )
    executor = adapter.executor
    assert isinstance(executor, GPUDraftStepExecutor)
    assert isinstance(executor.speculator, MultiModuleMTPSpeculator)
    layers = [
        layer
        for layer in executor.speculator.model.modules()
        if isinstance(layer, Attention)
    ]
    assert len(layers) == depth
    assert [layer.sliding_window for layer in layers] == expected_windows
    assert all(layer.get_attn_backend().get_name() == "TRITON_ATTN" for layer in layers)
    # Both dense spec types use this backend's token-major page representation.
    assert all(layer.kv_cache.shape[1:] == (2, 16, 64) for layer in layers)
    assert all(layer.kv_cache.dtype == torch.bfloat16 for layer in layers)
    modules = executor.speculator.model.model.layers
    assert len(modules) == depth
    for left, right in itertools.combinations(modules, 2):
        assert not torch.equal(
            left.self_attn.qkv_proj.weight, right.self_attn.qkv_proj.weight
        )
        assert not torch.equal(
            left.mlp.gate_up_proj.weight, right.mlp.gate_up_proj.weight
        )
    groups = executor.kv_config.kv_cache_groups
    assert len(groups) == len(set(expected_windows))
    layout, ranges = [], []
    for layer, window in zip(layers, expected_windows):
        matches = [
            (i, group)
            for i, group in enumerate(groups)
            if layer.layer_name in group.layer_names
        ]
        assert len(matches) == 1
        group_id, group = matches[0]
        spec = group.kv_cache_spec
        assert isinstance(spec, (FullAttentionSpec, SlidingWindowSpec))
        assert type(spec).__name__ == (
            "FullAttentionSpec" if window is None else "SlidingWindowSpec"
        )
        assert getattr(spec, "sliding_window", None) == window
        assert spec.block_size == 16 and spec.dtype == torch.bfloat16
        cache = layer.kv_cache
        size = (
            1 + sum((n - 1) * stride for n, stride in zip(cache.shape, cache.stride()))
        ) * cache.element_size()
        ranges.append((cache.data_ptr(), cache.data_ptr() + size))
        layout.append(
            {
                "layer": layer.layer_name,
                "window": window,
                "group": group_id,
                "spec": type(spec).__name__,
                "cache_shape": list(cache.shape),
                "cache_strides": list(cache.stride()),
            }
        )
    for (left, left_end), (right, right_end) in itertools.combinations(ranges, 2):
        assert left_end <= right or right_end <= left
    assert {name for group in groups for name in group.layer_names} == {
        layer.layer_name for layer in layers
    }
    schema = adapter.feature_schema()
    assert adapter.capabilities().num_prefill_lookahead == depth
    batch_ids = itertools.count()
    reference_ids = itertools.count(100)
    checks = []
    prefill_chunks = []

    @torch.inference_mode()
    def target_features(tokens):
        output = target(
            torch.tensor([tokens], dtype=torch.int64),
            use_cache=False,
            output_hidden_states=True,
        )
        hidden = output.hidden_states[-1][0].clone()
        assert hidden.shape == (len(tokens), 128)
        assert torch.isfinite(hidden).all()
        return hidden

    def frames(tokens, positions, hidden, lookahead=None):
        values = {
            "token_ids": torch.tensor(tokens, dtype=torch.int32),
            "positions": torch.tensor(positions, dtype=torch.int64),
            "last_hidden_states": hidden,
            "next_prefill_tokens": lookahead,
        }
        return tuple(
            None if values[slot.kind] is None else tensor_to_frame(i, values[slot.kind])
            for i, slot in enumerate(schema.slots)
        )

    def prefill(key, tokens, hidden, *, chunk_size=None):
        adapter.open_sequence(key)
        offset = 0
        while offset < len(tokens):
            # Nonfinal chunks need D real future prompt tokens. If a tiny
            # scheduling budget cannot leave them, finish with a larger budget.
            remaining = len(tokens) - offset
            count = remaining if chunk_size is None else min(chunk_size, remaining)
            if 0 < remaining - count < depth:
                count = remaining - depth
                if count <= 0:
                    count = remaining
            end = offset + count
            final = end == len(tokens)
            lookahead = None if final else torch.tensor([tokens[end : end + depth]])
            assert count > 0
            assert final or (lookahead is not None and lookahead.shape == (1, depth))
            adapter.prefill(
                key,
                PrefillInputs(
                    offset,
                    count,
                    final,
                    frames(
                        tokens[offset:end],
                        range(offset, end),
                        hidden[offset:end],
                        lookahead,
                    ),
                ),
            )
            prefill_chunks.append(
                {
                    "prompt_length": len(tokens),
                    "offset": offset,
                    "count": count,
                    "final": final,
                    "lookahead": 0 if lookahead is None else lookahead.numel(),
                }
            )
            offset = end

    def advance(keys, token_rows, position_rows, hidden_rows, accepted, recovery, k):
        result = adapter.advance_and_propose(
            StandaloneProposalBatch(
                batch_id=next(batch_ids),
                keys=tuple(keys),
                active_rows=tuple(range(len(keys))),
                num_speculative_tokens=depth,
                accepted_counts=tensor_to_frame(
                    0, torch.tensor(accepted, dtype=torch.int32)
                ),
                recovery_tokens=tensor_to_frame(
                    0, torch.tensor(recovery, dtype=torch.int32)
                ),
                features=frames(
                    list(itertools.chain.from_iterable(token_rows)),
                    list(itertools.chain.from_iterable(position_rows)),
                    torch.cat(hidden_rows),
                ),
                query_lens=tuple(map(len, token_rows)),
                effective_num_speculative_tokens=k,
            )
        )
        assert result.row_statuses == (SpeculatorStatusCode.OK,) * len(keys)
        assert frame_to_tensor(result.valid_lengths).tolist() == [k] * len(keys)
        return frame_to_tensor(result.token_ids)[:, :k].tolist()

    def kv_prefix(key, length):
        req_index = adapter.planner.req_index(key)
        blocks = adapter.planner.block_pool.blocks(req_index)
        indices = torch.tensor(blocks, dtype=torch.int64, device=executor.device)
        result = []
        for module, layer in enumerate(layers):
            # Module j's last j rows can still depend on unverified candidates.
            # Compare only its authoritative materialized prefix.
            valid = max(length - module, 0)
            cache = layer.kv_cache.index_select(0, indices).transpose(1, 2)
            prefix = cache.flatten(0, 1)[:valid].cpu().clone()
            assert torch.isfinite(prefix).all()
            result.append(prefix)
        return result

    def compare_reference(key, tokens, hidden, recovery, k, actual, label):
        assert adapter.planner.num_computed(key) == len(tokens)
        incremental = kv_prefix(key, len(tokens))
        reference = SequenceKey("oracle", next(reference_ids), 0)
        prefill(reference, tokens, hidden)
        expected = advance([reference], [[]], [[]], [hidden[:0]], [0], [recovery], k)[0]
        assert actual == expected, (label, key, k, actual, expected)
        for module, (left, right) in enumerate(
            zip(incremental, kv_prefix(reference, len(tokens)))
        ):
            assert torch.equal(left, right), (label, key, module, len(tokens))
        checks.append(
            {
                "case": label,
                "length": len(tokens),
                "effective_k": k,
                "kv_lengths": [len(tensor) for tensor in incremental],
            }
        )
        adapter.close_sequences((reference,))

    keys = [SequenceKey("incremental", i, 0) for i in range(4)]
    tokens = [
        [3 + (i * 7 + row) % 125 for i in range(n)]
        for row, n in enumerate((1, depth - 1, 125, 126))
    ]
    hidden = [target_features(row) for row in tokens]
    for key, ids, features in zip(keys, tokens, hidden):
        prefill(key, ids, features, chunk_size=32)
    recovery = [17, 23, 29, 37]
    drafts: list[list[int]] = [[], [], [], []]
    max_model_len = config.model_config.max_model_len
    max_length_truncations = []
    outcomes = []
    rounds = [depth, 0, depth, 1, depth, 0, *range(1, depth + 1)]
    active = set(range(len(keys)))
    for step, k in enumerate(rounds):
        finished = {row for row in active if len(tokens[row]) >= max_model_len - 1}
        adapter.close_sequences(tuple(keys[row] for row in finished))
        active -= finished
        token_rows, position_rows, hidden_rows, accepted = {}, {}, {}, {}
        next_recovery = [3 + (step * 11 + row * 13) % 125 for row in range(4)]
        for row in sorted(active):
            if step == 0:
                query, positions, features = [], [], hidden[row][:0]
                accepted[row] = 0
            else:
                start = len(tokens[row])
                # Match scheduler admission, including the target's bonus slot.
                max_query_tokens = max_model_len - start - 1
                assert max_query_tokens >= 1
                submitted_drafts = drafts[row][: max_query_tokens - 1]
                # Cover rejection, every acceptance length and K=0 catch-up.
                accepted[row] = (
                    0
                    if step == 1
                    else (len(submitted_drafts) - row) % (len(submitted_drafts) + 1)
                )
                if len(submitted_drafts) < len(drafts[row]):
                    max_length_truncations.append(
                        {
                            "round": step,
                            "row": row,
                            "start": start,
                            "returned_k": len(drafts[row]),
                            "submitted_k": len(submitted_drafts),
                        }
                    )
                outcomes.append(
                    {
                        "round": step,
                        "row": row,
                        "returned_k": len(drafts[row]),
                        "submitted_k": len(submitted_drafts),
                        "accepted": accepted[row],
                    }
                )
                query = [recovery[row], *submitted_drafts]
                positions = list(range(start, start + len(query)))
                assert positions[-1] < max_model_len - 1
                # The target helper returns inference tensors; take a mutable
                # owned copy before deliberately poisoning rejected input rows.
                features = target_features(tokens[row] + query)[start:].clone()
                valid = accepted[row] + 1
                tokens[row] += query[:valid]
                hidden[row] = torch.cat((hidden[row], features[:valid]))
                # Rejected hidden rows must never enter any native module.
                features[valid:] = torch.nan
            token_rows[row] = query
            position_rows[row] = positions
            hidden_rows[row] = features
        order = [
            row for row in ((3, 1, 0, 2) if step % 2 else (2, 0, 3, 1)) if row in active
        ]
        output = advance(
            [keys[i] for i in order],
            [token_rows[i] for i in order],
            [position_rows[i] for i in order],
            [hidden_rows[i] for i in order],
            [accepted[i] for i in order],
            [next_recovery[i] for i in order],
            k,
        )
        for index, row in enumerate(order):
            drafts[row] = output[index]
            compare_reference(
                keys[row],
                tokens[row],
                hidden[row],
                next_recovery[row],
                k,
                drafts[row],
                f"round-{step}",
            )
        recovery = next_recovery
    assert max_length_truncations == [
        {"round": 1, "row": 2, "start": 125, "returned_k": depth, "submitted_k": 1},
        {"round": 1, "row": 3, "start": 126, "returned_k": depth, "submitted_k": 0},
    ]
    assert {outcome["accepted"] for outcome in outcomes} == set(range(depth + 1))
    adapter.close_sequences(tuple(keys))
    assert adapter.planner.num_open == 0
    # Tiny nonfinal chunks and both block/window edges use real prompt lookahead.
    for length in (15, 16, 17, 31, 32, 33):
        key = SequenceKey("boundary", length, 0)
        ids = [3 + (i * 11) % 125 for i in range(length)]
        features = target_features(ids)
        prefill(key, ids, features, chunk_size=1 if length == 15 else 32)
        output = advance([key], [[]], [[]], [features[:0]], [0], [19], depth)[0]
        compare_reference(key, ids, features, 19, depth, output, f"prompt-{length}")
        adapter.close_sequences((key,))
    # Reuse the same external ID and request slot with a fresh generation.
    key = SequenceKey("incremental", 0, 1)
    ids = [7]
    features = target_features(ids)
    prefill(key, ids, features)
    output = advance([key], [[]], [[]], [features[:0]], [0], [19], 0)[0]
    compare_reference(key, ids, features, 19, 0, output, "reused-k0")
    ids += [19]
    features = target_features(ids)
    output = advance([key], [[19]], [[1]], [features[1:]], [0], [31], depth)[0]
    compare_reference(key, ids, features, 31, depth, output, "reused-max-k")
    adapter.close_sequences((key,))
    assert adapter.planner.num_open == 0
    assert adapter.planner.block_pool.num_free_blocks == executor.num_blocks - 1
    Path(result_path).write_text(
        json.dumps(
            {
                "speculator": type(executor.speculator).__name__,
                "modules": len(layers),
                "layout": layout,
                "disjoint_layer_storage": True,
                "target_features": "real ExaoneMoeForCausalLM eager CPU forward",
                "comparison": "exact proposals and authoritative per-module KV",
                "checks": checks,
                "max_length_truncations": max_length_truncations,
                "outcomes": outcomes,
                "prefill_chunks": prefill_chunks,
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize(
    "mtp_family,depth",
    [("exaone_two", 2), ("exaone_three", 3)],
    ids=["two-full", "three-mixed"],
)
def test_remote_multi_module_mtp_dynamic_k_matches_full_prefix(
    mtp_family, depth, request, monkeypatch, tmp_path, record_property
):
    """Keep K=0..D correct across rejection, reorder, short prompts and slot reuse."""
    import json

    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    model = request.getfixturevalue(f"tiny_{mtp_family}_mtp_model")
    result = tmp_path / "multi-module-oracle.json"
    log_path = tmp_path / "multi-module-oracle.log"
    command = [
        sys.executable,
        "-c",
        "import sys; from tests.v1.e2e.spec_decode.remote.test_remote_mtp "
        "import _run_multi_module_oracle; _run_multi_module_oracle(*sys.argv[1:])",
        model,
        str(result),
    ]
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            assert process.wait(timeout=300) == 0, (
                f"native MTP oracle failed: {log_path}"
            )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    receipt = json.loads(result.read_text())
    assert receipt["modules"] == depth
    assert receipt["speculator"] == "MultiModuleMTPSpeculator"
    assert [layer["window"] for layer in receipt["layout"]] == (
        [None, None] if depth == 2 else [None, 16, 32]
    )
    assert receipt["disjoint_layer_storage"]
    assert len(receipt["checks"]) == 2 * depth + 25
    assert {check["effective_k"] for check in receipt["checks"]} == set(
        range(depth + 1)
    )
    assert max(check["length"] for check in receipt["checks"]) == 127
    assert {15, 16, 17, 31, 32, 33, *range(1, depth)} <= {
        check["length"] for check in receipt["checks"]
    }
    for check in receipt["checks"]:
        assert check["kv_lengths"] == [
            max(check["length"] - module, 0) for module in range(depth)
        ]
    assert receipt["max_length_truncations"] == [
        {"round": 1, "row": 2, "start": 125, "returned_k": depth, "submitted_k": 1},
        {"round": 1, "row": 3, "start": 126, "returned_k": depth, "submitted_k": 0},
    ]
    assert {outcome["accepted"] for outcome in receipt["outcomes"]} == set(
        range(depth + 1)
    )
    chunks = receipt["prefill_chunks"]
    assert (
        sum(
            chunk["prompt_length"] == 15 and chunk["count"] == 1 and not chunk["final"]
            for chunk in chunks
        )
        == 15 - depth
    )
    assert all(chunk["lookahead"] == depth for chunk in chunks if not chunk["final"])
    assert all(chunk["final"] for chunk in chunks if chunk["prompt_length"] < depth)
    record_property("native.full_prefix_oracle", receipt)


def _separate_mtp_draft(source, destination):
    """Keep real checkpoint weights while changing the draft's scalar settings."""
    import json
    import shutil

    shutil.copytree(source, destination, copy_function=os.link)
    path = destination / "config.json"
    config = json.loads(path.read_text())
    config["rms_norm_eps"] = 0.001
    config["rope_parameters"]["rope_theta"] *= 2
    path.unlink()
    path.write_text(json.dumps(config))
    return str(destination)


@pytest.mark.parametrize("mtp_family", ["hyv3", "ernie"])
def test_separate_mtp_checkpoint_settings_reach_public_llm(
    vllm_runner, monkeypatch, tmp_path, request, mtp_family
):
    """A separately published MTP head must retain its own normalization/RoPE."""
    import json

    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    source = request.getfixturevalue(f"tiny_{mtp_family}_mtp_model")
    draft = _separate_mtp_draft(source, tmp_path / "separate-draft")
    target_config = json.loads((Path(source) / "config.json").read_text())
    draft_config = json.loads((Path(draft) / "config.json").read_text())

    def settings(worker):
        model = worker.model_runner.get_draft_model()
        layer = next(iter(model.model.layers.values()))
        norm = layer.enorm if mtp_family == "hyv3" else layer.mtp_emb_norm
        return (
            worker.vllm_config.model_config.hf_config.rms_norm_eps,
            norm.variance_epsilon,
            layer.mtp_block.self_attn.rotary_emb.base,
        )

    with vllm_runner(
        source,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        attention_config={"backend": "TRITON_ATTN"},
        moe_backend="triton",
        kv_cache_memory_bytes=32 << 20,
        gpu_memory_utilization=0.2,
        speculative_config={
            "method": "mtp",
            "model": draft,
            "num_speculative_tokens": 1,
        },
    ) as runner:
        assert runner.llm.collective_rpc(settings) == [
            (
                target_config["rms_norm_eps"],
                draft_config["rms_norm_eps"],
                draft_config["rope_parameters"]["rope_theta"],
            )
        ]
