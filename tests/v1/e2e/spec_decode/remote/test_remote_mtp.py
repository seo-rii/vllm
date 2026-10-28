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
from typing import Any

import pytest

from vllm import SamplingParams
from vllm.platforms import current_platform
from vllm.transformers_utils.utils import maybe_model_redirect

from ..utils import get_spec_decode_metric_value

pytestmark = pytest.mark.usefixtures("enable_pickle")

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
    eagle3_draft: str | None = None,
    revision: str | None = REVISION,
    num_speculative_tokens: int = 1,
    attention_backend: str = "FLASH_ATTN",
    target_attention_backend: str | None = None,
    draft_kv_cache_gib: float = 0.25,
    max_feature_tokens: int = 64,
    max_model_len: int = 128,
    trust_remote_code: bool = True,
    moe_backend: str | None = None,
    batch_invariant: bool | None = None,
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
            eagle3_draft or model,
            "--method",
            "eagle3" if eagle3_draft is not None else "mtp",
            "--num-speculative-tokens",
            str(num_speculative_tokens),
            "--device",
            "cuda:0",
            "--dtype",
            "bfloat16",
            "--max-batch-size",
            "4",
            "--max-model-len",
            str(max_model_len),
            "--max-feature-tokens",
            str(max_feature_tokens),
            "--draft-kv-cache-gib",
            str(draft_kv_cache_gib),
            "--block-size",
            "16",
            "--attention-backend",
            attention_backend,
            "--target-attention-backend",
            target_attention_backend or attention_backend,
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
                str(num_speculative_tokens),
                "--max-batch-size",
                "4",
                "--max-model-len",
                "128",
                "--max-feature-tokens",
                str(max_feature_tokens),
                "--draft-kv-cache-gib",
                str(draft_kv_cache_gib),
                "--attention-backend",
                attention_backend,
                "--target-attention-backend",
                target_attention_backend or attention_backend,
            ]
        environment = None
        if batch_invariant is not None:
            environment = dict(
                os.environ, VLLM_BATCH_INVARIANT=str(int(batch_invariant))
            )
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment,
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
    "transport,target_kv,num_workers,max_num_seqs,registration_oom",
    [
        ("inline", False, 1, 4, False),
        ("cuda_ipc", False, 1, 4, False),
        ("inline", False, 2, 4, False),
        ("cuda_ipc", False, 2, 4, False),
        ("inline", True, 1, 4, False),
        ("inline", True, 2, 4, False),
        ("inline", True, 1, 1, False),
        ("inline", True, 1, 1, True),
    ],
)
def test_remote_mtp_matches_target_and_uses_drafts(
    vllm_runner,
    monkeypatch,
    tmp_path,
    transport,
    target_kv,
    num_workers,
    max_num_seqs,
    registration_oom,
):
    """Exercise real chunked prefill, verification and completed-request reuse."""
    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if registration_oom:
        monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        monkeypatch.setenv("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")
    elif max_num_seqs == 1:
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
            failure_policy="target_only" if registration_oom else "error",
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
            metric_names = (
                "vllm:spec_decode_num_draft_tokens",
                "vllm:spec_decode_num_accepted_tokens",
            )
            before_recovery = None
            if registration_oom:

                class RegistrationRequestId(str):
                    injected = False

                    def __hash__(self):
                        frame = sys._getframe(1)
                        if (
                            not self.injected
                            and frame.f_code.co_name == "_dispatch_snapshot"
                            and frame.f_locals.get("keys")
                        ):
                            self.injected = True
                            raise MemoryError("target-KV request map insertion")
                        return super().__hash__()

                engine = remote.llm.llm_engine
                request_id = RegistrationRequestId("registration-oom")
                engine.add_request(request_id, prompts[0], sampling)
                completed: list[Any] = []
                while engine.has_unfinished_requests():
                    completed.extend(
                        output for output in engine.step() if output.finished
                    )
                assert request_id.injected
                assert len(completed) == 1
                assert list(completed[0].outputs[0].token_ids) == list(
                    expected[0].outputs[0].token_ids
                )
                engine.do_log_stats()
                before_recovery = {
                    name: get_spec_decode_metric_value(remote.llm.get_metrics(), name)
                    for name in metric_names
                }
            sibling = None
            if target_kv and max_num_seqs == 1 and not registration_oom:
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
                if before_recovery is not None:
                    remote.llm.llm_engine.do_log_stats()
                    metrics = remote.llm.get_metrics()
                    for name in metric_names:
                        assert (
                            get_spec_decode_metric_value(metrics, name)
                            > before_recovery[name]
                        )
                    before_recovery = None
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
            for name in metric_names:
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
                        num_speculative_tokens=3,
                        attention_backend="TRITON_ATTN",
                        draft_kv_cache_gib=0.03125,
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


@pytest.mark.parametrize(
    "target_windows,draft_windows",
    [((None,), (16,)), ((32, 32), (None, 16))],
)
def test_separate_exaone_mtp_checkpoint_retains_attention_policy_through_llm(
    vllm_runner, monkeypatch, tmp_path, target_windows, draft_windows
):
    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    target = _tiny_exaone_mtp_model(tmp_path / "target", target_windows)
    draft = _tiny_exaone_mtp_model(tmp_path / "draft", draft_windows)
    kwargs = dict(
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=2,
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
        result = [list(output.outputs[0].token_ids) for output in outputs]
        assert all(len(row) == 8 for row in result)
        return result

    def draft_attention_windows(worker):
        model = worker.model_runner.get_draft_model()
        return [layer.self_attn.attn.sliding_window for layer in model.model.layers]

    with vllm_runner(target, **kwargs) as baseline:
        expected = generate(baseline)
    with vllm_runner(
        target,
        speculative_config={
            "method": "mtp",
            "model": draft,
            "num_speculative_tokens": len(draft_windows),
        },
        **kwargs,
    ) as native:
        assert native.llm.collective_rpc(draft_attention_windows) == [
            list(draft_windows)
        ]
        assert generate(native) == expected
        native.llm.llm_engine.do_log_stats()
        assert (
            get_spec_decode_metric_value(
                native.llm.get_metrics(), "vllm:spec_decode_num_draft_tokens"
            )
            > 0
        )


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


def _check_deepseek_mtp_weights(target, draft, model_path):
    """Account for every checkpoint tensor and native packed parameter."""
    import torch
    from safetensors.torch import load_file

    weights = load_file(Path(model_path) / "model.safetensors")
    checked = set()
    parameter_counts = {}
    assert type(target).__name__ == "DeepseekV3ForCausalLM"
    assert type(draft).__name__ == "DeepSeekMTP"

    def equal(actual, names):
        expected = torch.cat([weights[name] for name in names]).to(actual.dtype)
        assert torch.equal(actual.detach().cpu(), expected), names
        checked.update(names)

    for role, model in (("target", target), ("draft", draft)):
        parameters = dict(model.named_parameters())
        for name, parameter in parameters.items():
            source = name.replace(".mtp_block.", ".")
            if source in weights:
                equal(parameter, [source])
            elif source.endswith(".fused_qkv_a_proj.weight"):
                equal(
                    parameter,
                    [
                        source.replace("fused_qkv_a_proj", projection)
                        for projection in ("q_a_proj", "kv_a_proj_with_mqa")
                    ],
                )
            elif source.endswith(".gate_up_proj.weight"):
                equal(
                    parameter,
                    [
                        source.replace("gate_up_proj", projection)
                        for projection in ("gate_proj", "up_proj")
                    ],
                )
            elif source.endswith((".w13_weight", ".w2_weight")):
                prefix, projection = source.rsplit(".", 1)
                prefix = prefix.removesuffix(".routed_experts")
                for expert, value in enumerate(parameter):
                    names = (
                        ("gate_proj", "up_proj")
                        if projection == "w13_weight"
                        else ("down_proj",)
                    )
                    equal(
                        value,
                        [f"{prefix}.{expert}.{name}.weight" for name in names],
                    )
            elif source.endswith((".W_UV", ".W_UK_T")):
                prefix = source.rsplit(".self_attn.", 1)[0]
                checkpoint_name = f"{prefix}.self_attn.kv_b_proj.weight"
                config = model.config
                original = (
                    weights[checkpoint_name]
                    .to(parameter.dtype)
                    .T.reshape(
                        config.kv_lora_rank,
                        config.num_attention_heads,
                        config.qk_nope_head_dim + config.v_head_dim,
                    )
                )
                key, value = original.split(
                    [config.qk_nope_head_dim, config.v_head_dim], dim=-1
                )
                expected = (
                    value.transpose(0, 1)
                    if source.endswith(".W_UV")
                    else key.permute(1, 2, 0)
                )
                assert torch.equal(parameter.detach().cpu(), expected), name
                checked.add(checkpoint_name)
            else:
                raise AssertionError(f"unchecked native {role} parameter: {name}")
        parameter_counts[role] = len(parameters)

    assert draft.model.embed_tokens is target.model.embed_tokens
    assert draft.lm_head is target.lm_head
    assert tuple(draft.model.layers) == ("2",)
    layer = draft.model.layers["2"]
    assert layer.shared_head.head is target.lm_head
    equal(target.model.embed_tokens.weight, ["model.layers.2.embed_tokens.weight"])
    equal(target.lm_head.weight, ["model.layers.2.shared_head.head.weight"])
    assert checked == weights.keys(), weights.keys() - checked
    return dict(
        target=type(target).__name__,
        draft=type(draft).__name__,
        parameters=parameter_counts,
        checkpoint_tensors=len(checked),
        exact_parameter_roundtrip=True,
        shared_embedding_and_head=True,
        mtp_absolute_layer=2,
    )


def _save_deepseek_mtp_model(directory, ffn_layout):
    """Save a complete initialized HF V3 target and independent native MTP block."""
    import json

    import torch
    from transformers import DeepseekV3Config, DeepseekV3ForCausalLM
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        DeepseekV3DecoderLayer,
        DeepseekV3RMSNorm,
    )

    assert current_platform.is_cpu() and not torch.cuda.is_initialized()
    assert ffn_layout in ("dense", "moe")
    torch.set_num_threads(1)
    root = Path(directory)
    dense = ffn_layout == "dense"
    config = DeepseekV3Config(
        architectures=["DeepseekV3ForCausalLM"],
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        n_routed_experts=0 if dense else 4,
        n_shared_experts=0 if dense else 1,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        first_k_dense_replace=3 if dense else 1,
        n_group=1,
        topk_group=1,
        kv_lora_rank=512,
        q_lora_rank=128,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        num_nextn_predict_layers=1,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        attention_bias=False,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        dtype="bfloat16",
        topk_method="noaux_tc",
        scoring_func="sigmoid",
    )
    assert not hasattr(config, "index_topk")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261026)
        target = DeepseekV3ForCausalLM(config)
        _two_candidate_head(target.lm_head.weight)
        # HF initializes expert tensors in the owning model's post_init().
        # Use its independently initialized absolute-N layer, not raw empty
        # expert Parameters from a standalone decoder constructor.
        mtp_config = DeepseekV3Config.from_dict(config.to_dict(), num_hidden_layers=3)
        mtp = DeepseekV3ForCausalLM(mtp_config).model.layers[2]
        assert isinstance(mtp, DeepseekV3DecoderLayer)
        state = dict(target.state_dict())
        state.update(
            {
                f"model.layers.2.{name}": value
                for name, value in mtp.state_dict().items()
            }
        )
        for name in ("enorm", "hnorm", "shared_head.norm"):
            norm = DeepseekV3RMSNorm(128, config.rms_norm_eps)
            state[f"model.layers.2.{name}.weight"] = norm.weight.detach().clone()
        fusion = torch.nn.Linear(256, 128, bias=False)
        _conditioned_mtp_fusion(fusion.weight)
        state["model.layers.2.eh_proj.weight"] = fusion.weight.detach().clone()
        state["model.layers.2.embed_tokens.weight"] = state[
            "model.embed_tokens.weight"
        ].clone()
        state["model.layers.2.shared_head.head.weight"] = state[
            "lm_head.weight"
        ].clone()
    weights = _unpack_mtp_experts(state)
    _save_mtp_checkpoint(root / "model", config, weights)
    assert not torch.cuda.is_initialized()
    receipt = dict(
        target_architecture=config.architectures[0],
        draft_architecture="DeepSeekMTPModel",
        target_hf_class=type(target).__name__,
        mtp_hf_class=type(mtp).__name__,
        mtp_source="independently_initialized_hf_layer_2_of_3",
        fusion="embedding_1_4_target_hidden_3_4",
        checkpoint_tensors=len(weights),
        exact_checkpoint_roundtrip=True,
        mtp_absolute_layer=2,
        ffn_layout=ffn_layout,
        routed_experts=config.n_routed_experts,
        first_k_dense_replace=config.first_k_dense_replace,
        latent_width=576,
        seed=20261026,
        head_candidate_ids=[3, 4],
        head_scale=32,
        cuda_initialized=False,
    )
    (root / "fixture.json").write_text(json.dumps(receipt) + "\n")


@pytest.fixture
def tiny_deepseek_mtp_model(tmp_path, ffn_layout):
    """Build real initialized weights without changing the serving CUDA platform."""
    root = (tmp_path / f"deepseek-v3-{ffn_layout}").resolve()
    root.mkdir()
    source = Path(__file__).resolve().parents[5]
    command = [
        sys.executable,
        "-c",
        "from tests.v1.e2e.spec_decode.remote.test_remote_mtp "
        "import _save_deepseek_mtp_model; "
        "import sys; _save_deepseek_mtp_model(*sys.argv[1:])",
        str(root),
        ffn_layout,
    ]
    log_path = root / "fixture.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            env=dict(
                os.environ,
                PYTHONPATH=str(source),
                CUDA_VISIBLE_DEVICES="",
                VLLM_TARGET_DEVICE="cpu",
                VLLM_BATCH_INVARIANT="0",
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            assert process.wait(timeout=210) == 0, (
                f"DeepSeek fixture failed: {log_path}"
            )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    return str(root / "model")


def _save_qwen35_mtp_model(directory, *, routed=False):
    """Real text/GDN weights, an independent attention head and inherited RoPE."""
    import json

    import torch
    from transformers import (
        Qwen3_5ForCausalLM,
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeTextConfig,
        Qwen3_5TextConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    assert not torch.cuda.is_initialized()
    config_cls = Qwen3_5MoeTextConfig if routed else Qwen3_5TextConfig
    model_cls = Qwen3_5MoeForCausalLM if routed else Qwen3_5ForCausalLM
    config = config_cls(
        architectures=[model_cls.__name__],
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
        mtp_num_hidden_layers=1,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [1, 1, 2],
            "mrope_interleaved": True,
        },
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        dtype="bfloat16",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261028)
        target = model_cls(config)
        _two_candidate_head(target.lm_head.weight)
        owner_config = config_cls.from_dict(
            config.to_dict(), num_hidden_layers=1, layer_types=["full_attention"]
        )
        owner = model_cls(owner_config)
        weights = dict(target.state_dict())
        weights.update(
            {
                f"mtp.layers.0.{name}": value
                for name, value in owner.model.layers[0].state_dict().items()
            }
        )
        for name in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden"):
            norm = Qwen3_5RMSNorm(128, eps=config.rms_norm_eps)
            assert torch.count_nonzero(norm.weight) == 0
            weights[f"mtp.{name}.weight"] = norm.weight.detach().clone()
        fusion = torch.nn.Linear(256, 128, bias=False)
        _conditioned_mtp_fusion(fusion.weight)
        weights["mtp.fc.weight"] = fusion.weight.detach().clone()
    root = Path(directory)
    model = _save_mtp_checkpoint(root / "model", config, _unpack_mtp_experts(weights))
    receipt = dict(
        synthetic_profile=True,
        seed=20261028,
        target_architecture=model_cls.__name__,
        draft_architecture="Qwen3_5MoeMTP" if routed else "Qwen3_5MTP",
        target_layers=["linear_attention", "full_attention"],
        mtp_layers=["full_attention"],
        checkpoint_tensors=len(_unpack_mtp_experts(weights)),
        target_hf_class=type(target).__name__,
        head_hf_class=type(owner.model.layers[0]).__name__,
        fusion="embedding_1_4_target_hidden_3_4",
        head_candidate_ids=[3, 4],
        head_scale=32,
        saved_mrope_section=config.rope_parameters["mrope_section"],
        saved_mrope_interleaved=True,
        routed=routed,
        cuda_initialized=False,
    )
    (root / "fixture.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return model, receipt


def _tiny_qwen35_mtp_model(tmp_path, *, routed):
    command = [
        sys.executable,
        "-c",
        "import sys; from tests.v1.e2e.spec_decode.remote.test_remote_mtp "
        "import _save_qwen35_mtp_model; "
        "_save_qwen35_mtp_model(sys.argv[1], routed=sys.argv[2] == 'moe')",
        str(tmp_path),
        "moe" if routed else "dense",
    ]
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        VLLM_TARGET_DEVICE="cpu",
        VLLM_BATCH_INVARIANT="0",
    )
    with (tmp_path / "fixture.log").open("w") as log:
        subprocess.run(
            command,
            env=environment,
            check=True,
            timeout=180,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return str(tmp_path / "model")


@pytest.fixture
def tiny_qwen35_dense_mtp_model(tmp_path):
    return _tiny_qwen35_mtp_model(tmp_path, routed=False)


@pytest.fixture
def tiny_qwen35_moe_mtp_model(tmp_path):
    return _tiny_qwen35_mtp_model(tmp_path, routed=True)


@pytest.mark.parametrize("profile", ["dense", "moe"])
@pytest.mark.parametrize("k", [1, 3])
def test_remote_qwen35_text_mtp_matches_native_and_target(
    profile, k, vllm_runner, request, monkeypatch, tmp_path, record_property
):
    """Use native flat-text RoPE and GDN rollback with a full-attention head."""
    import json

    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    model = request.getfixturevalue(f"tiny_qwen35_{profile}_mtp_model")
    record_property("fixture", json.loads((tmp_path / "fixture.json").read_text()))
    _compare_native_and_remote_draft(
        vllm_runner,
        monkeypatch,
        tmp_path,
        request,
        f"qwen35_{profile}",
        None,
        k,
        record_property,
        target_model=model,
    )


@pytest.mark.parametrize(
    "fixture,projection",
    [("dense", "q_proj")] + [("moe", part) for part in ("q_proj", "k_proj", "v_proj")],
)
def test_qwen35_mtp_rejects_incomplete_checkpoint_through_llm(
    fixture, projection, request, tmp_path
):
    """Reject a missing shard during public LLM loading, before serving tokens."""
    import json

    from safetensors.torch import load_file, save_file

    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    model = request.getfixturevalue(f"tiny_qwen35_{fixture}_mtp_model")
    path = Path(model) / "model.safetensors"
    weights = load_file(path)
    missing = f"mtp.layers.0.self_attn.{projection}.weight"
    del weights[missing]
    save_file(weights, path, metadata={"format": "pt"})
    options = dict(
        model=model,
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
        mamba_backend="triton",
        use_replayssm=False,
        gdn_prefill_backend="triton",
        kv_cache_memory_bytes=32 << 20,
        gpu_memory_utilization=0.2,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 1,
            "draft_load_config": {
                "model_loader_extra_config": {"enable_weights_track": None}
            },
        },
    )
    program = """
import json
import sys
from vllm import LLM
try:
    LLM(**json.loads(sys.argv[1]))
except ValueError as error:
    assert "not initialized from checkpoint" in str(error), error
    assert sys.argv[2] in str(error), error
else:
    raise AssertionError("LLM admitted an incomplete MTP checkpoint")
"""
    log_path = tmp_path / "incomplete-mtp.log"
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


def _check_qwen35_model_weights(model, model_path, *, mtp=False):
    """Check packed attention, GDN and expert tensors against the saved bytes."""
    import torch
    from safetensors.torch import load_file

    weights = load_file(Path(model_path) / "model.safetensors")
    checked = set()
    for name, parameter in model.named_parameters():
        source = (
            name.replace("model.", "mtp.", 1)
            if mtp and name != "model.embed_tokens.weight"
            else name
        )
        if source in weights:
            names = [source]
        else:
            projections = next(
                (
                    (packed, shards)
                    for packed, shards in (
                        ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
                        ("gate_up_proj", ("gate_proj", "up_proj")),
                        ("in_proj_qkvz", ("in_proj_qkv", "in_proj_z")),
                        ("in_proj_ba", ("in_proj_b", "in_proj_a")),
                    )
                    if source.endswith(f".{packed}.weight")
                ),
                None,
            )
            if projections is not None:
                packed, shards = projections
                names = [source.replace(packed, shard) for shard in shards]
            else:
                assert source.endswith((".w13_weight", ".w2_weight")), name
                prefix, projection = source.rsplit(".", 1)
                prefix = prefix.removesuffix(".routed_experts")
                shards = (
                    ("gate_proj", "up_proj")
                    if projection == "w13_weight"
                    else ("down_proj",)
                )
                expected = torch.stack(
                    [
                        torch.cat([weights[f"{prefix}.{e}.{p}.weight"] for p in shards])
                        for e in range(parameter.shape[0])
                    ]
                )
                names = [
                    f"{prefix}.{e}.{p}.weight"
                    for e in range(parameter.shape[0])
                    for p in shards
                ]
                assert torch.equal(
                    parameter.detach().cpu(), expected.to(parameter.dtype)
                ), name
                checked.update(names)
                continue
        expected = torch.cat(
            [weights[key].reshape(-1, weights[key].shape[-1]) for key in names]
        )
        assert torch.equal(
            parameter.detach().cpu(),
            expected.reshape(parameter.shape).to(parameter.dtype),
        ), name
        checked.update(names)
    required = (
        {name for name in weights if name.startswith("mtp.")}
        | {"model.embed_tokens.weight", "lm_head.weight"}
        if mtp
        else {name for name in weights if not name.startswith("mtp.")}
    )
    assert checked == required, (required - checked, checked - required)
    return dict(
        parameters=len(list(model.parameters())), checkpoint_keys=sorted(checked)
    )


def _save_step3p5_mtp_model(directory, profile="one_full"):
    import json

    import torch
    from transformers import Step3p7TextConfig, Step3p7TextModel
    from transformers.models.step3p7.modeling_step3p7 import Step3p7RMSNorm

    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    root = Path(directory)
    assert profile in ("one_full", "three_swa")
    heads = 3 if profile == "three_swa" else 1
    target_types = (
        ["full_attention", "sliding_attention"]
        if heads == 3
        else ["full_attention"] * 2
    )
    head_types = ["sliding_attention" if heads == 3 else "full_attention"] * heads
    max_length = 256 if heads == 3 else 128
    config = Step3p7TextConfig(
        architectures=["Step3p5ForCausalLM"],
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=max_length,
        num_nextn_predict_layers=heads,
        layer_types=target_types + head_types,
        per_layer_config={} if heads == 3 else None,
        mlp_layer_types=["dense", "sparse"] + ["dense"] * heads,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        share_expert_dim=128,
        use_moe_router_bias=True,
        need_fp32_gate=True,
        norm_expert_weight=True,
        moe_router_activation="sigmoid",
        use_head_wise_attn_gate=True,
        gating=True,
        swiglu_limits=[0] * (2 + heads),
        swiglu_limits_shared=[0] * (2 + heads),
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        dtype="bfloat16",
        **(
            {"num_sliding_attention_heads": 4, "sliding_window": 128}
            if heads == 3
            else {}
        ),
    )
    assert config.model_type == "step3p5"
    assert config.layer_types == target_types
    assert config.mtp_layer_types == head_types
    assert config.mlp_layer_types == ["dense", "sparse"]
    assert config.mtp_mlp_layer_types == ["dense"] * heads

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261027)
        target = Step3p7TextModel(config)
        head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        _two_candidate_head(head.weight)
        expanded = Step3p7TextConfig.from_dict(
            config.to_dict(),
            num_hidden_layers=2 + heads,
            num_nextn_predict_layers=0,
            layer_types=target_types + head_types,
            mlp_layer_types=["dense", "sparse"] + ["dense"] * heads,
            swiglu_limits=[0] * (2 + heads),
            swiglu_limits_shared=[0] * (2 + heads),
        )
        mtp_owner = Step3p7TextModel(expanded)
        state = {f"model.{name}": value for name, value in target.state_dict().items()}
        state["lm_head.weight"] = (
            head.weight.detach().to(torch.bfloat16).clone()
            if heads == 3
            else head.weight.detach().clone()
        )
        for index in range(2, 2 + heads):
            mtp = mtp_owner.layers[index]
            state.update(
                {
                    f"model.layers.{index}.{name}": value
                    for name, value in mtp.state_dict().items()
                }
            )
            for name in ("enorm", "hnorm", "shared_head.norm"):
                norm = Step3p7RMSNorm(config.hidden_size, config.rms_norm_eps)
                assert torch.count_nonzero(norm.weight) == 0
                prefix = "transformer." if name == "shared_head.norm" else ""
                state[f"model.layers.{index}.{prefix}{name}.weight"] = (
                    norm.weight.detach().clone()
                )
            fusion = torch.nn.Linear(
                config.hidden_size * 2, config.hidden_size, bias=False
            )
            _conditioned_mtp_fusion(fusion.weight)
            state[f"model.layers.{index}.eh_proj.weight"] = (
                fusion.weight.detach().clone()
            )
            state[f"model.layers.{index}.transformer.shared_head.output.weight"] = (
                state["lm_head.weight"] * index
            )

    weights = {}
    for name, value in state.items():
        if ".mlp.experts.gate_up_proj" in name:
            assert name == "model.layers.1.mlp.experts.gate_up_proj"
            gate, up = value.chunk(2, dim=1)
            weights["model.layers.1.moe.gate_proj.weight"] = gate
            weights["model.layers.1.moe.up_proj.weight"] = up
        elif name == "model.layers.1.mlp.experts.down_proj":
            weights["model.layers.1.moe.down_proj.weight"] = value
        elif name == "model.layers.1.mlp.gate.weight":
            weights["model.layers.1.moe.gate.weight"] = value
        elif name == "model.layers.1.mlp.gate.e_score_correction_bias":
            weights["model.layers.1.moe.router_bias"] = value
        elif name.startswith("model.layers.1.mlp.shared_experts."):
            weights[name.replace(".mlp.shared_experts.", ".share_expert.")] = value
        else:
            weights[name] = value
    model_path = Path(_save_mtp_checkpoint(root / "model", config, weights))
    saved_config = json.loads((model_path / "config.json").read_text())
    if heads == 1:
        saved_config["per_layer_config"] = None
    (model_path / "config.json").write_text(json.dumps(saved_config, indent=2) + "\n")
    for index in range(2, 2 + heads):
        head_value = weights[
            f"model.layers.{index}.transformer.shared_head.output.weight"
        ]
        assert not torch.equal(head_value, weights["lm_head.weight"])
        assert torch.equal(head_value, weights["lm_head.weight"] * index)
    receipt = dict(
        synthetic_profile=True,
        target_architecture="Step3p5ForCausalLM",
        draft_architecture="Step3p5MTP",
        target_hf_class=type(target).__name__,
        mtp_hf_class=type(mtp).__name__,
        mtp_source=f"independently_initialized_hf_tail_{heads}_of_{heads + 2}",
        fusion="embedding_1_4_target_hidden_3_4",
        owned_head_scales=list(range(2, 2 + heads)),
        head_paths=[
            f"model.layers.{index}.shared_head.head" for index in range(2, 2 + heads)
        ],
        checkpoint_heads=[
            f"model.layers.{index}.transformer.shared_head.output.weight"
            for index in range(2, 2 + heads)
        ],
        head_candidate_ids=[3, 4],
        head_scale=32,
        checkpoint_tensors=len(weights),
        exact_checkpoint_roundtrip=True,
        mtp_absolute_layer=2,
        target_mlp=["dense", "sparse"],
        mtp_mlp=["dense"] * heads,
        attention_layers=target_types + head_types,
        profile=profile,
        mtp_heads=heads,
        max_model_len=max_length,
        mtp_windows=[128 if heads == 3 else None] * heads,
        target_num_attention_heads=4,
        num_attention_groups=2,
        head_dim=32,
        rms_norm_semantics="(1 + weight), checkpoint zero weights",
        homogeneous_config="public per_layer_config:null"
        if heads == 1
        else "canonical per_layer_config:{}; SWA head count resolves directly",
        seed=20261027,
        cuda_initialized=torch.cuda.is_initialized(),
    )
    assert not receipt["cuda_initialized"]
    (root / "fixture.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return model_path, receipt


def _check_step3p5_model_weights(model, model_path, *, mtp):
    """Account for every native parameter and each packed checkpoint tensor."""
    import torch
    from safetensors.torch import load_file

    weights = load_file(Path(model_path) / "model.safetensors")
    checked = set()
    parameters = dict(model.named_parameters())
    for name, parameter in parameters.items():
        source = name.replace(".mtp_block.", ".")
        if mtp and ".shared_head." in source:
            source = source.replace(".shared_head.", ".transformer.shared_head.")
            source = source.replace(".shared_head.head.", ".shared_head.output.")
        source = source.replace(".moe.share_expert.", ".share_expert.")
        if source in weights:
            inputs = [source]
        elif source.endswith(".qkv_proj.weight"):
            inputs = [
                source.replace("qkv_proj", proj)
                for proj in ("q_proj", "k_proj", "v_proj")
            ]
        elif source.endswith(".gate_up_proj.weight"):
            inputs = [
                source.replace("gate_up_proj", proj)
                for proj in ("gate_proj", "up_proj")
            ]
        else:
            assert source.endswith((".w13_weight", ".w2_weight")), name
            prefix, projection = source.rsplit(".", 1)
            assert prefix.endswith(".moe.experts.routed_experts")
            prefix = prefix.removesuffix(".experts.routed_experts")
            projections = (
                ("gate_proj", "up_proj")
                if projection == "w13_weight"
                else ("down_proj",)
            )
            inputs = [f"{prefix}.{proj}.weight" for proj in projections]
        values = [weights[key] for key in inputs]
        expected = values[0] if len(values) == 1 else torch.cat(values, dim=-2)
        assert torch.equal(parameter.detach().cpu(), expected.to(parameter.dtype)), name
        checked.update(inputs)
    config = model.config if mtp else model.model.config
    tail = tuple(
        f"model.layers.{i}." for i in range(2, 2 + config.num_nextn_predict_layers)
    )
    required = (
        {key for key in weights if key.startswith(tail)}
        if mtp
        else {key for key in weights if not key.startswith(tail)}
    )
    if mtp:
        required.add("model.embed_tokens.weight")
    assert checked == required, (required - checked, checked - required)
    return dict(parameters=len(parameters), checkpoint_keys=sorted(checked))


def _check_step3p5_mtp_weights(target, draft, model_path, *, shared=True):
    """Verify complete native values and checkpoint-owned heads after loading."""
    import torch

    from vllm.model_executor.models.step3p5 import Step3p5ForCausalLM
    from vllm.model_executor.models.step3p5_mtp import Step3p5MTP

    assert type(target) is Step3p5ForCausalLM and type(draft) is Step3p5MTP
    proofs = {
        role: _check_step3p5_model_weights(model, model_path, mtp=role == "draft")
        for role, model in (("target", target), ("draft", draft))
    }
    heads = draft.config.num_nextn_predict_layers
    assert heads in (1, 3)
    keys = set().union(*(proof["checkpoint_keys"] for proof in proofs.values()))
    assert len(keys) == 32 + 17 * heads
    assert {role: proof["parameters"] for role, proof in proofs.items()} == {
        "target": 25,
        "draft": 1 + 14 * heads,
    }
    assert draft.has_own_lm_head and not hasattr(draft, "lm_head")
    assert (draft.model.embed_tokens is target.model.embed_tokens) is shared
    assert torch.equal(
        draft.model.embed_tokens.weight, target.model.embed_tokens.weight
    )
    owned: list[torch.nn.Module] = []
    for index in range(2, 2 + heads):
        layer = draft.model.layers[str(index)]
        head = layer.shared_head.head
        assert head is not target.lm_head
        assert head.weight.data_ptr() != target.lm_head.weight.data_ptr()
        assert torch.equal(head.weight, target.lm_head.weight * index)
        assert not torch.equal(head.weight, target.lm_head.weight)
        assert all(head is not previous for previous in owned)
        assert all(not torch.equal(head.weight, previous.weight) for previous in owned)
        owned.append(head)
        assert all(
            torch.count_nonzero(norm.weight) == 0
            for norm in (
                layer.enorm,
                layer.hnorm,
                layer.shared_head.norm,
            )
        )
    return dict(
        target=type(target).__name__,
        draft=type(draft).__name__,
        parameters={role: proof["parameters"] for role, proof in proofs.items()},
        checkpoint_keys=len(keys),
        exact_parameter_roundtrip=True,
        shared_embedding=shared,
        checkpoint_owned_heads=heads,
        owned_head_scales=list(range(2, 2 + heads)),
        owned_head_bytes=sum(
            head.weight.numel() * head.weight.element_size() for head in owned
        ),
        distinct_owned_head_objects_and_bytes=True,
        top_level_draft_head=False,
        gemma_rms_checkpoint_weights_zero=True,
    )


def _step3p5_attention_geometry(model):
    from vllm.model_executor.layers.attention import Attention

    return [
        dict(
            window=layer.sliding_window,
            query_heads=layer.num_heads,
            kv_heads=layer.num_kv_heads,
            head_dim=layer.head_size,
            backend=layer.get_attn_backend().get_name(),
            cache_shape=tuple(layer.kv_cache.shape),
            cache_dtype=str(layer.kv_cache.dtype),
        )
        for layer in model.modules()
        if isinstance(layer, Attention)
    ]


def _step3p5_standalone_ownership(model_path, k, profile, output):
    """Check the actual standalone loader before running its production server."""
    import json

    import torch
    from safetensors.torch import load_file

    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.spec_decode.remote.runtime.gpu import GPUDraftStepExecutor
    from vllm.v1.spec_decode.remote.runtime.weights import model_weight_bytes

    heads = 3 if profile == "three_swa" else 1
    config = EngineArgs(
        model=model_path,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=256 if heads == 3 else 128,
        max_num_seqs=4,
        max_num_batched_tokens=32,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=True,
        block_size=16,
        attention_config={"backend": "TRITON_ATTN"},
        moe_backend="triton",
        speculative_config=dict(
            method="mtp",
            model=model_path,
            num_speculative_tokens=k,
            draft_tensor_parallel_size=1,
            draft_sample_method="greedy",
            remote_draft=dict(
                endpoint="unix:///tmp/step3p5-ownership.sock", transport="inline"
            ),
        ),
    ).create_engine_config()
    executor = GPUDraftStepExecutor(
        config, device=torch.device("cuda:0"), kv_cache_bytes=32 << 20
    )
    assert executor.contract.requires_target_lm_head is False
    draft = executor.speculator.model
    assert type(draft).__name__ == "Step3p5MTP"
    assert not hasattr(draft, "lm_head")
    assert draft.model.num_mtp_layers == heads
    proof = _check_step3p5_model_weights(draft, model_path, mtp=True)
    assert proof["parameters"] == 1 + 14 * heads
    weights = load_file(Path(model_path) / "model.safetensors")
    assert torch.equal(
        draft.model.embed_tokens.weight.cpu(), weights["model.embed_tokens.weight"]
    )
    owned: list[torch.nn.Module] = []
    for index in range(2, 2 + heads):
        head = draft.model.layers[str(index)].shared_head.head
        assert torch.equal(head.weight.cpu(), weights["lm_head.weight"] * index)
        assert all(head is not previous for previous in owned)
        assert all(not torch.equal(head.weight, previous.weight) for previous in owned)
        owned.append(head)
    draft_bytes, embedding_bytes, head_bytes = model_weight_bytes(draft)
    assert embedding_bytes == 128 * 128 * 2
    assert head_bytes == heads * embedding_bytes
    assert draft_bytes + embedding_bytes + head_bytes == sum(
        parameter.numel() * parameter.element_size() for parameter in draft.parameters()
    )
    geometry = _step3p5_attention_geometry(draft)
    assert [row["window"] for row in geometry] == ([128] * 3 if heads == 3 else [None])
    assert all(
        row["query_heads"] == 4
        and row["kv_heads"] == 2
        and row["head_dim"] == 32
        and row["backend"] == "TRITON_ATTN"
        and row["cache_dtype"] == "torch.bfloat16"
        for row in geometry
    )
    executor.warmup()
    assert config.speculative_config is not None
    result = dict(
        model=type(draft).__name__,
        parameters=proof["parameters"],
        checkpoint_keys=proof["checkpoint_keys"],
        exact_parameter_roundtrip=True,
        requires_target_lm_head=False,
        top_level_draft_head=False,
        checkpoint_owned_heads=heads,
        owned_head_scales=list(range(2, 2 + heads)),
        owned_head_bytes=head_bytes,
        embedding_bytes=embedding_bytes,
        draft_specific_bytes=draft_bytes,
        distinct_owned_head_objects_and_bytes=True,
        embedding_checkpoint_values=True,
        geometry=geometry,
        actual_warmup=True,
        num_prefill_lookahead_tokens=config.num_prefill_lookahead_tokens,
        use_multi_module_mtp=config.speculative_config.use_multi_module_mtp(),
    )
    assert (
        result["num_prefill_lookahead_tokens"] == 1
        and not result["use_multi_module_mtp"]
    )
    Path(output).write_text(json.dumps(result, indent=2) + "\n")


@pytest.mark.parametrize(
    "profile,k",
    [("one_full", 1), ("one_full", 3), ("three_swa", 1)],
    ids=["one_full-1", "one_full-3", "three_swa-1"],
)
def test_remote_step3p5_mtp_matches_native_and_target(
    profile,
    k,
    vllm_runner,
    request,
    monkeypatch,
    tmp_path,
    record_property,
):
    """Keep checkpoint-owned MTP output heads through native and remote loading."""
    import json

    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    directory = tmp_path / "step3p5"
    command = [
        sys.executable,
        "-c",
        (
            "import sys; from tests.v1.e2e.spec_decode.remote.test_remote_mtp "
            "import _save_step3p5_mtp_model; "
            "from pathlib import Path; p=Path(sys.argv[1]); p.mkdir(); "
            "_save_step3p5_mtp_model(p, sys.argv[2])"
        ),
        str(directory),
        profile,
    ]
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        VLLM_TARGET_DEVICE="cpu",
        VLLM_BATCH_INVARIANT="0",
    )
    subprocess.run(command, env=environment, check=True, timeout=180)
    model = str(directory / "model")
    record_property("fixture", json.loads((directory / "fixture.json").read_text()))
    ownership = tmp_path / "step3p5-standalone.json"
    command = [
        sys.executable,
        "-c",
        (
            "import sys; from tests.v1.e2e.spec_decode.remote.test_remote_mtp "
            "import _step3p5_standalone_ownership; "
            "_step3p5_standalone_ownership("
            "sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])"
        ),
        model,
        str(k),
        profile,
        str(ownership),
    ]
    with (tmp_path / "step3p5-standalone.log").open("w") as log:
        subprocess.run(
            command, check=True, timeout=180, stdout=log, stderr=subprocess.STDOUT
        )
    record_property("remote.step3p5_identity", json.loads(ownership.read_text()))
    _compare_native_and_remote_draft(
        vllm_runner,
        monkeypatch,
        tmp_path,
        request,
        "step3p5_" + profile,
        None,
        k,
        record_property,
        target_model=model,
    )


def _mimo_v2_flash_fixture_data():
    import torch
    from transformers import AutoConfig

    config = AutoConfig.for_model(
        "mimo_v2_flash",
        **{
            "architectures": ["MiMoV2FlashForCausalLM"],
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 192,
            "v_head_dim": 128,
            "swa_num_attention_heads": 2,
            "swa_num_key_value_heads": 2,
            "swa_head_dim": 192,
            "swa_v_head_dim": 128,
            "hybrid_layer_pattern": [0, 1],
            "sliding_window": 128,
            "sliding_window_size": 128,
            "attention_chunk_size": 128,
            "attention_bias": False,
            "add_swa_attention_sink_bias": True,
            "add_full_attention_sink_bias": False,
            "attention_value_scale": 0.707,
            "partial_rotary_factor": 0.334,
            "rope_theta": 5000000,
            "swa_rope_theta": 10000,
            "layernorm_epsilon": 1e-05,
            "max_position_embeddings": 256,
            "hidden_act": "silu",
            "moe_layer_freq": [0, 1],
            "moe_intermediate_size": 128,
            "n_routed_experts": 4,
            "n_shared_experts": None,
            "num_experts_per_tok": 2,
            "norm_topk_prob": True,
            "scoring_func": "sigmoid",
            "n_group": 1,
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "routed_scaling_factor": None,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
        },
    )
    generator = torch.Generator(device="cpu").manual_seed(20261024)
    weights = {}

    def weight(name, shape, *, norm=False, router=False):
        value = (
            torch.ones(shape)
            if norm
            else torch.randn(shape, generator=generator) * 0.02
        )
        weights[name] = value.to(
            torch.float32 if router else torch.bfloat16
        ).contiguous()

    def attention(prefix, kv_heads, sink):
        for projection, rows in [
            ("q_proj", 384),
            ("k_proj", kv_heads * 192),
            ("v_proj", kv_heads * 128),
        ]:
            weight(prefix + "." + projection + ".weight", (rows, 64))
        weight(prefix + ".o_proj.weight", (64, 256))
        if sink:
            weight(prefix + ".attention_sink_bias", (2,))

    def mlp(prefix):
        for name in ["gate_proj", "up_proj"]:
            weight(prefix + "." + name + ".weight", (128, 64))
        weight(prefix + ".down_proj.weight", (64, 128))

    weight("model.embed_tokens.weight", (128, 64))
    weight("lm_head.weight", (128, 64))
    weight("model.norm.weight", (64,), norm=True)
    for i, kv_heads in enumerate([1, 2]):
        prefix = f"model.layers.{i}"
        weight(prefix + ".input_layernorm.weight", (64,), norm=True)
        weight(prefix + ".post_attention_layernorm.weight", (64,), norm=True)
        attention(prefix + ".self_attn", kv_heads, i == 1)
        if i == 0:
            mlp(prefix + ".mlp")
        else:
            weight(prefix + ".mlp.gate.weight", (4, 64), router=True)
            weight(prefix + ".mlp.gate.e_score_correction_bias", (4,), router=True)
            for expert in range(4):
                mlp(prefix + f".mlp.experts.{expert}")
    mtp = "model.mtp.layers.0"
    weight(mtp + ".eh_proj.weight", (64, 128))
    _conditioned_mtp_fusion(weights[mtp + ".eh_proj.weight"])
    for name in [
        "enorm",
        "hnorm",
        "input_layernorm",
        "pre_mlp_layernorm",
        "final_layernorm",
    ]:
        weight(mtp + "." + name + ".weight", (64,), norm=True)
    attention(mtp + ".self_attn", 2, True)
    mlp(mtp + ".mlp")
    assert len(weights) == 47
    target_keys = {name for name in weights if ".mtp." not in name}
    draft_keys = {
        name
        for name in weights
        if ".mtp." in name or name in ["model.embed_tokens.weight", "lm_head.weight"]
    }
    assert len(target_keys) == 33 and len(draft_keys) == 16
    assert all(torch.isfinite(value).all() for value in weights.values())
    _two_candidate_head(weights["lm_head.weight"])
    return config, weights


@pytest.fixture
def tiny_mimo_v2_flash_mtp_model(tmp_path):
    """Complete synthetic BF16 Flash target and first native MTP head, not quality."""
    config, weights = _mimo_v2_flash_fixture_data()
    import json

    model = _save_mtp_checkpoint(tmp_path / "mimo-v2-flash", config, weights)
    (tmp_path / "fixture.json").write_text(
        json.dumps(
            dict(
                seed=20261024,
                head_candidate_ids=[3, 4],
                head_scale=32,
                architecture="MiMoV2FlashForCausalLM",
                synthetic_bf16=True,
                fusion="embedding_1_4_target_hidden_3_4",
                checkpoint_tensors=47,
                exact_checkpoint_roundtrip=True,
                native_used_heads=1,
                max_model_len=256,
                window=128,
                qk_head_dim=192,
                v_head_dim=128,
                target_kv_heads=[1, 2],
                draft_kv_heads=[2],
            )
        )
        + "\n"
    )
    return model


def _check_mimo_v2_flash_model_weights(model, model_path, role):
    """Check every original and fused parameter without claiming execution stage."""
    import torch
    from safetensors.torch import load_file

    assert role in ("target", "mtp")
    weights = load_file(Path(model_path) / "model.safetensors")
    assert len(weights) == 47
    required = {
        name
        for name in weights
        if (
            ".mtp." not in name
            if role == "target"
            else ".mtp." in name
            or name in {"model.embed_tokens.weight", "lm_head.weight"}
        )
    }
    checked = set()
    parameters = []
    expert_mapping = model.get_expert_mapping() if role == "target" else []
    for name, parameter in model.named_parameters():
        if name in weights:
            expected = weights[name]
            inputs = [name]
        elif ".qkv_proj." in name:
            inputs = [
                name.replace("qkv_proj", projection)
                for projection in ["q_proj", "k_proj", "v_proj"]
            ]
            expected = torch.cat([weights[key] for key in inputs])
        elif ".gate_up_proj." in name:
            inputs = [
                name.replace("gate_up_proj", projection)
                for projection in ["gate_proj", "up_proj"]
            ]
            expected = torch.cat([weights[key] for key in inputs])
        elif name.endswith("w13_weight") or name.endswith("w2_weight"):
            inputs = []
            parts = {}
            for native_name, source_name, expert, shard in expert_mapping:
                for key in required:
                    if (
                        source_name in key
                        and key.replace(source_name, native_name) == name
                    ):
                        parts[expert, shard] = weights[key]
                        inputs.append(key)
            if name.endswith("w13_weight"):
                assert set(parts) == {
                    (e, shard) for e in range(4) for shard in ["w1", "w3"]
                }
                expected = torch.stack(
                    [torch.cat([parts[e, "w1"], parts[e, "w3"]]) for e in range(4)]
                )
            else:
                assert set(parts) == {(e, "w2") for e in range(4)}
                expected = torch.stack([parts[e, "w2"] for e in range(4)])
        else:
            raise AssertionError(f"unaccounted native parameter: {name}")
        assert parameter.shape == expected.shape, (
            name,
            parameter.shape,
            expected.shape,
        )
        assert torch.equal(parameter.detach().cpu(), expected.to(parameter.dtype)), name
        checked.update(inputs)
        parameters.append(
            dict(
                name=name,
                shape=list(parameter.shape),
                dtype=str(parameter.dtype),
                checkpoint_keys=sorted(inputs),
            )
        )
    assert checked == required, (role, required - checked, checked - required)
    return dict(parameters=len(parameters), checkpoint_keys=sorted(checked))


def _check_mimo_v2_flash_mtp_weights(target, draft, model_path):
    """Exact postprocessed values and native sharing after the real GPU loader."""
    from vllm.model_executor.models.mimo_v2 import MiMoV2FlashForCausalLM
    from vllm.model_executor.models.mimo_v2_mtp import MiMoV2MTP

    assert type(target) is MiMoV2FlashForCausalLM
    assert type(draft) is MiMoV2MTP
    proofs = {
        role: _check_mimo_v2_flash_model_weights(model, model_path, role)
        for role, model in (("target", target), ("mtp", draft))
    }
    counts = {role: proof["parameters"] for role, proof in proofs.items()}
    assert counts == {"target": 18, "mtp": 13}
    all_checked = set().union(*(proof["checkpoint_keys"] for proof in proofs.values()))
    assert len(all_checked) == 47
    assert draft.model.embed_tokens is target.model.embed_tokens
    assert draft.lm_head is target.lm_head
    assert draft.model.num_mtp_layers == 1 and tuple(draft.model.mtp.layers) == ("0",)
    return dict(
        target=type(target).__name__,
        draft=type(draft).__name__,
        checkpoint_keys=len(all_checked),
        parameters=counts,
        exact_parameter_roundtrip=True,
        first_native_head_only=True,
        shared_embedding_and_head=True,
    )


def _mimo_v2_flash_gpu_geometry(model, *, mtp=False):
    """Observe actual allocated native cache geometry without changing its values."""
    import torch

    from vllm.model_executor.layers.attention import Attention

    rows = []
    for layer in model.modules():
        if not isinstance(layer, Attention):
            continue
        assert layer.get_attn_backend().get_name() == "TRITON_ATTN_DIFFKV"
        assert (layer.head_size, layer.head_size_v) == (192, 128)
        cache = layer.kv_cache
        assert cache.ndim == 4 and cache.dtype == torch.bfloat16
        assert cache.shape[1] == layer.num_kv_heads and cache.shape[-1] == 320
        key, value = cache.split([192, 128], dim=-1)
        rows.append(
            dict(
                layer=layer.layer_name,
                kv_heads=layer.num_kv_heads,
                query_heads=layer.num_heads,
                qk=192,
                value=128,
                window=layer.sliding_window,
                sink=layer.has_sink,
                cache_shape=list(cache.shape),
                key_shape=list(key.shape),
                value_shape=list(value.shape),
                cache_stride=list(cache.stride()),
            )
        )
    assert [r["kv_heads"] for r in rows] == ([2] if mtp else [1, 2])
    assert [r["window"] for r in rows] == ([128] if mtp else [None, 128])
    assert [r["sink"] for r in rows] == ([True] if mtp else [False, True])
    return rows


@pytest.mark.parametrize("k", [1, 3])
def test_remote_mimo_v2_flash_mtp_matches_native_and_target(
    vllm_runner,
    monkeypatch,
    tmp_path,
    request,
    k,
    tiny_mimo_v2_flash_mtp_model,
    record_property,
):
    """Compare native first-head Flash drafting and remote inline across windows."""
    import json

    model = tiny_mimo_v2_flash_mtp_model
    record_property(
        "fixture", json.loads((Path(model).parent / "fixture.json").read_text())
    )
    _compare_native_and_remote_draft(
        vllm_runner,
        monkeypatch,
        tmp_path,
        request,
        "mimo_v2_flash",
        None,
        k,
        record_property,
        target_model=model,
    )


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


def _save_eagle3_models(directory: str, target_family: str) -> None:
    """Export real native EAGLE weights; amplify two head rows for acceptance coverage.

    Attention, MLP and feature projection stay seeded, nontrivial native layers.
    This controlled checkpoint measures state/placement correctness, not quality.
    """
    import json

    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        MistralConfig,
        MistralForCausalLM,
        Qwen2Config,
        Qwen2ForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
        TokenizersBackend,
    )

    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        SpeculativeConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.model_loader import get_model
    from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model

    assert current_platform.is_cpu() and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    root = Path(directory)
    target_path, draft_path = root / "target", root / "draft"
    common = dict(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
    )
    assert target_family in ("llama", "qwen3", "qwen2", "mistral")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261025)
        if target_family == "llama":
            target = LlamaForCausalLM(LlamaConfig(num_hidden_layers=4, **common))
        elif target_family == "qwen3":
            target = Qwen3ForCausalLM(
                Qwen3Config(
                    num_hidden_layers=4,
                    layer_types=["full_attention"] * 4,
                    **common,
                )
            )
        elif target_family == "qwen2":
            target = Qwen2ForCausalLM(
                Qwen2Config(
                    num_hidden_layers=4,
                    use_sliding_window=False,
                    layer_types=["full_attention"] * 4,
                    **common,
                )
            )
        else:
            target = MistralForCausalLM(
                MistralConfig(num_hidden_layers=4, sliding_window=None, **common)
            )
        target = target.to(dtype=torch.bfloat16).eval()
        with torch.no_grad():
            target.lm_head.weight[3].mul_(32)
            target.lm_head.weight[4].copy_(-target.lm_head.weight[3])
    target.save_pretrained(target_path)
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        num_hidden_layers=1,
        draft_vocab_size=128,
        target_hidden_size=128,
        num_aux_hidden_states=3,
        eagle_aux_hidden_state_layer_ids=[1, 2, 4],
        eagle_config={"use_aux_hidden_state": True},
        **common,
    ).save_pretrained(draft_path)
    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(WordLevel({f"token{i}": i for i in range(128)})),
        bos_token="token1",
        eos_token="token2",
        pad_token="token0",
    )
    for path in (target_path, draft_path):
        tokenizer.save_pretrained(path)

    def make_config():
        model = ModelConfig(
            model=str(target_path),
            dtype="bfloat16",
            max_model_len=128,
            enforce_eager=True,
            skip_tokenizer_init=True,
        )
        parallel = ParallelConfig()
        return VllmConfig(
            model_config=model,
            parallel_config=parallel,
            speculative_config=SpeculativeConfig(
                model=str(draft_path),
                method="eagle3",
                num_speculative_tokens=3,
                target_model_config=model,
                target_parallel_config=parallel,
                draft_sample_method="greedy",
                draft_tensor_parallel_size=1,
            ),
            device_config=DeviceConfig("cpu"),
            cache_config=CacheConfig(block_size=32, enable_prefix_caching=False),
            scheduler_config=SchedulerConfig(
                max_num_seqs=4,
                max_num_batched_tokens=32,
                max_model_len=128,
                async_scheduling=False,
                is_encoder_decoder=False,
            ),
        )

    config = make_config()
    try:
        with set_current_vllm_config(config):
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=(root / "rendezvous").as_uri(),
                backend="gloo",
            )
            initialize_model_parallel(backend="gloo")
        with (
            set_current_vllm_config(config),
            set_default_torch_dtype(torch.bfloat16),
            torch.device("cpu"),
            torch.random.fork_rng(devices=[]),
            torch.no_grad(),
        ):
            torch.manual_seed(20261026)
            draft = Eagle3LlamaForCausalLM(vllm_config=config)
            for value in draft.parameters():
                value.fill_(1) if value.ndim == 1 else value.normal_(mean=0, std=0.02)
            draft.model.embed_tokens.weight.copy_(target.model.embed_tokens.weight)
            draft.lm_head.weight.copy_(target.lm_head.weight)
        expected = {
            name: value.detach().clone() for name, value in draft.named_parameters()
        }
        save_file(
            {
                name.removeprefix("model."): value.contiguous()
                for name, value in expected.items()
            },
            draft_path / "model.safetensors",
        )
        loaded_config = make_config()
        with set_current_vllm_config(loaded_config):
            native_target = get_model(vllm_config=loaded_config)
            loaded = load_eagle_model(native_target, loaded_config)
        actual = dict(loaded.named_parameters())
        assert actual.keys() == expected.keys()
        assert all(torch.equal(actual[name], value) for name, value in expected.items())
        assert loaded.model.embed_tokens is native_target.model.embed_tokens
        assert loaded.lm_head is native_target.lm_head
        assert loaded.model.layers[0].self_attn.qkv_proj.weight.shape == (256, 256)
        captured = {}

        def capture(index):
            def hook(module, args, output):
                captured[index] = (
                    output[0] if isinstance(output, tuple) else output
                ).detach()

            return hook

        handles = [
            target.model.layers[index - 1].register_forward_hook(capture(index))
            for index in (1, 2, 4)
        ]
        try:
            with set_current_vllm_config(loaded_config), torch.inference_mode():
                output = target(
                    torch.tensor([[3 + (i * 7) % 125 for i in range(17)]]),
                    output_hidden_states=True,
                    use_cache=False,
                )
                assert tuple(captured) == (1, 2, 4)
                assert torch.equal(captured[1], output.hidden_states[1])
                assert torch.equal(captured[2], output.hidden_states[2])
                assert not torch.equal(captured[4], output.hidden_states[-1])
                features = torch.cat([captured[i] for i in (1, 2, 4)], dim=-1)[0]
                combined = loaded.combine_hidden_states(features)
                assert features.shape == (17, 384) and combined.shape == (17, 128)
                assert torch.equal(
                    combined,
                    torch.nn.functional.linear(features, loaded.model.fc.weight),
                )
        finally:
            for handle in handles:
                handle.remove()
    finally:
        try:
            destroy_model_parallel()
        finally:
            destroy_distributed_environment()
    assert not torch.cuda.is_initialized()
    (root / "fixture.json").write_text(
        json.dumps(
            dict(
                model=type(loaded).__name__,
                target_family=target_family,
                target_model=type(native_target).__name__,
                parameters=len(expected),
                exact_parameter_roundtrip=True,
                shared_embedding_and_head=True,
                capture_ids=[1, 2, 4],
                final_capture_prenorm=True,
                feature_shape=list(features.shape),
                combined_shape=list(combined.shape),
                cuda_initialized=False,
                head_candidate_ids=[3, 4],
                head_scale=32,
            )
        )
        + "\n"
    )


def _tiny_eagle3_models(tmp_path, target_family):
    """Keep CPU constructor/TP setup isolated from the serving worker's platform."""
    root = (tmp_path / f"{target_family}-eagle3").resolve()
    root.mkdir()
    source = Path(__file__).resolve().parents[5]
    command = [
        sys.executable,
        "-c",
        "import pathlib,sys; import vllm; "
        "assert pathlib.Path(vllm.__file__).resolve().is_relative_to(sys.argv[1]); "
        "from tests.v1.e2e.spec_decode.remote.test_remote_mtp "
        "import _save_eagle3_models; _save_eagle3_models(*sys.argv[2:])",
        str(source),
        str(root),
        target_family,
    ]
    log_path = root / "fixture.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            env=dict(
                os.environ,
                PYTHONPATH=str(source),
                CUDA_VISIBLE_DEVICES="",
                VLLM_TARGET_DEVICE="cpu",
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            assert process.wait(timeout=210) == 0, f"EAGLE fixture failed: {log_path}"
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    return str(root / "target"), str(root / "draft")


@pytest.fixture
def tiny_eagle3_models(tmp_path, eagle_target):
    return _tiny_eagle3_models(tmp_path, eagle_target)


@pytest.fixture
def tiny_independent_qwen3_models(tmp_path, request, mtp_family):
    target_path = request.getfixturevalue(f"tiny_{mtp_family}_mtp_model")
    return _tiny_independent_qwen3_models(tmp_path, target_path)


def _conditioned_mtp_fusion(weight):
    """Preserve target feature direction while retaining token conditioning."""
    import torch

    width = weight.shape[0]
    assert weight.shape == (width, 2 * width)
    identity = torch.eye(width, dtype=weight.dtype, device=weight.device)
    with torch.no_grad():
        weight.copy_(torch.cat((identity / 4, 3 * identity / 4), dim=1))


def _two_candidate_head(weight):
    import torch

    with torch.no_grad():
        weight[3].mul_(32)
        weight[4].copy_(-weight[3])


def _tiny_independent_qwen3_models(tmp_path, target_path):
    """Exercise accepted states with real models and two amplified output rows.

    Keep every target attention/recurrent weight and the original fixture intact.
    The independent draft has its own head geometry; this is not a quality test.
    """
    import hashlib
    import json
    import shutil

    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from transformers import AutoConfig

    def digests(path):
        return {
            str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(path.rglob("*"))
            if file.is_file()
        }

    source = Path(target_path)
    original = digests(source)
    config = AutoConfig.from_pretrained(source)
    assert not config.tie_word_embeddings
    target = tmp_path / "controlled-target"
    shutil.copytree(source, target)
    checkpoint = target / "model.safetensors"
    with safe_open(checkpoint, framework="pt", device="cpu") as reader:
        metadata = reader.metadata()
    weights = load_file(checkpoint)
    original_head = weights["lm_head.weight"].clone()
    _two_candidate_head(weights["lm_head.weight"])
    assert torch.equal(weights["lm_head.weight"][:3], original_head[:3])
    assert torch.equal(weights["lm_head.weight"][5:], original_head[5:])
    replacement = target / "controlled.safetensors"
    save_file(weights, replacement, metadata=metadata)
    replacement.replace(checkpoint)
    assert digests(source) == original
    target_identity = digests(target)
    assert all(
        target_identity[name] == digest
        for name, digest in original.items()
        if name != "model.safetensors"
    )
    draft = _tiny_independent_qwen3_draft(tmp_path, target)
    (tmp_path / "independent-control.json").write_text(
        json.dumps(
            dict(
                original_target=original,
                controlled_target=target_identity,
                draft=digests(Path(draft)),
                target_architecture=config.architectures,
                target_hidden_size=config.hidden_size,
                draft_hidden_size=64,
                head_candidate_ids=[3, 4],
                head_scale=32,
                original_target_unchanged=True,
                changed_target_rows=[3, 4],
            )
        )
        + "\n"
    )
    return str(target), draft


def _tiny_independent_qwen3_draft(tmp_path, target_path):
    """Load a separate dense LM with the target's exact tokenizer artifacts."""
    import shutil

    import torch
    from transformers import AutoConfig, Qwen3Config, Qwen3ForCausalLM

    target_path = Path(target_path)
    target = AutoConfig.from_pretrained(target_path)
    assert (
        target.vocab_size,
        target.bos_token_id,
        target.eos_token_id,
        target.pad_token_id,
    ) == (128, 1, 2, 0)
    config = Qwen3Config(
        vocab_size=target.vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=128,
        layer_types=["full_attention"] * 2,
        bos_token_id=target.bos_token_id,
        eos_token_id=target.eos_token_id,
        pad_token_id=target.pad_token_id,
        tie_word_embeddings=False,
    )
    assert config.hidden_size != target.hidden_size
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261025)
        draft = Qwen3ForCausalLM(config)
    _two_candidate_head(draft.lm_head.weight)
    path = tmp_path / "independent-qwen3"
    draft.save_pretrained(path)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "tokenizer.model",
    ):
        if (target_path / name).is_file():
            shutil.copyfile(target_path / name, path / name)
            assert (path / name).read_bytes() == (target_path / name).read_bytes()
    assert all(
        (path / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json")
    )
    return str(path)


@pytest.fixture
def tiny_independent_mla_models(tmp_path, request):
    return _tiny_independent_mla_models(
        tmp_path,
        dense_ffn=getattr(request, "param", "moe") == "dense",
    )


def _tiny_independent_mla_models(tmp_path, *, dense_ffn=False):
    """Real independent MLA weights; fixed heads exercise accepted-state paths."""
    import hashlib
    import json
    import shutil

    import torch
    from transformers import (
        DeepseekV2Config,
        DeepseekV2ForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
    )

    common = dict(
        vocab_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        dtype="bfloat16",
    )
    target_config = Qwen3Config(
        hidden_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        layer_types=["full_attention"] * 2,
        **common,
    )
    draft_config = DeepseekV2Config(
        architectures=["DeepseekV2ForCausalLM"],
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=2,
        kv_lora_rank=512,
        q_lora_rank=128,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        n_routed_experts=0 if dense_ffn else 4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        first_k_dense_replace=2 if dense_ffn else 1,
        n_group=1,
        topk_group=1,
        topk_method="greedy",
        norm_topk_prob=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        attention_bias=False,
        **common,
    )
    paths = []
    for name, cls, config, seed in (
        ("qwen3-target", Qwen3ForCausalLM, target_config, 20261024),
        ("deepseek-v2-draft", DeepseekV2ForCausalLM, draft_config, 20261025),
    ):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = cls(config)
        _two_candidate_head(model.lm_head.weight)
        # These are complete LM checkpoints, without synthesized MTP modules.
        config.architectures = [cls.__name__]
        paths.append(
            _save_mtp_checkpoint(
                tmp_path / name, config, _unpack_mtp_experts(model.state_dict())
            )
        )
    target, draft = map(Path, paths)
    for source in target.iterdir():
        if source.name not in ("model.safetensors", "config.json"):
            shutil.copyfile(source, draft / source.name)
            assert source.read_bytes() == (draft / source.name).read_bytes()
    (tmp_path / "mla-fixture.json").write_text(
        json.dumps(
            dict(
                head_candidate_ids=[3, 4],
                head_scale=32,
                seeds=[20261024, 20261025],
                target_architecture="Qwen3ForCausalLM",
                draft_architecture="DeepseekV2ForCausalLM",
                independent_full_model=True,
                ffn_layout="dense" if dense_ffn else "moe",
                routed_experts=0 if dense_ffn else 4,
                first_k_dense_replace=2 if dense_ffn else 1,
                target_hidden_size=64,
                draft_hidden_size=128,
                latent_width=576,
                models={
                    role: {
                        file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                        for file in sorted(path.iterdir())
                        if file.is_file()
                    }
                    for role, path in (("target", target), ("draft", draft))
                },
            )
        )
        + "\n"
    )
    return paths[0], paths[1]


@pytest.fixture
def tiny_independent_recurrent_models(tmp_path):
    return _tiny_independent_recurrent_models(tmp_path)


def _tiny_independent_recurrent_models(tmp_path):
    """Complete ordinary Nemotron-H weights, separate from every MTP fixture."""
    import hashlib
    import json
    import shutil

    import torch
    from transformers import (
        NemotronHConfig,
        NemotronHForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
    )

    common = dict(
        vocab_size=128,
        intermediate_size=256,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        dtype="bfloat16",
    )
    target_config = Qwen3Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        layer_types=["full_attention"] * 2,
        **common,
    )
    draft_config = NemotronHConfig(
        architectures=["NemotronHForCausalLM"],
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        layers_block_type=["linear_attention", "full_attention"],
        mamba_num_heads=4,
        mamba_head_dim=64,
        ssm_state_size=64,
        n_groups=1,
        conv_kernel=4,
        chunk_size=128,
        n_routed_experts=0,
        n_shared_experts=0,
        num_nextn_predict_layers=0,
        mtp_layers_block_type=[],
        **common,
    )
    assert draft_config.hybrid_override_pattern == "M*"
    paths = []
    for name, cls, config, seed in (
        ("qwen3-target", Qwen3ForCausalLM, target_config, 20261024),
        ("nemotron-h-draft", NemotronHForCausalLM, draft_config, 20261025),
    ):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = cls(config)
        _two_candidate_head(model.lm_head.weight)
        weights = dict(model.state_dict())
        assert not any(name.startswith("mtp.") for name in weights)
        config.architectures = [cls.__name__]
        paths.append(_save_mtp_checkpoint(tmp_path / name, config, weights))
    target, draft = map(Path, paths)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(target / name, draft / name)
        assert (target / name).read_bytes() == (draft / name).read_bytes()
    (tmp_path / "recurrent-fixture.json").write_text(
        json.dumps(
            dict(
                head_candidate_ids=[3, 4],
                head_scale=32,
                seeds=[20261024, 20261025],
                target_architecture="Qwen3ForCausalLM",
                draft_architecture="NemotronHForCausalLM",
                independent_full_model=True,
                num_nextn_predict_layers=0,
                hybrid_pattern="M*",
                target_hidden_size=64,
                draft_hidden_size=128,
                mamba_conv_channels=384,
                mamba_conv_history=3,
                mamba_ssm_shape=[4, 64, 64],
                models={
                    role: {
                        file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                        for file in sorted(path.iterdir())
                        if file.is_file()
                    }
                    for role, path in (("target", target), ("draft", draft))
                },
            )
        )
        + "\n"
    )
    return paths[0], paths[1]


def _check_independent_recurrent_weights(model, model_path):
    """Verify every native parameter against the actual complete checkpoint."""
    import torch
    from safetensors.torch import load_file

    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

    assert isinstance(model, NemotronHForCausalLM)
    assert model.model.config.hybrid_override_pattern == "M*"
    assert model.model.config.num_nextn_predict_layers == 0
    weights = load_file(Path(model_path) / "model.safetensors")
    mapped: dict[str, list[tuple[str, str | None, torch.Tensor]]] = {}
    for name, tensor in weights.items():
        destinations = list(model.hf_to_vllm_mapper.apply(((name, tensor),)))
        assert len(destinations) == 1, name
        destination, value = destinations[0]
        shard = getattr(value, "shard_id", None)
        assert shard is None or isinstance(shard, str) and shard in ("q", "k", "v")
        mapped.setdefault(destination, []).append((name, shard, value))
    parameters = dict(model.named_parameters())
    assert mapped.keys() == parameters.keys()
    checked = set()
    for name, parameter in parameters.items():
        entries = mapped[name]
        if len(entries) == 3:
            shards = {shard: (source, tensor) for source, shard, tensor in entries}
            assert set(shards) == {"q", "k", "v"}, name
            names = [shards[shard][0] for shard in ("q", "k", "v")]
            expected = torch.cat([shards[shard][1] for shard in ("q", "k", "v")])
        else:
            assert len(entries) == 1 and entries[0][1] is None, name
            source, _, expected = entries[0]
            names = [source]
            if source.endswith(".A_log"):
                assert name.endswith(".A")
                expected = -torch.exp(expected.float())
        assert torch.equal(parameter.detach().cpu(), expected.to(parameter.dtype)), name
        checked.update(names)
    assert checked == weights.keys(), weights.keys() - checked
    return sorted(checked)


def _check_independent_recurrent_state(model, model_state, config):
    """Inspect the ordinary model's actual native attention and recurrent views."""
    import torch

    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    assert type(model) is NemotronHForCausalLM
    assert type(model_state) is MambaHybridModelState
    assert config.speculative_config is None and config.num_speculative_tokens == 0
    assert config.model_config.hf_config.hybrid_override_pattern == "M*"
    assert config.model_config.hf_config.num_nextn_predict_layers == 0
    assert config.cache_config.mamba_cache_mode == "none"
    assert not config.cache_config.use_replayssm
    assert not config.cache_config.use_kda_recoverssm
    assert config.mamba_config.backend.value == "triton"
    recurrent = [layer for layer in model.modules() if isinstance(layer, MambaMixer2)]
    attention = [layer for layer in model.modules() if isinstance(layer, Attention)]
    assert len(recurrent) == len(attention) == 1
    states = recurrent[0].kv_cache
    state_spec = recurrent[0].get_kv_cache_spec(config)
    assert isinstance(state_spec, MambaSpec)
    assert state_spec.num_speculative_blocks == 0
    assert state_spec.dtypes == (torch.bfloat16, torch.float32)
    assert tuple(tuple(state.shape[1:]) for state in states) == state_spec.shapes
    assert tuple(state.dtype for state in states) == state_spec.dtypes
    assert sorted(state_spec.shapes[0]) == [3, 384]
    assert state_spec.shapes[1] == (4, 64, 64)
    attn_spec = attention[0].get_kv_cache_spec(config)
    assert type(attn_spec) is FullAttentionSpec
    assert (attn_spec.num_kv_heads, attn_spec.head_size, attn_spec.block_size) == (
        2,
        32,
        config.cache_config.block_size,
    )
    assert attn_spec.dtype == torch.bfloat16
    assert attention[0].get_attn_backend().get_name() == "TRITON_ATTN"
    return dict(
        model=type(model).__name__,
        state=type(model_state).__name__,
        speculative_config=None,
        num_speculative_tokens=0,
        hybrid_pattern="M*",
        mamba_cache_mode=state_spec.mamba_cache_mode,
        mamba_backend="triton",
        state_shapes=state_spec.shapes,
        state_dtypes=[str(dtype) for dtype in state_spec.dtypes],
        state_slots=states[0].shape[0],
        attention_backend="TRITON_ATTN",
        attention_heads=attn_spec.num_kv_heads,
        attention_head_size=attn_spec.head_size,
        attention_block_size=attn_spec.block_size,
        configured_attention_block_size=config.cache_config.block_size,
    )


def _check_independent_mla_weights(model, model_path):
    """Check the complete real checkpoint, including native packed projections."""
    import torch
    from safetensors.torch import load_file

    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    weights = load_file(Path(model_path) / "model.safetensors")
    checked = set()

    def equal(actual, names):
        # The native loader casts checkpoint tensors to each real parameter's
        # dtype; the router checkpoint deliberately retains FP32 storage.
        assert actual.dtype in (torch.bfloat16, torch.float32)
        expected = torch.cat([weights[name] for name in names]).to(actual.dtype)
        assert torch.equal(actual.detach().cpu(), expected), names
        checked.update(names)

    for name, parameter in model.named_parameters():
        if name in weights:
            equal(parameter, [name])
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}"
        equal(
            layer.self_attn.fused_qkv_a_proj.weight,
            [
                f"{prefix}.self_attn.q_a_proj.weight",
                f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
            ],
        )
        experts = [
            module
            for module in layer.mlp.modules()
            if isinstance(module, RoutedExperts)
        ]
        dense = (
            [(layer.mlp.shared_experts, f"{prefix}.mlp.shared_experts")]
            if experts
            else [(layer.mlp, f"{prefix}.mlp")]
        )
        for mlp, name in dense:
            equal(
                mlp.gate_up_proj.weight,
                [f"{name}.gate_proj.weight", f"{name}.up_proj.weight"],
            )
        if experts:
            assert len(experts) == 1
            method = experts[0].quant_method
            assert isinstance(method, UnquantizedFusedMoEMethod)
            assert method.unquantized_backend.name == "TRITON"
            for expert in range(4):
                base = f"{prefix}.mlp.experts.{expert}"
                equal(
                    experts[0].w13_weight[expert],
                    [f"{base}.gate_proj.weight", f"{base}.up_proj.weight"],
                )
                equal(experts[0].w2_weight[expert], [f"{base}.down_proj.weight"])
    assert checked == weights.keys(), weights.keys() - checked
    return sorted(checked)


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
    _compare_native_and_remote_draft(
        vllm_runner,
        monkeypatch,
        tmp_path,
        request,
        mtp_family,
        mtp_window,
        k,
        record_property,
    )


@pytest.mark.parametrize("ffn_layout", ["dense", "moe"])
@pytest.mark.parametrize("k", [1, 3])
def test_remote_deepseek_mtp_matches_native_and_target(
    vllm_runner,
    monkeypatch,
    tmp_path,
    request,
    ffn_layout,
    k,
    tiny_deepseek_mtp_model,
    record_property,
):
    """Preserve dense MLA KV, initialized MTP weights and repeated-head feedback."""
    import json

    model = tiny_deepseek_mtp_model
    record_property(
        "fixture", json.loads((Path(model).parent / "fixture.json").read_text())
    )
    _compare_native_and_remote_draft(
        vllm_runner,
        monkeypatch,
        tmp_path,
        request,
        f"deepseek_v3_{ffn_layout}",
        None,
        k,
        record_property,
        target_model=model,
    )


@pytest.mark.parametrize("mtp_family", ["qwen3_next", "nemotron_h", "glm4_moe_lite"])
@pytest.mark.parametrize("k", [1, 3])
def test_remote_independent_draft_with_hybrid_or_mla_target(
    vllm_runner,
    monkeypatch,
    tmp_path,
    request,
    mtp_family,
    k,
    tiny_independent_qwen3_models,
    record_property,
):
    """Keep target state local while a distinct dense AR model drafts token deltas."""
    import json

    target, draft = tiny_independent_qwen3_models
    record_property(
        "fixture", json.loads((tmp_path / "independent-control.json").read_text())
    )
    _compare_native_and_remote_draft(
        vllm_runner,
        monkeypatch,
        tmp_path,
        request,
        mtp_family,
        None,
        k,
        record_property,
        target_model=target,
        independent_draft=draft,
    )


@pytest.mark.parametrize("tiny_independent_mla_models", ["dense", "moe"], indirect=True)
@pytest.mark.parametrize("k", [1, 3])
def test_remote_independent_mla_draft_preserves_target(
    vllm_runner,
    monkeypatch,
    tmp_path,
    request,
    k,
    tiny_independent_mla_models,
    record_property,
):
    """Check exact target output and accepted drafts for dense and routed MLA."""
    import json

    target, draft = tiny_independent_mla_models
    fixture = json.loads((tmp_path / "mla-fixture.json").read_text())
    record_property("fixture", fixture)
    _compare_native_and_remote_draft(
        vllm_runner,
        monkeypatch,
        tmp_path,
        request,
        "qwen3",
        None,
        k,
        record_property,
        target_model=target,
        independent_draft=draft,
        independent_draft_kind="deepseek_v2",
        native_control=False,
    )


@pytest.mark.parametrize("k", [1, 3])
def test_remote_independent_recurrent_draft_preserves_target(
    vllm_runner,
    monkeypatch,
    tmp_path,
    request,
    k,
    tiny_independent_recurrent_models,
    record_property,
):
    """Use complete recurrent draft weights without moving target-owned state."""
    import json

    target, draft = tiny_independent_recurrent_models
    record_property(
        "fixture", json.loads((tmp_path / "recurrent-fixture.json").read_text())
    )
    _compare_native_and_remote_draft(
        vllm_runner,
        monkeypatch,
        tmp_path,
        request,
        "qwen3",
        None,
        k,
        record_property,
        target_model=target,
        independent_draft=draft,
        independent_draft_kind="nemotron_h",
        native_control=False,
    )


@pytest.mark.parametrize("eagle_target", ["llama", "qwen3", "qwen2", "mistral"])
@pytest.mark.parametrize("k", [1, 3])
def test_remote_eagle3_matches_native_and_target(
    vllm_runner,
    monkeypatch,
    tmp_path,
    request,
    eagle_target,
    k,
    tiny_eagle3_models,
    record_property,
):
    """Preserve real captured features and shared weights across draft placement."""
    import json

    target, draft = tiny_eagle3_models
    record_property(
        "fixture", json.loads((Path(draft).parent / "fixture.json").read_text())
    )
    _compare_native_and_remote_draft(
        vllm_runner,
        monkeypatch,
        tmp_path,
        request,
        eagle_target,
        None,
        k,
        record_property,
        target_model=target,
        eagle3_draft=draft,
    )


def _compare_native_and_remote_draft(
    vllm_runner,
    monkeypatch,
    tmp_path,
    request,
    mtp_family,
    mtp_window,
    k,
    record_property,
    *,
    independent_draft=None,
    independent_draft_kind="qwen3",
    target_model=None,
    eagle3_draft=None,
    native_control=True,
):
    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    independent = independent_draft is not None
    mla_draft = independent and independent_draft_kind == "deepseek_v2"
    recurrent_draft = independent and independent_draft_kind == "nemotron_h"
    if not native_control:
        import json

        assert mla_draft or recurrent_draft
        assert mtp_family == "qwen3"
        draft_config = json.loads((Path(independent_draft) / "config.json").read_text())
        if mla_draft:
            assert draft_config["n_routed_experts"] in (0, 4)
        else:
            assert draft_config["architectures"] == ["NemotronHForCausalLM"]
            assert draft_config["num_nextn_predict_layers"] == 0
    eagle3 = eagle3_draft is not None
    deepseek_mtp = mtp_family in ("deepseek_v3_dense", "deepseek_v3_moe")
    mimo_v2_flash = mtp_family == "mimo_v2_flash"
    step3p5 = mtp_family in ("step3p5_one_full", "step3p5_three_swa")
    step3p5_swa = mtp_family == "step3p5_three_swa"
    qwen35 = mtp_family in ("qwen35_dense", "qwen35_moe")
    detailed = (
        independent or eagle3 or deepseek_mtp or mimo_v2_flash or step3p5 or qwen35
    )
    hybrid_target = mtp_family in (
        "qwen3_next",
        "nemotron_h",
        "qwen35_dense",
        "qwen35_moe",
    )
    # GDN/Mamba backends do not implement batch-invariant kernels.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0" if hybrid_target else "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    model = target_model or request.getfixturevalue(f"tiny_{mtp_family}_mtp_model")
    mtp_windows = {
        "exaone_two": (None, None),
        "exaone_three": (None, 16, 32),
        "step3p5_one_full": (None,),
        "step3p5_three_swa": (128, 128, 128),
    }.get(mtp_family, (128 if mimo_v2_flash else mtp_window,))
    num_modules = len(mtp_windows)
    multi_module = num_modules > 1 and not step3p5
    dense_mla = mtp_family == "glm4_moe_lite" or deepseek_mtp
    attention_backend = (
        "TRITON_ATTN_DIFFKV"
        if mimo_v2_flash
        else "TRITON_MLA"
        if dense_mla
        else "TRITON_ATTN"
    )
    attention_config: dict[str, Any] = {"backend": attention_backend}
    if dense_mla or mla_draft:
        attention_config["mla_prefill_backend"] = "FLASH_ATTN"
        if independent:
            # Native AR uses the public per-kind map to give the MLA target and
            # independent dense draft their actual attention implementations.
            attention_config.update(
                backend="TRITON_ATTN",
                backend_per_kind={
                    "mla_attention": "TRITON_MLA",
                    "full_attention": "TRITON_ATTN",
                },
            )
    kwargs = dict(
        trust_remote_code=False,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=256 if mimo_v2_flash or step3p5_swa else 128,
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
    if independent:
        kwargs["tokenizer_mode"] = "hf"
    moe_backend = (
        None
        if mtp_family.startswith("exaone")
        or eagle3
        or mla_draft
        or recurrent_draft
        or mtp_family == "deepseek_v3_dense"
        or mtp_family == "qwen35_dense"
        else "triton"
    )
    if moe_backend is not None:
        kwargs["moe_backend"] = moe_backend
    if hybrid_target:
        kwargs.update(mamba_backend="triton", use_replayssm=False)
    if mtp_family == "qwen3_next" or qwen35:
        kwargs["gdn_prefill_backend"] = "triton"
        monkeypatch.setenv("VLLM_GDN_DECODE_KERNEL", "triton")
    prompts = [
        {"prompt_token_ids": [3 + (i * 7) % 125 for i in range(length)]}
        for length in (
            (31, 32, 33, 127, 128, 129, 193)
            if mimo_v2_flash or step3p5_swa
            else (15, 16, 17, 31, 32, 33, 65)
        )
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
        logprobs=5 if hybrid_target or dense_mla or detailed else None,
    )
    # MLA prefill and absorbed decode use different floating-point operations.
    compare_target_logprobs = hybrid_target or dense_mla
    record_property(
        "target_comparison",
        "top5_logprobs" if compare_target_logprobs else "exact_tokens",
    )
    record_property(
        "placement_comparison",
        "exact_tokens" if native_control else "remote_target_exact_tokens",
    )
    if not native_control:
        record_property(
            "native.unavailable",
            "Native mixed Qwen3/MLA execution builds draft metadata with the "
            "target head dimension; routed drafts also require an expert group "
            "that the dense target does not initialize. No native placement "
            "parity is claimed; standalone native proposal replay is separate."
            if mla_draft
            else "Mixed Qwen3/recurrent local speculation is outside this contract; "
            "an ordinary standalone native Nemotron-H proposal reference is separate.",
        )
    record_property("batch_invariant", not hybrid_target)

    def generate(runner, mode):
        outputs = [
            runner.llm.generate(batch, sampling, use_tqdm=False) for batch in batches
        ]
        if detailed:
            record_property(
                f"{mode}.outputs",
                [
                    [
                        {
                            "prompt_token_ids": prompt["prompt_token_ids"],
                            "token_ids": list(output.outputs[0].token_ids),
                            "text": output.outputs[0].text,
                            "logprobs": [
                                {
                                    token: {
                                        "logprob": value.logprob,
                                        "rank": value.rank,
                                        "decoded_token": value.decoded_token,
                                    }
                                    for token, value in step.items()
                                }
                                for step in output.outputs[0].logprobs
                            ],
                        }
                        for prompt, output in zip(prompts, batch)
                    ]
                    for prompts, batch in zip(batches, outputs)
                ],
            )
        if detailed and mode != "target":
            observations = []
            for batch in outputs:
                for output in batch:
                    metrics = output.outputs[0].spec_decode_metrics
                    assert metrics is not None
                    observation = metrics.to_dict()
                    assert observation["num_spec_tokens"] == k
                    assert len(metrics.histogram) == k + 1
                    assert metrics.per_step_accepted
                    assert len(metrics.per_step_accepted) == len(
                        metrics.per_step_drafted
                    )
                    assert all(
                        0 <= a <= d <= k
                        for a, d in zip(
                            metrics.per_step_accepted, metrics.per_step_drafted
                        )
                    )
                    observations.append(observation)
            record_property(f"{mode}.request_speculation", observations)
        return [
            [
                (
                    list(output.outputs[0].token_ids),
                    output.outputs[0].text,
                    output.outputs[0].logprobs,
                )
                for output in batch
            ]
            for batch in outputs
        ]

    def compare_outputs(actual, reference, mode):
        from tests.models.utils import check_logprobs_close

        assert len(actual) == len(reference) == len(batches)
        mismatches = 0
        first_mismatch = None
        for batch_index, (outputs, expected_outputs, prompts) in enumerate(
            zip(actual, reference, batches)
        ):
            assert len(outputs) == len(expected_outputs) == len(prompts)
            assert all(len(output[0]) == 8 for output in outputs)
            mismatches += sum(
                token != expected_token
                for output, expected_output in zip(outputs, expected_outputs)
                for token, expected_token in zip(output[0], expected_output[0])
            )
            if detailed and first_mismatch is None:
                for row, (output, expected_output) in enumerate(
                    zip(outputs, expected_outputs)
                ):
                    for position, (token, expected_token) in enumerate(
                        zip(output[0], expected_output[0])
                    ):
                        if token != expected_token:
                            first_mismatch = {
                                "batch": batch_index,
                                "row": row,
                                "position": position,
                                "actual_prefix": output[0][: position + 1],
                                "target_prefix": expected_output[0][: position + 1],
                            }
                            break
                    if first_mismatch is not None:
                        break
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
        if detailed:
            record_property(f"{mode}.first_target_token_mismatch", first_mismatch)

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
        if detailed or num_modules == 3:
            assert counts[2] > 0, "this draft fixture must exercise actual acceptance"
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
        from vllm.model_executor.layers.attention import Attention
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
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
            "model": type(runner.get_model()).__name__,
            "state": type(runner.model_state).__name__,
            "k": worker.vllm_config.num_speculative_tokens,
            "layers": layers,
            "attention_backends": [
                layer.get_attn_backend().get_name()
                for layer in runner.get_model().modules()
                if isinstance(layer, (Attention, MLAAttention))
            ],
        }

    def check_recurrence(runner, mode):
        policies = runner.llm.collective_rpc(recurrent_policy)
        record_property(f"{mode}.recurrent_policy", policies)
        assert len(policies) == 1
        policy = policies[0]
        assert policy["k"] == (0 if mode == "target" else k)
        if eagle3:
            assert (
                policy["model"]
                == {
                    "llama": "LlamaForCausalLM",
                    "qwen3": "Qwen3ForCausalLM",
                    "qwen2": "Qwen2ForCausalLM",
                    "mistral": "MistralForCausalLM",
                }[mtp_family]
            )
            assert policy["attention_backends"] == ["TRITON_ATTN"] * 4
        if independent:
            assert (
                policy["model"]
                == {
                    "qwen3_next": "Qwen3NextForCausalLM",
                    "nemotron_h": "NemotronHForCausalLM",
                    "glm4_moe_lite": "Glm4MoeLiteForCausalLM",
                    "qwen3": "Qwen3ForCausalLM",
                }[mtp_family]
            )
            assert policy["attention_backends"] == [attention_backend] * (
                1 if hybrid_target else 2
            )
        if deepseek_mtp:
            assert policy["model"] == "DeepseekV3ForCausalLM"
            assert policy["attention_backends"] == ["TRITON_MLA"] * 2
        if mimo_v2_flash:
            assert policy["model"] == "MiMoV2FlashForCausalLM"
            assert policy["attention_backends"] == ["TRITON_ATTN_DIFFKV"] * 2
        if step3p5:
            assert policy["model"] == "Step3p5ForCausalLM"
            assert policy["attention_backends"] == ["TRITON_ATTN"] * 2
        if qwen35:
            assert policy["model"] == (
                "Qwen3_5MoeForCausalLM"
                if mtp_family == "qwen35_moe"
                else "Qwen3_5ForCausalLM"
            )
            assert policy["attention_backends"] == ["TRITON_ATTN"]
        if hybrid_target:
            assert policy["state"] == "MambaHybridModelState"
            assert len(policy["layers"]) == 1
            layer = policy["layers"][0]
            is_gdn = mtp_family == "qwen3_next" or qwen35
            assert layer["type"] == ("GDN_ATTN" if is_gdn else "MAMBA2")
            conv_shape, state_shape = layer["shapes"]
            assert sorted(conv_shape) == [3 + policy["k"], 512 if is_gdn else 384]
            assert tuple(state_shape) == ((2, 128, 128) if is_gdn else (4, 64, 64))
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

    def deepseek_identity(worker):
        from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
        from vllm.model_executor.models.deepseek_v2 import (
            DeepseekV2MLP,
            DeepseekV2MoE,
        )
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

        runner = worker.model_runner
        assert isinstance(runner.speculator, MTPSpeculator)
        draft = runner.get_draft_model()
        assert isinstance(draft, DeepSeekMTP)
        assert draft.model.num_mtp_layers == 1
        assert draft.model.mtp_start_layer_idx == 2
        assert not hasattr(draft.config, "index_topk")
        dense = mtp_family == "deepseek_v3_dense"
        assert draft.config.n_routed_experts == (0 if dense else 4)
        assert draft.config.first_k_dense_replace == (3 if dense else 1)
        block = draft.model.layers["2"].mtp_block
        assert isinstance(block.mlp, DeepseekV2MLP if dense else DeepseekV2MoE)
        return _check_deepseek_mtp_weights(runner.get_model(), draft, model)

    def mimo_identity(worker):
        from vllm.v1.attention.backends.triton_attn_diffkv import (
            TritonAttentionDiffKVMetadataBuilder,
        )
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

        runner = worker.model_runner
        assert isinstance(runner.speculator, MTPSpeculator)
        builders = [
            group.get_metadata_builder()
            for groups in runner.speculator.attn_groups
            for group in groups
        ]
        assert builders and all(
            isinstance(builder, TritonAttentionDiffKVMetadataBuilder)
            for builder in builders
        )
        return dict(
            weights=_check_mimo_v2_flash_mtp_weights(
                runner.get_model(), runner.get_draft_model(), model
            ),
            metadata_builders=[type(builder).__name__ for builder in builders],
            geometry=_mimo_v2_flash_gpu_geometry(runner.get_draft_model(), mtp=True),
        )

    def qwen35_identity(worker):
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

        runner = worker.model_runner
        assert isinstance(runner.speculator, MTPSpeculator)
        target, draft = runner.get_model(), runner.get_draft_model()
        assert type(draft).__name__ == (
            "Qwen3_5MoeMTP" if mtp_family == "qwen35_moe" else "Qwen3_5MTP"
        )
        assert draft.model.config is target.model.config
        assert not worker.vllm_config.model_config.uses_mrope
        assert not worker.vllm_config.model_config.is_multimodal_model
        assert len(draft.model.layers) == draft.model.num_mtp_layers == 1
        assert draft.model.embed_tokens is target.model.embed_tokens
        assert draft.lm_head is target.lm_head
        target_weights = _check_qwen35_model_weights(target, model)
        draft_weights = _check_qwen35_model_weights(draft, model, mtp=True)
        return dict(
            target_weights=target_weights,
            draft_weights=draft_weights,
            shared_embedding_and_head=True,
            target_rope=target.model.config.rope_parameters,
            draft_rope=draft.model.config.rope_parameters,
            scalar_native_positions=True,
        )

    def step3p5_identity(worker):
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

        runner = worker.model_runner
        spec = worker.vllm_config.speculative_config
        assert isinstance(runner.speculator, MTPSpeculator)
        assert type(runner.model_state).__name__ == "DefaultModelState"
        assert not spec.use_multi_module_mtp() and not spec.use_remote_draft()
        assert worker.vllm_config.num_speculative_tokens == k
        assert worker.vllm_config.num_prefill_lookahead_tokens == 1
        target, draft = runner.get_model(), runner.get_draft_model()
        geometry = {
            "target": _step3p5_attention_geometry(target),
            "draft": _step3p5_attention_geometry(draft),
        }
        assert [row["window"] for row in geometry["target"]] == (
            [None, 128] if step3p5_swa else [None, None]
        )
        assert [row["window"] for row in geometry["draft"]] == list(mtp_windows)
        assert all(
            row["query_heads"] == 4
            and row["kv_heads"] == 2
            and row["head_dim"] == 32
            and row["backend"] == "TRITON_ATTN"
            and row["cache_dtype"] == "torch.bfloat16"
            for rows in geometry.values()
            for row in rows
        )
        return dict(
            weights=_check_step3p5_mtp_weights(target, draft, model),
            geometry=geometry,
            speculator=type(runner.speculator).__name__,
            model_state=type(runner.model_state).__name__,
            use_multi_module_mtp=False,
            num_prefill_lookahead_tokens=1,
            num_speculative_tokens=k,
        )

    def independent_draft_identity(worker):
        from vllm.model_executor.layers.attention import Attention
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
        from vllm.v1.attention.backends.mla.triton_mla import TritonMLAMetadataBuilder
        from vllm.v1.attention.backends.triton_attn import (
            TritonAttentionMetadataBuilder,
        )
        from vllm.v1.worker.gpu.model_states import resolve_model_state_cls
        from vllm.v1.worker.gpu.model_states.default import DefaultModelState
        from vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator import (
            StandaloneARSpeculator,
        )

        runner = worker.model_runner
        speculator = runner.speculator
        assert isinstance(speculator, StandaloneARSpeculator)
        draft = runner.get_draft_model()
        assert type(draft).__name__ == (
            "DeepseekV2ForCausalLM" if mla_draft else "Qwen3ForCausalLM"
        )
        config = speculator.attn_vllm_config
        assert config.model_config.model == independent_draft
        assert config.model_config.get_hidden_size() == (128 if mla_draft else 64)
        assert config.model_config.get_num_kv_heads(config.parallel_config) == 1
        assert config.model_config.get_head_size() == (576 if mla_draft else 32)
        assert config.cache_config.cache_dtype == "auto"
        assert worker.vllm_config.speculative_config.kv_cache_dtype == "auto"
        assert resolve_model_state_cls(config, draft) is DefaultModelState
        if mla_draft:
            assert worker.vllm_config.speculative_config.method == "draft_model"
            assert config.model_config.use_mla and not draft.use_mha
            from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLP

            assert draft.config.n_routed_experts == 0
            assert draft.config.first_k_dense_replace == 2
            assert len(draft.model.layers) == 2
            assert all(
                isinstance(layer.mlp, DeepseekV2MLP) for layer in draft.model.layers
            )
            assert mla_layout(draft) == [expected_mla_layout] * 2
            for groups in speculator.attn_groups:
                for group in groups:
                    builder = group.get_metadata_builder()
                    assert isinstance(builder, TritonMLAMetadataBuilder)
                    assert builder.mla_dims.kv_lora_rank == 512
            assert len([m for m in draft.modules() if isinstance(m, MLAAttention)]) == 2
            return {
                "speculator": type(speculator).__name__,
                "model": type(draft).__name__,
                "resolved_model_state": "DefaultModelState",
                "hidden_size": 128,
                "kv_heads": 1,
                "head_size": 576,
                "backend": "TRITON_MLA",
                "metadata_builders": [
                    type(group.get_metadata_builder()).__name__
                    for groups in speculator.attn_groups
                    for group in groups
                ],
                "kv_cache_dtype": config.cache_config.cache_dtype,
                "dense_mlp_layers": 2,
                "mla_layout": mla_layout(draft),
                "checkpoint_weights": _check_independent_mla_weights(
                    draft, independent_draft
                ),
            }
        layers = [layer for layer in draft.modules() if isinstance(layer, Attention)]
        assert len(layers) == 2
        assert all(
            layer.num_kv_heads == 1 and layer.head_size == 32 for layer in layers
        )
        assert all(layer.sliding_window is None for layer in layers)
        assert all(
            layer.get_attn_backend().get_name() == "TRITON_ATTN" for layer in layers
        )
        builders = []
        for groups in speculator.attn_groups:
            for group in groups:
                builder = group.get_metadata_builder()
                assert isinstance(builder, TritonAttentionMetadataBuilder)
                assert (builder.num_heads_q, builder.num_heads_kv, builder.headdim) == (
                    2,
                    1,
                    32,
                )
                builders.append(type(builder).__name__)
        assert builders
        return {
            "speculator": type(speculator).__name__,
            "model": type(draft).__name__,
            "resolved_model_state": "DefaultModelState",
            "hidden_size": 64,
            "kv_heads": 1,
            "head_size": 32,
            "backend": "TRITON_ATTN",
            "metadata_builders": builders,
            "kv_cache_dtype": config.cache_config.cache_dtype,
        }

    def token_remote_identity(worker):
        from vllm.v1.worker.gpu.spec_decode.token_remote import (
            TokenOnlyRemoteSpeculator,
        )

        runner = worker.model_runner
        assert isinstance(runner.speculator, TokenOnlyRemoteSpeculator)
        assert runner.draft_backend is runner.speculator
        assert runner.get_draft_model() is None
        spec = worker.vllm_config.speculative_config
        assert spec.method == "draft_model"
        assert spec.draft_model_config.model == independent_draft
        return type(runner.speculator).__name__

    def eagle_identity(worker):
        from vllm.model_executor.layers.attention import Attention
        from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
        from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
        from vllm.v1.worker.gpu.spec_decode.remote import RemoteSpeculator

        runner = worker.model_runner
        target = runner.get_model()
        assert (
            type(target).__name__
            == {
                "llama": "LlamaForCausalLM",
                "qwen3": "Qwen3ForCausalLM",
                "qwen2": "Qwen2ForCausalLM",
                "mistral": "MistralForCausalLM",
            }[mtp_family]
        )
        spec = worker.vllm_config.speculative_config
        capture_ids = target.model.aux_hidden_state_layers
        assert capture_ids == (() if spec is None else (1, 2, 4))
        result = {"target": type(target).__name__, "capture_ids": capture_ids}
        if spec is None:
            return result
        assert spec.method == "eagle3" and spec.draft_model_config.model == eagle3_draft
        result["speculator"] = type(runner.speculator).__name__
        if spec.use_remote_draft():
            assert isinstance(runner.speculator, RemoteSpeculator)
            assert runner.get_draft_model() is None
        else:
            assert isinstance(runner.speculator, EagleSpeculator)
            draft = runner.get_draft_model()
            assert isinstance(draft, Eagle3LlamaForCausalLM)
            assert draft.model.use_aux_hidden_state
            assert draft.model.num_aux_hidden_states == 3
            assert draft.model.fc_input_size == 384
            assert draft.model.fc.weight.shape == (128, 384)
            assert draft.model.layers[0].self_attn.qkv_proj.weight.shape == (256, 256)
            assert draft.model.embed_tokens is target.model.embed_tokens
            assert draft.lm_head is target.lm_head
            backends = [
                layer.get_attn_backend().get_name()
                for layer in draft.modules()
                if isinstance(layer, Attention)
            ]
            assert backends == ["TRITON_ATTN"]
            result.update(
                draft=type(draft).__name__,
                fc_shape=tuple(draft.model.fc.weight.shape),
                shared_embedding_and_head=True,
                attention_backends=backends,
            )
        return result

    config = dict(
        method="draft_model" if independent else "eagle3" if eagle3 else "mtp",
        model=independent_draft or eagle3_draft or model,
        num_speculative_tokens=k,
        draft_tensor_parallel_size=1,
        draft_sample_method="greedy",
        attention_backend=(
            "TRITON_MLA"
            if mla_draft
            else "TRITON_ATTN"
            if independent
            else attention_backend
        ),
    )
    if independent:
        # Public native control: select draft-owned metadata geometry while
        # preserving the target's automatic BF16 cache precision.
        config["kv_cache_dtype"] = "auto"
    with vllm_runner(model, **kwargs) as baseline:
        windows = baseline.llm.apply_model(target_windows)
        assert windows == [
            [None, 128]
            if mimo_v2_flash or step3p5_swa
            else [None, None]
            if step3p5
            else [None] * (4 if eagle3 else 1 if hybrid_target else max(2, num_modules))
        ]
        record_property("target.attention_windows", windows)
        if mimo_v2_flash:
            record_property(
                "target.diffkv_geometry",
                baseline.llm.apply_model(_mimo_v2_flash_gpu_geometry),
            )
        if eagle3:
            record_property(
                "target.eagle3", baseline.llm.collective_rpc(eagle_identity)
            )
        if dense_mla:
            layout = baseline.llm.apply_model(mla_layout)
            assert layout == [[expected_mla_layout] * 2]
            record_property("target.mla_layout", layout)
        backends = baseline.llm.apply_model(moe_backends)
        expected_moe_count = (
            2 if mtp_family in ("qwen3_next", "qwen35_moe") else int(bool(moe_backend))
        )
        assert backends == [["TRITON"] * expected_moe_count]
        record_property("target.moe_backends", backends)
        check_recurrence(baseline, "target")
        expected = generate(baseline, "target")
    assert all(len(output[0]) == 8 for batch in expected for output in batch)
    if native_control:
        with vllm_runner(
            model,
            speculative_config=config,
            per_request_spec_decode_metrics="detailed" if detailed else "none",
            **kwargs,
        ) as native:
            policy = native.llm.collective_rpc(draft_kv_policy)
            assert policy == [
                [
                    (
                        window,
                        "MLAAttentionSpec"
                        if mla_draft or (dense_mla and not independent)
                        else "FullAttentionSpec"
                        if window is None
                        else "SlidingWindowSpec",
                        window,
                    )
                    for window in ((None, None) if independent else mtp_windows)
                ]
            ]
            record_property("native.kv_policy", policy)
            if multi_module:
                identity = native.llm.collective_rpc(draft_module_identity)
                record_property("native.module_identity", identity)
            if independent:
                identity = native.llm.collective_rpc(independent_draft_identity)
                record_property("native.independent_draft", identity)
            if eagle3:
                record_property(
                    "native.eagle3", native.llm.collective_rpc(eagle_identity)
                )
            if deepseek_mtp:
                record_property(
                    "native.deepseek_weights",
                    native.llm.collective_rpc(deepseek_identity),
                )
            if mimo_v2_flash:
                record_property(
                    "native.mimo_identity", native.llm.collective_rpc(mimo_identity)
                )
            if step3p5:
                record_property(
                    "native.step3p5_identity",
                    native.llm.collective_rpc(step3p5_identity),
                )
            if qwen35:
                record_property(
                    "native.qwen35_identity", native.llm.collective_rpc(qwen35_identity)
                )
            if dense_mla and not independent:
                layout = native.llm.collective_rpc(draft_mla_layout)
                assert layout == [[expected_mla_layout]]
                record_property("native.mla_layout", layout)
            backends = native.llm.collective_rpc(draft_moe_backends)
            draft_has_moe = (
                not independent
                and not eagle3
                and not mtp_family.startswith("exaone")
                and mtp_family != "ernie"
                and mtp_family != "deepseek_v3_dense"
                and mtp_family != "qwen35_dense"
                and not mimo_v2_flash
                and not step3p5
            )
            assert backends == [["TRITON"] if draft_has_moe else []]
            record_property("native.moe_backends", backends)
            recurrent_layers = native.llm.collective_rpc(draft_recurrent_layers)
            assert recurrent_layers == [0]
            record_property("native.recurrent_layers", recurrent_layers)
            native_recurrence = check_recurrence(native, "native")
            native_outputs = generate(native, "native")
            compare_outputs(native_outputs, expected, "native")
            native_counts = speculation_counts(native, "native")
    server_log = tmp_path / f"{mtp_family}-draft.log"
    with draft_server(
        model,
        server_log,
        token_only_draft=independent_draft,
        eagle3_draft=eagle3_draft,
        revision=None,
        num_speculative_tokens=k,
        attention_backend=(
            "TRITON_MLA"
            if mla_draft
            else "TRITON_ATTN"
            if independent
            else attention_backend
        ),
        target_attention_backend=attention_backend,
        draft_kv_cache_gib=0.03125,
        max_feature_tokens=128 if recurrent_draft else 32,
        max_model_len=256 if mimo_v2_flash or step3p5_swa else 128,
        trust_remote_code=False,
        moe_backend=moe_backend,
        batch_invariant=False if recurrent_draft else None,
    ) as endpoint:
        transports = ("inline", "cuda_ipc") if multi_module else ("inline",)
        for transport in transports:
            mode = "remote" if transport == "inline" else "remote_ipc"
            remote_options = dict(
                transport=transport,
                failure_policy="error",
                request_timeout_ms=5000,
                startup_timeout_ms=600000,
            )
            remote_config = dict(config)
            if independent:
                remote_config["execution"] = dict(
                    placement="remote",
                    policy="on_demand",
                    endpoints=[endpoint],
                    **remote_options,
                )
            else:
                remote_config["remote_draft"] = dict(
                    endpoint=endpoint, **remote_options
                )
            with vllm_runner(
                model,
                speculative_config=remote_config,
                per_request_spec_decode_metrics="detailed" if detailed else "none",
                **kwargs,
            ) as remote:
                actual_config = remote.llm.llm_engine.vllm_config.speculative_config
                assert actual_config.use_remote_draft()
                assert actual_config.get_remote_draft_endpoints() == (endpoint,)
                if recurrent_draft:
                    assert actual_config.method == "draft_model"
                    draft_config = actual_config.draft_model_config
                    assert draft_config.architecture == "NemotronHForCausalLM"
                    assert draft_config.hf_config.num_nextn_predict_layers == 0
                remote_recurrence = check_recurrence(remote, mode)
                if native_control:
                    assert remote_recurrence == native_recurrence
                if independent:
                    record_property(
                        "remote.backend",
                        remote.llm.collective_rpc(token_remote_identity),
                    )
                if eagle3:
                    record_property(
                        "remote.eagle3", remote.llm.collective_rpc(eagle_identity)
                    )
                remote_outputs = generate(remote, mode)
                compare_outputs(remote_outputs, expected, mode)
                if native_control:
                    for actual_batch, native_batch in zip(
                        remote_outputs, native_outputs
                    ):
                        assert [output[0] for output in actual_batch] == [
                            output[0] for output in native_batch
                        ]
                        if dense_mla or detailed:
                            assert [output[2] for output in actual_batch] == [
                                output[2] for output in native_batch
                            ]
                    if dense_mla or detailed:
                        record_property("placement.logprobs", "exact_returned_top5")
                remote_counts = speculation_counts(remote, mode)
                if native_control:
                    if detailed:
                        record_property(
                            "placement.speculation_counts_equal",
                            remote_counts == native_counts,
                        )
                    assert remote_counts == native_counts
    if multi_module:
        server_lines = server_log.read_text().splitlines()
        negotiated = [
            line.rsplit(" over ", 1)[1].strip()
            for line in server_lines
            if "opened for verifier" in line and " over " in line
        ]
        assert negotiated == ["inline", "cuda_ipc"]
        record_property("remote.negotiated_transports", negotiated)
        warmup = (
            "Standalone multi-module warmup: steps=3 "
            f"replayed_rows={num_modules - 1} floating_feature_device=cuda:0"
        )
        assert sum(warmup in line for line in server_lines) == 1
        record_property("remote.device_replay_warmup", warmup)

    if eagle3:
        identity = (
            "Standalone EAGLE3 features: model=Eagle3LlamaForCausalLM "
            "target_layers=(1, 2, 4) input_width=384 output_width=128"
        )
        assert (
            sum(identity in line for line in server_log.read_text().splitlines()) == 1
        )
        record_property("remote.eagle3_from_server_log", identity)
    if independent and not recurrent_draft:
        import ast

        identity_prefix = "Token-only draft loaded: "
        identities = [
            line.split(identity_prefix, 1)[1]
            for line in server_log.read_text().splitlines()
            if identity_prefix in line
        ]
        assert len(identities) == 1
        identity, layout_text = identities[0].split(" kv_layouts=", 1)
        expected_model = "DeepseekV2ForCausalLM" if mla_draft else "Qwen3ForCausalLM"
        expected_backend = "TRITON_MLA" if mla_draft else "TRITON_ATTN"
        assert identity == (
            f"speculator=StandaloneARSpeculator model={expected_model} "
            f"state=DefaultModelState attention_backends=['{expected_backend}']"
        )
        layouts = ast.literal_eval(layout_text)
        assert layouts and sum(layout["layers"] for layout in layouts) == 2
        assert all(
            layout["kind"] == ("MLAAttentionSpec" if mla_draft else "FullAttentionSpec")
            and layout["kv_heads"] == 1
            and layout["head_size"] == (576 if mla_draft else 32)
            and layout["window"] is None
            and layout["dtype"] == "torch.bfloat16"
            for layout in layouts
        )
        record_property(
            "remote.loaded_draft", {"identity": identity, "kv_layouts": layouts}
        )
    if recurrent_draft:
        import json

        prefix = "Recurrent token-only draft loaded: "
        identities = [
            json.loads(line.split(prefix, 1)[1])
            for line in server_log.read_text().splitlines()
            if prefix in line
        ]
        assert len(identities) == 1
        identity = identities[0]
        assert identity["executor"] == "RecurrentGPUStepExecutor"
        assert identity["model"] == "NemotronHForCausalLM"
        assert identity["state"] == "MambaHybridModelState"
        assert identity["speculative_config"] is None
        assert identity["num_speculative_tokens"] == 0
        assert identity["mamba_backend"] == "triton"
        assert identity["replay_policy"] == "full_computed_prefix"
        assert identity["max_num_reqs"] == 4 and identity["row_capacity"] == 1
        assert identity["max_model_len"] == identity["max_num_tokens"] == 128
        layouts = identity["kv_layouts"]
        assert {layout["kind"] for layout in layouts} == {
            "FullAttentionSpec",
            "MambaSpec",
        }
        recurrent = next(layout for layout in layouts if layout["kind"] == "MambaSpec")
        assert recurrent["blocks"] == 5
        assert recurrent["num_speculative_blocks"] == 0
        assert recurrent["cache_mode"] == "none"
        assert recurrent["dtypes"] == ["torch.bfloat16", "torch.float32"]
        assert sorted(recurrent["shapes"][0]) == [3, 384]
        assert recurrent["shapes"][1] == [4, 64, 64]
        assert len(identity["native_states"]) == 1
        states = next(iter(identity["native_states"].values()))
        assert states == [
            dict(shape=[5, *shape], dtype=dtype)
            for shape, dtype in zip(recurrent["shapes"], recurrent["dtypes"])
        ]
        record_property("remote.loaded_recurrent_draft", identity)
    if dense_mla and not independent:
        assert any(
            "Using " in line and "TRITON_MLA" in line and "backend." in line
            for line in server_log.read_text().splitlines()
        )
        record_property("remote.decode_backend_from_server_log", "TRITON_MLA")


def _run_independent_recurrent_oracle(
    target_path, draft_path, result_path, attention_blocks, live_slots
):
    """Exercise native compact caches with authoritative computed-prefix replay."""
    import itertools
    import json

    import torch

    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
    from vllm.v1.spec_decode.remote.protocol import SequenceKey, SpeculatorStatusCode
    from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
    from vllm.v1.spec_decode.remote.runtime.recurrent_gpu import (
        RecurrentGPUStepExecutor,
        create_recurrent_executor_config,
    )
    from vllm.v1.spec_decode.remote.runtime.token_state import (
        TokenAdvanceRow,
        TokenStateMachine,
    )
    from vllm.v1.spec_decode.remote.server import AdapterError

    attention_blocks, live_slots = int(attention_blocks), int(live_slots)
    assert (attention_blocks, live_slots) in ((2, 4), (9, 2))
    torch.set_num_threads(1)
    config = EngineArgs(
        model=target_path,
        tokenizer_mode="hf",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=live_slots,
        max_num_batched_tokens=128,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=True,
        block_size=16,
        attention_config={"backend": "TRITON_ATTN"},
        speculative_config={
            "method": "draft_model",
            "model": draft_path,
            "num_speculative_tokens": 3,
            "draft_tensor_parallel_size": 1,
            "attention_backend": "TRITON_ATTN",
        },
    ).create_engine_config()
    assert config.speculative_config is not None
    assert config.speculative_config.method == "draft_model"
    ordinary = create_recurrent_executor_config(config)
    state_spec = MambaSpec(
        block_size=ordinary.cache_config.mamba_block_size,
        shapes=NemotronHForCausalLM.get_mamba_state_shape_from_config(ordinary),
        dtypes=NemotronHForCausalLM.get_mamba_state_dtype_from_config(ordinary),
        page_size_padded=ordinary.cache_config.mamba_page_size_padded,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=1,
    )
    attention_spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=32, dtype=torch.bfloat16
    )
    budget = (live_slots + 1) * state_spec.page_size_bytes + (
        attention_blocks * attention_spec.page_size_bytes
    )
    executor = RecurrentGPUStepExecutor(
        config, device=torch.device("cuda:0"), kv_cache_bytes=budget
    )
    identity = _check_independent_recurrent_state(
        executor.model, executor.model_state, executor.vllm_config
    )
    checked_weights = _check_independent_recurrent_weights(executor.model, draft_path)
    assert executor.num_blocks == attention_blocks
    assert executor.kv_config.recurrent_num_blocks == live_slots + 1
    assert identity["state_slots"] == live_slots + 1
    assert executor.kv_config.kv_cache_tensors[0].size == budget
    assert executor.max_num_reqs == 1
    assert identity["attention_block_size"] == 16
    assert [
        group.kv_cache_spec
        for group in executor.kv_config.kv_cache_groups
        if type(group.kv_cache_spec) is FullAttentionSpec
    ] == [attention_spec]
    assert [
        group.kv_cache_spec
        for group in executor.kv_config.kv_cache_groups
        if isinstance(group.kv_cache_spec, MambaSpec)
    ] == [state_spec]
    assert state_spec.max_memory_usage_bytes(ordinary) == state_spec.page_size_bytes
    assert state_spec.max_num_blocks_per_req(ordinary, 128) == 1
    mamba = next(
        layer for layer in executor.model.modules() if isinstance(layer, MambaMixer2)
    )
    attention = next(
        layer for layer in executor.model.modules() if isinstance(layer, Attention)
    )
    assert attention.kv_cache.shape[1:] == (2, 16, 64)
    spans = []
    for tensor in executor.kv_caches.values():
        size = 1 + sum(
            (dim - 1) * stride for dim, stride in zip(tensor.shape, tensor.stride())
        )
        spans.append(
            (tensor.data_ptr(), tensor.data_ptr() + size * tensor.element_size())
        )
    assert len(spans) == 2
    assert spans[0][1] <= spans[1][0] or spans[1][1] <= spans[0][0]
    assert (
        len({t.untyped_storage().data_ptr() for t in executor.kv_caches.values()}) == 1
    )
    assert all(
        t.untyped_storage().nbytes() == budget for t in executor.kv_caches.values()
    )
    executor.warmup()
    assert executor.histories.num_sequences == executor.histories.token_count == 0
    pool = DraftBlockPool(attention_blocks, 16)
    state = TokenStateMachine(
        executor,
        pool,
        max_num_reqs=live_slots,
        max_model_len=128,
        vocab_size=128,
        max_k=3,
    )
    checks, references, outcomes, batch_ids = [], [], [], itertools.count()
    state_references = []
    max_attention_id = 0
    reference_ids = itertools.count(100)

    def allocation():
        torch.accelerator.synchronize(executor.device)
        return dict(
            allocated=torch.accelerator.memory_allocated(executor.device),
            reserved=torch.accelerator.memory_reserved(executor.device),
        )

    @torch.inference_mode()
    def poison(key):
        index = state.sequences[key].req_index
        for cache in mamba.kv_cache:
            cache[index + 1].fill_(torch.nan)
        blocks = pool.blocks(index)
        if blocks:
            attention.kv_cache.index_fill_(
                0, torch.tensor(blocks, device=executor.device), torch.nan
            )

    def snapshot(key, computed):
        nonlocal max_attention_id
        index = state.sequences[key].req_index
        assert executor.histories.tokens(key) == tuple(computed)
        blocks = pool.blocks(index)
        max_attention_id = max(max_attention_id, *blocks)
        native_state = [cache[index + 1].cpu().clone() for cache in mamba.kv_cache]
        cache = attention.kv_cache.index_select(
            0, torch.tensor(blocks, device=executor.device)
        )
        native_state.append(cache.transpose(1, 2).flatten(0, 1)[: len(computed)].cpu())
        assert all(torch.isfinite(value).all() for value in native_state)
        return native_state

    def same_state(actual, expected, label):
        assert len(actual) == len(expected) == 3
        assert all(torch.equal(a, b) for a, b in zip(actual, expected)), label

    def independent_reference(computed, actual, label):
        reference = SequenceKey("recurrent-reference", next(reference_ids), 0)
        state.open(reference)
        poison(reference)
        state.prefill(reference, 0, tuple(computed), is_final=True)
        same_state(actual, snapshot(reference, computed), label)
        state.close((reference,))

    def observe(key, authoritative, proposal, label, *, compare=True):
        # The final sampled token has not been forwarded. This golden prefix is
        # built independently from the accepted transcript and returned proposal.
        computed = [*authoritative, *proposal[:-1]]
        progress = state.sequences[key]
        assert progress.committed == progress.materialized == len(authoritative)
        assert progress.speculative_written_end == len(computed)
        assert progress.pending_last_token == (
            (len(computed), proposal[-1]) if proposal else None
        )
        actual = snapshot(key, computed)
        if compare:
            # Rebuild only after every live request closes. A long prefix may
            # consume all eight token pages even though a state slot is free.
            state_references.append((computed, actual, label))
        if proposal:
            references.append(
                dict(
                    case=label,
                    committed_token_ids=list(authoritative),
                    token_ids=list(proposal),
                )
            )
        checks.append(
            dict(
                case=label,
                committed=len(authoritative),
                computed=len(computed),
                k=len(proposal),
                req_index=progress.req_index,
                state_index=progress.req_index + 1,
                independent_state_reference=compare,
                poisoned_before_replay=True,
            )
        )
        return computed, actual

    before = allocation()
    torch.accelerator.reset_peak_memory_stats(executor.device)
    keys = [SequenceKey("recurrent-live", row, 0) for row in range(live_slots)]
    for key in keys:
        state.open(key)
    key = keys[-1] if attention_blocks == 2 else keys[0]
    if attention_blocks == 9:
        state.close(tuple(keys[1:]))
    else:
        assert state.sequences[key].req_index == 3
        assert state.sequences[key].req_index + 1 >= attention_blocks
    authoritative = [7]
    poison(key)
    state.prefill(key, 0, tuple(authoritative), is_final=True)
    proposal: list[int] = []
    previous_id: int | None = None
    schedule = [
        (3, 0, 0, (5,)),
        (0, 1, 0, (6,)),
        (3, 0, 0, (8, 9)),
        (1, 3, 2, (10,)),
        (3, 1, 1, (11,)),
        (0, 3, 3, (12,)),
        (0, 0, 0, (13, 14)),
    ]
    for step, (k, offered, accepted, recovery) in enumerate(schedule):
        delta = (*proposal[:accepted], *recovery)
        advance_row = TokenAdvanceRow(
            key, len(authoritative), previous_id, offered, accepted, delta
        )
        poison(key)
        result = state.advance(next(batch_ids), advance_row, k)
        authoritative.extend(delta)
        proposal = list(result.token_ids)
        previous_id = result.proposal_id
        observe(
            key,
            authoritative,
            proposal,
            f"correction-{step}",
            compare=attention_blocks == 9,
        )
        outcomes.append(
            dict(k=k, offered=offered, accepted=accepted, delta=list(delta))
        )
    if attention_blocks == 2:
        assert len(authoritative) == 16
        prior = executor.histories.tokens(key)
        forwards = executor.completed_forwards
        with pytest.raises(AdapterError) as caught:
            state.advance(
                next(batch_ids),
                TokenAdvanceRow(key, 16, previous_id, 0, 0, (15,)),
                1,
            )
        assert caught.value.status is SpeculatorStatusCode.OUT_OF_MEMORY
        assert executor.histories.tokens(key) == prior
        assert executor.completed_forwards == forwards
        assert state.sequences[key].ready
    state.close(tuple(keys))
    assert not state.sequences and executor.histories.num_sequences == 0
    if attention_blocks == 9:
        for length in (15, 16, 17, 31, 32, 33):
            key = SequenceKey("boundary", length, 0)
            state.open(key)
            authoritative = [3 + (index * 7) % 125 for index in range(length)]
            for offset in range(0, length, 15):
                poison(key)
                chunk = tuple(authoritative[offset : offset + 15])
                state.prefill(
                    key, offset, chunk, is_final=offset + len(chunk) == length
                )
            poison(key)
            result = state.advance(
                next(batch_ids), TokenAdvanceRow(key, length, None, 0, 0, (19,)), 3
            )
            authoritative.append(19)
            observe(key, authoritative, list(result.token_ids), f"boundary-{length}")
            state.close((key,))
        # Two live rows exercise caller order with distinct state slots/pages.
        live = [SequenceKey("reorder", row, 0) for row in range(2)]
        transcripts = [[7] * 15, [9] * 17]
        proposals: list[list[int]] = [[], []]
        ids: list[int | None] = [None, None]
        for item, tokens in zip(live, transcripts):
            state.open(item)
            poison(item)
            state.prefill(item, 0, tuple(tokens), is_final=True)
        assert not set(pool.blocks(state.sequences[live[0]].req_index)) & set(
            pool.blocks(state.sequences[live[1]].req_index)
        )
        for step, order in enumerate(((0, 1), (1, 0))):
            updates = []
            for row in order:
                offered = len(proposals[row]) if row else min(1, len(proposals[row]))
                accepted = int(step == 1 and row == 1)
                delta = (*proposals[row][:accepted], 23 + row)
                updates.append(
                    (
                        next(batch_ids),
                        TokenAdvanceRow(
                            live[row],
                            len(transcripts[row]),
                            ids[row],
                            offered,
                            accepted,
                            delta,
                        ),
                        3 if step == 0 else 1,
                    )
                )
                transcripts[row].extend(delta)
                poison(live[row])
            results = state.advance_many(tuple(updates))
            for row, result in zip(order, results):
                assert not isinstance(result, Exception), results
                proposals[row] = list(result.token_ids)
                ids[row] = result.proposal_id
                observe(
                    live[row],
                    transcripts[row],
                    proposals[row],
                    f"reorder-{step}-{row}",
                )
        state.close(tuple(live))
        for length in (125, 126, 127):
            key = SequenceKey("max-context", length, 0)
            state.open(key)
            poison(key)
            state.prefill(key, 0, tuple([7] * length), is_final=True)
            poison(key)
            result = state.advance(
                next(batch_ids), TokenAdvanceRow(key, length, None, 0, 0, (19,)), 3
            )
            assert len(result.token_ids) == 127 - length
            observe(
                key,
                [*([7] * length), 19],
                list(result.token_ids),
                f"max-context-{length}",
            )
            state.close((key,))
        assert max_attention_id >= live_slots + 1
    cycles, previous = [], None
    for generation in (1, 2, 3):
        key = SequenceKey("recurrent-live", live_slots - 1, generation)
        state.open(key)
        poison(key)
        state.prefill(key, 0, (7,), is_final=True)
        poison(key)
        result = state.advance(
            next(batch_ids), TokenAdvanceRow(key, 1, None, 0, 0, (5,)), 1
        )
        _, actual = observe(
            key,
            [7, 5],
            list(result.token_ids),
            f"reuse-{generation}",
            compare=attention_blocks == 9,
        )
        if previous is not None:
            same_state(actual, previous, "poisoned generation reuse")
        previous = actual
        state.close((key,))
        assert executor.histories.num_sequences == executor.histories.token_count == 0
        assert pool.num_free_blocks == attention_blocks - 1
        cycles.append(allocation())
    assert cycles[-1]["allocated"] == cycles[-2]["allocated"]
    assert not state.sequences and executor.histories.num_sequences == 0
    for computed, actual, label in state_references:
        independent_reference(computed, actual, label)
    assert not state.sequences and executor.histories.num_sequences == 0
    assert pool.num_free_blocks == attention_blocks - 1
    Path(result_path).write_text(
        json.dumps(
            dict(
                identity=identity,
                loaded_identity=executor.loaded_identity,
                checkpoint_weights=checked_weights,
                attention_blocks=attention_blocks,
                recurrent_blocks=live_slots + 1,
                kv_budget_bytes=budget,
                disjoint_backing_ranges=True,
                max_attention_block_id=max_attention_id,
                checks=checks,
                independent_state_references=[
                    label for _, _, label in state_references
                ],
                outcomes=outcomes,
                proposal_references=references,
                histories_after_close=executor.histories.num_sequences,
                computed_tokens_after_close=executor.histories.token_count,
                free_blocks_after_close=pool.num_free_blocks,
                completed_forwards=executor.completed_forwards,
                replayed_tokens=executor.replayed_tokens,
                allocation=dict(
                    before=before,
                    after_close=allocation(),
                    close_reopen_cycles=cycles,
                    peak_allocated=torch.accelerator.max_memory_allocated(
                        executor.device
                    ),
                    peak_reserved=torch.accelerator.max_memory_reserved(
                        executor.device
                    ),
                ),
            ),
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize(
    "attention_blocks,live_slots", [(2, 4), (9, 2)], ids=["b2-r5", "b9-r3"]
)
def test_remote_independent_recurrent_state_matches_native_full_prefix(
    vllm_runner,
    monkeypatch,
    tmp_path,
    tiny_independent_recurrent_models,
    attention_blocks,
    live_slots,
    record_property,
):
    """Use independent native state slots and ordinary one-token prefix references."""
    import json
    import math

    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    target, draft = tiny_independent_recurrent_models
    result_path = tmp_path / "recurrent-oracle.json"
    log_path = tmp_path / "recurrent-oracle.log"
    command = [
        sys.executable,
        "-c",
        "import sys; from tests.v1.e2e.spec_decode.remote.test_remote_mtp "
        "import _run_independent_recurrent_oracle; "
        "_run_independent_recurrent_oracle(*sys.argv[1:])",
        target,
        draft,
        str(result_path),
        str(attention_blocks),
        str(live_slots),
    ]
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            assert process.wait(timeout=300) == 0, (
                f"recurrent state oracle failed: {log_path}"
            )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    receipt = json.loads(result_path.read_text())
    record_property("native.recurrent_state_oracle", receipt)
    assert receipt["attention_blocks"] == attention_blocks
    assert receipt["recurrent_blocks"] == live_slots + 1
    assert len(receipt["checks"]) == (10 if attention_blocks == 2 else 23)
    assert receipt["independent_state_references"] == [
        check["case"]
        for check in receipt["checks"]
        if check["independent_state_reference"]
    ]
    assert (
        receipt["histories_after_close"] == receipt["computed_tokens_after_close"] == 0
    )
    assert receipt["free_blocks_after_close"] == attention_blocks - 1
    assert {outcome["accepted"] for outcome in receipt["outcomes"]} == {0, 1, 2, 3}
    assert receipt["disjoint_backing_ranges"]
    if attention_blocks == 2:
        assert all(
            check["state_index"] >= attention_blocks for check in receipt["checks"]
        )
    else:
        assert receipt["max_attention_block_id"] >= live_slots + 1
        assert {f"boundary-{length}" for length in (15, 16, 17, 31, 32, 33)} <= {
            check["case"] for check in receipt["checks"]
        }
    cases = receipt["proposal_references"]

    def identity(worker):
        from vllm.model_executor.layers.attention import Attention
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            MambaSpec,
            iter_layer_specs,
        )

        runner = worker.model_runner
        config = worker.vllm_config
        result = _check_independent_recurrent_state(
            runner.get_model(), runner.model_state, config
        )
        result["checkpoint_weights"] = _check_independent_recurrent_weights(
            runner.get_model(), draft
        )
        groups = runner.kv_cache_config.kv_cache_groups
        attention = next(
            layer
            for layer in runner.get_model().modules()
            if isinstance(layer, Attention)
        )
        public_attention_spec = attention.get_kv_cache_spec(config)
        assert type(public_attention_spec) is FullAttentionSpec
        full_specs = []
        native_groups = []
        for index, group in enumerate(groups):
            specs = list(iter_layer_specs(group.kv_cache_spec))
            assert len(specs) == 1
            spec = specs[0]
            assert isinstance(spec, (FullAttentionSpec, MambaSpec))
            if type(spec) is FullAttentionSpec:
                assert group.layer_names == [attention.layer_name]
                assert (
                    spec.num_kv_heads,
                    spec.head_size,
                    spec.dtype,
                ) == (
                    public_attention_spec.num_kv_heads,
                    public_attention_spec.head_size,
                    public_attention_spec.dtype,
                )
                full_specs.append(spec)
            native_groups.append(
                dict(
                    kind=type(spec).__name__,
                    layers=list(group.layer_names),
                    block_size=spec.block_size,
                    page_size_bytes=spec.page_size_bytes,
                    page_size_padded=spec.page_size_padded,
                    kernel_block_size=runner.kernel_block_sizes[index],
                )
            )
        assert len(full_specs) == 1
        assert {group["kind"] for group in native_groups} == {
            "FullAttentionSpec",
            "MambaSpec",
        }
        result["kv_groups"] = native_groups
        return result

    observations = []
    with vllm_runner(
        draft,
        tokenizer_mode="hf",
        skip_tokenizer_init=True,
        trust_remote_code=False,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=True,
        block_size=16,
        kv_cache_memory_bytes=32 << 20,
        gpu_memory_utilization=0.2,
        attention_config={"backend": "TRITON_ATTN"},
        mamba_backend="triton",
        mamba_cache_dtype="bfloat16",
        mamba_ssm_cache_dtype="float32",
        use_replayssm=False,
    ) as native:
        native_identity = native.llm.collective_rpc(identity)
        assert native_identity[0]["checkpoint_weights"] == receipt["checkpoint_weights"]
        for case in cases:
            prefix = list(case["committed_token_ids"])
            for position, expected in enumerate(case["token_ids"]):
                output = native.llm.generate(
                    [{"prompt_token_ids": list(prefix)}],
                    SamplingParams(
                        temperature=0,
                        max_tokens=1,
                        ignore_eos=True,
                        detokenize=False,
                        logprobs=5,
                    ),
                    use_tqdm=False,
                )[0]
                generated = output.outputs[0]
                assert output.prompt_token_ids is not None
                observed = dict(
                    case=case["case"],
                    position=position,
                    prefix=list(prefix),
                    native_prompt_token_ids=list(output.prompt_token_ids),
                    expected_token=expected,
                    token_ids=list(generated.token_ids),
                    finish_reason=generated.finish_reason,
                    logprobs=[
                        {
                            token: dict(logprob=value.logprob, rank=value.rank)
                            for token, value in values.items()
                        }
                        for values in generated.logprobs or []
                    ],
                )
                observations.append(observed)
                prefix.append(expected)
    record_property(
        "native.recurrent_proposal_replay",
        dict(
            identity=native_identity,
            comparison="exact_tokens_one_full_prefix_per_draft_position",
            requests=len(cases),
            tokens=len(observations),
            observations=observations,
        ),
    )
    for observed in observations:
        expected = observed["expected_token"]
        assert observed["native_prompt_token_ids"] == observed["prefix"]
        assert observed["finish_reason"] == "length"
        assert observed["token_ids"] == [expected], observed
        assert len(observed["logprobs"]) == 1
        values = observed["logprobs"][0]
        assert expected in values and len(values) >= 5
        assert all(math.isfinite(value["logprob"]) for value in values.values())


def _run_independent_mla_oracle(target_path, draft_path, result_path):
    """Compare real latent KV with authoritative replay, including poisoned tails."""
    import itertools
    import json

    import torch

    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.attention.mla_attention import (
        MLAAttention,
        MLACommonMetadata,
    )
    from vllm.v1.attention.backends.mla.triton_mla import TritonMLAMetadataBuilder
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.spec_decode.remote.protocol import SequenceKey, SpeculatorStatusCode
    from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
    from vllm.v1.spec_decode.remote.runtime.token_gpu import (
        TokenGPUStepExecutor,
        make_token_gpu_adapter,
    )
    from vllm.v1.spec_decode.remote.runtime.token_state import (
        TokenAdvanceRow,
        TokenForwardRow,
        TokenStateMachine,
    )
    from vllm.v1.spec_decode.remote.server import AdapterError

    torch.set_num_threads(1)
    config = EngineArgs(
        model=target_path,
        tokenizer_mode="hf",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=6,
        max_num_batched_tokens=128,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=True,
        block_size=16,
        attention_config={
            "backend": "TRITON_ATTN",
            "backend_per_kind": {
                "full_attention": "TRITON_ATTN",
                "mla_attention": "TRITON_MLA",
            },
            "mla_prefill_backend": "FLASH_ATTN",
        },
        speculative_config={
            "method": "draft_model",
            "model": draft_path,
            "num_speculative_tokens": 3,
            "draft_tensor_parallel_size": 1,
            "kv_cache_dtype": "auto",
            "attention_backend": "TRITON_MLA",
        },
    ).create_engine_config()
    assert config.speculative_config is not None
    assert config.speculative_config.method == "draft_model"
    adapter = make_token_gpu_adapter(
        config, device=torch.device("cuda:0"), kv_cache_bytes=32 << 20
    )
    state, executor = adapter.state, adapter.state.executor
    assert isinstance(executor, TokenGPUStepExecutor)
    pool = state.block_pool
    model = executor.speculator.model
    assert type(model).__name__ == "DeepseekV2ForCausalLM" and not model.use_mha
    assert type(executor.model_state).__name__ == "DefaultModelState"
    thresholds = []
    for groups in executor.attn_groups:
        for group in groups:
            builder = group.get_metadata_builder()
            assert isinstance(builder, TritonMLAMetadataBuilder)
            thresholds.append(builder.reorder_batch_threshold)
    assert thresholds and len(set(thresholds)) == 1
    decode_threshold = thresholds[0]
    checked_weights = _check_independent_mla_weights(model, draft_path)
    layers = [layer for layer in model.modules() if isinstance(layer, MLAAttention)]
    assert len(layers) == 2
    layout, ranges = [], []
    for layer in layers:
        spec = layer.get_kv_cache_spec(executor.vllm_config)
        assert type(spec) is MLAAttentionSpec
        assert (spec.num_kv_heads, spec.head_size, spec.head_size_v) == (1, 576, 0)
        assert spec.block_size == 16 and spec.dtype == torch.bfloat16
        assert layer.get_attn_backend().get_name() == "TRITON_MLA"
        assert layer.prefill_backend is not None
        assert layer.prefill_backend.get_name() == "FLASH_ATTN"
        assert not layer.impl.is_sparse
        cache = layer.kv_cache
        assert tuple(cache.shape) == (executor.num_blocks, 16, 576)
        size = (
            1 + sum((n - 1) * s for n, s in zip(cache.shape, cache.stride()))
        ) * cache.element_size()
        ranges.append((cache.data_ptr(), cache.data_ptr() + size))
        layout.append(
            dict(
                layer=layer.layer_name,
                shape=list(cache.shape),
                strides=list(cache.stride()),
            )
        )
    assert all(
        a1 <= b0 or b1 <= a0 for (a0, a1), (b0, b1) in itertools.combinations(ranges, 2)
    )
    metadata = []

    def observe_attention(layer, inputs):
        value = get_forward_context().attn_metadata[layer.layer_name]
        assert isinstance(value, MLACommonMetadata)
        metadata.append(
            dict(
                query_lens=torch.diff(value.query_start_loc).cpu().tolist(),
                num_decodes=value.num_decodes,
                num_prefills=value.num_prefills,
            )
        )

    hook = layers[0].register_forward_pre_hook(observe_attention)
    keys = [SequenceKey("mla-live", row, 0) for row in range(3)]
    refs = [SequenceKey("mla-reference", row, 0) for row in range(3)]
    transcripts = [
        [3 + (i * 7 + row) % 125 for i in range(n)]
        for row, n in enumerate((15, 16, 17))
    ]
    proposals: list[list[int]] = [[], [], []]
    previous_ids: list[int | None] = [None, None, None]
    checks, outcomes, orders, partitions, proposal_references = [], [], [], [], []

    def blocks(key):
        return pool.blocks(state.sequences[key].req_index)

    def prefix(key, length):
        indices = torch.tensor(blocks(key), dtype=torch.int64, device=executor.device)
        return [
            layer.kv_cache.index_select(0, indices).flatten(0, 1)[:length].cpu().clone()
            for layer in layers
        ]

    def owns_distinct_blocks():
        owned = [block for key in state.sequences for block in blocks(key)]
        assert 0 not in owned and len(owned) == len(set(owned))
        assert len(owned) + pool.num_free_blocks == pool.num_blocks - 1

    def forward_reference(row, start, tokens, sample):
        key = refs[row]
        req_index = state.sequences[key].req_index
        pool.ensure(req_index, start + len(tokens))
        return TokenForwardRow(
            key, req_index, start, tuple(tokens), blocks(key), sample
        )

    try:
        for row, tokens in enumerate(transcripts):
            for key in (keys[row], refs[row]):
                state.open(key)
                offset = 0
                chunk_size = (1, 15, 16)[row]
                while offset < len(tokens):
                    end = min(offset + chunk_size, len(tokens))
                    state.prefill(
                        key,
                        offset,
                        tuple(tokens[offset:end]),
                        is_final=end == len(tokens),
                    )
                    offset = end
        # The reference calls the real executor directly. Its transcript and
        # write positions do not come from TokenStateMachine's updated fields.
        for step, k in enumerate((0, 3, 3, 0, 3, 3, 1, 3)):
            order = (
                (2, 0, 1),
                (1, 2, 0),
                (0, 1, 2),
                (2, 0, 1),
                (1, 0, 2),
                (0, 2, 1),
                (2, 1, 0),
                (0, 1, 2),
            )[step]
            updates, corrections, expected_lengths = {}, {}, {}
            old_lengths = list(map(len, transcripts))
            for row in range(3):
                old = len(transcripts[row])
                previous = proposals[row]
                offered = min(len(previous), 1) if step == 6 else len(previous)
                accepted = (
                    min(offered, (0, 3, 1)[row])
                    if step == 2
                    else min(offered, (2, 0, 3)[row])
                )
                delta = previous[:accepted] + [3 + (step * 11 + row * 13) % 125]
                if not previous and step == 1:
                    delta += [17 + row] * row
                keep = old + min(accepted, max(len(previous) - 1, 0))
                before = {
                    other: prefix(keys[other], old_lengths[other]) for other in range(3)
                }
                # Poison only rejected/unwritten tail. Accepted latent rows
                # and all sibling allocations must remain untouched.
                with torch.inference_mode():
                    for position in range(keep, len(blocks(keys[row])) * 16):
                        for layer in layers:
                            layer.kv_cache[
                                blocks(keys[row])[position // 16], position % 16
                            ].fill_(torch.nan)
                for other in range(3):
                    for a, b in zip(
                        before[other], prefix(keys[other], old_lengths[other])
                    ):
                        assert torch.equal(a, b)
                updates[row] = (
                    step,
                    TokenAdvanceRow(
                        keys[row],
                        old,
                        previous_ids[row],
                        offered,
                        accepted,
                        tuple(delta),
                    ),
                    k,
                )
                corrections[row] = forward_reference(
                    row, keep, delta[keep - old :], k > 0
                )
                expected_lengths[row] = old + len(delta)
                transcripts[row].extend(delta)
                outcomes.append(
                    dict(
                        step=step,
                        row=row,
                        offered=offered,
                        accepted=accepted,
                        k=k,
                        correction=len(corrections[row].tokens),
                    )
                )
            orders.append(
                dict(
                    step=step,
                    rows=list(order),
                    query_lens=[len(corrections[row].tokens) for row in order],
                )
            )
            start = len(metadata)
            actual = state.advance_many(tuple(updates[row] for row in order))
            actual_partitions = metadata[start:]
            assert all(not isinstance(result, Exception) for result in actual), actual
            # Canonical decode/short-extend/prefill order is the native contract.
            # A noncanonical caller order must preserve row identity and state.
            reference_order = sorted(
                order, key=lambda row: len(corrections[row].tokens)
            )
            start = len(metadata)
            first = executor.forward_batch(
                tuple(corrections[row] for row in reference_order)
            )
            expected = {
                row: ([] if k == 0 else [token])
                for row, token in zip(reference_order, first)
            }
            for depth in range(1, k):
                rows = [
                    forward_reference(
                        row,
                        expected_lengths[row] + depth - 1,
                        [expected[row][-1]],
                        True,
                    )
                    for row in order
                ]
                for row, token in zip(order, executor.forward_batch(tuple(rows))):
                    expected[row].append(token)
            reference_partitions = metadata[start:]
            # Check actual execution paths before comparing floating-point KV.
            # The native MLA builder expects a uniform decode prefix; different
            # prefill/absorbed-decode paths are not a bitwise state oracle.
            assert len(actual_partitions) == max(k, 1)
            assert actual_partitions == reference_partitions
            for partition in actual_partitions:
                query_lens = partition["query_lens"]
                assert query_lens == sorted(query_lens)
                decodes = (
                    query_lens.count(query_lens[0])
                    if query_lens[0] <= decode_threshold
                    else 0
                )
                assert partition["num_decodes"] == decodes
                assert partition["num_prefills"] == len(query_lens) - decodes
            partitions.append(
                dict(
                    step=step, actual=actual_partitions, reference=reference_partitions
                )
            )
            for row, result in zip(order, actual):
                assert not isinstance(result, Exception)
                assert list(result.token_ids) == expected[row], (
                    step,
                    row,
                    result,
                    expected[row],
                )
                progress = state.sequences[keys[row]]
                committed = len(transcripts[row])
                assert (
                    result.committed
                    == progress.committed
                    == progress.materialized
                    == committed
                )
                assert progress.speculative_written_end == committed + max(k - 1, 0)
                assert progress.pending_last_token == (
                    (committed + k - 1, expected[row][-1]) if k else None
                )
                assert progress.reserved_end >= progress.speculative_written_end
                for a, b in zip(
                    prefix(keys[row], committed), prefix(refs[row], committed)
                ):
                    assert torch.isfinite(a).all() and torch.equal(a, b), (
                        step,
                        row,
                        committed,
                    )
                proposals[row] = list(result.token_ids)
                previous_ids[row] = result.proposal_id
                if result.token_ids:
                    proposal_references.append(
                        dict(
                            step=step,
                            row=row,
                            committed_token_ids=list(transcripts[row]),
                            token_ids=list(result.token_ids),
                        )
                    )
                checks.append(
                    dict(
                        step=step, row=row, committed=committed, k=k, poisoned_tail=True
                    )
                )
            owns_distinct_blocks()
        state.close(tuple(keys + refs))
        assert not state.sequences and pool.num_free_blocks == executor.num_blocks - 1
        truncations = []
        for row, length in enumerate((125, 126, 127)):
            key = SequenceKey("mla-live", row, 1)
            state.open(key)
            state.prefill(key, 0, tuple([3] * length), is_final=True)
            result = state.advance(
                100 + row, TokenAdvanceRow(key, length, None, 0, 0, (4,)), 3
            )
            assert len(result.token_ids) == 127 - length
            assert state.sequences[key].speculative_written_end <= 128
            truncations.append(
                dict(committed=result.committed, returned_k=len(result.token_ids))
            )
            state.close((key,))
        assert pool.num_free_blocks == executor.num_blocks - 1
        # Reuse the real allocation with a smaller logical budget only after
        # every preceding owner has closed; there is never overlapping ownership.
        small = TokenStateMachine(
            executor,
            DraftBlockPool(3, 16),
            max_num_reqs=2,
            max_model_len=128,
            vocab_size=128,
            max_k=3,
        )
        a, b = SequenceKey("capacity", 0, 0), SequenceKey("capacity", 1, 0)
        for key in (a, b):
            small.open(key)
            small.prefill(key, 0, tuple([3] * 16), is_final=True)
        previous_blocks = small.block_pool.blocks(small.sequences[a].req_index)
        with pytest.raises(AdapterError) as raised:
            small.advance(0, TokenAdvanceRow(a, 16, None, 0, 0, (4,)), 1)
        assert raised.value.status == SpeculatorStatusCode.OUT_OF_MEMORY
        assert small.block_pool.blocks(small.sequences[a].req_index) == previous_blocks
        assert small.sequences[a].committed == 16 and small.sequences[a].ready
        small.close((b,))
        assert (
            len(small.advance(0, TokenAdvanceRow(a, 16, None, 0, 0, (4,)), 1).token_ids)
            == 1
        )
        small.close((a,))
        assert not small.sequences and small.block_pool.num_free_blocks == 2
        Path(result_path).write_text(
            json.dumps(
                dict(
                    model=type(model).__name__,
                    state=type(executor.model_state).__name__,
                    layout=layout,
                    checkpoint_weights=checked_weights,
                    disjoint_layer_storage=True,
                    checks=checks,
                    outcomes=outcomes,
                    orders=orders,
                    metadata=metadata,
                    same_path_partitions=partitions,
                    proposal_references=proposal_references,
                    decode_threshold=decode_threshold,
                    truncations=truncations,
                    generation_reuse=True,
                    capacity_recovered=True,
                    all_blocks_recovered=True,
                )
            )
            + "\n"
        )
    finally:
        hook.remove()
        state.close(tuple(state.sequences))


def test_remote_independent_mla_state_matches_authoritative_replay(
    vllm_runner, monkeypatch, tmp_path, tiny_independent_mla_models, record_property
):
    """Check exact MLA state and standalone native proposals after corrections."""
    import json
    import math

    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    target, draft = tiny_independent_mla_models
    result, log_path = tmp_path / "mla-oracle.json", tmp_path / "mla-oracle.log"
    command = [
        sys.executable,
        "-c",
        "import sys; from tests.v1.e2e.spec_decode.remote.test_remote_mtp "
        "import _run_independent_mla_oracle; "
        "_run_independent_mla_oracle(*sys.argv[1:])",
        target,
        draft,
        str(result),
    ]
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            assert process.wait(timeout=300) == 0, (
                f"independent MLA oracle failed: {log_path}"
            )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    receipt = json.loads(result.read_text())
    assert receipt["model"] == "DeepseekV2ForCausalLM"
    assert len(receipt["checks"]) == 24
    assert {entry["accepted"] for entry in receipt["outcomes"]} == {0, 1, 2, 3}
    assert {entry["k"] for entry in receipt["checks"]} == {0, 1, 3}
    assert {(1, 2, 1), (2, 1, 2)} <= {
        tuple(entry["query_lens"]) for entry in receipt["orders"]
    }
    assert all(
        receipt[field]
        for field in (
            "capacity_recovered",
            "all_blocks_recovered",
            "generation_reuse",
            "disjoint_layer_storage",
        )
    )
    record_property("native.mla_state_oracle", receipt)
    cases = receipt["proposal_references"]
    assert len(cases) == 18 and sum(len(case["token_ids"]) for case in cases) == 48

    def native_identity(worker):
        import torch

        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
        from vllm.v1.attention.backends.mla.triton_mla import (
            TritonMLAMetadataBuilder,
        )
        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        config, runner = worker.vllm_config, worker.model_runner
        model = runner.model
        assert config.speculative_config is None
        assert config.model_config.model == draft
        assert config.model_config.use_mla and not model.use_mha
        assert config.model_config.get_hidden_size() == 128
        assert config.model_config.get_head_size() == 576
        assert config.cache_config.cache_dtype == "auto"
        assert type(model).__name__ == "DeepseekV2ForCausalLM"
        assert type(runner.model_state).__name__ == "DefaultModelState"
        layers = [m for m in model.modules() if isinstance(m, MLAAttention)]
        assert len(layers) == 2
        for layer in layers:
            spec = layer.get_kv_cache_spec(config)
            assert isinstance(spec, MLAAttentionSpec)
            assert (spec.num_kv_heads, spec.head_size, spec.head_size_v) == (1, 576, 0)
            assert spec.dtype == torch.bfloat16 and spec.block_size == 16
            assert layer.get_attn_backend().get_name() == "TRITON_MLA"
            assert layer.prefill_backend is not None
            assert layer.prefill_backend.get_name() == "FLASH_ATTN"
        builders = [
            g.get_metadata_builder() for groups in runner.attn_groups for g in groups
        ]
        assert builders and all(
            isinstance(b, TritonMLAMetadataBuilder) for b in builders
        )
        assert all(b.model_config is config.model_config for b in builders)
        return dict(
            model=type(model).__name__,
            model_path=config.model_config.model,
            head_size=config.model_config.get_head_size(),
            backend="TRITON_MLA",
            prefill_backend="FLASH_ATTN",
            checkpoint_weights=_check_independent_mla_weights(model, draft),
        )

    # The executor child has exited. A real standalone LLM now owns the draft
    # checkpoint and distributed state, without the mixed native runner's target
    # metadata ownership. Cross-path token equality supplements same-path KV.
    with vllm_runner(
        draft,
        tokenizer_mode="hf",
        skip_tokenizer_init=True,
        trust_remote_code=False,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=6,
        max_num_batched_tokens=128,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=True,
        block_size=16,
        kv_cache_memory_bytes=32 << 20,
        gpu_memory_utilization=0.2,
        kernel_config={"moe_backend": "triton", "enable_jit_warmup": False},
        attention_config={"backend": "TRITON_MLA", "mla_prefill_backend": "FLASH_ATTN"},
    ) as native:
        identity = native.llm.collective_rpc(native_identity)
        assert identity[0]["checkpoint_weights"] == receipt["checkpoint_weights"]
        outputs = native.llm.generate(
            [{"prompt_token_ids": case["committed_token_ids"]} for case in cases],
            [
                SamplingParams(
                    temperature=0,
                    max_tokens=len(case["token_ids"]),
                    ignore_eos=True,
                    detokenize=False,
                    logprobs=5,
                )
                for case in cases
            ],
            use_tqdm=False,
        )
    observations = [
        dict(
            **case,
            native_prompt_token_ids=output.prompt_token_ids,
            native_token_ids=list(output.outputs[0].token_ids),
            native_logprobs=[
                {
                    token: dict(logprob=value.logprob, rank=value.rank)
                    for token, value in values.items()
                }
                for values in output.outputs[0].logprobs or []
            ],
        )
        for case, output in zip(cases, outputs)
    ]
    record_property(
        "native.mla_proposal_replay",
        dict(
            identity=identity,
            comparison="exact_proposal_tokens_from_committed_prefix",
            requests=len(observations),
            tokens=sum(len(case["token_ids"]) for case in cases),
            observations=observations,
        ),
    )
    assert len(outputs) == len(cases)
    for case, output in zip(cases, outputs):
        assert output.prompt_token_ids == case["committed_token_ids"]
        generated = output.outputs[0]
        assert list(generated.token_ids) == case["token_ids"], case
        assert generated.finish_reason == "length"
        assert generated.logprobs is not None
        assert len(generated.logprobs) == len(generated.token_ids)
        for token, values in zip(generated.token_ids, generated.logprobs):
            assert token in values and len(values) >= 5
            assert all(math.isfinite(value.logprob) for value in values.values())


def _multi_module_ipc_client(pipe, endpoint, identity) -> None:
    """Own the verifier's CUDA exports; all feature consumption uses the socket."""
    import torch

    from vllm.v1.spec_decode.remote.protocol import SpeculatorStatusCode
    from vllm.v1.spec_decode.remote.session import RemoteDraftSession, frame_to_tensor
    from vllm.v1.spec_decode.remote.state import FeatureBatch, RemoteProposalBatch

    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    torch.accelerator.set_device_index(device)
    session = RemoteDraftSession(
        endpoint,
        identity,
        transport="cuda_ipc",
        failure_policy="error",
        request_timeout_ms=5000,
        startup_timeout_ms=60000,
    )

    def features(frames):
        slots = []
        for frame, spec in zip(frames, session.feature_schema.slots):
            tensor = None if frame is None else frame_to_tensor(frame)
            if (
                tensor is not None
                and tensor.is_floating_point()
                and not spec.per_sequence
            ):
                tensor = tensor.to(device)
            slots.append(tensor)
        return FeatureBatch(session.feature_schema.schema_id, tuple(slots))

    try:
        session.connect()
        assert session.selected_transport == "cuda_ipc"
        session.prepare_cuda_ipc(device)
        pipe.send(session.selected_transport)
        while True:
            operation, payload = pipe.recv()
            if operation == "stop":
                break
            if operation == "open":
                result = session.open_sequence(payload)
            elif operation == "prefill":
                key, chunk = payload
                session.prefill(key, features(chunk.features), is_final=chunk.is_final)
                result = None
            elif operation == "advance":
                batch = payload
                handle = session.dispatch(
                    RemoteProposalBatch(
                        batch_id=batch.batch_id,
                        keys=batch.keys,
                        accepted_counts=frame_to_tensor(batch.accepted_counts),
                        recovery_tokens=frame_to_tensor(batch.recovery_tokens),
                        features=features(batch.features),
                        query_lens=torch.tensor(batch.query_lens, dtype=torch.int32),
                        num_speculative_tokens=batch.effective_num_speculative_tokens,
                    )
                )
                proposal = session.collect(handle)
                assert proposal.row_statuses == (SpeculatorStatusCode.OK,) * len(
                    batch.keys
                )
                assert proposal.output.valid_lengths.tolist() == [
                    batch.effective_num_speculative_tokens
                ] * len(batch.keys)
                result = proposal.output.token_ids[
                    :, : batch.effective_num_speculative_tokens
                ].tolist()
            else:
                assert operation == "close"
                session.close_sequences(payload)
                result = None
            pipe.send(result)
    finally:
        try:
            session.close()
        finally:
            pipe.close()


@contextmanager
def _multi_module_ipc_oracle(adapter, config):
    """Serialize the real reference and IPC execution through one GPU owner."""
    import multiprocessing

    from vllm.v1.spec_decode.remote.conditioned_server import ConditionedDraftServer
    from vllm.v1.spec_decode.remote.protocol import RemoteServerLimits
    from vllm.v1.spec_decode.remote.session import VerifierIdentity

    contract = adapter.executor.contract
    limits = RemoteServerLimits(5, 256, 5, 128)
    server = ConditionedDraftServer(
        adapter,
        limits=limits,
        target_fingerprint=contract.target_fingerprint,
        tokenizer_fingerprint=contract.tokenizer_fingerprint,
    )
    identity = VerifierIdentity(
        verifier_instance_id="ipc-oracle",
        target_fingerprint=contract.target_fingerprint,
        tokenizer_fingerprint=contract.tokenizer_fingerprint,
        method="mtp",
        num_speculative_tokens=config.speculative_config.num_speculative_tokens,
        parallel_drafting=False,
        provided_features=tuple(slot.kind for slot in contract.schema.slots),
        max_num_seqs=4,
        max_num_batched_tokens=256,
        max_model_len=128,
        vocab_size=config.model_config.get_vocab_size(),
        draft_checkpoint_fingerprint=contract.draft_fingerprint,
        uses_multi_module=True,
    )
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = None
    with tempfile.TemporaryDirectory(prefix="vllm-mtp-ipc-") as directory:
        try:
            endpoint = server.start(f"unix://{directory}/draft.sock")
            process = context.Process(
                target=_multi_module_ipc_client, args=(child, endpoint, identity)
            )
            process.start()
            child.close()

            def receive():
                assert parent.poll(60), (
                    "IPC oracle client did not complete its operation"
                )
                return parent.recv()

            assert receive() == "cuda_ipc"

            def call(operation, payload):
                assert process is not None and process.is_alive()
                parent.send((operation, payload))
                return receive()

            yield call, server
            parent.send(("stop", None))
            process.join(timeout=10)
            assert process.exitcode == 0
            assert server.healthy
        finally:
            try:
                if process is not None and process.pid is not None:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=10)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=10)
                    assert not process.is_alive()
                    process.close()
            finally:
                try:
                    parent.close()
                    child.close()
                finally:
                    server.stop()


def _run_multi_module_oracle(
    model_path: str, result_path: str, transport: str = "direct"
) -> None:
    with ExitStack() as resources:
        _multi_module_oracle(model_path, result_path, transport, resources)


def _multi_module_oracle(model_path, result_path, transport, resources) -> None:
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
    assert transport in ("direct", "cuda_ipc")
    client = server = None
    wire_keys: dict[SequenceKey, SequenceKey] = {}
    if transport == "cuda_ipc":
        client, server = resources.enter_context(
            _multi_module_ipc_oracle(adapter, config)
        )

    def on_owner(action):
        return action() if server is None else server.work_queue.call("oracle", action)

    def allocation():
        torch.accelerator.synchronize(executor.device)
        return {
            "allocated": torch.accelerator.memory_allocated(executor.device),
            "reserved": torch.accelerator.memory_reserved(executor.device),
        }

    initial_allocation = on_owner(allocation)

    def close_sequences(keys):
        if client is not None:
            client("close", tuple(wire_keys[key] for key in keys))
        else:
            adapter.close_sequences(tuple(keys))

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
        remote = client is not None and key.verifier_instance_id != "oracle"
        if remote:
            assert client is not None
            actual = client("open", key.sequence_id)
            assert actual.sequence_id == key.sequence_id
            previous = [
                value.generation
                for old, value in wire_keys.items()
                if old.sequence_id == key.sequence_id
            ]
            assert not previous or actual.generation > max(previous)
            wire_keys[key] = actual
        else:
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
            chunk = PrefillInputs(
                offset,
                count,
                final,
                frames(
                    tokens[offset:end],
                    range(offset, end),
                    hidden[offset:end],
                    lookahead,
                ),
            )
            if remote:
                assert client is not None
                client("prefill", (wire_keys[key], chunk))
            else:
                adapter.prefill(key, chunk)
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
        remote = client is not None and keys[0].verifier_instance_id != "oracle"
        batch = StandaloneProposalBatch(
            batch_id=next(batch_ids),
            keys=tuple(wire_keys[key] for key in keys) if remote else tuple(keys),
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
        if remote:
            assert client is not None
            return client("advance", batch)
        result = adapter.advance_and_propose(batch)
        assert result.row_statuses == (SpeculatorStatusCode.OK,) * len(keys)
        assert frame_to_tensor(result.valid_lengths).tolist() == [k] * len(keys)
        return frame_to_tensor(result.token_ids)[:, :k].tolist()

    def kv_prefix(key, length):
        key = wire_keys.get(key, key)
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
        on_owner(
            lambda: compare_reference_on_owner(
                key, tokens, hidden, recovery, k, actual, label
            )
        )

    def compare_reference_on_owner(key, tokens, hidden, recovery, k, actual, label):
        assert adapter.planner.num_computed(wire_keys.get(key, key)) == len(tokens)
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
        close_sequences(tuple(keys[row] for row in finished))
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
    close_sequences(tuple(keys))
    assert on_owner(lambda: adapter.planner.num_open) == 0
    # Tiny nonfinal chunks and both block/window edges use real prompt lookahead.
    for length in (15, 16, 17, 31, 32, 33):
        key = SequenceKey("boundary", length, 0)
        ids = [3 + (i * 11) % 125 for i in range(length)]
        features = target_features(ids)
        prefill(key, ids, features, chunk_size=1 if length == 15 else 32)
        output = advance([key], [[]], [[]], [features[:0]], [0], [19], depth)[0]
        compare_reference(key, ids, features, 19, depth, output, f"prompt-{length}")
        close_sequences((key,))
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
    close_sequences((key,))
    assert on_owner(lambda: adapter.planner.num_open) == 0
    assert on_owner(lambda: adapter.planner.block_pool.num_free_blocks) == (
        executor.num_blocks - 1
    )
    final_allocation = on_owner(allocation)
    close_reopen_allocations = []
    if client is not None:
        previous_output = None
        # The first identical cycle settles persistent native buffers. The next
        # two must release history/received features without growing live storage.
        for generation in (2, 3, 4):
            key = SequenceKey("incremental", 0, generation)
            ids = [7]
            features = target_features(ids)
            prefill(key, ids, features)
            output = advance([key], [[]], [[]], [features[:0]], [0], [19], depth)[0]
            cycle_output = [output]
            for correction, recovery_token in ((19, 31), (31, 43)):
                start = len(ids)
                query = [correction, *output]
                features = target_features(ids + query)[start:].clone()
                features[1:] = torch.nan
                output = advance(
                    [key],
                    [query],
                    [list(range(start, start + len(query)))],
                    [features],
                    [0],
                    [recovery_token],
                    depth,
                )[0]
                ids.append(correction)
                cycle_output.append(output)
            if previous_output is not None:
                assert cycle_output == previous_output
            previous_output = cycle_output
            close_sequences((key,))
            assert on_owner(lambda: adapter.planner.num_open) == 0
            assert on_owner(lambda: adapter.planner.block_pool.num_free_blocks) == (
                executor.num_blocks - 1
            )
            close_reopen_allocations.append(on_owner(allocation))
        assert (
            close_reopen_allocations[-1]["allocated"]
            == (close_reopen_allocations[-2]["allocated"])
        )
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
                "transport": transport,
                "wire_generations": [
                    {"sequence_id": key.sequence_id, "generation": key.generation}
                    for key in wire_keys.values()
                ],
                "allocation": {
                    "scope": "draft process; client export arena is separate",
                    "before_scenarios": initial_allocation,
                    "after_close": final_allocation,
                    "close_reopen_cycles": close_reopen_allocations,
                },
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
@pytest.mark.parametrize("transport", ["direct", "cuda_ipc"])
def test_remote_multi_module_mtp_dynamic_k_matches_full_prefix(
    mtp_family, depth, transport, request, monkeypatch, tmp_path, record_property
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
        transport,
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
    assert receipt["transport"] == transport
    if transport == "cuda_ipc":
        generations = receipt["wire_generations"]
        assert len(generations) == 14
        reused = [key["generation"] for key in generations if key["sequence_id"] == 0]
        assert len(reused) == 5 and all(
            left < right for left, right in zip(reused, reused[1:])
        )
        cycles = receipt["allocation"]["close_reopen_cycles"]
        assert len(cycles) == 3
        assert cycles[-1]["allocated"] == cycles[-2]["allocated"]
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


def test_remote_mtp_rejects_short_draft_context_through_llm(
    vllm_runner, monkeypatch, tmp_path, tiny_ernie_mtp_model
):
    """Reject a short draft before opening its remote session or running RoPE."""
    import json

    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    draft = _separate_mtp_draft(tiny_ernie_mtp_model, tmp_path / "short-draft")
    config_path = Path(draft) / "config.json"
    draft_config = json.loads(config_path.read_text())
    draft_config["max_position_embeddings"] = 16
    config_path.write_text(json.dumps(draft_config))
    with (
        pytest.raises(ValueError, match="standalone MTP.*draft.*max_model_len"),
        vllm_runner(
            tiny_ernie_mtp_model,
            dtype="bfloat16",
            skip_tokenizer_init=True,
            max_model_len=64,
            max_num_seqs=2,
            max_num_batched_tokens=32,
            enable_chunked_prefill=True,
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
                "remote_draft": {
                    "endpoint": f"unix://{tmp_path / 'never-opened.sock'}",
                    "failure_policy": "error",
                },
            },
        ),
    ):
        pass


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
        enable_chunked_prefill=True,
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
