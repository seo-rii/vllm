# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real model coverage of process-separated MTP on one NVIDIA GPU."""

import asyncio
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
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


@contextmanager
def draft_server(
    model: str,
    log_path: Path,
    *,
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
    block_size: int = 16,
    trust_remote_code: bool = True,
    moe_backend: str | None = None,
    batch_invariant: bool | None = None,
    owned_processes: list[subprocess.Popen] | None = None,
    draft_tensor_parallel_size: int = 1,
    draft_devices: tuple[int, ...] | None = None,
    on_ready: Callable[[subprocess.Popen], None] | None = None,
):
    if token_only_draft is None and (
        draft_tensor_parallel_size != 1 or draft_devices is not None
    ):
        raise ValueError("draft TP devices are supported by the token-only server")
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
            str(block_size),
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
        if token_only_draft is not None:
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
            if draft_tensor_parallel_size != 1:
                command.extend(
                    ["--draft-tensor-parallel-size", str(draft_tensor_parallel_size)]
                )
            if draft_devices is not None:
                command.extend(
                    ["--devices", *(f"cuda:{device}" for device in draft_devices)]
                )
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
                if owned_processes is not None:
                    owned_processes.append(process)
                if on_ready is not None:
                    on_ready(process)
                yield f"unix://{path}"
            finally:
                if process.poll() is None:
                    # A fault-injection test may have stopped this owned child.
                    os.kill(process.pid, signal.SIGCONT)
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                        pytest.fail(f"draft server did not shut down: {log_path}")
                assert process.returncode == 0, f"draft server failed: {log_path}"


@pytest.mark.parametrize("transport", ["inline", "cuda_ipc"])
def test_remote_mtp_matches_target_and_uses_drafts(
    vllm_runner, monkeypatch, tmp_path, transport
):
    """Exercise real chunked prefill, verification and completed-request reuse."""
    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    model = maybe_model_redirect(MODEL)
    kwargs = dict(
        revision=REVISION,
        tokenizer_revision=REVISION,
        trust_remote_code=True,
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
        kv_cache_memory_bytes=1 << 30,
        gpu_memory_utilization=0.6,
        attention_config={"backend": "FLASH_ATTN"},
        disable_log_stats=False,
    )
    prompts = [
        {"prompt_token_ids": [11 + i % 4 for i in range(length)]}
        for length in (15, 16, 17, 65)
    ]
    sampling = SamplingParams(
        temperature=0, max_tokens=16, ignore_eos=True, detokenize=False
    )
    with vllm_runner(model, **kwargs) as baseline:
        expected = baseline.llm.generate(prompts, sampling, use_tqdm=False)
    with draft_server(model, tmp_path / "draft.log") as endpoint:
        config = dict(
            method="mtp",
            model=model,
            revision=REVISION,
            num_speculative_tokens=1,
            draft_tensor_parallel_size=1,
            draft_sample_method="greedy",
            attention_backend="FLASH_ATTN",
            remote_draft=dict(
                endpoint=endpoint,
                transport=transport,
                failure_policy="error",
                request_timeout_ms=5000,
                startup_timeout_ms=600000,
            ),
        )
        with vllm_runner(model, speculative_config=config, **kwargs) as remote:
            actual_config = remote.llm.llm_engine.vllm_config.speculative_config
            assert actual_config.use_remote_draft()
            assert actual_config.remote_draft.transport == transport
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
            remote.llm.llm_engine.do_log_stats()
            metrics = remote.llm.get_metrics()
            for name in (
                "vllm:spec_decode_num_draft_tokens",
                "vllm:spec_decode_num_accepted_tokens",
            ):
                assert get_spec_decode_metric_value(metrics, name) > 0


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


@pytest.fixture
def tiny_exaone_mtp_model(tmp_path, mtp_window):
    """Real seeded Exaone weights including the separately stored native MTP head."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        ExaoneMoeConfig,
        ExaoneMoeForCausalLM,
        TokenizersBackend,
    )
    from transformers.models.exaone_moe.modeling_exaone_moe import ExaoneMoeRMSNorm

    config = ExaoneMoeConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        layer_types=["full_attention", "full_attention"],
        mlp_layer_types=["dense", "dense"],
        sliding_window=32,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        num_nextn_predict_layers=1,
        mtp_layer_types=[
            "full_attention" if mtp_window is None else "sliding_attention"
        ],
        mtp_sliding_windows=[mtp_window or 32],
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
    weights.update(
        {
            f"mtp.layers.0.{name}": tensor.clone()
            for name, tensor in model.model.layers[0].state_dict().items()
        }
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
    + [pytest.param("ernie", None, 1, id="k1-ernie")],
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
    moe_backend = None if mtp_family == "exaone" else "triton"
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
        assert windows == [[None] if hybrid_target else [None, None]]
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
        spec_type = "FullAttentionSpec" if mtp_window is None else "SlidingWindowSpec"
        if dense_mla:
            spec_type = "MLAAttentionSpec"
        assert policy == [[(mtp_window, spec_type, mtp_window)]]
        record_property("native.kv_policy", policy)
        if dense_mla:
            layout = native.llm.collective_rpc(draft_mla_layout)
            assert layout == [[expected_mla_layout]]
            record_property("native.mla_layout", layout)
        backends = native.llm.collective_rpc(draft_moe_backends)
        draft_has_moe = mtp_family not in ("exaone", "ernie")
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
            assert actual_config.remote_draft.endpoint == endpoint
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


@contextmanager
def _paused_draft_server(process: subprocess.Popen):
    """Stop only this helper's child, and resume it even when validation fails."""
    assert process.poll() is None
    status = Path(f"/proc/{process.pid}/status")
    try:
        os.kill(process.pid, signal.SIGSTOP)
        deadline = time.monotonic() + 5
        while True:
            state = next(
                line.split()[1]
                for line in status.read_text().splitlines()
                if line.startswith("State:")
            )
            if state in ("T", "t"):
                break
            assert process.poll() is None, "owned draft server exited before stopping"
            assert time.monotonic() < deadline, "owned draft server did not stop"
            time.sleep(0.01)
        yield
    finally:
        if process.poll() is None:
            os.kill(process.pid, signal.SIGCONT)


def _save_token_prefix_models(directory):
    """Create the exact prefix-control weights in a fresh CPU process."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    directory = Path(directory)
    target_config = Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=128,
        max_position_embeddings=128,
        layer_types=["full_attention"] * 2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        dtype="bfloat16",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261024)
        target_model = Qwen3ForCausalLM(target_config)
    _two_candidate_head(target_model.lm_head.weight)
    target = _save_mtp_checkpoint(
        directory / "prefix-target", target_config, target_model.state_dict()
    )
    del target_model
    draft = _tiny_independent_qwen3_draft(directory, target)
    assert not torch.cuda.is_initialized()
    return target, draft


@pytest.mark.parametrize(
    "async_scheduling,prefix_caching",
    [(False, True), (True, False), (True, True)],
    ids=["sync-prefix-on", "async-prefix-off", "async-prefix-on"],
)
def test_remote_token_cached_prefix_matches_native_and_target(
    vllm_runner,
    monkeypatch,
    tmp_path,
    record_property,
    async_scheduling,
    prefix_caching,
):
    """Keep remote tokens exact with async scheduling and cached-prefix replay."""
    import hashlib

    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    command = [
        sys.executable,
        "-c",
        "import sys; from tests.v1.e2e.spec_decode.remote.test_remote_mtp "
        "import _save_token_prefix_models; _save_token_prefix_models(sys.argv[1])",
        str(tmp_path),
    ]
    with (tmp_path / "prefix-fixture.log").open("w") as log:
        subprocess.run(
            command,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", VLLM_TARGET_DEVICE="cpu"),
            check=True,
            timeout=180,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    target = str(tmp_path / "prefix-target")
    draft = str(tmp_path / "independent-qwen3")
    record_property(
        "prefix_replay.fixture",
        {
            role: {
                str(path.relative_to(directory)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(directory.rglob("*"))
                if path.is_file()
            }
            for role, directory in (("target", Path(target)), ("draft", Path(draft)))
        },
    )
    kwargs = dict(
        tokenizer_mode="hf",
        trust_remote_code=False,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=32,
        enable_chunked_prefill=True,
        enable_prefix_caching=prefix_caching,
        async_scheduling=async_scheduling,
        enforce_eager=True,
        block_size=16,
        kv_cache_memory_bytes=32 << 20,
        gpu_memory_utilization=0.2,
        attention_config={"backend": "TRITON_ATTN"},
        disable_log_stats=False,
    )
    spec = dict(
        method="draft_model",
        model=draft,
        num_speculative_tokens=3,
        draft_tensor_parallel_size=1,
        draft_sample_method="greedy",
        attention_backend="TRITON_ATTN",
    )
    prompts = [{"prompt_token_ids": [3 + (i * 7) % 125 for i in range(65)]}]
    sampling = SamplingParams(
        temperature=0, max_tokens=8, ignore_eos=True, detokenize=False
    )
    cached, tokens, warm_counts, warm_requests = {}, {}, {}, {}
    resolved = {}
    metric_names = (
        "vllm:spec_decode_num_drafts",
        "vllm:spec_decode_num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens",
    )

    def counts(runner):
        runner.llm.llm_engine.do_log_stats()
        metrics = runner.llm.get_metrics()
        return tuple(
            get_spec_decode_metric_value(metrics, name) for name in metric_names
        )

    def run_mode(mode, speculative_config=None):
        with vllm_runner(
            target,
            speculative_config=speculative_config,
            per_request_spec_decode_metrics=(
                "detailed" if speculative_config is not None else "none"
            ),
            **kwargs,
        ) as runner:
            actual = runner.llm.llm_engine.vllm_config
            assert actual.scheduler_config.async_scheduling is async_scheduling
            assert actual.cache_config.enable_prefix_caching is prefix_caching
            assert actual.model_config.enforce_eager and actual.use_v2_model_runner
            assert actual.parallel_config.tensor_parallel_size == 1
            draft_tp = None
            if actual.speculative_config is not None:
                draft_parallel = actual.speculative_config.draft_parallel_config
                assert draft_parallel is not None
                draft_tp = draft_parallel.tensor_parallel_size
                assert draft_tp == 1
            resolved[mode] = dict(
                async_scheduling=actual.scheduler_config.async_scheduling,
                prefix_caching=actual.cache_config.enable_prefix_caching,
                enforce_eager=actual.model_config.enforce_eager,
                use_v2_model_runner=actual.use_v2_model_runner,
                target_tp=actual.parallel_config.tensor_parallel_size,
                draft_tp=draft_tp,
            )
            cold = runner.llm.generate(prompts, sampling, use_tqdm=False)[0]
            before = counts(runner) if speculative_config is not None else None
            warm = runner.llm.generate(prompts, sampling, use_tqdm=False)[0]
            cached[mode] = warm.num_cached_tokens
            if prefix_caching:
                assert warm.num_cached_tokens > 32
            else:
                assert warm.num_cached_tokens == 0
            tokens[mode] = {
                label: list(output.outputs[0].token_ids)
                for label, output in (("cold", cold), ("warm", warm))
            }
            assert all(len(ids) == 8 for ids in tokens[mode].values())
            assert tokens[mode]["cold"] == tokens[mode]["warm"]
            if before is not None:
                warm_counts[mode] = tuple(a - b for a, b in zip(counts(runner), before))
                proposals, drafted, accepted = warm_counts[mode]
                assert proposals > 0 and drafted > 0 and 0 < accepted <= drafted
                detail = warm.outputs[0].spec_decode_metrics
                assert detail is not None and detail.num_spec_tokens == 3
                assert sum(detail.per_step_accepted) > 0
                warm_requests[mode] = detail.to_dict()

    run_mode("target")
    run_mode("native", spec)
    with draft_server(
        target,
        tmp_path / "prefix-draft.log",
        token_only_draft=draft,
        revision=None,
        num_speculative_tokens=3,
        attention_backend="TRITON_ATTN",
        draft_kv_cache_gib=0.03125,
        max_feature_tokens=32,
    ) as endpoint:
        run_mode(
            "remote",
            dict(
                spec,
                execution=dict(
                    placement="remote",
                    policy="on_demand",
                    endpoints=[endpoint],
                    transport="inline",
                    failure_policy="error",
                    request_timeout_ms=5000,
                    startup_timeout_ms=600000,
                ),
            ),
        )
    assert tokens["remote"] == tokens["native"] == tokens["target"]
    record_property(
        "prefix_replay.config",
        dict(
            k=3,
            prompt_tokens=65,
            max_num_batched_tokens=32,
            prefix_caching=prefix_caching,
            transport="inline",
            policy="error",
            async_scheduling=async_scheduling,
        ),
    )
    record_property("prefix_replay.resolved_config", resolved)
    record_property("prefix_replay.cached_tokens", cached)
    record_property("prefix_replay.output_token_ids", tokens)
    record_property("prefix_replay.warm_request_speculation", warm_requests)
    for index, label in enumerate(("proposals", "drafted", "accepted")):
        record_property(
            f"prefix_replay.{label}",
            {mode: values[index] for mode, values in warm_counts.items()},
        )


def test_remote_token_async_dynamic_k_preserves_concurrent_requests(
    vllm_runner, monkeypatch, tmp_path, record_property
):
    """Resume drafting after a concurrent K0 phase without losing own-KV state."""
    import hashlib

    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    command = [
        sys.executable,
        "-c",
        "import sys; from tests.v1.e2e.spec_decode.remote.test_remote_mtp "
        "import _save_token_prefix_models; _save_token_prefix_models(sys.argv[1])",
        str(tmp_path),
    ]
    with (tmp_path / "dynamic-k-fixture.log").open("w") as log:
        subprocess.run(
            command,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", VLLM_TARGET_DEVICE="cpu"),
            check=True,
            timeout=180,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    target = str(tmp_path / "prefix-target")
    draft = str(tmp_path / "independent-qwen3")
    record_property(
        "dynamic_k.fixture",
        {
            role: {
                str(path.relative_to(directory)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(directory.rglob("*"))
                if path.is_file()
            }
            for role, directory in (("target", Path(target)), ("draft", Path(draft)))
        },
    )
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
        async_scheduling=True,
        enforce_eager=True,
        block_size=16,
        kv_cache_memory_bytes=32 << 20,
        gpu_memory_utilization=0.2,
        attention_config={"backend": "TRITON_ATTN"},
        disable_log_stats=False,
    )
    schedule = [(1, 1, 3), (2, 4, 0)]
    spec = dict(
        method="draft_model",
        model=draft,
        num_speculative_tokens=3,
        num_speculative_tokens_per_batch_size=schedule,
        draft_tensor_parallel_size=1,
        draft_sample_method="greedy",
        attention_backend="TRITON_ATTN",
    )
    prompt = {"prompt_token_ids": [3] * 17}
    lengths = (8, 32)
    sampling = [
        SamplingParams(
            temperature=0, max_tokens=length, ignore_eos=True, detokenize=False
        )
        for length in lengths
    ]
    tokens, resolved, aggregate = {}, {}, {}
    details: dict[str, list[dict[str, Any]]] = {}
    timing: dict[str, list[dict[str, float]]] = {}

    def counts(runner):
        runner.llm.llm_engine.do_log_stats()
        metrics = runner.llm.get_metrics()
        return {
            key: get_spec_decode_metric_value(metrics, name)
            for key, name in (
                ("proposals", "vllm:spec_decode_num_drafts"),
                ("drafted", "vllm:spec_decode_num_draft_tokens"),
                ("accepted", "vllm:spec_decode_num_accepted_tokens"),
            )
        }

    def run_mode(mode, speculative_config=None):
        with vllm_runner(
            target,
            speculative_config=speculative_config,
            per_request_spec_decode_metrics=(
                "detailed" if speculative_config is not None else "none"
            ),
            **kwargs,
        ) as runner:
            actual = runner.llm.llm_engine.vllm_config
            assert actual.scheduler_config.async_scheduling
            assert actual.model_config.enforce_eager and actual.use_v2_model_runner
            assert not actual.cache_config.enable_prefix_caching
            assert actual.parallel_config.tensor_parallel_size == 1
            draft_tp = None
            if actual.speculative_config is not None:
                assert (
                    actual.speculative_config.num_speculative_tokens_per_batch_size
                    == schedule
                )
                draft_parallel = actual.speculative_config.draft_parallel_config
                assert draft_parallel is not None
                draft_tp = draft_parallel.tensor_parallel_size
                assert draft_tp == 1
            resolved[mode] = dict(
                async_scheduling=actual.scheduler_config.async_scheduling,
                prefix_caching=actual.cache_config.enable_prefix_caching,
                enforce_eager=actual.model_config.enforce_eager,
                use_v2_model_runner=actual.use_v2_model_runner,
                target_tp=actual.parallel_config.tensor_parallel_size,
                draft_tp=draft_tp,
                k_schedule=(
                    schedule if actual.speculative_config is not None else None
                ),
            )
            before = counts(runner) if speculative_config is not None else None
            outputs = runner.llm.generate([prompt, prompt], sampling, use_tqdm=False)
            tokens[mode] = [list(output.outputs[0].token_ids) for output in outputs]
            assert tuple(map(len, tokens[mode])) == lengths
            timing[mode] = []
            for output in outputs:
                metrics = output.metrics
                assert metrics is not None
                timing[mode].append(
                    dict(
                        scheduled=metrics.scheduled_ts,
                        first_token=metrics.first_token_ts,
                        last_token=metrics.last_token_ts,
                    )
                )
            short_timing, survivor_timing = timing[mode]
            assert 0 < short_timing["scheduled"] == survivor_timing["scheduled"]
            assert survivor_timing["first_token"] < short_timing["last_token"]
            if speculative_config is not None:
                details[mode] = []
                for output in outputs:
                    metrics = output.outputs[0].spec_decode_metrics
                    assert metrics is not None
                    details[mode].append(metrics.to_dict())
                assert before is not None
                aggregate[mode] = {
                    key: value - before[key] for key, value in counts(runner).items()
                }

    run_mode("target")
    run_mode("native", spec)
    with draft_server(
        target,
        tmp_path / "dynamic-k-draft.log",
        token_only_draft=draft,
        revision=None,
        num_speculative_tokens=3,
        attention_backend="TRITON_ATTN",
        draft_kv_cache_gib=0.03125,
        max_feature_tokens=64,
    ) as endpoint:
        run_mode(
            "remote",
            dict(
                spec,
                execution=dict(
                    placement="remote",
                    policy="on_demand",
                    endpoints=[endpoint],
                    transport="inline",
                    failure_policy="error",
                    request_timeout_ms=5000,
                    startup_timeout_ms=600000,
                ),
            ),
        )
    assert tokens["remote"] == tokens["native"] == tokens["target"]
    for mode in ("native", "remote"):
        counters = aggregate[mode]
        assert counters["proposals"] > 0
        assert 0 < counters["accepted"] <= counters["drafted"]
        assert counters["proposals"] == sum(
            row["num_spec_steps"] for row in details[mode]
        )
        assert counters["drafted"] == sum(
            row["num_draft_tokens"] for row in details[mode]
        )
        assert counters["accepted"] == sum(
            row["num_accepted_draft_tokens"] for row in details[mode]
        )
    short, survivor = details["remote"]
    assert short["num_draft_tokens"] == short["num_accepted_draft_tokens"] == 0
    assert 3 in survivor["per_step_drafted"]
    # K0 steps have no speculative metric row; the final prefill can supply
    # only one of the tokens omitted by those rows.
    unrecorded_tokens = (
        lengths[1] - survivor["num_spec_steps"] - survivor["num_accepted_draft_tokens"]
    )
    assert unrecorded_tokens > 1
    assert 0 < survivor["num_accepted_draft_tokens"] <= survivor["num_draft_tokens"]
    record_property("dynamic_k.schedule", schedule)
    record_property("dynamic_k.resolved_config", resolved)
    record_property("dynamic_k.output_token_ids", tokens)
    record_property("dynamic_k.request_speculation", details)
    record_property("dynamic_k.aggregate_speculation", aggregate)
    record_property("dynamic_k.unrecorded_output_tokens_lower_bound", unrecorded_tokens)
    record_property("dynamic_k.request_timing", timing)


def _fail_token_session_during_socket_cleanup(worker):
    import errno
    from unittest.mock import patch

    from vllm.v1.spec_decode.remote.token_session import TokenOnlyRemoteSession

    session = worker.model_runner.speculator._session
    assert isinstance(session, TokenOnlyRemoteSession)
    assert session.failure_policy == "target_only"
    keys = tuple(session.records)
    assert keys and any(session.active(key) for key in keys)
    connection = session._conn
    assert connection is not None
    close_socket = socket.socket.close
    injected = False

    def close_then_fail(sock):
        nonlocal injected
        close_socket(sock)
        if sock is connection._sock:
            injected = True
            raise OSError(errno.EIO, "injected socket cleanup failure")

    with patch.object(socket.socket, "close", close_then_fail):
        session.fail("injected primary token failure")
    return dict(
        cleanup_error_injected=injected,
        terminal_sequences=sum(not session.active(key) for key in keys),
        sequence_count=len(keys),
        connection_detached=session._conn is None and connection.closed,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lifecycle", ["abort-reuse", "timeout-recovery", "cleanup-error-fallback"]
)
async def test_remote_token_async_lifecycle_preserves_healthy_requests(
    vllm_runner, monkeypatch, tmp_path, record_property, lifecycle, enable_pickle
):
    """Keep healthy transcripts exact across cancellation and terminal fallback."""
    import hashlib

    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    command = [
        sys.executable,
        "-c",
        "import sys; from tests.v1.e2e.spec_decode.remote.test_remote_mtp "
        "import _save_token_prefix_models; _save_token_prefix_models(sys.argv[1])",
        str(tmp_path),
    ]
    with (tmp_path / "lifecycle-fixture.log").open("w") as log:
        subprocess.run(
            command,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", VLLM_TARGET_DEVICE="cpu"),
            check=True,
            timeout=180,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    target = str(tmp_path / "prefix-target")
    draft = str(tmp_path / "independent-qwen3")
    record_property(
        "async_lifecycle.fixture",
        {
            role: {
                str(path.relative_to(directory)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(directory.rglob("*"))
                if path.is_file()
            }
            for role, directory in (("target", Path(target)), ("draft", Path(draft)))
        },
    )
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
        async_scheduling=True,
        enforce_eager=True,
        block_size=16,
        kv_cache_memory_bytes=32 << 20,
        gpu_memory_utilization=0.2,
        attention_config={"backend": "TRITON_ATTN"},
        disable_log_stats=False,
    )
    prompts = [{"prompt_token_ids": [token] * 17} for token in (3, 7)]

    def sampling(length):
        return SamplingParams(
            temperature=0,
            max_tokens=length,
            ignore_eos=True,
            detokenize=False,
            output_kind=RequestOutputKind.CUMULATIVE,
        )

    with vllm_runner(target, **kwargs) as runner:
        expected = [
            list(output.outputs[0].token_ids)
            for output in runner.llm.generate(
                prompts, [sampling(96), sampling(32)], use_tqdm=False
            )
        ]
    assert tuple(map(len, expected)) == (96, 32)
    assert expected[0][:32] != expected[1], "request reuse must change the transcript"

    owned_processes: list[subprocess.Popen] = []
    with draft_server(
        target,
        tmp_path / "lifecycle-draft.log",
        token_only_draft=draft,
        revision=None,
        num_speculative_tokens=3,
        attention_backend="TRITON_ATTN",
        draft_kv_cache_gib=0.03125,
        max_feature_tokens=64,
        owned_processes=owned_processes,
    ) as endpoint:
        spec = dict(
            method="draft_model",
            model=draft,
            num_speculative_tokens=3,
            draft_tensor_parallel_size=1,
            draft_sample_method="greedy",
            attention_backend="TRITON_ATTN",
            execution=dict(
                placement="remote",
                policy="on_demand",
                endpoints=[endpoint],
                transport="inline",
                failure_policy="error" if lifecycle == "abort-reuse" else "target_only",
                request_timeout_ms=1000,
                startup_timeout_ms=600000,
            ),
        )
        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=target,
                speculative_config=spec,
                per_request_spec_decode_metrics="detailed",
                **kwargs,
            )
        )
        tasks: list[asyncio.Task] = []
        observed: dict[str, Any] = {}
        outputs: dict[str, Any] = {}
        events = {}
        first_output = asyncio.Event()

        async def collect(request_id, prompt, length):
            final = None
            async for output in engine.generate(prompt, sampling(length), request_id):
                observed[request_id] = output
                if (
                    request_id == "reused"
                    and output.outputs[0].token_ids
                    and not first_output.is_set()
                ):
                    await engine.pause_generation(mode="keep", clear_cache=False)
                    first_output.set()
                final = output
            assert final is not None and final.finished
            assert final.request_id == request_id
            return final

        def assert_active(output, reference):
            assert output.outputs[0].finish_reason == "length"
            assert list(output.outputs[0].token_ids) == reference
            metrics = output.outputs[0].spec_decode_metrics
            assert metrics is not None and metrics.num_spec_tokens == 3
            accepted = metrics.to_dict()["num_accepted_draft_tokens"]
            assert 0 < accepted <= metrics.num_draft_tokens
            assert metrics.num_draft_tokens == sum(metrics.per_step_drafted)
            assert accepted == sum(metrics.per_step_accepted)

        try:
            actual = engine.vllm_config
            assert actual.scheduler_config.async_scheduling
            assert actual.model_config.enforce_eager and actual.use_v2_model_runner
            assert not actual.cache_config.enable_prefix_caching
            assert actual.parallel_config.tensor_parallel_size == 1
            assert actual.speculative_config is not None
            assert actual.speculative_config.use_remote_draft()
            assert actual.speculative_config.draft_parallel_config is not None
            assert (
                actual.speculative_config.draft_parallel_config.tensor_parallel_size
                == 1
            )
            record_property(
                "async_lifecycle.resolved_config",
                dict(
                    async_scheduling=actual.scheduler_config.async_scheduling,
                    prefix_caching=actual.cache_config.enable_prefix_caching,
                    enforce_eager=actual.model_config.enforce_eager,
                    use_v2_model_runner=actual.use_v2_model_runner,
                    target_tp=actual.parallel_config.tensor_parallel_size,
                    draft_tp=actual.speculative_config.draft_parallel_config.tensor_parallel_size,
                    execution=spec["execution"],
                ),
            )
            async with asyncio.timeout(120):
                if lifecycle == "abort-reuse":
                    await engine.pause_generation(mode="keep", clear_cache=False)
                    assert await engine.is_paused()
                    tasks = [
                        asyncio.create_task(collect("reused", prompts[0], 96)),
                        asyncio.create_task(collect("healthy", prompts[1], 32)),
                    ]
                    async with asyncio.timeout(10):
                        while engine.get_num_unfinished_requests() != 2:
                            for task in tasks:
                                if task.done():
                                    task.result()
                                    pytest.fail(
                                        "request completed while generation was paused"
                                    )
                            await asyncio.sleep(0.01)
                    events["queued_before_resume"] = (
                        engine.get_num_unfinished_requests()
                    )
                    await engine.resume_generation()
                    await first_output.wait()
                    assert await engine.is_paused()
                    partial = observed["reused"]
                    assert not partial.finished
                    assert 0 < len(partial.outputs[0].token_ids) < 96
                    assert engine.get_num_unfinished_requests() in (1, 2)
                    assert not tasks[0].done()
                    events["tokens_before_abort"] = len(partial.outputs[0].token_ids)
                    events["healthy_finished_before_abort"] = tasks[1].done()
                    await engine.abort("reused")
                    await engine.resume_generation()
                    aborted, healthy = await asyncio.gather(*tasks)
                    assert aborted.outputs[0].finish_reason == "abort"
                    aborted_tokens = list(aborted.outputs[0].token_ids)
                    assert 0 < len(aborted_tokens) < 96
                    assert aborted_tokens == expected[0][: len(aborted_tokens)]
                    assert_active(healthy, expected[1])
                    reused = await collect("reused", prompts[1], 32)
                    assert_active(reused, expected[1])
                    outputs.update(aborted=aborted, healthy=healthy, reused=reused)
                elif lifecycle == "cleanup-error-fallback":
                    tasks = [asyncio.create_task(collect("reused", prompts[0], 96))]
                    await first_output.wait()
                    assert await engine.is_paused()
                    partial = observed["reused"]
                    partial_length = len(partial.outputs[0].token_ids)
                    assert not partial.finished and 0 < partial_length < 96
                    reports = await engine.collective_rpc(
                        _fail_token_session_during_socket_cleanup
                    )
                    assert len(reports) == 1
                    report = reports[0]
                    assert report["cleanup_error_injected"]
                    assert report["connection_detached"]
                    assert report["terminal_sequences"] == report["sequence_count"] > 0
                    events["cleanup_failure"] = report
                    await engine.resume_generation()
                    fallback = await tasks[0]
                    assert fallback.outputs[0].finish_reason == "length"
                    assert list(fallback.outputs[0].token_ids) == expected[0]
                    assert len(fallback.outputs[0].token_ids) > partial_length
                    recovered = await collect("recovered", prompts[1], 32)
                    assert_active(recovered, expected[1])
                    outputs.update(fallback=fallback, recovered=recovered)
                else:
                    warm = await collect("warm", prompts[1], 32)
                    assert_active(warm, expected[1])
                    assert len(owned_processes) == 1
                    with _paused_draft_server(owned_processes[0]):
                        fallback = await collect("timeout", prompts[0], 32)
                        assert fallback.outputs[0].finish_reason == "length"
                        assert list(fallback.outputs[0].token_ids) == expected[0][:32]
                        metrics = fallback.outputs[0].spec_decode_metrics
                        assert metrics is not None
                        assert metrics.num_draft_tokens == 0
                        assert metrics.to_dict()["num_accepted_draft_tokens"] == 0
                        assert (
                            "State:\tT"
                            in Path(
                                f"/proc/{owned_processes[0].pid}/status"
                            ).read_text()
                        )
                        events["completed_while_owned_server_stopped"] = True
                    recovered = await collect("recovered", prompts[1], 32)
                    assert_active(recovered, expected[1])
                    outputs.update(warm=warm, fallback=fallback, recovered=recovered)
            assert engine.get_num_unfinished_requests() == 0
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            engine.shutdown(timeout=30)

    record_property("async_lifecycle.case", lifecycle)
    record_property("async_lifecycle.events", events)
    record_property("async_lifecycle.reference_token_ids", expected)
    record_property(
        "async_lifecycle.finish_reasons",
        {name: output.outputs[0].finish_reason for name, output in outputs.items()},
    )
    record_property(
        "async_lifecycle.output_token_ids",
        {name: list(output.outputs[0].token_ids) for name, output in outputs.items()},
    )
    record_property(
        "async_lifecycle.request_speculation",
        {
            name: output.outputs[0].spec_decode_metrics.to_dict()
            for name, output in outputs.items()
            if output.outputs[0].spec_decode_metrics is not None
        },
    )


def _two_candidate_head(weight):
    import torch

    with torch.no_grad():
        weight[3].mul_(32)
        weight[4].copy_(-weight[3])


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


@pytest.mark.distributed(num_gpus=2)
@pytest.mark.parametrize("draft_devices", [(0, 1), (1, 0)], ids=["0-1", "1-0"])
def test_remote_token_draft_tp_matches_target_and_uses_drafts(
    vllm_runner, monkeypatch, tmp_path, record_property, draft_devices
):
    """Use two native draft ranks, including reversed local device binding."""
    import hashlib
    import json

    import torch

    assert current_platform.is_cuda(), "this test belongs to the NVIDIA CI step"
    assert torch.accelerator.device_count() >= 2, "this test requires two GPUs"
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "INFO")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    command = [
        sys.executable,
        "-c",
        "import sys; from tests.v1.e2e.spec_decode.remote.test_remote_mtp "
        "import _save_token_prefix_models; _save_token_prefix_models(sys.argv[1])",
        str(tmp_path),
    ]
    with (tmp_path / "tp-fixture.log").open("w") as log:
        subprocess.run(
            command,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", VLLM_TARGET_DEVICE="cpu"),
            check=True,
            timeout=180,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    target = str(tmp_path / "prefix-target")
    draft = str(tmp_path / "independent-qwen3")
    record_property(
        "draft_tp.fixture",
        {
            role: {
                str(path.relative_to(directory)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(directory.rglob("*"))
                if path.is_file()
            }
            for role, directory in (("target", Path(target)), ("draft", Path(draft)))
        },
    )
    kwargs = dict(
        tokenizer_mode="hf",
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
    spec = dict(
        method="draft_model",
        model=draft,
        num_speculative_tokens=3,
        draft_tensor_parallel_size=1,
        draft_sample_method="greedy",
        attention_backend="TRITON_ATTN",
    )
    sampling = SamplingParams(
        temperature=0, max_tokens=16, ignore_eos=True, detokenize=False
    )
    tokens, counts, resolved = {}, {}, {}

    def run_mode(mode, speculative_config=None):
        with vllm_runner(
            target,
            speculative_config=speculative_config,
            per_request_spec_decode_metrics=(
                "detailed" if speculative_config is not None else "none"
            ),
            **kwargs,
        ) as runner:
            actual = runner.llm.llm_engine.vllm_config
            assert actual.parallel_config.tensor_parallel_size == 1
            assert actual.model_config.enforce_eager and actual.use_v2_model_runner
            assert not actual.cache_config.enable_prefix_caching
            assert not actual.scheduler_config.async_scheduling
            draft_tp = None
            if actual.speculative_config is not None:
                draft_parallel = actual.speculative_config.draft_parallel_config
                draft_tp = draft_parallel.tensor_parallel_size
                assert draft_tp == (2 if mode == "remote" else 1)
            resolved[mode] = dict(target_tp=1, draft_tp=draft_tp)
            output = runner.llm.generate(
                [{"prompt_token_ids": [3] * 65}], sampling, use_tqdm=False
            )[0]
            tokens[mode] = list(output.outputs[0].token_ids)
            assert len(tokens[mode]) == 16
            if speculative_config is not None:
                runner.llm.llm_engine.do_log_stats()
                metrics = runner.llm.get_metrics()
                counts[mode] = {
                    name: get_spec_decode_metric_value(
                        metrics, f"vllm:spec_decode_num_{name}"
                    )
                    for name in ("drafts", "draft_tokens", "accepted_tokens")
                }
                assert counts[mode]["drafts"] > 0
                assert (
                    0 < counts[mode]["accepted_tokens"] <= counts[mode]["draft_tokens"]
                )
                detail = output.outputs[0].spec_decode_metrics
                assert detail is not None and detail.num_spec_tokens == 3
                assert sum(detail.per_step_accepted) > 0
                assert counts[mode]["drafts"] == len(detail.per_step_drafted)
                assert counts[mode]["draft_tokens"] == sum(detail.per_step_drafted)
                assert counts[mode]["accepted_tokens"] == sum(detail.per_step_accepted)
                record_property(f"draft_tp.{mode}.request_metrics", detail.to_dict())

    run_mode("target")
    run_mode("native", spec)
    server_log = tmp_path / "tp-draft.log"
    with draft_server(
        target,
        server_log,
        token_only_draft=draft,
        revision=None,
        num_speculative_tokens=3,
        attention_backend="TRITON_ATTN",
        draft_kv_cache_gib=0.03125,
        max_feature_tokens=32,
        draft_tensor_parallel_size=2,
        draft_devices=draft_devices,
    ) as endpoint:
        run_mode(
            "remote",
            dict(
                spec,
                draft_tensor_parallel_size=2,
                execution=dict(
                    placement="remote",
                    policy="on_demand",
                    endpoints=[endpoint],
                    transport="inline",
                    failure_policy="error",
                    request_timeout_ms=5000,
                    startup_timeout_ms=600000,
                ),
            ),
        )
    assert tokens["remote"] == tokens["native"] == tokens["target"]
    marker = "TOKEN_DRAFT_TP_READY "
    ready = [
        json.loads(line.split(marker, 1)[1])
        for line in server_log.read_text().splitlines()
        if marker in line
    ]
    assert len(ready) == 1
    topology = ready[0]
    assert topology["world_size"] == 2
    assert topology["devices"] == [f"cuda:{device}" for device in draft_devices]
    assert topology["num_blocks"] >= 2 and topology["block_size"] == 16
    assert topology["max_num_reqs"] == 4 and topology["max_num_tokens"] == 32
    assert [rank["rank"] for rank in topology["ranks"]] == [0, 1]
    assert len({rank["pid"] for rank in topology["ranks"]}) == 2
    assert len({rank["device_uuid"] for rank in topology["ranks"]}) == 2
    for rank, device in zip(topology["ranks"], topology["devices"]):
        assert rank["pid"] > 0 and rank["world_size"] == 2
        assert rank["device"] == device
        assert rank["device_uuid"] and rank["num_blocks"] >= topology["num_blocks"]
    record_property("draft_tp.ready", topology)
    record_property("draft_tp.resolved_config", resolved)
    record_property("draft_tp.output_token_ids", tokens)
    record_property("draft_tp.speculation", counts)
