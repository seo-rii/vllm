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
    "mtp_family,mtp_window",
    [
        ("exaone", None),
        ("exaone", 16),
        ("glm4_moe", None),
        ("qwen3_next", None),
        ("nemotron_h", None),
        ("hyv3", None),
    ],
    ids=["exaone-full", "exaone-swa", "glm4-moe", "qwen3-next", "nemotron-h", "hyv3"],
)
@pytest.mark.parametrize("k", [1, 3], ids=["k1", "k3"])
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
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    model = request.getfixturevalue(f"tiny_{mtp_family}_mtp_model")
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
    moe_backend = None if mtp_family == "exaone" else "triton"
    if moe_backend is not None:
        kwargs["moe_backend"] = moe_backend
    hybrid_target = mtp_family in ("qwen3_next", "nemotron_h")
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

        return [
            layer.sliding_window
            for layer in model.modules()
            if isinstance(layer, Attention)
        ]

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

        draft = worker.model_runner.get_draft_model()
        policies = []
        for attention in draft.modules():
            if not isinstance(attention, Attention):
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
        attention_backend="TRITON_ATTN",
    )
    with vllm_runner(model, **kwargs) as baseline:
        windows = baseline.llm.apply_model(target_windows)
        assert windows == [[None] if hybrid_target else [None, None]]
        record_property("target.attention_windows", windows)
        backends = baseline.llm.apply_model(moe_backends)
        expected_moe_count = 2 if mtp_family == "qwen3_next" else int(bool(moe_backend))
        assert backends == [["TRITON"] * expected_moe_count]
        record_property("target.moe_backends", backends)
        check_recurrence(baseline, "target")
        expected = generate(baseline)
    assert all(len(tokens) == 8 for batch in expected for tokens in batch)
    with vllm_runner(model, speculative_config=config, **kwargs) as native:
        policy = native.llm.collective_rpc(draft_kv_policy)
        spec_type = "FullAttentionSpec" if mtp_window is None else "SlidingWindowSpec"
        assert policy == [[(mtp_window, spec_type, mtp_window)]]
        record_property("native.kv_policy", policy)
        backends = native.llm.collective_rpc(draft_moe_backends)
        assert backends == [["TRITON"] if moe_backend else []]
        record_property("native.moe_backends", backends)
        recurrent_layers = native.llm.collective_rpc(draft_recurrent_layers)
        assert recurrent_layers == [0]
        record_property("native.recurrent_layers", recurrent_layers)
        native_recurrence = check_recurrence(native, "native")
        assert generate(native) == expected
        native_counts = speculation_counts(native, "native")
    with draft_server(
        model,
        tmp_path / f"{mtp_family}-draft.log",
        revision=None,
        num_speculative_tokens=k,
        attention_backend="TRITON_ATTN",
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
            assert generate(remote) == expected
            assert speculation_counts(remote, "remote") == native_counts
