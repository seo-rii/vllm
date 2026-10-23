# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the real snapshot assistant, without claiming GPU serving."""

from types import SimpleNamespace

import pytest
import torch
from transformers import Gemma4AssistantConfig, Gemma4TextConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.v1.spec_decode.remote.runtime.target_kv_gpu import Gemma4SnapshotExecutor
from vllm.v1.spec_decode.remote.runtime.target_kv_schema import (
    create_target_kv_contract,
)
from vllm.v1.spec_decode.remote.target_kv import import_target_kv


def _native_config(
    tmp_path,
    *,
    kv_heads=2,
    full_kv_heads=None,
    head_dim=128,
    full_head_dim=256,
    window=1024,
    max_model_len=640,
):
    """Build real offline model/runtime configs; this does not load model weights."""
    from safetensors.torch import save_file

    from vllm.config import (
        AttentionConfig,
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    text = Gemma4TextConfig(
        hidden_size=2304,
        num_hidden_layers=30,
        num_kv_shared_layers=10,
        num_key_value_heads=kv_heads,
        num_global_key_value_heads=full_kv_heads,
        attention_k_eq_v=full_kv_heads is not None,
        num_attention_heads=8,
        head_dim=head_dim,
        global_head_dim=full_head_dim,
        sliding_window=window,
        architectures=["Gemma4ForCausalLM"],
    )
    assistant = Gemma4AssistantConfig(
        text_config=Gemma4TextConfig(
            hidden_size=384,
            num_hidden_layers=4,
            num_kv_shared_layers=4,
            num_key_value_heads=kv_heads,
            num_global_key_value_heads=full_kv_heads,
            attention_k_eq_v=full_kv_heads is not None,
            num_attention_heads=8,
            head_dim=head_dim,
            global_head_dim=full_head_dim,
            sliding_window=window,
            layer_types=["sliding_attention"] * 3 + ["full_attention"],
            hidden_size_per_layer_input=0,
            vocab_size_per_layer_input=0,
        ),
        backbone_hidden_size=text.hidden_size,
        architectures=["Gemma4AssistantForCausalLM"],
    )
    target_path, draft_path = tmp_path / "target", tmp_path / "assistant"
    for path, hf in ((target_path, text), (draft_path, assistant)):
        hf.save_pretrained(path)
        save_file(
            {"identity_only": torch.zeros(1, dtype=torch.bfloat16)},
            str(path / "model.safetensors"),
        )
        (path / "tokenizer_config.json").write_text("{}")
    target = ModelConfig(
        str(target_path),
        dtype="bfloat16",
        max_model_len=max_model_len,
        enforce_eager=True,
    )
    parallel = ParallelConfig()
    speculative = SpeculativeConfig(
        target_model_config=target,
        target_parallel_config=parallel,
        model=str(draft_path),
        method="mtp",
        num_speculative_tokens=1,
        draft_tensor_parallel_size=1,
    )
    return VllmConfig(
        model_config=target,
        parallel_config=parallel,
        speculative_config=speculative,
        scheduler_config=SchedulerConfig(
            max_model_len=max_model_len,
            is_encoder_decoder=False,
            max_num_seqs=1,
            max_num_batched_tokens=64,
            async_scheduling=False,
        ),
        cache_config=CacheConfig(
            block_size=16, cache_dtype="auto", enable_prefix_caching=False
        ),
        attention_config=AttentionConfig(backend="TRITON_ATTN"),
        device_config=DeviceConfig("cpu"),
    )


@pytest.mark.parametrize(
    "kv_heads,full_kv_heads,head_dim,full_head_dim,window",
    [
        (1, None, 64, 128, 256),
        (2, None, 128, 256, 1024),
        (4, None, 256, 512, 2048),
        (4, 2, 128, 256, 1024),
    ],
)
def test_native_configs_derive_gemma_geometry_and_snapshot_import(
    tmp_path, kv_heads, full_kv_heads, head_dim, full_head_dim, window
):
    """Compatible Gemma4 sizes retain native owner mapping and independent KV."""
    from vllm.v1.spec_decode.remote.protocol import SequenceKey
    from vllm.v1.spec_decode.remote.target_kv import (
        SnapshotIdentity,
        SnapshotLayer,
        TargetKVSnapshot,
    )
    from vllm.v1.spec_decode.remote.target_kv_protocol import snapshot_wire_plan

    config = _native_config(
        tmp_path,
        kv_heads=kv_heads,
        full_kv_heads=full_kv_heads,
        head_dim=head_dim,
        full_head_dim=full_head_dim,
        window=window,
    )
    contract = create_target_kv_contract(config)
    assert contract.hidden_size == 2304
    assert contract.draft_num_attention_heads == 8
    assert contract.source_indices == {"target.layers.19": 19, "target.layers.17": 17}
    assert contract.draft_source_names == ("target.layers.19",) * 3 + (
        "target.layers.17",
    )
    specs = contract.expected_specs
    assert specs["target.layers.19"].head_size == head_dim
    assert specs["target.layers.19"].sliding_window == window
    assert specs["target.layers.17"].head_size == full_head_dim
    assert specs["target.layers.19"].num_kv_heads == kv_heads
    assert specs["target.layers.17"].num_kv_heads == (full_kv_heads or kv_heads)
    assert len({s.page_size_bytes for s in specs.values()}) == 1
    identity = SnapshotIdentity(SequenceKey("geometry", 0, 0), 1, 0)
    for length in (1, 31, 32, 33, 640):
        plans = snapshot_wire_plan(
            specs, length, contract.snapshot_budget, contract.receiver_budget
        )
        layers = tuple(
            SnapshotLayer(
                p.name,
                p.spec,
                p.logical_pages,
                torch.ones(p.shape, dtype=torch.bfloat16),
            )
            for p in plans
        )
        snapshot = TargetKVSnapshot(identity, length, layers)
        imported = import_target_kv(
            snapshot,
            expected_identity=identity,
            expected_materialized_length=length,
            expected_specs=specs,
            budget=contract.receiver_budget,
            device=torch.device("cpu"),
        )
        for source, destination in zip(snapshot.layers, imported.layers):
            assert destination.cache.shape[1:] == source.pages.shape[1:]
            assert destination.cache.untyped_storage().data_ptr() != (
                source.pages.untyped_storage().data_ptr()
            )


@pytest.mark.parametrize(
    "side,field,value",
    [
        ("target", "num_attention_heads", 3),
        ("target", "sliding_window", 0),
        ("assistant", "num_attention_heads", 3),
        ("assistant", "sliding_window", 2048),
        ("assistant", "num_kv_shared_layers", 1),
        ("assistant", "vocab_size", 123),
        ("assistant", "enable_moe_block", True),
        ("assistant", "max_position_embeddings", 639),
    ],
)
def test_general_gemma_contract_rejects_incompatible_native_configs(
    tmp_path, side, field, value
):
    config = _native_config(tmp_path)
    model = (
        config.model_config
        if side == "target"
        else config.speculative_config.draft_model_config
    )
    setattr(model.hf_config.get_text_config(), field, value)
    with pytest.raises(ValueError, match="Gemma4|assistant attention"):
        create_target_kv_contract(config)


def test_native_page_padding_is_rejected_before_snapshot_allocation(tmp_path):
    config = _native_config(tmp_path, head_dim=128, full_head_dim=192)
    with pytest.raises(ValueError, match="dense causal BF16"):
        create_target_kv_contract(config)


def test_large_gqa_geometry_must_fit_the_negotiated_snapshot_budget(tmp_path):
    config = _native_config(
        tmp_path, kv_heads=8, head_dim=256, full_head_dim=512, max_model_len=8192
    )
    with pytest.raises(ValueError, match="snapshot exceeds page or byte budget"):
        create_target_kv_contract(config)


def _config():
    text = Gemma4TextConfig(
        hidden_size=1536,
        num_hidden_layers=35,
        num_kv_shared_layers=20,
        num_key_value_heads=1,
        num_attention_heads=8,
        head_dim=256,
        global_head_dim=512,
        sliding_window=512,
        layer_types=(["sliding_attention"] * 4 + ["full_attention"]) * 7,
    )
    assistant = Gemma4TextConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_kv_shared_layers=4,
        num_key_value_heads=1,
        num_attention_heads=4,
        head_dim=256,
        global_head_dim=512,
        sliding_window=512,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
        use_double_wide_mlp=False,
    )
    assistant = SpeculativeConfig.hf_config_override(
        Gemma4AssistantConfig(
            text_config=assistant,
            backbone_hidden_size=1536,
            architectures=["Gemma4AssistantForCausalLM"],
        )
    )

    def model(hf, name):
        return SimpleNamespace(
            hf_config=hf,
            model=name,
            revision="a" * 40,
            dtype=torch.bfloat16,
            quantization=None,
            tokenizer="target",
            tokenizer_revision="a" * 40,
            tokenizer_mode="auto",
            enforce_eager=True,
            max_model_len=640,
            enable_prompt_embeds=False,
            is_multimodal_model=False,
            get_vocab_size=lambda: 262144,
            get_hidden_size=lambda: 1536,
            try_get_generation_config=lambda: {},
        )

    return SimpleNamespace(
        model_config=model(text, "target"),
        speculative_config=SimpleNamespace(
            draft_model_config=model(assistant, "assistant"),
            method="mtp",
            num_speculative_tokens=1,
            draft_sample_method="greedy",
            parallel_drafting=False,
            enable_adaptive_verification=False,
            draft_tensor_parallel_size=1,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=1,
            async_scheduling=False,
            disable_hybrid_kv_cache_manager=False,
        ),
        attention_config=SimpleNamespace(backend="TRITON_ATTN"),
        cache_config=SimpleNamespace(
            block_size=16,
            cache_dtype="auto",
            enable_prefix_caching=False,
        ),
        lora_config=None,
    )


def test_native_hf_override_and_page_unification_match_gemma_e2b():
    contract = create_target_kv_contract(_config())
    assert contract.source_indices == {"target.layers.13": 13, "target.layers.14": 14}
    assert contract.draft_source_names == ("target.layers.13",) * 3 + (
        "target.layers.14",
    )
    swa, full = contract.expected_specs.values()
    assert (swa.block_size, swa.head_size, swa.sliding_window) == (32, 256, 512)
    assert (full.block_size, full.head_size, full.sliding_window) == (16, 512, None)
    assert swa.page_size_bytes == full.page_size_bytes
    assert create_target_kv_contract(_config()).fingerprint == contract.fingerprint
    changed = _config()
    changed.speculative_config.draft_model_config.revision = "b" * 40
    assert create_target_kv_contract(changed).fingerprint != contract.fingerprint


def _resolved_model_revision(source, resolved):
    """Represent Hub provenance across ResolvedRevision constructor versions."""
    from huggingface_hub import ResolvedRevision

    revision = ResolvedRevision(resolved, initial="main")
    revision._repo_id, revision._repo_type = source, "model"
    return revision


def test_tokenizer_revision_uses_resolved_commit():
    config = _config()
    target = config.model_config
    target.tokenizer = "test/separate-tokenizer"
    original = create_target_kv_contract(config).fingerprint
    target.tokenizer_revision = _resolved_model_revision(target.tokenizer, "a" * 40)
    assert create_target_kv_contract(config).fingerprint == original
    target.tokenizer = "test/different-tokenizer"
    with pytest.raises(ValueError, match="different repository"):
        create_target_kv_contract(config)
    target.tokenizer = "test/separate-tokenizer"
    target.tokenizer_revision = _resolved_model_revision(target.tokenizer, "b" * 40)
    assert create_target_kv_contract(config).fingerprint != original
    target.tokenizer_revision = "a" * 40
    target.tokenizer_mode = "slow"
    assert create_target_kv_contract(config).fingerprint != original


@pytest.mark.parametrize(
    "first,second,equal",
    [
        ({"suppress_tokens": [1]}, {"suppress_tokens": [2]}, False),
        ({}, {"suppress_tokens": [1]}, False),
        ({}, {"suppress_tokens": None}, True),
        ({}, {"suppress_tokens": []}, True),
        ({"suppress_tokens": [2, 1, 2]}, {"suppress_tokens": [1, 2]}, True),
        (
            {"suppress_tokens": [1], "max_new_tokens": 10},
            {"suppress_tokens": [1], "max_new_tokens": 20},
            True,
        ),
    ],
)
def test_contract_tracks_effective_assistant_token_suppression(first, second, equal):
    config = _config()
    draft = config.speculative_config.draft_model_config
    draft.try_get_generation_config = lambda: first
    original = create_target_kv_contract(config).fingerprint
    draft.try_get_generation_config = lambda: second
    changed = create_target_kv_contract(config).fingerprint
    assert (original == changed) is equal


def test_local_tokenizer_identity_tracks_contents_not_directory(tmp_path):
    config = _config()
    tokenizer = tmp_path / "first"
    tokenizer.mkdir()
    artifact = tokenizer / "tokenizer.json"
    artifact.write_text('{"model":{"vocab":{"a":0,"b":1}}}')
    config.model_config.tokenizer = str(tokenizer)
    original = create_target_kv_contract(config).fingerprint
    copied = tmp_path / "copied"
    copied.mkdir()
    (copied / artifact.name).write_bytes(artifact.read_bytes())
    config.model_config.tokenizer = str(copied)
    assert create_target_kv_contract(config).fingerprint == original
    (copied / artifact.name).write_text('{"model":{"vocab":{"b":0,"a":1}}}')
    assert create_target_kv_contract(config).fingerprint != original


@pytest.mark.parametrize(
    "group,name,value",
    [
        ("parallel_config", "tensor_parallel_size", 2),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("parallel_config", "prefill_context_parallel_size", 2),
        ("scheduler_config", "max_num_seqs", 2),
        ("scheduler_config", "async_scheduling", True),
        ("scheduler_config", "disable_hybrid_kv_cache_manager", True),
        ("cache_config", "block_size", 32),
        ("cache_config", "cache_dtype", "fp8"),
        ("cache_config", "enable_prefix_caching", True),
        ("model_config", "dtype", torch.float16),
        ("model_config", "enforce_eager", False),
        ("model_config", "enable_prompt_embeds", True),
        ("model_config", "max_model_len", 8193),
        ("attention_config", "backend", "FLASH_ATTN"),
        ("speculative_config", "num_speculative_tokens", 2),
        ("speculative_config", "enable_adaptive_verification", True),
    ],
)
def test_unsupported_config_is_rejected_before_loading(group, name, value):
    config = _config()
    setattr(getattr(config, group), name, value)
    with pytest.raises(ValueError, match="remote target-KV"):
        create_target_kv_contract(config)


def test_multimodal_wrapper_requires_language_model_only_not_zero_limits():
    config = _config()
    config.model_config.is_multimodal_model = True
    config.model_config.multimodal_config = SimpleNamespace(
        language_model_only=False, limit_per_prompt={"image": 0, "audio": 0, "video": 0}
    )
    with pytest.raises(ValueError, match="language_model_only"):
        create_target_kv_contract(config)
    config.model_config.multimodal_config.language_model_only = True
    assert create_target_kv_contract(config).hidden_size == 1536


def test_warmup_uses_real_snapshot_import_at_one_and_max_context(monkeypatch):
    executor = Gemma4SnapshotExecutor.__new__(Gemma4SnapshotExecutor)
    executor.contract = create_target_kv_contract(_config())
    lengths = []

    def execute(
        snapshot,
        *,
        expected_identity,
        expected_materialized_length,
        recovery_token,
        hidden_state,
    ):
        imported = import_target_kv(
            snapshot,
            expected_identity=expected_identity,
            expected_materialized_length=expected_materialized_length,
            expected_specs=executor.contract.expected_specs,
            budget=executor.contract.receiver_budget,
            device=torch.device("cpu"),
        )
        lengths.append(imported.materialized_length)
        assert recovery_token == 0
        assert hidden_state.shape == (1, 1536)
        assert not hidden_state.count_nonzero()
        return 0

    monkeypatch.setattr(executor, "execute", execute)
    executor.warmup()
    assert lengths == [1, 640]


@pytest.mark.parametrize("resolved", [False, True])
def test_embedding_loader_requests_only_target_embedding_and_keeps_draft_head(
    monkeypatch, resolved
):
    import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
    import vllm.v1.spec_decode.remote.runtime.target_kv_gpu as gpu

    config = _config().model_config
    config.hf_config._commit_hash = "c" * 40
    config.hf_config_path = "test/config-only"
    if resolved:
        config.revision = _resolved_model_revision(config.model, "a" * 40)
    config.get_vocab_size, config.get_hidden_size = lambda: 5, lambda: 3
    weight = torch.arange(15, dtype=torch.bfloat16).reshape(5, 3)
    selected = []

    def fetch(source, revision, names):
        assert source == config.model and revision == "a" * 40
        selected.extend(names)
        return {names[0]: weight}

    class Embedding(torch.nn.Module):
        def __init__(self, *shape, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.empty(*shape, dtype=torch.bfloat16), requires_grad=False
            )

        def weight_loader(self, destination, source):
            destination.copy_(source)

    monkeypatch.setattr(gpu, "selected_checkpoint_weights", fetch)
    monkeypatch.setattr(embedding_module, "VocabParallelEmbedding", Embedding)
    assistant = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=object()), lm_head=object()
    )
    head = assistant.lm_head
    assistant.model.embed_tokens = gpu.materialize_gemma4_embedding(
        config, torch.device("cpu")
    )
    assert assistant.lm_head is head
    torch.testing.assert_close(assistant.model.embed_tokens.weight, weight)
    assert selected == [
        "model.language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
    ]


@pytest.mark.parametrize("failure", [None, "oom", "fatal", "fence"])
def test_executor_fences_consumers_before_unbinding_or_quarantines(
    monkeypatch, failure
):
    """Exercise production execute/import; only model and CUDA execution are doubles."""
    import threading
    from contextlib import nullcontext

    import vllm.forward_context as forward
    import vllm.v1.spec_decode.remote.runtime.target_kv_gpu as gpu
    from vllm.v1.spec_decode.remote.protocol import SequenceKey
    from vllm.v1.spec_decode.remote.target_kv import (
        SnapshotCompletionError,
        SnapshotIdentity,
        SnapshotLayer,
        TargetKVSnapshot,
    )

    executor = Gemma4SnapshotExecutor.__new__(Gemma4SnapshotExecutor)
    executor.contract = create_target_kv_contract(_config())
    executor.config = _config()
    executor.device = torch.device("cpu:0")
    executor.healthy = True
    executor._lock = threading.Lock()
    executor._quarantine = []
    executor._empty_caches = tuple(torch.empty(0) for _ in range(4))
    executor._attentions = tuple(
        SimpleNamespace(layer_name=f"draft.{i}", num_kv_heads=1, kv_cache=empty)
        for i, empty in enumerate(executor._empty_caches)
    )
    executor._scratch = tuple((torch.empty(0),) * 3 for _ in range(4))
    executor._input_ids = torch.empty(1, dtype=torch.int32)
    executor._positions = torch.empty(1, dtype=torch.int64)
    executor._seq_lens = torch.empty(1, dtype=torch.int32)
    executor._hidden = torch.empty((1, 1536), dtype=torch.bfloat16)
    executor._query_start = torch.tensor([0, 1], dtype=torch.int32)
    executor._slots = torch.tensor([-1], dtype=torch.int64)
    events = []
    identity = SnapshotIdentity(SequenceKey("test", 1, 1), 1, 1)
    snapshot = TargetKVSnapshot(
        identity,
        1,
        tuple(
            SnapshotLayer(
                name,
                spec,
                (0,),
                torch.zeros(
                    (1, 1, spec.block_size, 2 * spec.head_size), dtype=torch.bfloat16
                ),
            )
            for name, spec in executor.contract.expected_specs.items()
        ),
    )

    def synchronize():
        events.append("fence")
        assert all(attn.kv_cache.numel() for attn in executor._attentions)
        if failure == "fence":
            raise RuntimeError("completion unknown")

    class Model:
        def __call__(self, **kwargs):
            events.append("model")
            assert kwargs["positions"].tolist() == [0]
            assert kwargs["input_ids"].tolist() == [7]
            assert all(attn.kv_cache.numel() for attn in executor._attentions)
            if failure == "oom":
                raise torch.OutOfMemoryError("injected forward allocation failure")
            if failure == "fatal":
                raise RuntimeError("CUDA error: illegal memory access")
            return torch.tensor([[1.0, 2.0, 3.0]]), None

        def compute_logits(self, hidden):
            return hidden

    executor.model = Model()
    monkeypatch.setattr(torch.accelerator, "device_index", lambda device: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(synchronize=synchronize),
    )
    monkeypatch.setattr(gpu, "set_current_vllm_config", lambda config: nullcontext())
    monkeypatch.setattr(
        forward, "set_forward_context", lambda *args, **kwargs: nullcontext()
    )
    kwargs = dict(
        expected_identity=identity,
        expected_materialized_length=1,
        recovery_token=7,
        hidden_state=torch.zeros((1, 1536), dtype=torch.bfloat16),
    )
    if failure:
        expected = SnapshotCompletionError if failure == "fence" else RuntimeError
        with pytest.raises(expected):
            executor.execute(snapshot, **kwargs)
    else:
        assert executor.execute(snapshot, **kwargs) == 2
    assert events[0] == "model"
    if failure == "fatal":
        assert events == ["model"]
    else:
        assert "fence" in events
    if failure in ("fatal", "fence"):
        assert not executor.healthy
        assert len(executor._quarantine) == 1

        def forbidden_cuda_device(device):
            pytest.fail("an unhealthy executor must not enter the CUDA context")

        monkeypatch.setattr(torch.accelerator, "device_index", forbidden_cuda_device)
        with pytest.raises(RuntimeError, match="process restart"):
            executor.execute(snapshot, **kwargs)
        assert all(attn.kv_cache.numel() for attn in executor._attentions)
    else:
        assert executor.healthy
        assert not executor._quarantine
        assert all(
            attn.kv_cache is empty
            for attn, empty in zip(executor._attentions, executor._empty_caches)
        )


def test_cli_requires_text_only_and_never_defaults_to_remote_checkpoint_code(
    monkeypatch,
):
    import vllm.v1.spec_decode.remote.runtime.target_kv_cli as cli

    captured = {}
    events = []
    contract = create_target_kv_contract(_config())
    config = object()

    def engine_args(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(create_engine_config=lambda: config)

    class Executor:
        def __init__(self, actual, *, device):
            assert actual is config
            self.contract = contract

        def close(self):
            events.append("closed")

    class Server:
        healthy = True

        def __init__(self, executor, **kwargs):
            assert kwargs["expected_specs"] == contract.expected_specs

        def start(self, endpoint):
            events.append("started")
            return endpoint

        def wait(self):
            events.append("waited")

        def stop(self):
            events.append("stopped")

    monkeypatch.setattr(cli, "EngineArgs", engine_args)
    monkeypatch.setattr(cli, "Gemma4SnapshotExecutor", Executor)
    monkeypatch.setattr(cli, "TargetKVDraftServer", Server)
    args = [
        "--endpoint",
        "unix:///tmp/test-gemma.sock",
        "--target-model",
        "target",
        "--draft-model",
        "assistant",
    ]
    with pytest.raises(SystemExit):
        cli.main(args)
    assert not captured
    cli.main([*args, "--language-model-only"])
    assert captured["language_model_only"]
    assert not captured["trust_remote_code"]
    assert captured["max_num_seqs"] == 1
    assert captured["block_size"] == 16
    assert captured["dtype"] == "bfloat16"
    assert not captured["async_scheduling"]
    assert not captured["enable_prefix_caching"]
    assert events == ["started", "waited", "stopped", "closed"]


def test_maximum_supported_context_fits_snapshot_and_receiver_caps():
    from vllm.v1.spec_decode.remote.target_kv_protocol import snapshot_wire_plan

    config = _config()
    config.model_config.max_model_len = 8192
    contract = create_target_kv_contract(config)
    for length in (1, 16, 17, 31, 32, 33, 511, 512, 513, 8191, 8192):
        snapshot_wire_plan(
            contract.expected_specs,
            length,
            contract.snapshot_budget,
            contract.receiver_budget,
        )
