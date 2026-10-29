# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RemoteDraftConfig field validation."""

import pytest
from pydantic import ValidationError

from tests.v1.spec_decode.remote.runtime_fixtures import (
    cpu_parallel_state as cpu_parallel_state,
)
from vllm.config.speculative import (
    RemoteDraftConfig,
    SpeculativeConfig,
    SpeculativeExecutionConfig,
)


def test_defaults_are_production_safe():
    config = RemoteDraftConfig(endpoint="unix:///run/vllm/drafter.sock")
    assert config.transport == "auto"
    assert config.failure_policy == "target_only"
    assert config.request_timeout_ms > 0
    assert config.startup_timeout_ms > 0


def test_inline_transport_is_explicitly_supported():
    config = RemoteDraftConfig(
        endpoint="unix:///run/vllm/drafter.sock", transport="inline"
    )
    assert config.transport == "inline"


def test_endpoint_is_required():
    with pytest.raises(ValidationError):
        RemoteDraftConfig()


@pytest.mark.parametrize("endpoint", ["", "   "])
def test_empty_endpoint_rejected(endpoint):
    with pytest.raises(ValidationError):
        RemoteDraftConfig(endpoint=endpoint)


@pytest.mark.parametrize(
    "endpoint", ["http://127.0.0.1:5555", "tcp://127.0.0.1", "drafter.sock"]
)
def test_malformed_endpoint_rejected(endpoint):
    with pytest.raises(ValidationError, match="endpoint"):
        RemoteDraftConfig(endpoint=endpoint)


@pytest.mark.parametrize("transport", ["pinned_host", "zmq", "carrier_pigeon"])
def test_unimplemented_transport_rejected(transport):
    with pytest.raises(ValidationError):
        RemoteDraftConfig(endpoint="unix:///x.sock", transport=transport)


def test_nonpositive_timeout_rejected():
    with pytest.raises(ValidationError):
        RemoteDraftConfig(endpoint="unix:///x.sock", request_timeout_ms=0)


def test_cuda_ipc_requires_unix_socket():
    assert (
        RemoteDraftConfig(endpoint="unix:///x.sock", transport="cuda_ipc").transport
        == "cuda_ipc"
    )
    with pytest.raises(ValidationError, match="UNIX socket"):
        RemoteDraftConfig(endpoint="tcp://127.0.0.1:9999", transport="cuda_ipc")


@pytest.mark.parametrize(
    "options",
    [
        {"endpoints": ["unix:///draft.sock"]},
        {"transport": "inline"},
        {"failure_policy": "target_only"},
        {"request_timeout_ms": 100},
        {"startup_timeout_ms": 30_000},
    ],
)
def test_local_execution_rejects_remote_options(options):
    with pytest.raises(ValidationError, match="local execution"):
        SpeculativeExecutionConfig(placement="local", **options)


@pytest.mark.parametrize("endpoints", [[], ["unix:///a", "unix:///b"]])
def test_remote_execution_requires_one_endpoint(endpoints):
    with pytest.raises(ValidationError, match="exactly one endpoint"):
        SpeculativeExecutionConfig(placement="remote", endpoints=endpoints)


@pytest.mark.parametrize("policy", ["outcome_cache", "shadow"])
def test_unimplemented_execution_policies_are_not_silently_downgraded(policy):
    with pytest.raises(ValidationError):
        SpeculativeExecutionConfig(policy=policy)


def test_new_remote_configuration_preserves_legacy_defaults():
    config = SpeculativeExecutionConfig(
        placement="remote", endpoints=("unix:///draft.sock",)
    ).remote_config()
    assert config == RemoteDraftConfig(endpoint="unix:///draft.sock")


def test_remote_configuration_validates_transport_and_preserves_overrides():
    execution = SpeculativeExecutionConfig(
        placement="remote",
        endpoints=("unix:///draft.sock",),
        transport="cuda_ipc",
        failure_policy="error",
        request_timeout_ms=25,
        startup_timeout_ms=2000,
    )
    assert execution.remote_config() == RemoteDraftConfig(
        endpoint="unix:///draft.sock",
        transport="cuda_ipc",
        failure_policy="error",
        request_timeout_ms=25,
        startup_timeout_ms=2000,
    )
    with pytest.raises(ValidationError, match="UNIX socket"):
        SpeculativeExecutionConfig(
            placement="remote",
            endpoints=("tcp://127.0.0.1:99",),
            transport="cuda_ipc",
        )


def test_local_execution_does_not_apply_remote_method_restrictions():
    options = dict(method="ngram", num_speculative_tokens=3, prompt_lookup_max=3)
    native = SpeculativeConfig(**options)
    wrapped = SpeculativeConfig(**options, execution={"placement": "local"})
    assert not native.use_unified_draft_backend()
    assert wrapped.use_unified_draft_backend()
    assert not wrapped.use_remote_draft()
    assert wrapped.get_remote_draft_config() is None
    assert wrapped.method == native.method


def test_legacy_and_new_placement_cannot_be_combined_before_model_loading():
    with pytest.raises(ValidationError, match="cannot be combined"):
        SpeculativeConfig(
            model="this-model-must-not-be-loaded",
            remote_draft={"endpoint": "unix:///draft.sock"},
            execution={"placement": "local"},
        )


@pytest.mark.parametrize(
    "override,reason",
    [
        ({}, None),
        ({"draft_tensor_parallel_size": 3}, "draft_tensor_parallel_size"),
        ({"draft_sample_method": "probabilistic"}, "greedy"),
        ({"parallel_drafting": True}, "parallel_drafting"),
        ({"use_heterogeneous_vocab": True}, "heterogeneous"),
    ],
)
def test_token_only_config_rejects_unimplemented_semantics(override, reason):
    # Isolate validation after model resolution without loading a checkpoint.
    spec = object.__new__(SpeculativeConfig)
    spec.method = "draft_model"
    spec.remote_draft = None
    spec.execution = SpeculativeExecutionConfig(
        placement="remote", endpoints=("unix:///draft.sock",)
    )
    spec.draft_tensor_parallel_size = 1
    spec.draft_sample_method = "greedy"
    spec.parallel_drafting = False
    spec.use_heterogeneous_vocab = False
    for name, value in override.items():
        setattr(spec, name, value)
    if reason is None:
        spec._validate_remote_draft()
        spec.execution.transport = "cuda_ipc"
        with pytest.raises(ValueError, match="inline transport"):
            spec._validate_remote_draft()
    else:
        with pytest.raises(ValueError, match=reason):
            spec._validate_remote_draft()


def test_legacy_remote_does_not_silently_switch_to_token_only_protocol():
    spec = object.__new__(SpeculativeConfig)
    spec.method = "draft_model"
    spec.remote_draft = RemoteDraftConfig(endpoint="unix:///draft.sock")
    spec.execution = None
    spec.draft_tensor_parallel_size = 1
    spec.draft_sample_method = "greedy"
    spec.draft_model_config = None
    assert spec.get_remote_draft_config() is spec.remote_draft
    with pytest.raises(ValueError, match="method 'draft_model'"):
        spec._validate_remote_draft()


# One contract table drives public API, runtime schema, CLI, and supported TP1
# native-load checks. Device enumeration is the only simulated CLI service.
_TOKEN_COMPATIBILITY_CASES = [
    pytest.param(1, "float32", "on_demand", {}, None, id="dense-tp1-fp32"),
    pytest.param(1, "bfloat16", "on_demand", {}, None, id="dense-tp1-bf16"),
    pytest.param(2, "bfloat16", "on_demand", {}, None, id="dense-tp2-bf16"),
    pytest.param(2, "float32", "on_demand", {}, "unquantized bfloat16", id="tp2-fp32"),
    pytest.param(
        2,
        "bfloat16",
        "on_demand",
        {"draft_type": "mistral", "sliding_window": 16},
        "dense full-attention",
        id="tp2-sliding",
    ),
    pytest.param(
        2,
        "bfloat16",
        "on_demand",
        {"sliding_window": 16, "layer_types": ["full_attention", "sliding_attention"]},
        "dense full-attention",
        id="tp2-mixed-windows",
    ),
    pytest.param(
        2,
        "bfloat16",
        "on_demand",
        {"draft_type": "nemotron_h"},
        "dense full-attention",
        id="tp2-recurrent",
    ),
    pytest.param(
        2,
        "bfloat16",
        "outcome_cache",
        {},
        "outcome cache requires.*TP1",
        id="cache-tp2",
    ),
]


def _token_cli_contract(arguments):
    """Stop real CLI startup after contract validation, before any GPU owner."""
    import sys

    from vllm.v1.spec_decode.remote.runtime.schema import RuntimeContract
    from vllm.v1.spec_decode.remote.runtime.token_cli import create_adapter, parser
    from vllm.v1.spec_decode.remote.runtime.token_schema import (
        create_token_runtime_contract,
    )

    class Validated(Exception):
        pass

    observed = []

    def stop_after_validation(frame, event, result):
        if (
            event == "return"
            and frame.f_code is create_token_runtime_contract.__code__
            and isinstance(result, RuntimeContract)
        ):
            observed.append((frame.f_locals["config"], result))
            raise Validated

    previous = sys.getprofile()
    sys.setprofile(stop_after_validation)
    try:
        try:
            create_adapter(parser().parse_args(arguments))
        except Validated:
            assert len(observed) == 1
            return observed[0]
    finally:
        sys.setprofile(previous)
    pytest.fail("CLI did not validate a token runtime contract")


@pytest.mark.parametrize(
    "draft_tp,dtype,policy,draft_options,reason", _TOKEN_COMPATIBILITY_CASES
)
def test_token_compatibility_contract_matches_public_schema_cli_and_native_load(
    tmp_path, monkeypatch, request, draft_tp, dtype, policy, draft_options, reason
):
    from dataclasses import replace

    import torch

    from tests.v1.spec_decode.remote.test_token_runtime import token_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform
    from vllm.v1.spec_decode.remote.protocol import FeatureSlot
    from vllm.v1.spec_decode.remote.runtime.token_schema import (
        create_token_runtime_contract,
    )

    assets = token_config(
        tmp_path, dtype=dtype, enforce_eager=True, head_dim=64, **draft_options
    )
    target = assets.model_config.model
    draft = assets.speculative_config.draft_model_config.model
    endpoint = "unix:///tmp/compatibility.sock"
    options = {
        "method": "draft_model",
        "model": draft,
        "num_speculative_tokens": 3,
        "draft_tensor_parallel_size": draft_tp,
        "execution": {
            "placement": "remote",
            "policy": policy,
            "endpoints": [endpoint],
            "transport": "inline",
        },
    }

    def public():
        config = EngineArgs(
            model=target,
            tokenizer_mode="hf",
            dtype=dtype,
            skip_tokenizer_init=True,
            max_model_len=64,
            max_num_seqs=4,
            max_num_batched_tokens=64,
            enable_prefix_caching=False,
            async_scheduling=False,
            enforce_eager=True,
            speculative_config=options,
        ).create_engine_config()
        return config, create_token_runtime_contract(config)

    def schema():
        config = replace(
            assets,
            speculative_config=SpeculativeConfig(
                target_model_config=assets.model_config,
                target_parallel_config=assets.parallel_config,
                **options,
            ),
        )
        return config, create_token_runtime_contract(config)

    arguments = [
        "--endpoint",
        endpoint,
        "--target-model",
        target,
        "--draft-model",
        draft,
        "--dtype",
        dtype,
        "--policy",
        policy,
        "--max-model-len",
        "64",
        "--max-feature-tokens",
        "64",
        "--draft-tensor-parallel-size",
        str(draft_tp),
    ]
    if draft_tp == 2:
        arguments.extend(("--devices", "cuda:0", "cuda:1"))
    contracts = []
    for construct in (public, schema, lambda: _token_cli_contract(arguments)):
        with monkeypatch.context() as hardware:
            hardware.setattr(torch.accelerator, "device_count", lambda: 2)
            hardware.setattr(
                current_platform, "visible_device_id_to_physical_device_id", int
            )
            if reason is not None:
                with pytest.raises(ValueError, match=reason):
                    construct()
                continue
            config, contract = construct()
        assert (
            config.speculative_config.draft_parallel_config.tensor_parallel_size
            == draft_tp
        )
        assert config.speculative_config.draft_model_config.dtype == getattr(
            torch, dtype
        )
        assert contract.schema.slots == (FeatureSlot("token_ids", "int32", ()),)
        contracts.append(contract)
    if reason is not None:
        return
    assert len({contract.tokenizer_fingerprint for contract in contracts}) == 1
    assert len({contract.draft_fingerprint for contract in contracts}) == 1
    if draft_tp != 1:
        return  # Native TP2 execution requires the GPU matrix.

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader import get_model
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.spec_decode.remote.runtime.token_gpu import (
        create_token_executor_config,
        validate_token_model_inputs,
    )
    from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec

    request.getfixturevalue("cpu_parallel_state")
    resource = create_token_executor_config(config)
    with monkeypatch.context() as hardware, set_current_vllm_config(resource):
        hardware.setattr(torch.accelerator, "empty_cache", lambda: None)
        model = get_model(vllm_config=resource)
        validate_token_model_inputs(resource, model)
        specs = get_kv_cache_spec(resource)
    assert type(model).__name__ == "Qwen3ForCausalLM"
    assert len(specs) == resource.model_config.get_total_num_hidden_layers()
    assert all(type(spec) is FullAttentionSpec for spec in specs.values())
    assert all(
        isinstance(spec, FullAttentionSpec) and spec.dtype == getattr(torch, dtype)
        for spec in specs.values()
    )


@pytest.mark.parametrize("target_tp", [1, 2])
@pytest.mark.parametrize("entrypoint", ["engine_args", "vllm_config", "schema"])
def test_token_target_topology_is_checked_before_worker_creation(
    tmp_path, target_tp, entrypoint
):
    """Every public construction path must reject the worker's unsupported TP."""
    from dataclasses import replace

    from tests.v1.spec_decode.remote.test_token_runtime import token_config
    from vllm.config import ParallelConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.spec_decode.remote.protocol import FeatureSlot
    from vllm.v1.spec_decode.remote.runtime.token_schema import (
        create_token_runtime_contract,
    )

    assets = token_config(tmp_path)
    parallel = ParallelConfig(tensor_parallel_size=target_tp)
    options = {
        "model": assets.speculative_config.draft_model_config.model,
        "method": "draft_model",
        "num_speculative_tokens": 3,
        "draft_tensor_parallel_size": 1,
        "execution": {
            "placement": "remote",
            "endpoints": ["unix:///tmp/token-topology.sock"],
            "transport": "inline",
        },
    }

    def construct():
        if entrypoint == "engine_args":
            config = EngineArgs(
                model=assets.model_config.model,
                tokenizer_mode="hf",
                dtype="float32",
                skip_tokenizer_init=True,
                max_model_len=64,
                max_num_seqs=4,
                max_num_batched_tokens=64,
                tensor_parallel_size=target_tp,
                enforce_eager=True,
                enable_prefix_caching=False,
                async_scheduling=False,
                speculative_config=options,
            ).create_engine_config()
        elif entrypoint == "vllm_config":
            config = replace(
                assets,
                parallel_config=parallel,
                speculative_config=SpeculativeConfig(
                    target_model_config=assets.model_config,
                    target_parallel_config=parallel,
                    **options,
                ),
            )
        else:
            config = replace(assets, parallel_config=parallel)
        return create_token_runtime_contract(config)

    if target_tp == 1:
        assert construct().schema.slots == (FeatureSlot("token_ids", "int32", ()),)
    else:
        with pytest.raises(ValueError, match="token-only.*target TP1"):
            construct()


def test_llm_rejects_remote_token_target_tp2_before_starting_workers(tmp_path):
    from tests.v1.spec_decode.remote.test_token_runtime import token_config
    from vllm import LLM

    assets = token_config(tmp_path)
    with pytest.raises(ValueError, match="token-only.*target TP1"):
        LLM(
            model=assets.model_config.model,
            tokenizer_mode="hf",
            dtype="float32",
            max_model_len=64,
            max_num_seqs=4,
            max_num_batched_tokens=64,
            tensor_parallel_size=2,
            enforce_eager=True,
            enable_prefix_caching=False,
            async_scheduling=False,
            speculative_config={
                "model": assets.speculative_config.draft_model_config.model,
                "method": "draft_model",
                "num_speculative_tokens": 3,
                "draft_tensor_parallel_size": 1,
                "execution": {
                    "placement": "remote",
                    "endpoints": ["unix:///tmp/token-topology.sock"],
                    "transport": "inline",
                },
            },
        )
