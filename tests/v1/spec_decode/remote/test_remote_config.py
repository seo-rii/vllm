# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RemoteDraftConfig field validation."""

import pytest
from pydantic import ValidationError

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


def test_remote_cost_options_preserve_typed_configuration_and_reject_local():
    from vllm.config.speculative import RemoteDraftCostConfig

    execution = SpeculativeExecutionConfig(
        placement="remote",
        endpoints=("unix:///draft.sock",),
        cost_aware={"min_samples": 3, "min_speedup": 1.2},
    )
    assert execution.remote_config().cost_aware == RemoteDraftCostConfig(
        min_samples=3,
        min_speedup=1.2,
    )
    with pytest.raises(ValidationError, match="local execution"):
        SpeculativeExecutionConfig(cost_aware={})


@pytest.mark.parametrize(
    "options",
    [
        {"min_samples": 0},
        {"min_samples": 65},
        {"ema_alpha": 0},
        {"ema_alpha": float("nan")},
        {"min_speedup": 0.9},
        {"switch_margin": -1},
        {"probe_interval": 0},
    ],
)
def test_invalid_remote_cost_options_are_rejected(options):
    from vllm.config.speculative import RemoteDraftCostConfig

    with pytest.raises(ValidationError):
        RemoteDraftCostConfig(**options)


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
        ({"draft_tensor_parallel_size": 2}, "draft_tensor_parallel_size"),
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
