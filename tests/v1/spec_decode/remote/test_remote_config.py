# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RemoteDraftConfig field validation."""

import pytest
from pydantic import ValidationError

from vllm.config.speculative import RemoteDraftConfig


def test_defaults_are_production_safe():
    config = RemoteDraftConfig(endpoint="unix:///run/vllm/drafter.sock")
    assert config.transport == "auto"
    assert config.failure_policy == "target_only"
    assert config.request_timeout_ms > 0
    assert config.startup_timeout_ms > 0


def test_endpoint_is_required():
    with pytest.raises(ValidationError):
        RemoteDraftConfig()


def test_invalid_transport_rejected():
    with pytest.raises(ValidationError):
        RemoteDraftConfig(endpoint="unix:///x.sock", transport="carrier_pigeon")


def test_nonpositive_timeout_rejected():
    with pytest.raises(ValidationError):
        RemoteDraftConfig(endpoint="unix:///x.sock", request_timeout_ms=0)
