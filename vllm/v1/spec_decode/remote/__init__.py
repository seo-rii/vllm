# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dedicated-device placement for target-conditioned own-KV speculators.

This package contains the placement-independent building blocks for running
EAGLE / P-EAGLE / native-MTP style speculators in a standalone process on a
dedicated same-host GPU outside the target tensor-parallel group:

- capabilities: what a speculator adapter/checkpoint supports and which
  target features it needs every round.
- protocol: control-plane message types exchanged between the verifier and
  the standalone speculator server.

Modules in this package must stay importable without CUDA and without the
heavyweight parts of vLLM so that protocol and capability logic can be unit
tested in isolation.
"""

from vllm.v1.spec_decode.remote.capabilities import (
    REMOTE_DRAFT_SUPPORTED_METHODS,
    SpeculatorPlacementCapabilities,
    TargetFeatureKind,
    placement_incompatibilities,
    remote_draft_config_incompatibilities,
)
from vllm.v1.spec_decode.remote.protocol import (
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    MessageEnvelope,
    ProtocolError,
    ProtocolVersionError,
    SequenceKey,
    SpeculatorStatusCode,
    decode_envelope,
    decode_payload,
    encode_message,
)

__all__ = [
    "PROTOCOL_MAJOR",
    "PROTOCOL_MINOR",
    "REMOTE_DRAFT_SUPPORTED_METHODS",
    "MessageEnvelope",
    "ProtocolError",
    "ProtocolVersionError",
    "SequenceKey",
    "SpeculatorPlacementCapabilities",
    "SpeculatorStatusCode",
    "TargetFeatureKind",
    "decode_envelope",
    "decode_payload",
    "encode_message",
    "placement_incompatibilities",
    "remote_draft_config_incompatibilities",
]
