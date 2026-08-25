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
