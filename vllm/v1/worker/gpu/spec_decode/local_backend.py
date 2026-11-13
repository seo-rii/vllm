# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in local adapter that keeps the native speculator and stream ordering."""

import itertools
from collections.abc import Collection
from dataclasses import dataclass

import torch

from vllm.v1.spec_decode.contracts import DraftStep, LocalExecutionContext
from vllm.v1.spec_decode.proposal import ProposalHandle, SpeculatorOutput
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator


@dataclass(eq=False)
class _LocalHandle(ProposalHandle):
    owner: object
    round_id: int
    step: DraftStep
    context: LocalExecutionContext | None
    tokens: torch.Tensor | None


class LocalDraftBackend:
    def __init__(self, speculator: BaseSpeculator):
        self.speculator = speculator
        self._owner = object()
        self._rounds = itertools.count()
        self._pending: _LocalHandle | None = None
        self._closed = False

    def dispatch(
        self, step: DraftStep, context: LocalExecutionContext
    ) -> ProposalHandle:
        if self._closed:
            raise RuntimeError("local draft backend is closed")
        if self._pending is not None:
            raise RuntimeError("local draft proposal is still in flight")
        if (
            not 0
            <= step.num_speculative_tokens
            <= self.speculator.num_speculative_steps
        ):
            raise ValueError("round draft budget exceeds the configured maximum")
        if step != DraftStep.from_batch(
            context.input_batch, step.num_speculative_tokens
        ):
            raise ValueError("draft step metadata no longer matches its local context")
        tokens = self.speculator.propose(**context.propose_kwargs(step))
        handle = _LocalHandle(self._owner, next(self._rounds), step, context, tokens)
        self._pending = handle
        return handle

    def collect(self, handle: ProposalHandle) -> SpeculatorOutput:
        handle = self._owned_handle(handle)
        try:
            tokens = handle.tokens
            assert tokens is not None
            valid = (
                (tokens >= 0)
                .cumprod(dim=1, dtype=torch.int32)
                .sum(dim=1, dtype=torch.int32)
            )
            valid.clamp_(max=handle.step.num_speculative_tokens)
            return SpeculatorOutput(tokens, valid)
        finally:
            self._discard_handle(handle)

    def cancel(self, handle: ProposalHandle, reason: str) -> None:
        # Native work is already queued on the original stream; discard its result.
        self._discard_handle(self._owned_handle(handle))

    def _owned_handle(self, handle: ProposalHandle) -> _LocalHandle:
        if not isinstance(handle, _LocalHandle) or handle.owner is not self._owner:
            raise RuntimeError("draft handle belongs to a different owner")
        if self._pending is not handle or handle.tokens is None:
            raise RuntimeError("stale, cancelled or already collected draft handle")
        return handle

    def _discard_handle(self, handle: _LocalHandle) -> None:
        if self._pending is handle:
            self._pending = None
        handle.context = None
        handle.tokens = None

    def close_requests(self, req_ids: Collection[str]) -> None:
        handle = self._pending
        if handle is not None and set(req_ids).intersection(handle.step.request_ids):
            self._discard_handle(handle)
        self.speculator.close_requests(req_ids)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pending is not None:
            self._discard_handle(self._pending)
        self.speculator.close()
