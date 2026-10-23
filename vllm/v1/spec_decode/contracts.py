# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-step drafting contracts; tensor and runner inputs remain borrowed."""

from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from vllm.v1.spec_decode.proposal import ProposalHandle, SpeculatorOutput

if TYPE_CHECKING:
    import torch

    from vllm.v1.worker.gpu.dp_utils import DPSyncState
    from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass(frozen=True)
class DraftStep:
    """Neutral row metadata copied before dispatch consumes a mutable batch."""

    request_ids: tuple[str, ...]
    query_lens: tuple[int, ...]
    is_prefilling: tuple[bool, ...]
    num_speculative_tokens: int

    @classmethod
    def from_batch(
        cls, batch: "InputBatch", num_speculative_tokens: int
    ) -> "DraftStep":
        starts = batch.query_start_loc_np
        return cls(
            tuple(batch.req_ids),
            tuple(int(starts[i + 1]) - int(starts[i]) for i in range(batch.num_reqs)),
            tuple(bool(value) for value in batch.is_prefilling_np[: batch.num_reqs]),
            num_speculative_tokens,
        )


@dataclass(frozen=True)
class LocalExecutionContext:
    """Borrowed local inputs; retain until collect, never mutate or serialize them.

    The record freezes its bindings, not tensors, mappings, or InputBatch.
    Dispatch consumes the batch metadata immediately. Native and remote handles
    retain this context until collection or explicit cancellation.
    """

    input_batch: "InputBatch"
    attn_metadata: dict[str, Any] | None
    slot_mappings: dict[str, "torch.Tensor"] | None
    last_hidden_states: "torch.Tensor | None"
    aux_hidden_states: "list[torch.Tensor] | None"
    num_sampled: "torch.Tensor"
    num_rejected: "torch.Tensor"
    last_sampled: "torch.Tensor"
    next_prefill_tokens: "torch.Tensor"
    temperature: "torch.Tensor"
    seeds: "torch.Tensor"
    dp_sync_state: "DPSyncState | None" = None
    dummy_run: bool = False
    skip_attn_for_dummy_run: bool = False
    mm_inputs: "tuple[list[torch.Tensor], torch.Tensor] | None" = None
    is_profile: bool = False

    def propose_kwargs(self, step: DraftStep) -> dict[str, Any]:
        return {**vars(self), "num_speculative_tokens": step.num_speculative_tokens}


class DraftBackend(Protocol):
    """Same-step calls made in the same order by every participating TP rank.

    Handle misuse is a caller error, including inconsistent calls across ranks.
    Remote preparation and execution failures rendezvous inside collect.
    """

    def dispatch(
        self, step: DraftStep, context: LocalExecutionContext
    ) -> ProposalHandle: ...

    def collect(self, handle: ProposalHandle) -> SpeculatorOutput: ...

    def cancel(self, handle: ProposalHandle, reason: str) -> None: ...

    def close_requests(self, req_ids: Collection[str]) -> None: ...

    def close(self) -> None: ...
