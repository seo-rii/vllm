# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-phase speculator proposal API.

Splitting proposal generation into dispatch and collect lets a remote
(dedicated-GPU) speculator overlap draft compute with target-side
bookkeeping. Local speculators keep their synchronous behavior: the
default dispatch runs the proposal immediately and returns an
ImmediateProposalHandle whose collect is free.
"""

from dataclasses import dataclass

import torch


@dataclass
class SpeculatorOutput:
    """Result of one proposal round.

    ``valid_lengths`` decouples failure from token values: row ``i``
    contributes ``token_ids[i, :valid_lengths[i]]`` draft tokens, and
    ``valid_lengths[i] == 0`` means the request continues with target-only
    decoding this round. Failures are never encoded as sentinel token IDs.
    """

    token_ids: torch.Tensor
    """[batch, num_speculative_tokens] proposed draft token IDs."""
    valid_lengths: torch.Tensor
    """[batch] number of usable draft tokens per request."""

    @staticmethod
    def from_dense(token_ids: torch.Tensor) -> "SpeculatorOutput":
        """Wrap a fully-valid [batch, K] proposal tensor."""
        batch_size, num_tokens = token_ids.shape
        valid_lengths = torch.full(
            (batch_size,),
            num_tokens,
            dtype=torch.int32,
            device=token_ids.device,
        )
        return SpeculatorOutput(token_ids, valid_lengths)

    @staticmethod
    def target_only(
        batch_size: int,
        num_speculative_tokens: int,
        device: torch.device,
    ) -> "SpeculatorOutput":
        """An all-failed round: every request decodes target-only."""
        return SpeculatorOutput(
            token_ids=torch.zeros(
                batch_size,
                num_speculative_tokens,
                dtype=torch.int64,
                device=device,
            ),
            valid_lengths=torch.zeros(
                batch_size, dtype=torch.int32, device=device
            ),
        )


class ProposalHandle:
    """Opaque handle returned by dispatch, redeemed by collect."""


@dataclass
class ImmediateProposalHandle(ProposalHandle):
    """Handle for a proposal that was computed synchronously at dispatch."""

    output: SpeculatorOutput
