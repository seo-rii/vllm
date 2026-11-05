# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-phase speculator proposal results.

Splitting proposal generation into dispatch and collect lets a remote
(dedicated-GPU) speculator overlap draft compute with target-side
bookkeeping: dispatch returns a ProposalHandle that collect redeems for a
SpeculatorOutput.
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
    Native token buffers may be borrowed; consume them before the next dispatch.
    """

    token_ids: torch.Tensor
    """[batch, num_speculative_tokens] proposed draft token IDs."""
    valid_lengths: torch.Tensor
    """[batch] number of usable draft tokens per request."""

    def __post_init__(self) -> None:
        # Structural checks only; value-level checks would force a device
        # sync on the hot path.
        if self.token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must be 2-D [batch, K], got {self.token_ids.ndim}-D"
            )
        if self.valid_lengths.ndim != 1:
            raise ValueError(
                f"valid_lengths must be 1-D [batch], got {self.valid_lengths.ndim}-D"
            )
        if self.token_ids.shape[0] != self.valid_lengths.shape[0]:
            raise ValueError(
                f"batch size mismatch: token_ids has {self.token_ids.shape[0]} "
                f"rows, valid_lengths has {self.valid_lengths.shape[0]}"
            )
        for name, tensor in (
            ("token_ids", self.token_ids),
            ("valid_lengths", self.valid_lengths),
        ):
            if tensor.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must have an int32 or int64 integer dtype")
        if self.token_ids.device != self.valid_lengths.device:
            raise ValueError(
                f"token_ids ({self.token_ids.device}) and valid_lengths "
                f"({self.valid_lengths.device}) must be on the same device"
            )

    def to_legacy_tokens(self) -> torch.Tensor:
        """Encode the valid prefix as the runner's existing -1 suffix format."""
        usable = torch.arange(self.token_ids.shape[1], device=self.token_ids.device)
        return torch.where(
            usable[None, :] < self.valid_lengths[:, None], self.token_ids, -1
        )


class ProposalHandle:
    """Opaque handle returned by dispatch, redeemed by collect."""
