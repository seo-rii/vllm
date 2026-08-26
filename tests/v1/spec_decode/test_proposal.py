# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-phase proposal API: valid_lengths semantics, output invariants,
and the synchronous default dispatch/collect path."""

import pytest
import torch

from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.proposal import (
    ImmediateProposalHandle,
    ProposalHandle,
    SpeculatorOutput,
)


def test_from_dense_marks_all_rows_fully_valid():
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    output = SpeculatorOutput.from_dense(token_ids)
    assert output.token_ids is token_ids
    assert output.valid_lengths.tolist() == [3, 3]


def test_target_only_zeroes_valid_lengths_not_tokens_only():
    output = SpeculatorOutput.target_only(2, 3, torch.device("cpu"))
    assert output.token_ids.shape == (2, 3)
    assert output.valid_lengths.tolist() == [0, 0]


def test_batch_size_mismatch_rejected():
    with pytest.raises(ValueError, match="batch size mismatch"):
        SpeculatorOutput(
            token_ids=torch.zeros(2, 3, dtype=torch.long),
            valid_lengths=torch.zeros(3, dtype=torch.int32),
        )


def test_wrong_rank_rejected():
    with pytest.raises(ValueError, match="2-D"):
        SpeculatorOutput(
            token_ids=torch.zeros(6, dtype=torch.long),
            valid_lengths=torch.zeros(6, dtype=torch.int32),
        )


def test_non_integer_dtype_rejected():
    with pytest.raises(ValueError, match="integer dtype"):
        SpeculatorOutput(
            token_ids=torch.zeros(2, 3),
            valid_lengths=torch.zeros(2, dtype=torch.int32),
        )


def test_default_dispatch_collect_is_synchronous_propose():
    proposer = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
    draft = torch.tensor([[7, 8], [9, 10]])
    seen_kwargs = {}

    def fake_propose(**kwargs):
        seen_kwargs.update(kwargs)
        return draft

    proposer.propose = fake_propose

    handle = proposer.dispatch_proposal(num_speculative_tokens=2)
    assert isinstance(handle, ProposalHandle)
    assert isinstance(handle, ImmediateProposalHandle)
    assert seen_kwargs == {"num_speculative_tokens": 2}

    output = proposer.collect_proposal(handle)
    assert torch.equal(output.token_ids, draft)
    assert output.valid_lengths.tolist() == [2, 2]
