# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-phase proposal API: valid_lengths semantics and the synchronous
default dispatch/collect path."""

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
