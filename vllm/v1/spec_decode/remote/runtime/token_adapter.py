# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-only adapter, deliberately unavailable through the v2 proposal API."""

from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.runtime.schema import RuntimeContract
from vllm.v1.spec_decode.remote.runtime.token_state import TokenStateMachine
from vllm.v1.spec_decode.remote.server import PrefillInputs, StandaloneDraftAdapter
from vllm.v1.spec_decode.remote.transport import DataFrame, frame_to_ints


class TokenOnlyDraftAdapter(StandaloneDraftAdapter):
    def __init__(self, state: TokenStateMachine, contract: RuntimeContract) -> None:
        self.state = state
        self.contract = contract

    def capabilities(self):
        return self.contract.capabilities

    def feature_schema(self):
        return self.contract.schema

    def supported_methods(self):
        return ("draft_model",)

    def draft_checkpoint_fingerprint(self):
        return self.contract.draft_fingerprint

    def warmup(self) -> None:
        self.state.executor.warmup()

    def close(self) -> None:
        self.state.executor.close()

    def open_sequence(self, key: SequenceKey) -> None:
        self.state.open(key)

    def prefill(self, key: SequenceKey, inputs: PrefillInputs) -> None:
        frame = inputs.features[0]
        if not isinstance(frame, DataFrame):
            raise ValueError("missing prompt tokens")
        self.state.prefill(
            key, inputs.offset, tuple(frame_to_ints(frame)), is_final=inputs.is_final
        )

    def advance_and_propose(self, batch):
        raise ValueError(
            "token-only drafting requires the explicit v3 proposal protocol"
        )

    def close_sequences(self, keys: tuple[SequenceKey, ...]) -> None:
        self.state.close(keys)
