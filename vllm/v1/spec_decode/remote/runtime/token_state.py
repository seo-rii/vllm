# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact logical/materialized prefix state for an independent token-only LM."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from vllm.v1.spec_decode.remote.protocol import SequenceKey, SpeculatorStatusCode
from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool, DraftKVExhausted
from vllm.v1.spec_decode.remote.server import AdapterError


@dataclass(frozen=True)
class TokenAdvanceRow:
    key: SequenceKey
    base_length: int
    previous_proposal_id: int | None
    offered: int
    accepted: int
    delta: tuple[int, ...]


@dataclass
class TokenProgress:
    req_index: int
    committed: int = 0
    materialized: int = 0
    speculative_written_end: int = 0
    reserved_end: int = 0
    proposal_id: int | None = None
    proposal_tokens: tuple[int, ...] = ()
    pending_last_token: tuple[int, int] | None = None
    ready: bool = False


@dataclass(frozen=True)
class TokenRowResult:
    token_ids: tuple[int, ...]
    committed: int
    proposal_id: int


class TokenStepExecutor(ABC):
    def warmup(self) -> None:  # noqa: B027
        """Finish device initialization before the listener becomes ready."""

    @abstractmethod
    def forward(
        self,
        key: SequenceKey,
        req_index: int,
        start: int,
        tokens: tuple[int, ...],
        blocks: tuple[int, ...],
        *,
        sample: bool,
    ) -> int | None:
        """Complete a real token forward; optionally sample its last position."""


class TokenStateMachine:
    """Bounded sequence slots and own-KV; no outcome cache or route pool."""

    def __init__(
        self,
        executor: TokenStepExecutor,
        block_pool: DraftBlockPool,
        *,
        max_num_reqs: int,
        max_model_len: int,
        vocab_size: int,
        max_k: int,
    ) -> None:
        self.executor = executor
        self.block_pool = block_pool
        self.max_model_len = max_model_len
        self.vocab_size = vocab_size
        self.max_k = max_k
        self.sequences: dict[SequenceKey, TokenProgress] = {}
        # Retain capacity so returning a slot cannot grow a Python list.
        self._free = list(range(max_num_reqs - 1, -1, -1))
        self._num_free = max_num_reqs

    def open(self, key: SequenceKey) -> None:
        if key in self.sequences:
            raise AdapterError(SpeculatorStatusCode.SEQUENCE_DESYNC, "already open")
        if not self._num_free:
            raise AdapterError(SpeculatorStatusCode.QUEUE_FULL, "draft slots exhausted")
        remaining = self._num_free - 1
        self.sequences[key] = TokenProgress(self._free[remaining])
        self._num_free = remaining

    def close(self, keys: tuple[SequenceKey, ...]) -> None:
        for key in keys:
            state = self.sequences.get(key)
            if state is not None:
                returned = self._num_free + 1
                self._free[self._num_free] = state.req_index
                self.block_pool.release(state.req_index)
                del self.sequences[key]
                self._num_free = returned

    def _tokens(self, tokens: tuple[int, ...]) -> None:
        if not tokens or any(
            type(t) is not int or not 0 <= t < self.vocab_size for t in tokens
        ):
            raise AdapterError(
                SpeculatorStatusCode.PROTOCOL_ERROR, "invalid token delta"
            )

    def _reserve(self, state: TokenProgress, end: int) -> tuple[int, ...]:
        if end > self.max_model_len:
            raise AdapterError(
                SpeculatorStatusCode.OUT_OF_MEMORY, "context limit exceeded"
            )
        try:
            self.block_pool.ensure(state.req_index, end)
        except DraftKVExhausted as error:
            raise AdapterError(
                SpeculatorStatusCode.OUT_OF_MEMORY, str(error)
            ) from error
        blocks = self.block_pool.blocks(state.req_index)
        state.reserved_end = len(blocks) * self.block_pool.block_size
        return blocks

    def prefill(
        self, key: SequenceKey, offset: int, tokens: tuple[int, ...], *, is_final: bool
    ) -> None:
        state = self.sequences[key]
        self._tokens(tokens)
        if state.ready or offset != state.committed:
            raise AdapterError(
                SpeculatorStatusCode.SEQUENCE_DESYNC, "prefill offset/finality mismatch"
            )
        blocks = self._reserve(state, offset + len(tokens))
        self.executor.forward(
            key, state.req_index, offset, tokens, blocks, sample=False
        )
        state.committed += len(tokens)
        state.materialized = state.speculative_written_end = state.committed
        state.ready = is_final

    def advance(self, batch_id: int, row: TokenAdvanceRow, k: int) -> TokenRowResult:
        state = self.sequences[row.key]
        self._tokens(row.delta)
        if (
            not state.ready
            or row.base_length != state.committed
            or row.previous_proposal_id != state.proposal_id
        ):
            raise AdapterError(
                SpeculatorStatusCode.SEQUENCE_DESYNC,
                "proposal identity or committed prefix mismatch",
            )
        if state.proposal_id is not None and batch_id <= state.proposal_id:
            raise AdapterError(
                SpeculatorStatusCode.ROUND_MISMATCH, "proposal did not advance"
            )
        if not 0 <= row.accepted <= row.offered <= len(state.proposal_tokens):
            raise AdapterError(
                SpeculatorStatusCode.SEQUENCE_DESYNC,
                "verification exceeds previous proposal",
            )
        if (
            len(row.delta) <= row.accepted
            or row.delta[: row.accepted] != state.proposal_tokens[: row.accepted]
        ):
            raise AdapterError(
                SpeculatorStatusCode.SEQUENCE_DESYNC, "accepted token prefix mismatch"
            )
        if state.proposal_tokens and len(row.delta) != row.accepted + 1:
            raise AdapterError(
                SpeculatorStatusCode.SEQUENCE_DESYNC, "unexpected authoritative gap"
            )
        if not 0 <= k <= self.max_k:
            raise AdapterError(
                SpeculatorStatusCode.PROTOCOL_ERROR, "invalid draft budget"
            )
        committed = state.committed + len(row.delta)
        valid_k = min(k, max(0, self.max_model_len - committed))
        blocks = self._reserve(state, committed + max(valid_k - 1, 0))
        keep = min(state.speculative_written_end, state.committed + row.accepted)
        missing = row.delta[keep - state.committed :]
        token = self.executor.forward(
            row.key, state.req_index, keep, missing, blocks, sample=valid_k > 0
        )
        state.committed = state.materialized = state.speculative_written_end = committed
        drafts: list[int] = []
        for index in range(valid_k):
            if token is None or not 0 <= token < self.vocab_size:
                raise AdapterError(
                    SpeculatorStatusCode.INTERNAL_ERROR, "invalid sampled token"
                )
            drafts.append(token)
            if index + 1 < valid_k:
                token = self.executor.forward(
                    row.key,
                    state.req_index,
                    state.speculative_written_end,
                    (token,),
                    blocks,
                    sample=True,
                )
                state.speculative_written_end += 1
        state.proposal_tokens = tuple(drafts)
        state.proposal_id = batch_id
        state.pending_last_token = (
            (state.speculative_written_end, drafts[-1]) if drafts else None
        )
        return TokenRowResult(tuple(drafts), committed, batch_id)
