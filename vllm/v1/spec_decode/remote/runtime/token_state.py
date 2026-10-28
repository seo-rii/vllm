# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact logical/materialized prefix state for an independent token-only LM."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from vllm.v1.spec_decode.remote.errors import runtime_unusable
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


@dataclass(frozen=True)
class TokenForwardRow:
    key: SequenceKey
    req_index: int
    start: int
    tokens: tuple[int, ...]
    blocks: tuple[int, ...]
    sample: bool


@dataclass
class _TokenUpdate:
    batch_id: int
    row: TokenAdvanceRow
    state: TokenProgress
    committed: int
    valid_k: int
    blocks: tuple[int, ...]
    written_end: int
    correction: TokenForwardRow
    drafts: list[int]
    started: bool = False


class TokenStepExecutor(ABC):
    def warmup(self) -> None:  # noqa: B027
        """Finish device initialization before the listener becomes ready."""

    def close_sequences(self, keys: tuple[SequenceKey, ...]) -> None:  # noqa: B027
        """Retire executor-owned state before request slots are reused."""

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

    @property
    def max_num_tokens(self) -> int | None:
        return None

    @property
    def max_num_reqs(self) -> int | None:
        return None

    def forward_batch(
        self, rows: tuple[TokenForwardRow, ...]
    ) -> tuple[int | None | Exception, ...]:
        """Serial executors can report row errors; GPU forwards fail as a group."""
        results: list[int | None | Exception] = []
        for row in rows:
            try:
                results.append(
                    self.forward(
                        row.key,
                        row.req_index,
                        row.start,
                        row.tokens,
                        row.blocks,
                        sample=row.sample,
                    )
                )
            except Exception as error:
                results.append(error)
                if runtime_unusable(error):
                    results.extend([error] * (len(rows) - len(results)))
                    break
        return tuple(results)


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
                self.executor.close_sequences((key,))
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

    def _prepare_advance(
        self, batch_id: int, row: TokenAdvanceRow, k: int
    ) -> _TokenUpdate:
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
        return _TokenUpdate(
            batch_id,
            row,
            state,
            committed,
            valid_k,
            blocks,
            committed,
            TokenForwardRow(
                row.key, state.req_index, keep, missing, blocks, sample=valid_k > 0
            ),
            [],
        )

    def advance_many(
        self,
        updates: tuple[tuple[int, TokenAdvanceRow, int], ...],
        *,
        deadlines: tuple[float | None, ...] | None = None,
    ) -> tuple[TokenRowResult | Exception, ...]:
        """Validate independently, then batch correction and each draft depth.

        A failed shared forward invalidates every participating row. No work
        with uncertain KV contents is retried. Deadlines bound submission;
        they cannot interrupt a CUDA kernel that is already executing.
        """
        if deadlines is None:
            deadlines = (None,) * len(updates)
        if len(deadlines) != len(updates):
            raise ValueError("token update deadlines do not match the rows")
        keys = [row.key for _, row, _ in updates]
        repeated = {key for key in keys if keys.count(key) > 1}
        results: list[TokenRowResult | Exception | None] = [None] * len(updates)
        prepared: dict[int, _TokenUpdate] = {}

        def expired(index: int) -> bool:
            deadline = deadlines[index]
            if deadline is not None and time.monotonic() >= deadline:
                results[index] = AdapterError(
                    SpeculatorStatusCode.TIMEOUT, "token execution deadline expired"
                )
                return True
            return False

        for index, (batch_id, row, k) in enumerate(updates):
            if expired(index):
                continue
            try:
                if row.key in repeated:
                    raise AdapterError(
                        SpeculatorStatusCode.SEQUENCE_DESYNC,
                        "sequence appears more than once in the token batch",
                    )
                update = self._prepare_advance(batch_id, row, k)
                limit = self.executor.max_num_tokens
                if limit is not None and len(update.correction.tokens) > limit:
                    raise AdapterError(
                        SpeculatorStatusCode.OUT_OF_MEMORY,
                        "token correction exceeds the executor token budget",
                    )
                prepared[index] = update
            except Exception as error:
                results[index] = error

        def execute(rows: list[tuple[int, TokenForwardRow]]) -> dict[int, int | None]:
            values: dict[int, int | None] = {}
            groups: list[list[tuple[int, TokenForwardRow]]] = [[]]
            count = 0
            for index, row in rows:
                if expired(index):
                    continue
                limit = self.executor.max_num_tokens
                row_limit = self.executor.max_num_reqs
                if (limit is not None and count + len(row.tokens) > limit) or (
                    row_limit is not None and len(groups[-1]) >= row_limit
                ):
                    groups.append([])
                    count = 0
                groups[-1].append((index, row))
                count += len(row.tokens)
            for group in groups:
                active = [(index, row) for index, row in group if not expired(index)]
                if not active:
                    continue
                try:
                    for index, _ in active:
                        prepared[index].started = True
                    sampled = self.executor.forward_batch(
                        tuple(row for _, row in active)
                    )
                    if len(sampled) != len(active):
                        raise RuntimeError(
                            "token executor returned a different row count"
                        )
                    fatal = None
                    for (index, _), token in zip(active, sampled):
                        if isinstance(token, Exception):
                            results[index] = token
                            if runtime_unusable(token):
                                fatal = token
                        else:
                            values[index] = token
                    if fatal is not None:
                        for index in prepared:
                            if results[index] is None:
                                results[index] = fatal
                        break

                except Exception as error:
                    for index, _ in active:
                        results[index] = error
                    # Context-fatal failures prohibit further CUDA calls. Even
                    # recoverable failures leave this group's KV untrustworthy.
                    if runtime_unusable(error):
                        for index in prepared:
                            if results[index] is None:
                                results[index] = error
                        break
            return values

        values = execute([(i, update.correction) for i, update in prepared.items()])
        for depth in range(max((u.valid_k for u in prepared.values()), default=0)):
            following = []
            for index, update in prepared.items():
                if results[index] is not None or depth >= update.valid_k:
                    continue
                token = values.get(index)
                if type(token) is not int or not 0 <= token < self.vocab_size:
                    results[index] = AdapterError(
                        SpeculatorStatusCode.INTERNAL_ERROR, "invalid sampled token"
                    )
                    continue
                update.drafts.append(token)
                if depth + 1 < update.valid_k:
                    following.append(
                        (
                            index,
                            TokenForwardRow(
                                update.row.key,
                                update.state.req_index,
                                update.written_end,
                                (token,),
                                update.blocks,
                                sample=True,
                            ),
                        )
                    )
                    update.written_end += 1
            values = execute(following)
        for index, update in prepared.items():
            if results[index] is not None:
                if update.started:
                    update.state.ready = False
                continue
            state = update.state
            state.committed = state.materialized = update.committed
            state.speculative_written_end = update.written_end
            state.proposal_tokens = tuple(update.drafts)
            state.proposal_id = update.batch_id
            state.pending_last_token = (
                (update.written_end, update.drafts[-1]) if update.drafts else None
            )
            results[index] = TokenRowResult(
                tuple(update.drafts), update.committed, update.batch_id
            )
        assert all(result is not None for result in results)
        return tuple(results)  # type: ignore[arg-type]

    def advance(self, batch_id: int, row: TokenAdvanceRow, k: int) -> TokenRowResult:
        result = self.advance_many(((batch_id, row, k),))[0]
        if isinstance(result, Exception):
            raise result
        return result
