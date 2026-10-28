# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded full-copy prediction of one all-accepted token-draft outcome."""

import itertools
import time
import uuid
from dataclasses import dataclass, replace

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.spec_decode.remote.errors import runtime_unusable
from vllm.v1.spec_decode.remote.protocol import SequenceKey, SpeculatorStatusCode
from vllm.v1.spec_decode.remote.runtime.token_state import (
    TokenAdvanceRow,
    TokenProgress,
    TokenRowResult,
    TokenStateMachine,
)
from vllm.v1.spec_decode.remote.server import AdapterError

logger = init_logger(__name__)


@dataclass(frozen=True)
class TokenCacheOptions:
    max_branches: int = 1
    kv_budget_bytes: int = 64 << 20
    copy_chunk_blocks: int = 1

    def __post_init__(self):
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.max_branches,
                self.kv_budget_bytes,
                self.copy_chunk_blocks,
            )
        ):
            raise ValueError("outcome cache budgets must be positive integers")


@dataclass(frozen=True)
class TokenPredictionPlan:
    key: SequenceKey
    session_epoch: int
    cache_epoch: str
    build_id: int
    contract_id: tuple[str, ...]
    prefix_revision: int
    proposal_id: int
    requested_k: int
    committed: int
    materialized: int
    written_end: int
    tokens: tuple[int, ...]
    pending_last_token: tuple[int, int] | None
    source_blocks: tuple[int, ...]


@dataclass
class _Branch:
    source: TokenPredictionPlan
    progress: TokenProgress
    phase: str = "copy"
    copied: int = 0
    bonus: int | None = None
    tokens: tuple[int, ...] = ()


class TokenOutcomeCache:
    """Single-owner cache; physical blocks have one allocation authority."""

    def __init__(
        self,
        state: TokenStateMachine,
        *,
        live_slots: int,
        max_branches: int,
        max_branch_blocks: int,
        copy_chunk_blocks: int,
        contract_id: tuple[str, ...],
    ):
        if any(
            type(value) is not int or value <= 0
            for value in (
                live_slots,
                max_branches,
                max_branch_blocks,
                copy_chunk_blocks,
            )
        ):
            raise ValueError("outcome cache budgets must be positive integers")
        if not contract_id or any(not value for value in contract_id):
            raise ValueError(
                "outcome cache requires immutable model/tokenizer identity"
            )
        limit = state.executor.max_num_reqs
        if limit is not None and live_slots + max_branches > limit:
            raise ValueError(
                "outcome cache slots must be reserved before model startup"
            )
        self.state = state
        self.max_branches = max_branches
        self.max_branch_blocks = max_branch_blocks
        self.copy_chunk_blocks = copy_chunk_blocks
        self.contract_id = contract_id
        self.cache_epoch = uuid.uuid4().hex
        self._ids = itertools.count()
        self._epochs: dict[SequenceKey, int] = {}
        self._builds: dict[SequenceKey, _Branch] = {}
        self._branches = TokenStateMachine(
            state.executor,
            state.block_pool,
            max_num_reqs=max_branches,
            req_index_offset=live_slots,
            max_model_len=state.max_model_len,
            vocab_size=state.vocab_size,
            max_k=state.max_k,
        )
        self.counters = dict(
            planned=0,
            started=0,
            ready=0,
            hits=0,
            misses=0,
            cancelled=0,
            retired=0,
            failed=0,
            budget_skipped=0,
            copied_blocks=0,
            build_forwards=0,
        )

    @property
    def branch_blocks(self) -> int:
        return sum(
            len(self.state.block_pool.blocks(branch.progress.req_index))
            for branch in self._builds.values()
        )

    def warmup(self) -> None:
        count = min(
            self.copy_chunk_blocks,
            self.max_branch_blocks,
            self.state.block_pool.num_blocks - 1,
        )
        self.state.executor.copy_blocks(
            tuple((0, index) for index in range(1, count + 1))
        )

    def bind_epoch(self, key: SequenceKey, epoch: int) -> None:
        if type(epoch) is not int or epoch <= 0:
            raise ValueError("cache sequence requires a positive session epoch")
        self.close((key,))
        self._epochs[key] = epoch

    def _retire(self, key: SequenceKey, *, consumed: bool = False) -> None:
        branch = self._builds.get(key)
        if branch is not None:
            self._branches.close((key,))
            del self._builds[key]
            self.counters["retired"] += 1
            if not consumed:
                self.counters["cancelled"] += 1

    def close(self, keys: tuple[SequenceKey, ...]) -> None:
        for key in keys:
            self._retire(key)
            self._epochs.pop(key, None)

    def cancel(self, source: TokenPredictionPlan) -> None:
        branch = self._builds.get(source.key)
        if branch is not None and branch.source == source:
            self._retire(source.key)

    def _make_room(self, required: int) -> None:
        for key in tuple(self._builds):
            if self.state.block_pool.num_free_blocks >= required:
                break
            self._retire(key)

    def prefill(
        self, key: SequenceKey, offset: int, tokens: tuple[int, ...], *, is_final: bool
    ) -> None:
        current = self.state.validate_prefill(key, offset, tokens)
        required = cdiv(offset + len(tokens), self.state.block_pool.block_size) - len(
            self.state.block_pool.blocks(current.req_index)
        )
        self._make_room(max(required, 0))
        self.state.prefill(key, offset, tokens, is_final=is_final)

    def _matches(self, source: TokenPredictionPlan) -> bool:
        current = self.state.sequences.get(source.key)
        return (
            current is not None
            and current.ready
            and self._epochs.get(source.key) == source.session_epoch
            and source.cache_epoch == self.cache_epoch
            and source.contract_id == self.contract_id
            and current.prefix_revision == source.prefix_revision
            and current.proposal_id == source.proposal_id
            and current.committed == source.committed
            and current.materialized == source.materialized
            and current.speculative_written_end == source.written_end
            and current.proposal_tokens == source.tokens
            and current.pending_last_token == source.pending_last_token
            and self.state.block_pool.blocks(current.req_index) == source.source_blocks
        )

    def _plan(self, key: SequenceKey, requested_k: int) -> TokenPredictionPlan | None:
        state = self.state.sequences[key]
        epoch = self._epochs.get(key)
        # Explicit K0 and an empty proposal are healthy on-demand rounds.
        if (
            epoch is None
            or requested_k == 0
            or not state.proposal_tokens
            or state.proposal_id is None
            or state.committed + len(state.proposal_tokens) + 1
            >= self.state.max_model_len
        ):
            return None
        self.counters["planned"] += 1
        return TokenPredictionPlan(
            key,
            epoch,
            self.cache_epoch,
            next(self._ids),
            self.contract_id,
            state.prefix_revision,
            state.proposal_id,
            requested_k,
            state.committed,
            state.materialized,
            state.speculative_written_end,
            state.proposal_tokens,
            state.pending_last_token,
            self.state.block_pool.blocks(state.req_index),
        )

    def _lookup(
        self, batch_id: int, row: TokenAdvanceRow, k: int
    ) -> TokenRowResult | None:
        current, committed, valid_k = self.state.validate_advance(batch_id, row, k)
        branch = self._builds.get(row.key)
        if branch is None or branch.phase != "ready":
            return None
        source = branch.source
        if (
            not self._matches(source)
            or k != source.requested_k
            or row.offered != len(source.tokens)
            or row.accepted != len(source.tokens)
            or row.delta != source.tokens + (branch.bonus,)
            or len(branch.tokens) != valid_k
            or branch.progress.committed != committed
        ):
            return None
        # Prepare metadata before the ownership transfer. Reuse the live slot;
        # the retired branch slot now owns and returns the old live pages.
        try:
            promoted = replace(
                branch.progress,
                req_index=current.req_index,
                proposal_id=batch_id,
                prefix_revision=current.prefix_revision + 1,
            )
            result = TokenRowResult(branch.tokens, committed, batch_id, origin="cache")
        except MemoryError:
            return None
        self.state.block_pool.swap(current.req_index, branch.progress.req_index)
        self.state.sequences[row.key] = promoted
        self._retire(row.key, consumed=True)
        self.counters["hits"] += 1
        logger.info(
            "Token outcome cache hit: sequence=%s generation=%s proposal=%s hits=%s",
            row.key.sequence_id,
            row.key.generation,
            batch_id,
            self.counters["hits"],
        )
        return result

    def advance_many(
        self,
        updates: tuple[tuple[int, TokenAdvanceRow, int], ...],
        *,
        deadlines: tuple[float | None, ...] | None = None,
    ) -> tuple[TokenRowResult | Exception, ...]:
        if deadlines is None:
            deadlines = (None,) * len(updates)
        if len(deadlines) != len(updates):
            raise ValueError("cache update deadlines do not match the rows")
        keys = [row.key for _, row, _ in updates]
        repeated = {key for key in keys if keys.count(key) > 1}
        outcomes: list[TokenRowResult | Exception | None] = [None] * len(updates)
        misses, indices, budgets = [], [], []
        required = 0
        for index, (batch_id, row, k) in enumerate(updates):
            try:
                if row.key in repeated:
                    raise AdapterError(
                        SpeculatorStatusCode.SEQUENCE_DESYNC,
                        "sequence appears more than once in the token batch",
                    )
                deadline = deadlines[index]
                if deadline is not None and time.monotonic() >= deadline:
                    raise AdapterError(
                        SpeculatorStatusCode.TIMEOUT, "token execution deadline expired"
                    )
                cached = self._lookup(batch_id, row, k)
                if cached is not None:
                    outcomes[index] = cached
                    continue
                self._retire(row.key)
                current, committed, valid_k = self.state.validate_advance(
                    batch_id, row, k
                )
                needed = cdiv(
                    committed + max(valid_k - 1, 0), self.state.block_pool.block_size
                ) - len(self.state.block_pool.blocks(current.req_index))
                required += max(needed, 0)
                misses.append((batch_id, row, k))
                indices.append(index)
                budgets.append(deadlines[index])
                self.counters["misses"] += 1
            except Exception as error:
                outcomes[index] = error
        self._make_room(required)
        for index, outcome in zip(
            indices, self.state.advance_many(tuple(misses), deadlines=tuple(budgets))
        ):
            outcomes[index] = outcome
        for index, (_, row, k) in enumerate(updates):
            result = outcomes[index]
            if isinstance(result, TokenRowResult):
                try:
                    outcomes[index] = replace(result, prediction=self._plan(row.key, k))
                except MemoryError:
                    self.counters["failed"] += 1
                    logger.warning(
                        "Token outcome prediction metadata allocation failed"
                    )
        assert all(outcome is not None for outcome in outcomes)
        return tuple(outcomes)  # type: ignore[arg-type]

    def _start(self, source: TokenPredictionPlan) -> _Branch | None:
        if not self._matches(source):
            return None
        old = self._builds.get(source.key)
        if old is not None:
            if old.source == source:
                return old
            self._retire(source.key)
        committed = source.committed + len(source.tokens) + 1
        k = min(source.requested_k, self.state.max_model_len - committed)
        needed = committed + max(k - 1, 0)
        blocks = cdiv(needed, self.state.block_pool.block_size)
        if (
            blocks + self.branch_blocks > self.max_branch_blocks
            or blocks > self.state.block_pool.num_free_blocks
        ):
            self.counters["budget_skipped"] += 1
            return None
        try:
            self._branches.open(source.key)
        except AdapterError:
            self.counters["budget_skipped"] += 1
            return None
        progress = self._branches.sequences[source.key]
        try:
            self.state.block_pool.ensure(progress.req_index, needed)
            progress = replace(
                self.state.sequences[source.key],
                req_index=progress.req_index,
                reserved_end=blocks * self.state.block_pool.block_size,
            )
            branch = _Branch(source, progress)
            self._branches.sequences[source.key] = progress
            self._builds[source.key] = branch
        except BaseException:
            self._branches.close((source.key,))
            raise
        self.counters["started"] += 1
        return branch

    def step(self, source: TokenPredictionPlan) -> bool:
        """Complete one copy chunk or one forward; True requests another phase."""
        branch = self._start(source)
        if branch is None or branch.phase == "ready":
            return False
        if not self._matches(source):
            self._retire(source.key)
            return False
        try:
            blocks = self.state.block_pool.blocks(branch.progress.req_index)
            if branch.phase == "copy":
                count = cdiv(source.written_end, self.state.block_pool.block_size)
                end = min(count, branch.copied + self.copy_chunk_blocks)
                copies = tuple(
                    zip(
                        source.source_blocks[branch.copied : end],
                        blocks[branch.copied : end],
                    )
                )
                self.state.executor.copy_blocks(copies)
                self.counters["copied_blocks"] += len(copies)
                branch.copied = end
                if end == count:
                    branch.phase = "bonus"
                return True
            if branch.phase == "bonus":
                start = source.committed + len(source.tokens) - 1
                token = source.tokens[-1]
            elif not branch.tokens:
                start = source.committed + len(source.tokens)
                assert branch.bonus is not None
                token = branch.bonus
            else:
                start = branch.progress.speculative_written_end
                token = branch.tokens[-1]
            sampled = self.state.executor.forward(
                source.key,
                branch.progress.req_index,
                start,
                (token,),
                blocks,
                sample=True,
            )
            self.counters["build_forwards"] += 1
            if type(sampled) is not int or not 0 <= sampled < self.state.vocab_size:
                raise ValueError("cache build returned an invalid token")
            if branch.phase == "bonus":
                branch.bonus = sampled
                branch.phase = "draft"
                return True
            committed = source.committed + len(source.tokens) + 1
            branch.tokens += (sampled,)
            progress = branch.progress
            progress.committed = progress.materialized = committed
            progress.speculative_written_end = start + 1
            progress.proposal_tokens = branch.tokens
            progress.pending_last_token = (start + 1, sampled)
            progress.ready = True
            k = min(source.requested_k, self.state.max_model_len - committed)
            if len(branch.tokens) == k:
                branch.phase = "ready"
                self.counters["ready"] += 1
                logger.info(
                    "Token outcome cache ready: sequence=%s generation=%s source=%s "
                    "build=%s branch_blocks=%s ready=%s",
                    source.key.sequence_id,
                    source.key.generation,
                    source.proposal_id,
                    source.build_id,
                    self.branch_blocks,
                    self.counters["ready"],
                )
                return False
            return True
        except Exception as error:
            if runtime_unusable(error):
                raise
            self.counters["failed"] += 1
            logger.warning("Token outcome prediction discarded: %s", error)
            self._retire(source.key)
            return False
