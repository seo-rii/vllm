# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded computed-token prefixes for replaying an independent recurrent LM."""

from dataclasses import dataclass

from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.runtime.token_state import TokenForwardRow


@dataclass(frozen=True)
class _ComputedPrefix:
    req_index: int
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class RecurrentTokenUpdate:
    key: SequenceKey
    req_index: int
    tokens: tuple[int, ...]
    previous: _ComputedPrefix


class RecurrentTokenHistory:
    """Stage prefix rewinds and publish them only after completed execution.

    A computed prefix can contain speculative tokens. Its caller validates the
    authoritative outcome before supplying the correction's start and tokens.
    Request-slot ownership also covers a prepared first forward, so closing it
    fences its receipt without retaining identities of previously closed keys.
    """

    def __init__(
        self, *, max_num_reqs: int, max_model_len: int, vocab_size: int
    ) -> None:
        if min(max_num_reqs, max_model_len, vocab_size) <= 0:
            raise ValueError("recurrent history budgets must be positive")
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.vocab_size = vocab_size
        self._prefixes: dict[SequenceKey, _ComputedPrefix] = {}
        self._owners: list[SequenceKey | None] = [None] * max_num_reqs

    @property
    def num_sequences(self) -> int:
        return len(self._prefixes)

    @property
    def token_count(self) -> int:
        return sum(len(prefix.tokens) for prefix in self._prefixes.values())

    def tokens(self, key: SequenceKey) -> tuple[int, ...]:
        return self._prefixes[key].tokens

    def prepare(self, row: TokenForwardRow) -> RecurrentTokenUpdate:
        if (
            type(row.req_index) is not int
            or not 0 <= row.req_index < self.max_num_reqs
            or type(row.start) is not int
            or row.start < 0
            or not row.tokens
            or any(
                type(token) is not int or not 0 <= token < self.vocab_size
                for token in row.tokens
            )
        ):
            raise ValueError("invalid recurrent replay row")
        previous = self._prefixes.get(row.key)
        if previous is None:
            if row.start != 0 or self._owners[row.req_index] is not None:
                raise ValueError("recurrent request has no matching computed prefix")
            previous = _ComputedPrefix(row.req_index, ())
        elif previous.req_index != row.req_index:
            raise ValueError("recurrent request changed its execution slot")
        if row.start > len(previous.tokens):
            raise ValueError("recurrent correction skips uncomputed tokens")
        if row.start + len(row.tokens) > self.max_model_len:
            raise ValueError("recurrent replay exceeds the model length")
        tokens = previous.tokens[: row.start] + row.tokens
        update = RecurrentTokenUpdate(row.key, row.req_index, tokens, previous)
        if row.key not in self._prefixes:
            self._prefixes[row.key] = previous
            self._owners[row.req_index] = row.key
        return update

    def commit(self, update: RecurrentTokenUpdate) -> None:
        if (
            self._prefixes.get(update.key) is not update.previous
            or self._owners[update.req_index] != update.key
        ):
            raise ValueError("stale recurrent replay completion")
        self._prefixes[update.key] = _ComputedPrefix(update.req_index, update.tokens)

    def close_sequences(self, keys: tuple[SequenceKey, ...]) -> None:
        for key in keys:
            previous = self._prefixes.pop(key, None)
            if previous is not None:
                self._owners[previous.req_index] = None
