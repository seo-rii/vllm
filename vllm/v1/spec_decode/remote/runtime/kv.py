# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft-only KV block pool.

The standalone process owns its draft KV cache outright: blocks are handed
out per request slot and returned when the sequence closes. Running out of
blocks fails the request that needed them; resident sequences are never
evicted to make room.
"""

import numpy as np

from vllm.utils.math_utils import cdiv

NULL_BLOCK_ID = 0
"""Reserved block that padded tokens write to; never handed out."""


class DraftKVExhausted(Exception):
    """The pool cannot grow a sequence to the requested length."""


class DraftBlockPool:
    """Fixed pool of page leases, with shared immutable prefix blocks."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks < 2:
            raise ValueError("a draft KV pool needs the null block plus one")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        # Only the prefix is free. Keep storage allocated so cleanup never
        # needs to grow a list while recovering from an allocation failure.
        self._free = list(range(num_blocks - 1, NULL_BLOCK_ID, -1))
        self._num_free = len(self._free)
        self._owned: dict[int, list[int]] = {}
        self._references = [0] * num_blocks

    @property
    def num_free_blocks(self) -> int:
        return self._num_free

    def blocks(self, req_index: int) -> tuple[int, ...]:
        return tuple(self._owned.get(req_index, ()))

    def exclusive_blocks(self, req_index: int) -> int:
        return sum(
            self._references[block] == 1 for block in self._owned.get(req_index, ())
        )

    def ensure(self, req_index: int, num_positions: int) -> list[int]:
        """Grow ``req_index`` to cover positions ``[0, num_positions)``.

        Returns the block ids appended, in order. Allocation is all or
        nothing: on DraftKVExhausted the slot keeps its previous blocks.
        """
        owned = self._owned.get(req_index, [])
        missing = cdiv(num_positions, self.block_size) - len(owned)
        if missing <= 0:
            return []
        if missing > self._num_free:
            raise DraftKVExhausted(
                f"need {missing} more draft KV blocks, {self._num_free} free"
            )
        # Prepare allocation-backed metadata before consuming free capacity.
        remaining = self._num_free - missing
        new_blocks = self._free[remaining : self._num_free][::-1]
        references = iter(new_blocks)
        self._owned[req_index] = owned + new_blocks
        for block in references:
            self._references[block] = 1
        self._num_free = remaining
        return new_blocks

    def fork_prefix(
        self,
        source: int,
        destination: int,
        shared_blocks: int,
        num_positions: int,
    ) -> list[int]:
        """Share immutable whole prefix pages and reserve a private writable tail.

        The caller must never write the shared prefix through either lease.
        Metadata and free capacity are unchanged if preparation fails.
        """
        owned = self._owned[source]
        count = cdiv(num_positions, self.block_size)
        if (
            source == destination
            or destination in self._owned
            or not 0 <= shared_blocks <= min(len(owned), count)
        ):
            raise ValueError("invalid draft KV prefix fork")
        missing = count - shared_blocks
        if missing > self._num_free:
            raise DraftKVExhausted(
                f"need {missing} private draft KV blocks, {self._num_free} free"
            )
        remaining = self._num_free - missing
        private = self._free[remaining : self._num_free][::-1]
        prefix = owned[:shared_blocks]
        references = iter(
            [(block, self._references[block] + 1) for block in prefix]
            + [(block, 1) for block in private]
        )
        self._owned[destination] = prefix + private
        for block, count in references:
            self._references[block] = count
        self._num_free = remaining
        return private

    def release(self, req_index: int) -> None:
        owned = self._owned.get(req_index)
        if owned is None:
            return
        returned = self._num_free
        for block_id in reversed(owned):
            self._references[block_id] -= 1
            if self._references[block_id] == 0:
                self._free[returned] = block_id
                returned += 1
        # Until all writes succeed these cells are outside the free prefix.
        del self._owned[req_index]
        self._num_free = returned

    def swap(self, first: int, second: int) -> None:
        """Exchange complete page leases without changing their reference counts."""
        if first == second:
            raise ValueError("draft KV lease swap requires distinct slots")
        left, right = self._owned[first], self._owned[second]
        self._owned[first], self._owned[second] = right, left

    def slot_mapping(self, req_index: int, positions: np.ndarray) -> np.ndarray:
        """KV slot of each position; positions must be covered by ``ensure``."""
        blocks = np.asarray(self._owned[req_index], dtype=np.int64)
        return blocks[positions // self.block_size] * self.block_size + (
            positions % self.block_size
        )
