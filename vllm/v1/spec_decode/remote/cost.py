# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side measured goodput selection for same-step token RPCs."""

import math
from dataclasses import dataclass, field

from vllm.config.speculative import RemoteDraftCostConfig


@dataclass
class _Estimate:
    count: int = 0
    seconds: float = 0
    tokens: float = 0

    def update(self, seconds: float, tokens: float, alpha: float) -> None:
        weight = 1.0 if self.count == 0 else alpha
        self.seconds += weight * (seconds - self.seconds)
        self.tokens += weight * (tokens - self.tokens)
        self.count += 1

    @property
    def goodput(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0


@dataclass
class _BatchCosts:
    estimates: dict[int, _Estimate] = field(default_factory=dict)
    selected: int = 0
    rounds: int = 0
    probe: int = 0


class RemoteDraftCostController:
    """Calibrate K=0, explore positive K, then compare measured round goodput.

    K=0 maintains the draft context during calibration. Its target interval
    alone estimates the eventual target-only path, which sends no draft RPCs.
    Observations must cover the same decode requests on consecutive rounds;
    callers discard samples spanning prefill, reorder, cancellation or fallback.
    """

    def __init__(self, options: RemoteDraftCostConfig, capacity: int):
        if capacity < 1:
            raise ValueError("cost-aware drafting requires positive draft capacity")
        self.options = options
        self.capacity = capacity
        self._batches: dict[int, _BatchCosts] = {}
        self._chosen = [0] * (capacity + 1)
        self.observations = 0
        self.transitions = 0

    def observe(
        self,
        *,
        batch_size: int,
        k: int,
        target_seconds: float,
        rpc_seconds: float,
        emitted_tokens: int,
    ) -> None:
        if (
            batch_size < 1
            or not 0 <= k <= self.capacity
            or emitted_tokens < batch_size
            or emitted_tokens > batch_size * (k + 1)
            or not math.isfinite(target_seconds)
            or not math.isfinite(rpc_seconds)
            or target_seconds <= 0
            or rpc_seconds < 0
        ):
            return
        bucket = self._batches.setdefault(batch_size, _BatchCosts())
        estimate = bucket.estimates.setdefault(k, _Estimate())
        estimate.update(
            target_seconds + (rpc_seconds if k else 0),
            emitted_tokens / batch_size,
            self.options.ema_alpha,
        )
        bucket.rounds += 1
        self.observations += 1

    def choose(self, batch_size: int, upper_bound: int) -> int | None:
        """Return K, or None to retire the current requests to target-only."""
        if batch_size < 1 or not 0 <= upper_bound <= self.capacity:
            raise ValueError("invalid cost-aware batch size or draft budget")
        if upper_bound == 0:
            self._chosen[0] += 1
            return 0
        bucket = self._batches.setdefault(batch_size, _BatchCosts())
        for k in range(upper_bound + 1):
            if bucket.estimates.get(k, _Estimate()).count < self.options.min_samples:
                bucket.selected = k
                self._chosen[k] += 1
                return k
        best = max(
            range(1, upper_bound + 1),
            key=lambda k: bucket.estimates[k].goodput,
        )
        baseline = bucket.estimates[0].goodput
        if bucket.estimates[best].goodput < baseline * self.options.min_speedup:
            self.transitions += batch_size
            return None
        current = bucket.selected
        if (
            current < 1
            or current > upper_bound
            or bucket.estimates[best].goodput
            > bucket.estimates[current].goodput * (1 + self.options.switch_margin)
        ):
            bucket.selected = best
        selected = bucket.selected
        if bucket.rounds and bucket.rounds % self.options.probe_interval == 0:
            selected = bucket.probe % (upper_bound + 1)
            bucket.probe += 1
        self._chosen[selected] += 1
        return selected

    def snapshot(self) -> dict[str, int | tuple[int, ...]]:
        return {
            "observations": self.observations,
            "target_only_transitions": self.transitions,
            "chosen_k_counts": tuple(self._chosen),
        }

    def reset_estimates(self) -> None:
        """An idle backend must recalibrate after its workload changes."""
        self._batches.clear()
