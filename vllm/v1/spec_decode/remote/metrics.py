# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded CPU aggregates carried by the normal engine statistics output."""

import math
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, field

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram

REMOTE_LATENCY_BUCKETS = (
    0.0001,
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
    10,
    30,
)


@dataclass
class RemoteHistogramStats:
    # Non-cumulative bins, including the final +Inf bin.
    buckets: list[int] = field(
        default_factory=lambda: [0] * (len(REMOTE_LATENCY_BUCKETS) + 1)
    )
    sum: float = 0.0

    def observe(self, seconds: float) -> None:
        if math.isfinite(seconds) and seconds >= 0:
            self.buckets[bisect_left(REMOTE_LATENCY_BUCKETS, seconds)] += 1
            self.sum += seconds


@dataclass
class RemoteDraftStats:
    backend: str
    rounds: dict[str, RemoteHistogramStats] = field(default_factory=dict)
    server: dict[str, dict[str, RemoteHistogramStats]] = field(default_factory=dict)
    bytes: dict[str, int] = field(default_factory=dict)
    failures: dict[str, dict[str, int]] = field(default_factory=dict)
    rows: dict[str, int] = field(default_factory=dict)
    resources: dict[str, int] = field(default_factory=dict)


def _observe_histogram(histogram: Histogram, stats: RemoteHistogramStats) -> None:
    # The SDK has no bulk-observe API. Keep its storage dependency here so that
    # per-step aggregation preserves the exact distribution, including +Inf.
    bounds = (*REMOTE_LATENCY_BUCKETS, math.inf)
    if (
        tuple(histogram._upper_bounds) != bounds
        or len(stats.buckets) != len(bounds)
        or any(type(count) is not int or count < 0 for count in stats.buckets)
        or not math.isfinite(stats.sum)
        or stats.sum < 0
        or (not any(stats.buckets) and stats.sum != 0)
    ):
        raise ValueError("invalid remote draft histogram aggregate")
    histogram._sum.inc(stats.sum)
    for bucket, count in zip(histogram._buckets, stats.buckets, strict=True):
        bucket.inc(count)


class RemoteMetrics:
    """Publish only in the frontend, alongside the existing engine metrics."""

    def __init__(
        self,
        labelnames: Sequence[str] = (),
        per_engine_labelvalues: dict[int, list[object]] | None = None,
        *,
        registry: CollectorRegistry = REGISTRY,
    ) -> None:
        self.per_engine_labelvalues = per_engine_labelvalues or {0: []}
        labels = (*labelnames, "backend", "operation")
        self.rounds = Histogram(
            "vllm:remote_draft_rpc_seconds",
            "Host wall time from sending remote prefill/proposal to its reply.",
            labels,
            buckets=REMOTE_LATENCY_BUCKETS,
            registry=registry,
        )
        self.server = Histogram(
            "vllm:remote_draft_server_seconds",
            "Remote server host wall time; execution includes existing device waits.",
            (*labels, "stage"),
            buckets=REMOTE_LATENCY_BUCKETS,
            registry=registry,
        )
        self.bytes = Counter(
            "vllm:remote_draft_bytes",
            "Framed socket bytes sent/received by the target, including controls.",
            (*labelnames, "backend", "direction"),
            registry=registry,
        )
        self.failures = Counter(
            "vllm:remote_draft_rpc_failures",
            "Failed remote RPCs, classified without request or endpoint labels.",
            (*labels, "reason"),
            registry=registry,
        )
        self.rows = Counter(
            "vllm:remote_draft_failed_rows",
            "Remote proposal rows rejected with a protocol status.",
            (*labelnames, "backend", "reason"),
            registry=registry,
        )
        self.resources = Gauge(
            "vllm:remote_draft_server_resources",
            "Latest server snapshot per engine, including acknowledged cleanup.",
            (*labelnames, "backend", "resource"),
            multiprocess_mode="mostrecent",
            registry=registry,
        )

    def observe(self, stats: RemoteDraftStats, engine_idx: int = 0) -> None:
        labels = (*self.per_engine_labelvalues[engine_idx], stats.backend)
        for operation, histogram in stats.rounds.items():
            _observe_histogram(self.rounds.labels(*labels, operation), histogram)
        for operation, stages in stats.server.items():
            for stage, histogram in stages.items():
                _observe_histogram(
                    self.server.labels(*labels, operation, stage), histogram
                )
        for direction, count in stats.bytes.items():
            self.bytes.labels(*labels, direction).inc(count)
        for operation, reasons in stats.failures.items():
            for reason, count in reasons.items():
                self.failures.labels(*labels, operation, reason).inc(count)
        for reason, count in stats.rows.items():
            self.rows.labels(*labels, reason).inc(count)
        for resource, value in stats.resources.items():
            self.resources.labels(*labels, resource).set(value)
