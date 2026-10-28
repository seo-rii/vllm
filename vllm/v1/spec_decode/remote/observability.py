# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side remote RPC measurements and optional distributed tracing."""

import contextlib
import contextvars
import math
import os
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from prometheus_client import REGISTRY, Counter, Gauge, Histogram

from vllm.logger import init_logger
from vllm.tracing import (
    extract_trace_context,
    is_tracing_available,
    maybe_init_worker_tracer,
)
from vllm.v1.spec_decode.remote.protocol import RemoteObservation, TraceContext
from vllm.v1.spec_decode.remote.transport import TransportError, TransportTimeout

logger = init_logger(__name__)

_BUCKETS = (
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


class RemoteMetrics:
    def __init__(self) -> None:
        labels = ("backend", "operation")
        self.rounds = Histogram(
            "vllm:remote_draft_rpc_seconds",
            "Host wall time from sending remote prefill/proposal to its reply.",
            labels,
            buckets=_BUCKETS,
        )
        self.server = Histogram(
            "vllm:remote_draft_server_seconds",
            "Remote server host wall time; execution includes existing device waits.",
            (*labels, "stage"),
            buckets=_BUCKETS,
        )
        self.bytes = Counter(
            "vllm:remote_draft_bytes",
            "Framed socket bytes sent/received by the target, including controls.",
            ("backend", "direction"),
        )
        self.failures = Counter(
            "vllm:remote_draft_rpc_failures",
            "Failed remote RPCs, classified without request or endpoint labels.",
            (*labels, "reason"),
        )
        self.rows = Counter(
            "vllm:remote_draft_failed_rows",
            "Remote proposal rows rejected with a protocol status.",
            ("backend", "reason"),
        )
        self.resources = Gauge(
            "vllm:remote_draft_server_resources",
            "Latest server snapshot per target worker; maximum across live workers.",
            ("backend", "resource"),
            multiprocess_mode="livemax",
        )


_metrics: RemoteMetrics | None = None
_metrics_lock = threading.Lock()


def remote_metrics() -> RemoteMetrics:
    global _metrics
    # LLM teardown can unregister vllm collectors between in-process engines.
    with _metrics_lock:
        if _metrics is None or _metrics.rounds not in REGISTRY._collector_to_names:
            _metrics = RemoteMetrics()
    return _metrics


def failure_reason(error: Exception) -> str:
    if isinstance(error, TransportTimeout):
        return "timeout"
    if isinstance(error, TransportError):
        return "connection"
    if isinstance(error, MemoryError):
        return "allocation"
    return "remote_error"


@dataclass
class _Rpc:
    operation: str
    started: float
    span: Any = None


class RemoteClientObservation:
    def __init__(self, backend: str) -> None:
        self.backend = backend
        self._metrics = remote_metrics()
        self.pending: dict[int, _Rpc] = {}
        self.contexts: dict[str, Any] = {}
        self.links: list[Any] = []
        self._connection: Any = None
        self._sent = self._received = 0

    @property
    def metrics(self) -> RemoteMetrics:
        if self._metrics.rounds not in REGISTRY._collector_to_names:
            self._metrics = remote_metrics()
        return self._metrics

    def register(self, request_id: str, headers: Mapping[str, str]) -> None:
        if is_tracing_available():
            context = extract_trace_context(headers)
            if context is not None:
                from opentelemetry import trace

                span_context = trace.get_current_span(context).get_span_context()
                if span_context.is_valid:
                    self.contexts[request_id] = span_context

    def forget(self, request_ids: Sequence[str]) -> None:
        for request_id in request_ids:
            self.contexts.pop(request_id, None)
        self.links = []

    def select(self, request_ids: Sequence[str]) -> None:
        if self.contexts:
            from opentelemetry.trace import Link

            self.links = [
                Link(self.contexts[request_id])
                for request_id in request_ids
                if request_id in self.contexts
            ]
        else:
            self.links = []

    def start(self, request_id: int, operation: str, *, trace_enabled: bool):
        span = wire = None
        if trace_enabled and is_tracing_available():
            from opentelemetry import trace
            from opentelemetry.propagate import inject

            if not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
                context = None
                if len(self.links) == 1:
                    context = trace.set_span_in_context(
                        trace.NonRecordingSpan(self.links[0].context)
                    )
                span = trace.get_tracer(__name__).start_span(
                    f"remote_draft.{operation}",
                    context=context,
                    kind=trace.SpanKind.CLIENT,
                    links=self.links if len(self.links) != 1 else (),
                    attributes={"vllm.remote.backend": self.backend},
                )
                if span.get_span_context().is_valid:
                    carrier: dict[str, str] = {}
                    inject(carrier, context=trace.set_span_in_context(span))
                    if "traceparent" in carrier:
                        wire = TraceContext(
                            carrier["traceparent"], carrier.get("tracestate")
                        )
        self.pending[request_id] = _Rpc(operation, time.monotonic(), span)
        return wire

    def finish(
        self,
        request_id: int,
        observation: RemoteObservation | None = None,
        *,
        reason: str | None = None,
    ) -> None:
        if observation is not None:
            for resource, value in (
                ("active_sequences", observation.active_sequences),
                ("kv_used_blocks", observation.kv_used_blocks),
                ("kv_capacity_blocks", observation.kv_capacity_blocks),
            ):
                if value is not None:
                    self.metrics.resources.labels(self.backend, resource).set(value)
        rpc = self.pending.pop(request_id, None)
        if rpc is None:
            return
        self.metrics.rounds.labels(self.backend, rpc.operation).observe(
            max(0, time.monotonic() - rpc.started)
        )
        if observation is not None:
            for stage, seconds in (
                ("queue", observation.queue_seconds),
                ("execute", observation.execute_seconds),
            ):
                if math.isfinite(seconds):
                    self.metrics.server.labels(
                        self.backend, rpc.operation, stage
                    ).observe(seconds)
        if reason is not None:
            self.metrics.failures.labels(self.backend, rpc.operation, reason).inc()
        if rpc.span is not None:
            if reason is not None:
                from opentelemetry.trace import Status, StatusCode

                rpc.span.set_status(Status(StatusCode.ERROR, reason))
            rpc.span.end()

    def fail(self, reason: str) -> None:
        for request_id in tuple(self.pending):
            try:
                self.finish(request_id, reason=reason)
            except Exception as error:
                logger.warning(
                    "Remote draft telemetry cleanup failed (%s)",
                    type(error).__name__,
                )
            finally:
                self.pending.pop(request_id, None)

    def close(self, connection: Any) -> None:
        try:
            if connection is not None:
                self.record_bytes(connection)
            self.fail("connection")
        except Exception as error:
            logger.warning(
                "Remote draft telemetry cleanup failed (%s)", type(error).__name__
            )
        finally:
            self.pending.clear()

    def record_bytes(self, connection: Any) -> None:
        if self._connection is not connection:
            self._connection = connection
            self._sent = self._received = 0
        sent = getattr(connection, "sent_bytes", 0)
        received = getattr(connection, "received_bytes", 0)
        if not isinstance(sent, int) or not isinstance(received, int):
            return
        if sent > self._sent:
            self.metrics.bytes.labels(self.backend, "sent").inc(sent - self._sent)
        if received > self._received:
            self.metrics.bytes.labels(self.backend, "received").inc(
                received - self._received
            )
        self._sent, self._received = sent, received


_current_server: contextvars.ContextVar["RemoteServerObservation | None"] = (
    contextvars.ContextVar("remote_draft_observation", default=None)
)


class RemoteServerObservation:
    def __init__(self) -> None:
        self.queue_seconds = 0.0
        self.execute_seconds = 0.0

    @contextlib.contextmanager
    def turn(self) -> Iterator[None]:
        started = time.monotonic()
        try:
            yield
        finally:
            self.execute_seconds += time.monotonic() - started


@contextlib.contextmanager
def server_observation(
    backend: str,
    operation: str,
    wire: TraceContext | None,
    *,
    enabled: bool = True,
) -> Iterator[None]:
    if not enabled:
        yield
        return
    observation = RemoteServerObservation()
    token = _current_server.set(observation)
    manager = contextlib.nullcontext()
    if wire is not None and is_tracing_available():
        from opentelemetry import trace

        context = extract_trace_context(
            {"traceparent": wire.traceparent, "tracestate": wire.tracestate or ""}
        )
        manager = trace.get_tracer(__name__).start_as_current_span(
            f"remote_draft.server.{operation}",
            context=context,
            kind=trace.SpanKind.SERVER,
            attributes={"vllm.remote.backend": backend},
        )
    try:
        with manager as span:
            yield
            if span is not None:
                span.set_attributes(
                    {
                        "vllm.remote.queue_seconds": observation.queue_seconds,
                        "vllm.remote.execute_seconds": observation.execute_seconds,
                    }
                )
    finally:
        _current_server.reset(token)


def current_server_observation() -> RemoteServerObservation | None:
    return _current_server.get()


def init_remote_tracing() -> None:
    if os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") and is_tracing_available():
        from opentelemetry import trace

        if isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
            maybe_init_worker_tracer(
                "vllm.remote_draft", "remote_draft", "remote_draft"
            )
