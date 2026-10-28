# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded fair scheduling for one token-draft GPU owner."""

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vllm.v1.spec_decode.remote.server import AdapterQueueFull, ServerShutdownTimeout


@dataclass
class DraftWork:
    owner: str
    action: Callable[[], Any] | None
    payload: Any = None
    num_bytes: int = 0
    num_rows: int = 0
    num_tokens: int = 0
    cleanup: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None
    _submitted: bool = False

    def wait(self) -> Any:
        self.done.wait()
        if self.error is not None:
            raise self.error
        return self.result


class DraftWorkQueue:
    """Round-robin verifier queues, including active work in admission credits.

    Completion waits do not abandon running GPU work. The server owns the bounded
    shutdown deadline and retains the adapter if its owner cannot finish.
    """

    def __init__(
        self,
        execute_batch: Callable[[tuple[DraftWork, ...]], None],
        *,
        max_jobs: int,
        max_bytes: int,
        max_jobs_per_owner: int,
        max_bytes_per_owner: int,
        max_cleanup_jobs: int,
        max_batch_rows: int,
        max_batch_tokens: int,
        batch_wait_s: float = 0.001,
        on_failure: Callable[[BaseException], None] | None = None,
    ) -> None:
        for name, value in (
            ("max_jobs", max_jobs),
            ("max_bytes", max_bytes),
            ("max_jobs_per_owner", max_jobs_per_owner),
            ("max_bytes_per_owner", max_bytes_per_owner),
            ("max_cleanup_jobs", max_cleanup_jobs),
            ("max_batch_rows", max_batch_rows),
            ("max_batch_tokens", max_batch_tokens),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(batch_wait_s) or batch_wait_s < 0:
            raise ValueError("batch_wait_s must be finite and nonnegative")
        self.execute_batch = execute_batch
        self.max_jobs = max_jobs
        self.max_bytes = max_bytes
        self.max_jobs_per_owner = max_jobs_per_owner
        self.max_bytes_per_owner = max_bytes_per_owner
        self.max_cleanup_jobs = max_cleanup_jobs
        self.max_batch_rows = max_batch_rows
        self.max_batch_tokens = max_batch_tokens
        self.batch_wait_s = batch_wait_s
        self.on_failure = on_failure
        self._condition = threading.Condition()
        self._queues: dict[str, deque[DraftWork]] = {}
        self._ready: deque[str] = deque()
        self._credits: dict[str, tuple[int, int]] = {}
        # Admission ownership survives queue removal, batch allocation, execution,
        # and credit release. Only a completed work item leaves this registry.
        self._owned: dict[int, DraftWork] = {}
        self._jobs = self._bytes = self._cleanup_jobs = 0
        self._closing = False
        self._initialized = False
        self._failure: BaseException | None = None
        self._thread: threading.Thread | None = None

    @property
    def depth(self) -> int:
        with self._condition:
            return self._jobs + self._cleanup_jobs

    @property
    def pending_bytes(self) -> int:
        with self._condition:
            return self._bytes

    def start(self, initialize: Callable[[], None]) -> None:
        ready = DraftWork("", initialize)
        start_error = None

        def run() -> None:
            try:
                initialize()
                with self._condition:
                    self._initialized = True
            except BaseException as error:
                ready.error = error
            finally:
                ready.done.set()
            if ready.error is None:
                self._run()

        with self._condition:
            if self._failure is not None:
                raise ServerShutdownTimeout(
                    "token GPU owner failed; process restart required"
                ) from self._failure
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("draft work queue is already started")
            if (
                self._jobs
                or self._bytes
                or self._cleanup_jobs
                or self._queues
                or self._ready
                or self._credits
                or self._owned
            ):
                raise RuntimeError("draft work queue retains unfinished work")
            self._closing = False
            self._initialized = False
            thread = threading.Thread(
                target=run, name="token-draft-gpu-owner", daemon=True
            )
            self._thread = thread
            try:
                thread.start()
            except BaseException as error:
                self._closing = True
                self._condition.notify_all()
                if isinstance(error, RuntimeError) and thread.ident is None:
                    self._thread = None
                    raise
                # A low-memory start can raise after the OS thread exists,
                # before Python publishes its ident. Retain its ownership.
                start_error = error
        if start_error is not None:
            self._fail(start_error)
            raise start_error
        try:
            ready.wait()
        except BaseException as error:
            if error is ready.error:
                # Failed initialization has no work loop; finish its owner before
                # exposing a recoverable error that permits an immediate retry.
                thread.join()
            raise

    def submit(self, work: DraftWork) -> DraftWork:
        if any(
            type(value) is not int or value < 0
            for value in (work.num_bytes, work.num_rows, work.num_tokens)
        ):
            raise ValueError("draft work sizes must be nonnegative integers")
        if work.action is None and (
            not 0 < work.num_rows <= self.max_batch_rows
            or not 0 < work.num_tokens <= self.max_batch_tokens
        ):
            raise AdapterQueueFull("token job exceeds the GPU batch capacity")
        with self._condition:
            if work._submitted:
                raise RuntimeError("draft work cannot be submitted twice")
            if (
                not self._initialized
                or self._closing
                or self._thread is None
                or not self._thread.is_alive()
            ):
                raise ServerShutdownTimeout(
                    "token draft owner is not ready or stopping"
                )
            previous = self._credits.get(work.owner)
            jobs, size = previous if previous is not None else (0, 0)
            if work.cleanup:
                if self._cleanup_jobs >= self.max_cleanup_jobs:
                    raise ServerShutdownTimeout("token cleanup queue is exhausted")
            elif (
                self._jobs >= self.max_jobs
                or self._bytes + work.num_bytes > self.max_bytes
                or jobs >= self.max_jobs_per_owner
                or size + work.num_bytes > self.max_bytes_per_owner
            ):
                raise AdapterQueueFull("token draft admission credit exhausted")
            queue = self._queues.get(work.owner)
            created = queue is None
            if queue is None:
                queue = deque((work,))
            credits = jobs + 1, size + work.num_bytes
            total_jobs = self._jobs + (not work.cleanup)
            total_bytes = self._bytes + (0 if work.cleanup else work.num_bytes)
            cleanup_jobs = self._cleanup_jobs + work.cleanup
            try:
                self._owned[id(work)] = work
                if not work.cleanup:
                    self._credits[work.owner] = credits
                if created:
                    self._queues[work.owner] = queue
                    self._ready.append(work.owner)
                else:
                    queue.append(work)
            except BaseException:
                self._owned.pop(id(work), None)
                if created:
                    self._queues.pop(work.owner, None)
                if not work.cleanup:
                    if previous is None:
                        self._credits.pop(work.owner, None)
                    else:
                        self._credits[work.owner] = previous
                raise
            work._submitted = True
            self._jobs, self._bytes = total_jobs, total_bytes
            self._cleanup_jobs = cleanup_jobs
            self._condition.notify()
        return work

    def call(self, owner: str, action: Callable[[], Any], **kwargs) -> Any:
        return self.submit(DraftWork(owner, action, **kwargs)).wait()

    def _pop(self) -> DraftWork:
        owner = self._ready.popleft()
        queue = self._queues[owner]
        work = queue.popleft()
        if queue:
            self._ready.append(owner)
        else:
            del self._queues[owner]
        return work

    def _next_batch(self) -> tuple[DraftWork, ...]:
        with self._condition:
            self._condition.wait_for(lambda: self._ready or self._closing)
            if not self._ready:
                return ()
            first = self._pop()
            batch = [first]
            if first.action is None:
                until = time.monotonic() + self.batch_wait_s
                rows, tokens = first.num_rows, first.num_tokens
                while rows < self.max_batch_rows and tokens < self.max_batch_tokens:
                    if not self._ready:
                        remaining = until - time.monotonic()
                        if remaining <= 0 or self._closing:
                            break
                        self._condition.wait(remaining)
                        if not self._ready:
                            break
                    next_work = self._queues[self._ready[0]][0]
                    if (
                        next_work.action is not None
                        or rows + next_work.num_rows > self.max_batch_rows
                        or tokens + next_work.num_tokens > self.max_batch_tokens
                    ):
                        break
                    batch.append(self._pop())
                    rows += next_work.num_rows
                    tokens += next_work.num_tokens
            return tuple(batch)

    def _execute(self, batch: tuple[DraftWork, ...]) -> None:
        try:
            if batch[0].action is None:
                self.execute_batch(batch)
            else:
                batch[0].result = batch[0].action()
        except BaseException as error:
            for work in batch:
                work.error = error
        finally:
            with self._condition:
                for work in batch:
                    if work.cleanup:
                        self._cleanup_jobs -= 1
                    else:
                        self._jobs -= 1
                        self._bytes -= work.num_bytes
                        jobs, size = self._credits[work.owner]
                        if jobs == 1:
                            del self._credits[work.owner]
                        else:
                            self._credits[work.owner] = jobs - 1, size - work.num_bytes
                    work.done.set()
                    del self._owned[id(work)]
                self._condition.notify_all()

    def _fail(self, error: BaseException) -> None:
        with self._condition:
            self._failure = error
            self._closing = True
        try:
            if self.on_failure is not None:
                self.on_failure(error)
        except BaseException as cleanup_error:
            self._failure = BaseExceptionGroup(
                "draft owner and failure notification failed", [error, cleanup_error]
            )
        finally:
            with self._condition:
                # Do not reconstruct the partially consumed batch or rely on
                # possibly half-updated credits while unwinding an allocation.
                for work in self._owned.values():
                    if not work.done.is_set():
                        if work.error is None:
                            work.error = error
                        work.done.set()
                self._owned.clear()
                self._queues.clear()
                self._ready.clear()
                self._credits.clear()
                self._jobs = self._bytes = self._cleanup_jobs = 0
                self._condition.notify_all()

    def _run(self) -> None:
        try:
            while batch := self._next_batch():
                self._execute(batch)
                del batch
        except BaseException as error:
            self._fail(error)

    def stop(self, deadline: float) -> None:
        with self._condition:
            self._closing = True
            thread = self._thread
            self._condition.notify_all()
        if thread is not None and thread.ident is not None:
            thread.join(max(0, deadline - time.monotonic()))
            if thread.is_alive():
                raise ServerShutdownTimeout(
                    "token GPU owner is still running; process restart required"
                )
        with self._condition:
            if self._failure is not None or self._owned or self.depth:
                raise ServerShutdownTimeout(
                    "token GPU owner failed; process restart required"
                ) from self._failure
