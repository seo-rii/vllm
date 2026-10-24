# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded cross-verifier execution for target-conditioned own-KV drafts."""

import time
from collections.abc import Callable
from typing import Any

from vllm.v1.spec_decode.remote.protocol import SpeculatorStatusCode
from vllm.v1.spec_decode.remote.runtime.adapter import TargetConditionedDraftAdapter
from vllm.v1.spec_decode.remote.runtime.work_queue import DraftWork, DraftWorkQueue
from vllm.v1.spec_decode.remote.server import (
    AdapterError,
    RemoteDraftServer,
    ServerShutdownTimeout,
    StandaloneProposalResult,
    _ProposalInput,
    _ServerSession,
)
from vllm.v1.spec_decode.remote.transport import FramedConnection, is_local_endpoint


class ConditionedDraftServer(RemoteDraftServer):
    """One GPU owner shares slots/KV and batches compatible verifier proposals."""

    adapter: TargetConditionedDraftAdapter

    def __init__(
        self,
        adapter: TargetConditionedDraftAdapter,
        *,
        max_pending_jobs: int = 64,
        max_queued_feature_bytes: int = 1 << 30,
        max_jobs_per_verifier: int = 8,
        max_feature_bytes_per_verifier: int = 1 << 30,
        max_sequences_per_verifier: int | None = None,
        batch_wait_ms: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(adapter, **kwargs)
        if self.limits.max_batch_size > adapter.planner.max_num_reqs:
            raise ValueError("server batch capacity exceeds the draft request slots")
        if max_sequences_per_verifier is None:
            max_sequences_per_verifier = self.limits.max_sequences
        if (
            type(max_sequences_per_verifier) is not int
            or not 0 < max_sequences_per_verifier <= self.limits.max_sequences
        ):
            raise ValueError("per-verifier sequence credit must fit the server limit")
        self.max_sequences_per_verifier = max_sequences_per_verifier
        self.work_queue = DraftWorkQueue(
            self._execute_proposals,
            max_jobs=max_pending_jobs,
            max_bytes=max_queued_feature_bytes,
            max_jobs_per_owner=max_jobs_per_verifier,
            max_bytes_per_owner=max_feature_bytes_per_verifier,
            max_cleanup_jobs=self.max_connections,
            max_batch_rows=self.limits.max_batch_size,
            max_batch_tokens=self.limits.max_feature_tokens,
            batch_wait_s=batch_wait_ms / 1000,
        )

    @property
    def queue_depth(self) -> int:
        return max(super().queue_depth, self.work_queue.depth)

    def _warmup_owner(self) -> None:
        device = getattr(self.adapter.executor, "device", None)
        if device is not None and device.type != "cpu":
            import torch

            torch.accelerator.set_device_index(device)
            torch.accelerator.synchronize(device)
        self.adapter.warmup()

    def _warmup(self) -> None:
        self.work_queue.start(self._warmup_owner)

    def _create_session(self, conn: FramedConnection) -> "_ConditionedServerSession":
        return _ConditionedServerSession(self, conn)

    def start(self, endpoint: str) -> str:
        if not is_local_endpoint(endpoint):
            raise ValueError("conditioned draft pooling requires the same host")
        # Do not stop an already running owner's queue on a duplicate start.
        if self._listener is not None or not self.healthy:
            raise RuntimeError("conditioned runtime is already started or unhealthy")
        try:
            return super().start(endpoint)
        except BaseException:
            self.work_queue.stop(time.monotonic() + self.shutdown_timeout_s)
            raise

    def stop(self) -> None:
        deadline = time.monotonic() + self.shutdown_timeout_s
        try:
            super().stop()
        finally:
            try:
                self.work_queue.stop(deadline)
            except ServerShutdownTimeout:
                self._shutdown_failed = True
                raise

    def _execute_proposals(self, work: tuple[DraftWork, ...]) -> None:
        # Different K budgets must not share a native rollout: padding a K=0
        # row to K=max would write extra speculative KV and consume its budget.
        groups: dict[int, list[DraftWork]] = {}
        for job in work:
            groups.setdefault(job.payload[1].effective_k, []).append(job)
        with self.adapter_turn():
            for jobs in groups.values():
                executions = [
                    session._begin_proposal(inputs)
                    for session, inputs in (job.payload for job in jobs)
                ]
                batches, locations = [], []
                for i, (job, execution) in enumerate(zip(jobs, executions)):
                    session, inputs = job.payload
                    if execution.batch is None:
                        continue
                    if not self.healthy:
                        execution.failure = AdapterError(
                            SpeculatorStatusCode.INTERNAL_ERROR, "runtime is unhealthy"
                        )
                    elif session._expire_owned_sequences(
                        inputs.deadline, inputs.message.keys
                    ):
                        execution.failure = AdapterError(
                            SpeculatorStatusCode.TIMEOUT,
                            "execution budget expired before proposal",
                        )
                    else:
                        batches.append(execution.batch)
                        locations.append(i)
                outcomes: list[StandaloneProposalResult | Exception | None] = [
                    None
                ] * len(jobs)
                if batches:
                    results = self.adapter.advance_many(batches)
                    for i, result in zip(locations, results):
                        if isinstance(result, Exception):
                            self._adapter_failed(result)
                        outcomes[i] = result
                for job, execution, outcome in zip(jobs, executions, outcomes):
                    session, _ = job.payload
                    job.result = session._finish_proposal(execution, outcome)


class _ConditionedServerSession(_ServerSession):
    server: ConditionedDraftServer

    def _run_action(
        self,
        action: Callable[[], Any],
        *,
        cleanup: bool = False,
        num_bytes: int = 0,
    ) -> Any:
        return self.server.work_queue.call(
            self.verifier_instance_id,
            action,
            cleanup=cleanup,
            num_bytes=num_bytes,
        )

    def _sequence_credit_available(self) -> bool:
        return (
            super()._sequence_credit_available()
            and self.server.registry.count(self.verifier_instance_id)
            < self.server.max_sequences_per_verifier
        )

    def _run_proposal(
        self, inputs: _ProposalInput
    ) -> StandaloneProposalResult | AdapterError:
        frames = [inputs.accepted, inputs.recovery, *inputs.frames]
        num_bytes = sum(
            frame.cuda_ipc.nbytes if frame.cuda_ipc is not None else len(frame.payload)
            for frame in frames
            if frame is not None
        )
        return self.server.work_queue.submit(
            DraftWork(
                self.verifier_instance_id,
                None,
                (self, inputs),
                num_bytes=num_bytes,
                num_rows=len(inputs.message.keys),
                num_tokens=max(1, sum(inputs.query_lens)),
            )
        ).wait()
