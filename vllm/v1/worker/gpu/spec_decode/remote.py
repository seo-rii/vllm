# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 adapter for a target-conditioned drafter on a dedicated device."""

import itertools
import time
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.v1.spec_decode.contracts import DraftStep, LocalExecutionContext
from vllm.v1.spec_decode.proposal import ProposalHandle, SpeculatorOutput
from vllm.v1.spec_decode.remote.capabilities import TargetFeatureKind
from vllm.v1.spec_decode.remote.errors import (
    cuda_context_fatal,
    host_allocation_failure,
)
from vllm.v1.spec_decode.remote.protocol import SequenceKey, SpeculatorStatusCode
from vllm.v1.spec_decode.remote.runtime.schema import create_runtime_contract
from vllm.v1.spec_decode.remote.session import (
    RemoteDraftSession,
    RemoteProposalHandle,
    VerifierIdentity,
)
from vllm.v1.spec_decode.remote.state import (
    FeatureBatch,
    RemoteProposalBatch,
    SequenceState,
)
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

logger = init_logger(__name__)


@dataclass
class _Request:
    key: SequenceKey
    num_computed: int = 0


@dataclass(eq=False)
class _RemoteHandle(ProposalHandle):
    owner: object
    round_id: int
    step: DraftStep
    context: LocalExecutionContext | None
    started: float
    session_handle: RemoteProposalHandle | None = None
    rows: tuple[int, ...] = ()
    output: SpeculatorOutput | None = None
    error: str | None = None
    preparation_deadline: float | None = None
    retained: tuple[Any, ...] = field(default_factory=tuple, repr=False)


class RemoteSpeculator(BaseSpeculator):
    """Keep all draft weights and KV on the server; TP rank zero owns its session."""

    supports_mm_inputs = False
    draft_logits = None

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self._validate_config(vllm_config)
        spec = vllm_config.speculative_config
        assert spec is not None
        remote_config = (
            spec.get_remote_draft_config()
            if hasattr(spec, "get_remote_draft_config")
            else spec.remote_draft
        )
        assert remote_config is not None
        assert spec.method is not None
        self.device = device
        self.num_speculative_steps = spec.num_speculative_tokens
        self._contract = create_runtime_contract(vllm_config)
        self._tp = get_tp_group()
        self._session: RemoteDraftSession | None = None
        self._requests: dict[str, _Request] = {}
        self._terminal_requests: set[str] = set()
        self._closing_requests: set[str] = set()
        self._sequence_ids = itertools.count()
        self._batch_ids = itertools.count()
        self._owner = object()
        self._rounds = itertools.count()
        self._pending: _RemoteHandle | None = None
        self._closed = False
        self._stats: dict[str, int | float] = {
            "proposal_calls": 0,
            "feature_bytes": 0,
            "proposal_seconds": 0.0,
            "session_seconds": 0.0,
        }
        error = None
        if self._tp.rank_in_group == 0:
            identity = VerifierIdentity(
                verifier_instance_id=random_uuid(),
                target_fingerprint=self._contract.target_fingerprint,
                tokenizer_fingerprint=self._contract.tokenizer_fingerprint,
                draft_checkpoint_fingerprint=self._contract.draft_fingerprint,
                method=spec.method,
                num_speculative_tokens=self.num_speculative_steps,
                parallel_drafting=False,
                provided_features=tuple(s.kind for s in self._contract.schema.slots),
                max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
                max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
                max_model_len=vllm_config.model_config.max_model_len,
                vocab_size=vllm_config.model_config.get_vocab_size(),
            )
            self._session = RemoteDraftSession.from_config(remote_config, identity)
            try:
                self._session.connect()
                if self._session.feature_schema != self._contract.schema:
                    raise ValueError(
                        "remote feature schema differs from checkpoint contract"
                    )
                # Reserve transport memory before the worker profiles free GPU
                # memory and assigns the remaining budget to the target KV cache.
                self._session.prepare_cuda_ipc(self.device)
                if self._session.cuda_ipc_reserved_bytes:
                    logger.info(
                        "Remote drafter CUDA IPC arena: %d bytes",
                        self._session.cuda_ipc_reserved_bytes,
                    )
            except Exception as exc:
                self.close()
                error = f"{type(exc).__name__}: {exc}"
        error = self._tp.broadcast_object(error, src=0)
        if error is not None:
            raise RuntimeError(f"remote drafter startup failed: {error}")

    @staticmethod
    def _validate_config(config: VllmConfig) -> None:
        assert config.speculative_config is not None
        parallel = config.parallel_config
        unsupported = []
        if parallel.tensor_parallel_size not in (1, 2):
            unsupported.append("target tensor_parallel_size must be 1 or 2")
        if any(
            getattr(parallel, name) != 1
            for name in (
                "pipeline_parallel_size",
                "data_parallel_size",
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
            )
        ):
            unsupported.append("PP, DP and context parallelism must be disabled")
        if config.cache_config.enable_prefix_caching:
            unsupported.append("enable_prefix_caching must be False")
        if config.scheduler_config.async_scheduling:
            unsupported.append("async_scheduling must be False")
        if config.lora_config is not None:
            unsupported.append("LoRA is unsupported")
        if config.model_config.is_multimodal_model:
            unsupported.append("multimodal models are unsupported")
        if getattr(config.model_config, "enable_prompt_embeds", False):
            unsupported.append("prompt embeddings are unsupported")
        if getattr(config, "watermark_config", None) is not None:
            unsupported.append("watermarking is unsupported")
        if config.speculative_config.draft_lm_head_quantization is not None:
            unsupported.append("draft_lm_head_quantization is unsupported")
        if config.speculative_config.enable_adaptive_verification:
            unsupported.append("adaptive verification is unsupported")
        if unsupported:
            raise ValueError("remote drafter: " + "; ".join(unsupported))

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        pass

    def capture(self) -> None:
        pass

    def close_requests(self, req_ids: Collection[str]) -> None:
        if not req_ids:
            return
        self._terminal_requests.difference_update(req_ids)
        error = None
        pending = self._pending
        cancelled = pending is not None and bool(
            set(req_ids).intersection(pending.step.request_ids)
        )
        if self._session is not None:
            try:
                self._closing_requests.update(
                    req_id for req_id in req_ids if req_id in self._requests
                )
                if cancelled:
                    # A lost completion cannot prove that IPC readers finished.
                    self._session.close()
                if cancelled or pending is None or pending.session_handle is None:
                    self._flush_request_closes()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if cancelled:
            assert pending is not None
            self._discard_handle(pending)
        error = self._tp.broadcast_object(error, src=0)
        if error is not None:
            raise RuntimeError(f"remote drafter request cleanup failed: {error}")

    def _flush_request_closes(self, *, deadline: float | None = None) -> None:
        if self._session is None or not self._closing_requests:
            return
        req_ids = tuple(self._closing_requests)
        keys = tuple(self._requests[req_id].key for req_id in req_ids)
        # Preserve ownership until the session has retired the keys. An unrelated
        # close must not consume an in-flight proposal response as a close ACK.
        self._session.close_sequences(keys, deadline=deadline)
        for req_id in req_ids:
            del self._requests[req_id]
            self._closing_requests.remove(req_id)

    def close(self) -> None:
        self._closed = True
        if self._session is not None:
            session = self._session
            self._session = None
            try:
                session.close()
            except Exception:
                logger.exception("Failed to close remote drafter session")
            finally:
                self._requests.clear()
                self._terminal_requests.clear()
                self._closing_requests.clear()
                logger.info("Remote drafter verifier statistics: %s", self._stats)
        if self._pending is not None:
            self._discard_handle(self._pending)

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        context = LocalExecutionContext(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dp_sync_state,
            dummy_run,
            skip_attn_for_dummy_run,
            mm_inputs,
            is_profile,
        )
        k = (
            self.num_speculative_steps
            if num_speculative_tokens is None
            else num_speculative_tokens
        )
        step = DraftStep.from_batch(input_batch, k)
        return self.collect(self.dispatch(step, context)).to_legacy_tokens()

    def dispatch(
        self, step: DraftStep, context: LocalExecutionContext
    ) -> ProposalHandle:
        if self._closed:
            raise RuntimeError("remote draft backend is closed")
        if self._pending is not None:
            raise RuntimeError("remote draft proposal is still in flight")
        handle = _RemoteHandle(
            self._owner, next(self._rounds), step, context, time.perf_counter()
        )
        self._pending = handle
        if context.dummy_run or context.is_profile:
            try:
                if context.is_profile and self._tp.rank_in_group == 0:
                    self._profile_features(
                        context.input_batch,
                        context.last_hidden_states,
                        context.aux_hidden_states,
                        context.next_prefill_tokens,
                    )
                handle.output = self._empty_output(
                    len(step.request_ids), device=self.device
                )
            except Exception:
                self._discard_handle(handle)
                raise
            return handle
        if self._tp.rank_in_group == 0:
            self._stats["proposal_calls"] += 1
            try:
                assert self._session is not None
                handle.preparation_deadline = self._session.preparation_deadline()
                if not 0 <= step.num_speculative_tokens <= self.num_speculative_steps:
                    raise ValueError(
                        "round draft budget exceeds the configured maximum"
                    )
                if step != DraftStep.from_batch(
                    context.input_batch, step.num_speculative_tokens
                ):
                    raise ValueError(
                        "draft step metadata no longer matches its local context"
                    )
                if torch.any(
                    context.temperature[context.input_batch.idx_mapping] != 0
                ).item():
                    raise ValueError(
                        "remote drafter currently requires greedy target sampling"
                    )
                try:
                    self._dispatch_step(handle, context)
                except Exception as exc:
                    self._feature_failure(handle, exc)
            except Exception as exc:
                handle.error = f"{type(exc).__name__}: {exc}"
        return handle

    def cancel(self, handle: ProposalHandle, reason: str) -> None:
        handle = self._owned_handle(handle)
        error = None
        try:
            if self._session is not None:
                try:
                    # Session teardown quarantines IPC exports on uncertain completion.
                    self._session.close()
                    self._flush_request_closes()
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            error = self._tp.broadcast_object(error, src=0)
            if error is not None:
                raise RuntimeError(f"remote drafter cancellation failed: {error}")
            logger.debug("Cancelled remote draft round %d: %s", handle.round_id, reason)
        finally:
            self._discard_handle(handle)

    def _owned_handle(self, handle: ProposalHandle) -> _RemoteHandle:
        if not isinstance(handle, _RemoteHandle) or handle.owner is not self._owner:
            raise RuntimeError("draft handle belongs to a different owner")
        if self._pending is not handle or handle.context is None:
            raise RuntimeError("stale, cancelled or already collected draft handle")
        return handle

    def collect(self, handle: ProposalHandle) -> SpeculatorOutput:
        handle = self._owned_handle(handle)
        context = handle.context
        assert context is not None
        try:
            if context.dummy_run or context.is_profile:
                assert handle.output is not None
                return handle.output
            payload = None
            if self._tp.rank_in_group == 0:
                try:
                    if handle.error is None and handle.session_handle is not None:
                        assert self._session is not None and handle.output is not None
                        started = time.perf_counter()
                        result = self._session.collect(handle.session_handle).output
                        self._stats["session_seconds"] += time.perf_counter() - started
                        handle.output.token_ids[list(handle.rows)] = result.token_ids
                        handle.output.valid_lengths[list(handle.rows)] = (
                            result.valid_lengths
                        )
                    output = handle.output if handle.error is None else None
                    if output is not None:
                        output.valid_lengths.clamp_(
                            max=handle.step.num_speculative_tokens
                        )
                    payload = (
                        handle.error,
                        None if output is None else output.token_ids.tolist(),
                        None if output is None else output.valid_lengths.tolist(),
                    )
                except Exception as exc:
                    try:
                        self._feature_failure(handle, exc)
                        assert handle.output is not None
                        payload = (
                            None,
                            handle.output.token_ids.tolist(),
                            handle.output.valid_lengths.tolist(),
                        )
                    except Exception as failure:
                        handle.error = f"{type(failure).__name__}: {failure}"
                        if (
                            self._session is not None
                            and handle.session_handle is not None
                        ):
                            try:
                                self._session.close()
                            except Exception:
                                logger.exception(
                                    "Failed to close remote proposal session"
                                )
                        payload = (handle.error, None, None)
                finally:
                    try:
                        # Deferred closes share the round budget instead of
                        # delaying its result for another full request timeout.
                        self._flush_request_closes(deadline=handle.preparation_deadline)
                    except Exception as exc:
                        cleanup_error = (
                            f"request cleanup failed: {type(exc).__name__}: {exc}"
                        )
                        if payload is None or payload[0] is None:
                            payload = (cleanup_error, None, None)
                        else:
                            logger.exception("Failed to retire closed remote requests")
            # Every TP rank reaches the same CPU broadcast, including dispatch errors.
            error, tokens, lengths = self._tp.broadcast_object(payload, src=0)
            if error is not None:
                raise RuntimeError(f"remote drafter proposal failed: {error}")
            return SpeculatorOutput(
                torch.tensor(tokens, dtype=torch.int64, device=self.device).reshape(
                    len(handle.step.request_ids), self.num_speculative_steps
                ),
                torch.tensor(lengths, dtype=torch.int32, device=self.device),
            )
        finally:
            self._discard_handle(handle)

    def _empty_output(
        self, num_reqs: int, *, device: torch.device | str = "cpu"
    ) -> SpeculatorOutput:
        return SpeculatorOutput(
            torch.zeros(
                (num_reqs, self.num_speculative_steps), dtype=torch.int64, device=device
            ),
            torch.zeros(num_reqs, dtype=torch.int32, device=device),
        )

    def _feature_failure(self, handle: _RemoteHandle, exc: Exception) -> None:
        if cuda_context_fatal(exc) or not (
            isinstance(exc, torch.OutOfMemoryError) or host_allocation_failure(exc)
        ):
            raise exc
        assert self._session is not None
        self._terminal_requests.update(handle.step.request_ids)
        self._session.fail_sequences(
            tuple(
                self._requests[req_id].key
                for req_id in handle.step.request_ids
                if req_id in self._requests
            ),
            status=SpeculatorStatusCode.OUT_OF_MEMORY,
            detail=f"remote feature preparation failed: {exc}",
            deadline=handle.preparation_deadline,
        )
        if handle.output is None:
            handle.output = self._empty_output(len(handle.step.request_ids))
        else:
            handle.output.valid_lengths.zero_()
        handle.session_handle = None

    def _discard_handle(self, handle: _RemoteHandle) -> None:
        if (
            handle.context is not None
            and not (handle.context.dummy_run or handle.context.is_profile)
            and self._tp.rank_in_group == 0
        ):
            self._stats["proposal_seconds"] += time.perf_counter() - handle.started
        handle.context = None
        handle.retained = ()
        handle.output = None
        if self._pending is handle:
            self._pending = None

    def _profile_features(
        self,
        batch: InputBatch,
        hidden: torch.Tensor | None,
        aux: list[torch.Tensor] | None,
        next_prefill_tokens: torch.Tensor,
    ) -> None:
        # Full-size compaction bounds both all-active and mixed fallback batches.
        tokens = list(range(batch.num_tokens))
        rows = list(range(batch.num_reqs))
        features = self._features(
            batch,
            hidden,
            aux,
            next_prefill_tokens,
            token_rows=tokens,
            sequence_rows=rows,
        )
        feature_bytes = self._stats["feature_bytes"]
        self._select(features, tokens, rows)
        self._stats["feature_bytes"] = feature_bytes

    def _dispatch_step(
        self, handle: _RemoteHandle, context: LocalExecutionContext
    ) -> None:
        batch = context.input_batch
        hidden, aux = context.last_hidden_states, context.aux_hidden_states
        num_sampled, num_rejected = context.num_sampled, context.num_rejected
        last_sampled, next_prefill_tokens = (
            context.last_sampled,
            context.next_prefill_tokens,
        )
        num_speculative_tokens = handle.step.num_speculative_tokens
        preparation_deadline = handle.preparation_deadline
        session = self._session
        assert session is not None
        session_started = time.perf_counter()
        starts = batch.query_start_loc_np
        new_requests = []
        for row, req_id in enumerate(batch.req_ids):
            if req_id in self._terminal_requests:
                continue
            if req_id not in self._requests:
                if (
                    not batch.is_prefilling_np[row]
                    or int(batch.positions[int(starts[row])]) != 0
                ):
                    raise ValueError(
                        "new remote sequence requires prefill from position zero"
                    )
                new_requests.append(req_id)
        if new_requests:
            opened_keys: tuple[SequenceKey, ...] = ()
            try:
                opened_keys = session.open_sequences(
                    tuple(next(self._sequence_ids) for _ in new_requests),
                    deadline=preparation_deadline,
                )
                for req_id, key in zip(new_requests, opened_keys):
                    self._requests[req_id] = _Request(key)
            except MemoryError:
                # Remote opens may already exist without a local request owner.
                # Fence the stream before dropping their local records; do not
                # spend a second close-RPC budget after the failed preparation.
                session.close()
                session.close_sequences(opened_keys)
                raise
        active_rows = [
            row
            for row, req_id in enumerate(batch.req_ids)
            if req_id not in self._terminal_requests
            and session.sequence_state(self._requests[req_id].key)
            in (SequenceState.PREFILLING, SequenceState.READY)
        ]
        handle.output = self._empty_output(batch.num_reqs)
        if not active_rows:
            self._stats["session_seconds"] += time.perf_counter() - session_started
            return
        session_seconds = time.perf_counter() - session_started
        active_tokens = (
            None
            if len(active_rows) == batch.num_reqs
            else [
                token
                for row in active_rows
                for token in range(int(starts[row]), int(starts[row + 1]))
            ]
        )
        features = self._features(
            batch,
            hidden,
            aux,
            next_prefill_tokens,
            token_rows=active_tokens,
            sequence_rows=(None if active_tokens is None else active_rows),
        )
        positions = features[TargetFeatureKind.POSITIONS.value]
        sampled = num_sampled.cpu().tolist()
        rejected = num_rejected.cpu().tolist()
        recovery = last_sampled[batch.idx_mapping].reshape(-1).cpu()
        session_started = time.perf_counter()
        chunks = []
        rows, keys, query_lens, accepted, sequence_rows = [], [], [], [], []
        token_indices: list[int] = []
        end = 0
        for feature_row, row in enumerate(active_rows):
            req_id = batch.req_ids[row]
            request = self._requests[req_id]
            query_len = int(starts[row + 1]) - int(starts[row])
            start, end = end, end + query_len
            if int(positions[start]) != request.num_computed:
                raise ValueError(f"remote context discontinuity for request {req_id}")
            if batch.is_prefilling_np[row]:
                remaining = int(batch.prefill_len_np[row]) - request.num_computed
                if remaining <= 0:
                    raise ValueError("remote prefill has no remaining prompt tokens")
                prompt_len = min(query_len, remaining)
                is_final = prompt_len == remaining
                # The scheduler may append speculative placeholders to a final
                # prompt chunk. Keep the physical end for the following row,
                # but never send those rejected positions as prompt context.
                padding = query_len - prompt_len
                if (
                    padding > self.num_speculative_steps
                    or rejected[row] != padding
                    or sampled[row] != int(is_final)
                ):
                    raise ValueError("unexpected sampling counts during remote prefill")
                chunks.append(
                    (
                        request.key,
                        self._select(
                            features, range(start, start + prompt_len), [feature_row]
                        ),
                        is_final,
                    )
                )
                request.num_computed += prompt_len
                if not is_final:
                    continue
                query_len = 0
            else:
                if sampled[row] < 1 or rejected[row] != query_len - sampled[row]:
                    raise ValueError(
                        "remote verification counts do not cover the query"
                    )
                token_indices.extend(range(start, end))
                request.num_computed += sampled[row]
            rows.append(row)
            keys.append(request.key)
            query_lens.append(query_len)
            accepted.append(sampled[row] - 1)
            sequence_rows.append(feature_row)
        session.prefill_many(chunks, deadline=preparation_deadline)
        if rows:
            proposal = RemoteProposalBatch(
                batch_id=next(self._batch_ids),
                keys=tuple(keys),
                accepted_counts=torch.tensor(accepted, dtype=torch.int32),
                recovery_tokens=recovery[rows],
                features=self._select(features, token_indices, sequence_rows),
                query_lens=torch.tensor(query_lens, dtype=torch.int32),
                num_speculative_tokens=num_speculative_tokens,
            )
            handle.session_handle = session.dispatch(
                proposal, deadline=preparation_deadline
            )
            handle.rows = tuple(rows)
            handle.retained = (features, chunks, proposal)
        else:
            handle.retained = (features, chunks)
        self._stats["session_seconds"] += (
            session_seconds + time.perf_counter() - session_started
        )

    def _features(
        self,
        batch: InputBatch,
        hidden: torch.Tensor | None,
        aux: list[torch.Tensor] | None,
        next_prefill_tokens: torch.Tensor,
        *,
        token_rows: Collection[int] | None = None,
        sequence_rows: Collection[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        token_rows = None if token_rows is None else list(token_rows)
        sequence_rows = None if sequence_rows is None else list(sequence_rows)
        source = {
            TargetFeatureKind.TOKEN_IDS.value: batch.input_ids[: batch.num_tokens],
            TargetFeatureKind.POSITIONS.value: batch.positions[: batch.num_tokens],
            TargetFeatureKind.LAST_HIDDEN_STATES.value: (
                None if hidden is None else hidden[: batch.num_tokens]
            ),
        }
        result = {}
        for slot in self._contract.schema.slots:
            if slot.kind == TargetFeatureKind.AUX_HIDDEN_STATES.value:
                assert aux is not None
                tensor = torch.cat(
                    [
                        state[: batch.num_tokens]
                        if token_rows is None
                        else state[token_rows]
                        for state in aux
                    ],
                    dim=-1,
                )
            elif slot.kind == TargetFeatureKind.NEXT_PREFILL_TOKENS.value:
                mapping = batch.idx_mapping
                if sequence_rows is not None:
                    mapping = mapping[sequence_rows]
                tensor = next_prefill_tokens[:1, mapping].T
            else:
                tensor = source[slot.kind]
                if tensor is None:
                    raise ValueError(
                        f"remote feature {slot.kind} requires target hidden states"
                    )
                if token_rows is not None:
                    tensor = tensor[token_rows]
            if tuple(tensor.shape[1:]) != slot.trailing_shape:
                raise ValueError(
                    f"unexpected remote feature shape for {slot.kind}: {tensor.shape}"
                )
            device = (
                self.device
                if self._session is not None
                and self._session.selected_transport == "cuda_ipc"
                and slot.kind
                in (
                    TargetFeatureKind.LAST_HIDDEN_STATES.value,
                    TargetFeatureKind.AUX_HIDDEN_STATES.value,
                )
                else torch.device("cpu")
            )
            result[slot.kind] = tensor.to(
                device=device, dtype=getattr(torch, slot.dtype)
            )
        return result

    def _select(
        self,
        features: dict[str, torch.Tensor],
        token_rows: Collection[int],
        sequence_rows: Collection[int],
    ) -> FeatureBatch:
        slots = tuple(
            features[slot.kind][
                list(sequence_rows if slot.per_sequence else token_rows)
            ]
            for slot in self._contract.schema.slots
        )
        self._stats["feature_bytes"] += sum(t.numel() * t.element_size() for t in slots)
        return FeatureBatch(self._contract.schema.schema_id, slots)
