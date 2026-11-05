# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-step independent AR backend: authoritative token deltas over v3."""

import itertools
import uuid
from collections.abc import Collection
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.v1.spec_decode.contracts import DraftStep, LocalExecutionContext
from vllm.v1.spec_decode.proposal import ProposalHandle, SpeculatorOutput
from vllm.v1.spec_decode.remote.errors import (
    cuda_context_fatal,
    host_allocation_failure,
)
from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.runtime.token_schema import (
    create_token_runtime_contract,
)
from vllm.v1.spec_decode.remote.runtime.token_state import TokenAdvanceRow
from vllm.v1.spec_decode.remote.token_pool import (
    TokenDraftPool,
    TokenPoolHandle,
    TokenPoolRow,
)
from vllm.v1.spec_decode.remote.token_protocol import TokenHello
from vllm.v1.spec_decode.remote.token_session import (
    TokenOnlyRemoteSession,
    TokenProposalHandle,
)
from vllm.v1.worker.gpu.spec_decode.remote import RemoteSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator


@dataclass
class _Request:
    key: SequenceKey
    target_computed: int = 0


@dataclass(eq=False)
class _TokenHandle(ProposalHandle):
    owner: object
    round_id: int
    step: DraftStep
    context: LocalExecutionContext | None
    rpc: TokenProposalHandle | TokenPoolHandle | None = None
    rows: tuple[int, ...] = ()
    output: SpeculatorOutput | None = None
    error: str | None = None


class TokenOnlyRemoteSpeculator(BaseSpeculator):
    supports_mm_inputs = False
    draft_logits = None

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        RemoteSpeculator._validate_config(vllm_config)
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("token-only remote drafting currently requires target TP1")
        spec = vllm_config.speculative_config
        assert spec is not None
        remote = spec.get_remote_draft_config()
        assert remote is not None
        if remote.transport not in ("inline", "auto"):
            raise ValueError("token-only remote drafting requires inline transport")
        self.device = device
        self.num_speculative_steps = spec.num_speculative_tokens
        self._tp = get_tp_group()
        self._owner = object()
        self._rounds = itertools.count()
        self._sequences = itertools.count()
        self._requests: dict[str, _Request] = {}
        self._terminal_requests: set[str] = set()
        self._pending: _TokenHandle | None = None
        self._closed = False
        self._session: TokenOnlyRemoteSession | TokenDraftPool | None = None
        error = None
        if self._tp.rank_in_group == 0:
            try:
                contract = create_token_runtime_contract(vllm_config)
                hello = TokenHello(
                    uuid.uuid4().hex,
                    contract.target_fingerprint,
                    contract.draft_fingerprint,
                    contract.tokenizer_fingerprint,
                    self.num_speculative_steps,
                    vllm_config.model_config.get_vocab_size(),
                    vllm_config.scheduler_config.max_num_seqs,
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    vllm_config.model_config.max_model_len,
                )
                endpoints = spec.get_remote_draft_endpoints()
                options = dict(
                    failure_policy=remote.failure_policy,
                    request_timeout_ms=remote.request_timeout_ms,
                    startup_timeout_ms=remote.startup_timeout_ms,
                )
                if len(endpoints) > 1:
                    assert spec.execution is not None
                    self._session = TokenDraftPool(
                        endpoints,
                        hello,
                        routing_policy=spec.execution.routing_policy,
                        **options,
                    )
                else:
                    self._session = TokenOnlyRemoteSession(
                        remote.endpoint, hello, **options
                    )
                self._session.connect()
            except Exception as exc:
                self.close()
                error = f"{type(exc).__name__}: {exc}"
        error = self._tp.broadcast_object(error, src=0)
        if error is not None:
            raise RuntimeError(f"token drafter startup failed: {error}")

    def get_routing_stats(self) -> dict[str, int | tuple[int, ...]]:
        if isinstance(self._session, TokenDraftPool):
            return self._session.get_routing_stats()
        return {}

    def init_cudagraph_manager(self, cudagraph_mode) -> None:
        pass

    def capture(self) -> None:
        pass

    def _empty(self, n, *, device="cpu") -> SpeculatorOutput:
        return SpeculatorOutput(
            torch.zeros(
                (n, self.num_speculative_steps), dtype=torch.int64, device=device
            ),
            torch.zeros(n, dtype=torch.int32, device=device),
        )

    def _allocation_failed(self, handle: _TokenHandle, error: Exception) -> None:
        if cuda_context_fatal(error) or not (
            isinstance(error, torch.OutOfMemoryError) or host_allocation_failure(error)
        ):
            raise error
        assert self._session is not None
        self._terminal_requests.update(handle.step.request_ids)
        self._session.fail(f"token metadata allocation failed: {error}")
        if handle.output is None:
            handle.output = self._empty(len(handle.step.request_ids))
        else:
            handle.output.valid_lengths.zero_()

    def dispatch(
        self, step: DraftStep, context: LocalExecutionContext
    ) -> ProposalHandle:
        if self._closed or self._pending is not None:
            raise RuntimeError(
                "token backend is closed or still has a pending proposal"
            )
        handle = _TokenHandle(self._owner, next(self._rounds), step, context)
        self._pending = handle
        if context.dummy_run or context.is_profile:
            handle.output = self._empty(len(step.request_ids), device=self.device)
            return handle
        if self._tp.rank_in_group == 0:
            try:
                handle.output = self._empty(len(step.request_ids))
                if not 0 <= step.num_speculative_tokens <= self.num_speculative_steps:
                    raise ValueError(
                        "token proposal budget exceeds configured capacity"
                    )
                if step != DraftStep.from_batch(
                    context.input_batch, step.num_speculative_tokens
                ):
                    raise ValueError(
                        "token step differs from its retained local context"
                    )
                if torch.any(
                    context.temperature[context.input_batch.idx_mapping] != 0
                ).item():
                    raise ValueError(
                        "token-only remote drafting requires greedy sampling"
                    )
                self._dispatch_tokens(handle, context)
            except Exception as exc:
                try:
                    self._allocation_failed(handle, exc)
                except Exception as error:
                    handle.error = f"{type(error).__name__}: {error}"
                    if self._session is not None:
                        self._session._invalidate()
        return handle

    def _dispatch_tokens(self, handle, context) -> None:
        session = self._session
        assert session is not None
        deadline = session.deadline()
        batch = context.input_batch
        starts = batch.query_start_loc_np
        if all(
            req_id in self._terminal_requests
            or (
                req_id in self._requests
                and not session.active(self._requests[req_id].key)
            )
            for req_id in handle.step.request_ids
        ):
            return
        # These are actual token IDs, never hidden states or embeddings.
        tokens = batch.input_ids[: batch.num_tokens].cpu().tolist()
        positions = batch.positions[: batch.num_tokens].cpu().tolist()
        sampled = context.num_sampled.cpu().tolist()
        rejected = context.num_rejected.cpu().tolist()
        recovery = context.last_sampled[batch.idx_mapping].reshape(-1).cpu().tolist()
        rows, prepared, indices = [], [], []
        pooled = isinstance(session, TokenDraftPool)
        for i, req_id in enumerate(handle.step.request_ids):
            if req_id in self._terminal_requests:
                continue
            start, end = int(starts[i]), int(starts[i + 1])
            q = end - start
            if q < 1:
                raise ValueError("target query cannot be empty")
            request = self._requests.get(req_id)
            open_required = request is None
            if request is None:
                if not handle.step.is_prefilling[i] or positions[start] != 0:
                    raise ValueError(
                        "new token requests must prefill from position zero"
                    )
                key = (
                    session.reserve_sequence(next(self._sequences), deadline=deadline)
                    if isinstance(session, TokenDraftPool)
                    else session.open_sequence(next(self._sequences), deadline=deadline)
                )
                try:
                    request = _Request(key)
                    self._requests[req_id] = request
                except MemoryError:
                    # No wrapper owns this remote key. Fence before dropping it,
                    # so cleanup cannot wait for another RPC after this budget.
                    session._invalidate()
                    session.close_sequences((key,))
                    raise
            if not session.active(request.key):
                continue
            if positions[start:end] != list(
                range(request.target_computed, request.target_computed + q)
            ):
                raise ValueError("token target positions are discontinuous")
            prefill = None
            final = False
            if handle.step.is_prefilling[i]:
                remaining = int(batch.prefill_len_np[i]) - request.target_computed
                if remaining <= 0:
                    raise ValueError("token prefill has no remaining prompt tokens")
                prompt_tokens = min(q, remaining)
                final = prompt_tokens == remaining
                # A final prompt chunk may be scheduled with speculative
                # placeholder slots. Rejection postprocessing rejects those
                # physical logits; they are not authoritative prompt tokens.
                padding = q - prompt_tokens
                if (
                    padding > self.num_speculative_steps
                    or rejected[i] != padding
                    or sampled[i] != int(final)
                ):
                    raise ValueError("unexpected sampling counts in token prefill")
                prefill = tuple(tokens[start : start + prompt_tokens])
                if not pooled:
                    session.prefill(
                        request.key, prefill, is_final=final, deadline=deadline
                    )
                request.target_computed += prompt_tokens
                if not final or not session.active(request.key):
                    if pooled and session.active(request.key):
                        prepared.append(
                            TokenPoolRow(
                                request.key, open_required, prefill, final, None
                            )
                        )
                        indices.append(i)
                    continue
                accepted, offered = 0, 0
                delta = (int(recovery[i]),)
            else:
                if sampled[i] < 1 or rejected[i] != q - sampled[i]:
                    raise ValueError("token verification counts do not cover the query")
                state = session.records[request.key]
                accepted = int(sampled[i]) - 1
                # The runner can physically schedule/mask a wider draft tail
                # than the last round's logical valid prefix, including K0.
                offered = min(q - 1, len(state.proposal_tokens))
                if not 0 <= accepted <= offered:
                    raise ValueError(
                        "target accepted outside the previous valid proposal"
                    )
                skip = state.committed - request.target_computed
                if skip != 1:
                    raise ValueError("token logical prefix lost its pending correction")
                delta = tuple(tokens[start + skip : start + int(sampled[i])]) + (
                    int(recovery[i]),
                )
                request.target_computed += int(sampled[i])
            if pooled:
                prepared.append(
                    TokenPoolRow(
                        request.key,
                        open_required,
                        prefill,
                        final,
                        (offered, accepted, delta),
                    )
                )
            else:
                state = session.records[request.key]
                rows.append(
                    TokenAdvanceRow(
                        request.key,
                        state.committed,
                        state.proposal_id,
                        offered,
                        accepted,
                        delta,
                    )
                )
            indices.append(i)
        handle.rows = tuple(indices)
        if prepared:
            assert isinstance(session, TokenDraftPool)
            handle.rpc = session.dispatch_prepared(
                handle.round_id,
                tuple(prepared),
                handle.step.num_speculative_tokens,
                deadline=deadline,
            )
        elif rows:
            handle.rpc = session.dispatch(
                handle.round_id,
                tuple(rows),
                handle.step.num_speculative_tokens,
                deadline=deadline,
            )

    def _owned(self, handle) -> _TokenHandle:
        if (
            not isinstance(handle, _TokenHandle)
            or handle.owner is not self._owner
            or self._pending is not handle
            or handle.context is None
        ):
            raise RuntimeError(
                "stale, foreign, cancelled or already collected token handle"
            )
        return handle

    def _discard(self, handle) -> None:
        handle.context = None
        handle.output = None
        handle.rpc = None
        self._pending = None

    def collect(self, handle: ProposalHandle) -> SpeculatorOutput:
        handle = self._owned(handle)
        context = handle.context
        assert context is not None
        try:
            if context.dummy_run or context.is_profile:
                assert handle.output is not None
                return handle.output
            payload = None
            if self._tp.rank_in_group == 0:
                try:
                    if handle.error is None and handle.rpc is not None:
                        assert self._session is not None and handle.output is not None
                        if isinstance(handle.rpc, TokenPoolHandle):
                            assert isinstance(self._session, TokenDraftPool)
                            result = self._session.collect(handle.rpc)
                        else:
                            assert isinstance(self._session, TokenOnlyRemoteSession)
                            result = self._session.collect(handle.rpc)
                        handle.output.token_ids[list(handle.rows)] = torch.tensor(
                            result.token_ids, dtype=torch.int64
                        )
                        handle.output.valid_lengths[list(handle.rows)] = torch.tensor(
                            result.valid_lengths, dtype=torch.int32
                        )
                    output = handle.output if handle.error is None else None
                    payload = (
                        handle.error,
                        None if output is None else output.token_ids.tolist(),
                        None if output is None else output.valid_lengths.tolist(),
                    )
                except Exception as exc:
                    try:
                        self._allocation_failed(handle, exc)
                        output = handle.output
                        assert output is not None
                        payload = (
                            None,
                            output.token_ids.tolist(),
                            output.valid_lengths.tolist(),
                        )
                    except Exception as fallback_error:
                        handle.error = (
                            f"{type(fallback_error).__name__}: {fallback_error}"
                        )
                        if self._session is not None:
                            self._session._invalidate()
                        payload = (handle.error, None, None)
            error, tokens, lengths = self._tp.broadcast_object(payload, src=0)
            if error is not None:
                raise RuntimeError(f"token drafter proposal failed: {error}")
            return SpeculatorOutput(
                torch.tensor(tokens, dtype=torch.int64, device=self.device).reshape(
                    len(handle.step.request_ids), self.num_speculative_steps
                ),
                torch.tensor(lengths, dtype=torch.int32, device=self.device),
            )
        finally:
            self._discard(handle)

    def cancel(self, handle: ProposalHandle, reason: str) -> None:
        handle = self._owned(handle)
        error = None
        try:
            if self._session is not None:
                try:
                    if handle.rpc is not None:
                        if isinstance(handle.rpc, TokenPoolHandle):
                            assert isinstance(self._session, TokenDraftPool)
                            self._session.cancel(handle.rpc)
                        else:
                            assert isinstance(self._session, TokenOnlyRemoteSession)
                            self._session.cancel(handle.rpc)
                    else:
                        self._session._invalidate()
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            error = self._tp.broadcast_object(error, src=0)
            if error is not None:
                raise RuntimeError(f"token cancellation failed: {error}")
        finally:
            self._discard(handle)

    def close_requests(self, req_ids: Collection[str]) -> None:
        if not req_ids:
            return
        self._terminal_requests.difference_update(req_ids)
        if self._pending is not None and set(req_ids).intersection(
            self._pending.step.request_ids
        ):
            self.cancel(self._pending, "request closed")
        error = None
        if self._session is not None:
            try:
                keys = tuple(
                    self._requests.pop(req_id).key
                    for req_id in req_ids
                    if req_id in self._requests
                )
                self._session.close_sequences(keys)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        error = self._tp.broadcast_object(error, src=0)
        if error is not None:
            raise RuntimeError(f"token request cleanup failed: {error}")

    def close(self) -> None:
        self._closed = True
        if self._session is not None:
            self._session.close()
            self._session = None
        self._requests.clear()
        self._terminal_requests.clear()
        if self._pending is not None:
            self._discard(self._pending)

    def propose(
        self,
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
        dp_sync_state=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
        num_speculative_tokens=None,
    ):
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
        return self.collect(
            self.dispatch(DraftStep.from_batch(input_batch, k), context)
        ).to_legacy_tokens()
