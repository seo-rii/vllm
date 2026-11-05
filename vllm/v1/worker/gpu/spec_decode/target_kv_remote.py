# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-step Gemma4 Q-only proposals from private remote target-KV snapshots."""

import itertools
import uuid
from collections.abc import Collection
from dataclasses import dataclass, replace

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.spec_decode.contracts import DraftStep, LocalExecutionContext
from vllm.v1.spec_decode.proposal import ProposalHandle, SpeculatorOutput
from vllm.v1.spec_decode.remote.errors import (
    host_allocation_failure,
    runtime_unusable,
)
from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.runtime.target_kv_schema import (
    create_target_kv_contract,
)
from vllm.v1.spec_decode.remote.target_kv import (
    SnapshotIdentity,
    TargetKVSource,
    select_draft_position,
    snapshot_target_kv,
)
from vllm.v1.spec_decode.remote.target_kv_pool import (
    TargetKVDraftPool,
    TargetKVPoolHandle,
)
from vllm.v1.spec_decode.remote.target_kv_protocol import (
    TargetKVHello,
    snapshot_batch_wire_plan,
)
from vllm.v1.spec_decode.remote.target_kv_session import (
    TargetKVProposalHandle,
    TargetKVRemoteSession,
)
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

logger = init_logger(__name__)


@dataclass(eq=False)
class _SnapshotHandle(ProposalHandle):
    owner: object
    step: DraftStep
    context: LocalExecutionContext | None
    rpc: TargetKVProposalHandle | TargetKVPoolHandle | None = None
    rpc_rows: tuple[int, ...] = ()
    error: Exception | None = None


class TargetKVRemoteSpeculator(BaseSpeculator):
    supports_mm_inputs = False
    draft_logits = None
    num_speculative_steps = 1

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.contract = create_target_kv_contract(vllm_config)
        self.config = vllm_config
        self.device = device
        self._owner = object()
        self._pending: _SnapshotHandle | None = None
        self._requests: dict[str, SequenceKey] = {}
        self._terminal_requests: set[str] = set()
        self._sequences = itertools.count()
        self._rounds = itertools.count()
        self._closed = False
        self.stats = {"dispatched": 0, "valid": 0, "fallback": 0}
        spec = vllm_config.speculative_config
        assert spec is not None
        remote = spec.get_remote_draft_config()
        assert remote is not None
        contract = self.contract
        endpoints = spec.get_remote_draft_endpoints()
        hello = TargetKVHello(
            uuid.uuid4().hex,
            contract.fingerprint,
            contract.hidden_size,
            contract.vocab_size,
            contract.snapshot_budget.max_model_len,
            contract.snapshot_budget.max_pages,
            contract.snapshot_budget.max_bytes,
            max_num_seqs=contract.max_num_seqs,
        )
        self._session: TargetKVDraftPool | TargetKVRemoteSession
        if len(endpoints) > 1:
            self._session = TargetKVDraftPool(
                endpoints,
                hello,
                expected_specs=contract.expected_specs,
                failure_policy=remote.failure_policy,
                request_timeout_ms=remote.request_timeout_ms,
                startup_timeout_ms=remote.startup_timeout_ms,
            )
        else:
            self._session = TargetKVRemoteSession(
                remote.endpoint,
                hello,
                expected_specs=contract.expected_specs,
                failure_policy=remote.failure_policy,
                request_timeout_ms=remote.request_timeout_ms,
                startup_timeout_ms=remote.startup_timeout_ms,
            )
        try:
            self._session.connect()
        except Exception:
            self._session.close()
            raise

    def init_cudagraph_manager(self, cudagraph_mode) -> None:
        pass

    def capture(self) -> None:
        pass

    @staticmethod
    def _select_hidden(hidden: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
        return hidden.index_select(
            0, torch.tensor(indices, dtype=torch.int64, device=hidden.device)
        )

    def _profile_snapshot(self, context: LocalExecutionContext) -> None:
        # Reserve the export peak alongside the still-live target output. No
        # cache views, request state or RPCs exist during target KV profiling.
        shapes = []
        length = self.contract.snapshot_budget.max_model_len
        pages = nbytes = 0
        for _ in range(self.contract.max_num_seqs):
            for spec in self.contract.expected_specs.values():
                count = (length + spec.block_size - 1) // spec.block_size
                if spec.sliding_window is not None:
                    count = min(
                        count,
                        (spec.sliding_window + 2 * spec.block_size - 2)
                        // spec.block_size,
                    )
                pages += count
                nbytes += count * spec.unpadded_page_size_bytes
                shapes.append(
                    (count, spec.num_kv_heads, spec.block_size, 2 * spec.head_size)
                )
        self.contract.snapshot_budget.check(length, pages, nbytes)
        retained = [
            torch.zeros(shape, dtype=torch.bfloat16, device=self.device)
            for shape in shapes
        ]
        hidden = context.last_hidden_states
        if hidden is None or hidden.ndim != 2 or not hidden.shape[0]:
            raise ValueError("target-KV profiling requires target hidden states")
        retained.append(self._select_hidden(hidden, (0,) * self.contract.max_num_seqs))
        # Inline transfer allocates CPU storage, while every exported GPU page
        # remains live until the synchronous send has consumed the snapshot.
        for tensor in retained:
            tensor.cpu()

    def dispatch(
        self, step: DraftStep, context: LocalExecutionContext
    ) -> ProposalHandle:
        if self._closed or self._pending is not None:
            raise RuntimeError("target-KV backend is closed or has a pending proposal")
        handle = _SnapshotHandle(self._owner, step, context)
        self._pending = handle
        try:
            if (
                len(step.request_ids) > self.contract.max_num_seqs
                or len(set(step.request_ids)) != len(step.request_ids)
                or step.num_speculative_tokens not in (0, 1)
            ):
                raise ValueError("target-KV serving exceeds request capacity or K0/K1")
            if context.is_profile:
                self._profile_snapshot(context)
            elif not context.dummy_run and step.request_ids:
                if step != DraftStep.from_batch(
                    context.input_batch, step.num_speculative_tokens
                ):
                    raise ValueError("target-KV step differs from retained batch")
                self._dispatch_snapshot(handle, context)
        except Exception as error:
            if (
                context.is_profile
                or runtime_unusable(error)
                or not (
                    isinstance(error, torch.OutOfMemoryError)
                    or host_allocation_failure(error)
                )
            ):
                self._session.close()
                handle.error = error
            else:
                try:
                    self._session.fail(
                        f"target-KV preparation allocation failed: {error}"
                    )
                    self.stats["fallback"] += len(step.request_ids)
                except Exception as policy_error:
                    handle.error = policy_error
        return handle

    def _dispatch_snapshot(self, handle, context) -> None:
        session = self._session
        deadline = session.deadline()
        batch = context.input_batch
        if context.mm_inputs is not None:
            raise ValueError("remote Gemma4 requires text-only inputs")
        new_ids = tuple(
            req_id
            for req_id in handle.step.request_ids
            if req_id not in self._requests and req_id not in self._terminal_requests
        )
        if new_ids:
            # Reserve terminal ownership before opening any remote sequences.
            self._terminal_requests.update(new_ids)
            keys: tuple[SequenceKey, ...] = ()
            try:
                keys = session.open_sequences(
                    tuple(next(self._sequences) for _ in new_ids), deadline=deadline
                )
                self._requests.update(zip(new_ids, keys, strict=True))
            except BaseException:
                for req_id in new_ids:
                    self._requests.pop(req_id, None)
                try:
                    session.close_sequences(keys, deadline=deadline)
                except Exception as error:
                    logger.warning("Target-KV registration rollback failed: %s", error)
                raise
            self._terminal_requests.difference_update(new_ids)
        active_rows = [
            row
            for row, req_id in enumerate(handle.step.request_ids)
            if req_id not in self._terminal_requests
            and session.active(self._requests[req_id])
        ]
        self.stats["fallback"] += len(handle.step.request_ids) - len(active_rows)
        if not active_rows:
            return
        if handle.step.num_speculative_tokens == 0:
            return
        if torch.any(context.temperature[batch.idx_mapping] != 0).item():
            raise ValueError("remote Gemma4 requires greedy sampling")
        positions = {}
        for row in active_rows:
            position = select_draft_position(
                query_start=int(batch.query_start_loc_np[row]),
                query_length=handle.step.query_lens[row],
                sequence_length=int(batch.seq_lens[row].item()),
                num_rejected=int(context.num_rejected[row].item()),
                num_sampled=int(context.num_sampled[row].item()),
            )
            if position is not None:
                if (
                    int(batch.positions[position.hidden_index].item())
                    != position.position
                ):
                    raise ValueError(
                        "target-KV query position differs from materialized context"
                    )
                positions[row] = position
        if not positions:
            return
        hidden = context.last_hidden_states
        if (
            hidden is None
            or hidden.ndim != 2
            or hidden.shape[1] != self.contract.hidden_size
            or hidden.dtype != torch.bfloat16
        ):
            raise ValueError("target-KV proposal requires the BF16 target hidden row")
        if context.aux_hidden_states:
            raise ValueError("Gemma4 does not consume auxiliary target features")
        snapshot_batch_wire_plan(
            self.contract.expected_specs,
            tuple(position.materialized_length for position in positions.values()),
            self.contract.snapshot_budget,
        )
        owners = []
        layers = self.config.compilation_config.static_forward_context
        for canonical, index in self.contract.source_indices.items():
            names = [
                name
                for name in layers
                if name.endswith(f".layers.{index}.self_attn.attn")
            ]
            if len(names) != 1 or context.attn_metadata is None:
                raise ValueError("cannot resolve the exact target-KV owner")
            name = names[0]
            owner = layers[name]
            metadata = context.attn_metadata[name]
            if (
                not isinstance(metadata, TritonAttentionMetadata)
                or not metadata.causal
                or metadata.mm_prefix_range_tensor is not None
                or metadata.rswa_prefix_lens is not None
            ):
                raise ValueError("target-KV requires causal text Triton metadata")
            cache = owner.kv_cache
            actual = replace(
                owner.get_kv_cache_spec(self.config), block_size=cache.shape[2]
            )
            if actual != self.contract.expected_specs[canonical]:
                raise ValueError("live target-KV geometry differs from the contract")
            owners.append((canonical, actual, cache, metadata.block_table))
        snapshots = []
        recovery = []
        layout = self.config.cache_config.get_resolved_kv_cache_layout()
        for row, position in positions.items():
            key = self._requests[handle.step.request_ids[row]]
            snapshots.append(
                snapshot_target_kv(
                    SnapshotIdentity(key, session.epoch_for(key), next(self._rounds)),
                    position.materialized_length,
                    [
                        TargetKVSource(name, spec, cache, tables[row], layout)
                        for name, spec, cache, tables in owners
                    ],
                    budget=self.contract.snapshot_budget,
                )
            )
            recovery.append(int(context.last_sampled[batch.idx_mapping[row]].item()))
        selected_hidden = self._select_hidden(
            hidden, tuple(position.hidden_index for position in positions.values())
        )
        if isinstance(session, TargetKVDraftPool):
            # Finish producer-stream reads before handing CPU storage to the
            # endpoint drivers, which must not read CUDA tensors on their streams.
            snapshots = [
                replace(
                    snapshot,
                    layers=tuple(
                        replace(layer, pages=layer.pages.cpu())
                        for layer in snapshot.layers
                    ),
                )
                for snapshot in snapshots
            ]
            selected_hidden = selected_hidden.cpu()
        handle.rpc = session.dispatch_batch(
            tuple(snapshots),
            tuple(recovery),
            selected_hidden,
            k=1,
            deadline=deadline,
        )
        handle.rpc_rows = tuple(positions)
        self.stats["dispatched"] += len(snapshots)

    def _owned(self, handle) -> _SnapshotHandle:
        if (
            not isinstance(handle, _SnapshotHandle)
            or handle.owner is not self._owner
            or self._pending is not handle
            or handle.context is None
        ):
            raise RuntimeError("stale, foreign or already consumed target-KV handle")
        return handle

    def _discard(self, handle) -> None:
        handle.context = None
        handle.rpc = None
        handle.rpc_rows = ()
        self._pending = None

    def collect(self, handle: ProposalHandle) -> SpeculatorOutput:
        handle = self._owned(handle)
        try:
            if handle.error is not None:
                raise handle.error
            n = len(handle.step.request_ids)
            tokens, lengths = [[0] for _ in range(n)], [0] * n
            if handle.rpc is not None:
                if isinstance(self._session, TargetKVDraftPool):
                    results = self._session.collect_batch(handle.rpc)
                else:
                    assert isinstance(handle.rpc, TargetKVProposalHandle)
                    results = self._session.collect_batch(handle.rpc)
                for row, result in zip(handle.rpc_rows, results, strict=True):
                    lengths[row] = result.valid_length
                    if result.valid_length:
                        tokens[row][0] = result.token_ids[0]
                        self.stats["valid"] += 1
                    else:
                        self.stats["fallback"] += 1
            return SpeculatorOutput(
                torch.tensor(tokens, dtype=torch.int64, device=self.device).reshape(
                    n, 1
                ),
                torch.tensor(lengths, dtype=torch.int32, device=self.device),
            )
        except BaseException as error:
            if handle.rpc is not None and handle.rpc.active:
                try:
                    if isinstance(self._session, TargetKVDraftPool):
                        self._session.cancel(handle.rpc)
                    else:
                        assert isinstance(handle.rpc, TargetKVProposalHandle)
                        self._session.cancel(handle.rpc)
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        "target-KV collection and cancellation failed",
                        [error, cleanup_error],
                    ) from error
            raise
        finally:
            self._discard(handle)

    def cancel(self, handle: ProposalHandle, reason: str) -> None:
        handle = self._owned(handle)
        try:
            if handle.rpc is not None:
                if isinstance(self._session, TargetKVDraftPool):
                    self._session.cancel(handle.rpc)
                else:
                    assert isinstance(handle.rpc, TargetKVProposalHandle)
                    self._session.cancel(handle.rpc)
        finally:
            self._discard(handle)

    def close_requests(self, req_ids: Collection[str]) -> None:
        if self._pending is not None and set(req_ids).intersection(
            self._pending.step.request_ids
        ):
            self.cancel(self._pending, "request closed")
        keys = tuple(
            self._requests.pop(req_id) for req_id in req_ids if req_id in self._requests
        )
        self._terminal_requests.difference_update(req_ids)
        if keys:
            self._session.close_sequences(keys)

    def close(self) -> None:
        self._closed = True
        self._session.close()
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
        step = DraftStep.from_batch(
            input_batch, 1 if num_speculative_tokens is None else num_speculative_tokens
        )
        return self.collect(self.dispatch(step, context)).to_legacy_tokens()
