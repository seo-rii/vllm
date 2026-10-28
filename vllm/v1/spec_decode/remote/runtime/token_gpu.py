# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-only native model/attention execution, without a target shell or engine."""

import inspect
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.config import VllmConfig, replace, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.spec_decode.remote.errors import CompletionUnknownError, runtime_unusable
from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.runtime.gpu import (
    GPUDraftStepExecutor,
    StandaloneParallelContext,
    make_input_batch,
)
from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
from vllm.v1.spec_decode.remote.runtime.step import (
    BlockTableUpdate,
    DraftStepPlan,
    DraftStepRow,
)
from vllm.v1.spec_decode.remote.runtime.token_adapter import TokenOnlyDraftAdapter
from vllm.v1.spec_decode.remote.runtime.token_schema import (
    create_token_runtime_contract,
)
from vllm.v1.spec_decode.remote.runtime.token_state import (
    TokenAdvanceRow,
    TokenForwardRow,
    TokenStateMachine,
    TokenStepExecutor,
)

if TYPE_CHECKING:
    from vllm.v1.spec_decode.remote.runtime.token_cache import TokenCacheOptions
    from vllm.v1.spec_decode.remote.runtime.tp_executor import TokenTPExecutor


def validate_token_model_inputs(config: VllmConfig, model: torch.nn.Module) -> None:
    """Fail closed when a native model needs inputs or state we do not supply."""
    from vllm.v1.worker.gpu.model_states import resolve_model_state_cls
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    try:
        inspect.signature(model.forward).bind(input_ids=None, positions=None)
    except TypeError as error:
        raise ValueError(
            "token-only draft must consume tokens and positions alone"
        ) from error
    if resolve_model_state_cls(config, model) is not DefaultModelState:
        raise ValueError("token-only draft requires the default decoder model state")


def create_token_executor_config(
    config: VllmConfig,
    *,
    parallel_context: StandaloneParallelContext | None = None,
    branch_slots: int = 0,
) -> VllmConfig:
    """Give the standalone model its own geometry and rank-specific topology."""
    spec = config.speculative_config
    assert spec is not None and spec.draft_model_config is not None
    resource = replace(
        config,
        model_config=spec.draft_model_config,
        parallel_config=spec.draft_parallel_config,
        quant_config=None,
        cache_config=(
            replace(config.cache_config, cache_dtype=spec.kv_cache_dtype)
            if spec.kv_cache_dtype is not None
            else config.cache_config
        ),
    )
    resource = spec.apply_draft_overrides(resource)
    if parallel_context is not None:
        resource = replace(
            resource,
            parallel_config=replace(
                resource.parallel_config, rank=parallel_context.rank
            ),
        )
    if type(branch_slots) is not int or branch_slots < 0:
        raise ValueError("branch slots must be a nonnegative integer")
    if branch_slots:
        resident_slots = resource.scheduler_config.max_num_seqs + branch_slots
        resource = replace(
            resource,
            scheduler_config=replace(
                resource.scheduler_config,
                max_num_seqs=resident_slots,
                max_num_batched_tokens=max(
                    resource.scheduler_config.max_num_batched_tokens, resident_slots
                ),
                max_model_len=resource.model_config.max_model_len,
                is_encoder_decoder=resource.model_config.is_encoder_decoder,
            ),
        )
    return resource


class TokenGPUStepExecutor(GPUDraftStepExecutor, TokenStepExecutor):
    def __init__(
        self,
        config: VllmConfig,
        *,
        device: torch.device,
        kv_cache_bytes: int,
        parallel_context: StandaloneParallelContext | None = None,
        branch_slots: int = 0,
    ):
        self._token_contract = create_token_runtime_contract(config)
        spec = config.speculative_config
        assert spec is not None and spec.draft_model_config is not None
        if (
            config.parallel_config.tensor_parallel_size != 1
            or spec.draft_parallel_config.tensor_parallel_size
            != (1 if parallel_context is None else parallel_context.world_size)
        ):
            raise ValueError(
                "token-only runtime topology differs from its worker group"
            )
        if spec.draft_model_config.quantization is not None:
            raise ValueError(
                "token-only first runtime requires unquantized draft weights"
            )
        resource = create_token_executor_config(
            config, parallel_context=parallel_context, branch_slots=branch_slots
        )
        super().__init__(
            resource,
            device=device,
            kv_cache_bytes=kv_cache_bytes,
            parallel_context=parallel_context,
        )

    def _create_contract(self, config):
        return self._token_contract

    def _load_speculator(self, config, device, shared_weights_source):
        from vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator import (
            StandaloneARSpeculator,
        )

        speculator = StandaloneARSpeculator(config, device)
        speculator.load_standalone_model()
        validate_token_model_inputs(config, speculator.model)
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        from vllm.v1.spec_decode.remote.runtime.kv_layout import (
            select_standalone_kv_spec,
        )
        from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec

        actual = select_standalone_kv_spec(get_kv_cache_spec(config))
        if type(actual) is not FullAttentionSpec:
            raise ValueError(
                "token-only loaded draft requires homogeneous full attention KV"
            )
        return speculator

    @property
    def max_num_reqs(self) -> int:
        return self.input_buffers.max_num_reqs

    @torch.inference_mode()
    def copy_blocks(self, copies: tuple[tuple[int, int], ...]) -> None:
        from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
        from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

        if not copies:
            return
        torch.accelerator.set_device_index(self.device)
        failure: BaseException | None = None
        try:
            copy_kv_cache_blocks_inplace(
                self.kv_caches.values(),
                self.num_blocks,
                [
                    KVCacheBlockCopy(source, destination)
                    for source, destination in copies
                ],
            )
        except BaseException as error:
            failure = error
            raise
        finally:
            if failure is None or not runtime_unusable(failure):
                try:
                    torch.accelerator.synchronize(self.device)
                except BaseException as completion_error:
                    unknown = CompletionUnknownError(
                        "outcome cache copy completion is unknown"
                    )
                    if failure is not None:
                        failure.add_note(
                            f"copy completion also failed: {completion_error}"
                        )
                        raise unknown from failure
                    raise unknown from completion_error

    def close_sequences(self, keys: tuple[SequenceKey, ...]) -> None:
        """TokenStateMachine owns the dense executor's request KV and slots."""

    def forward(self, key, req_index, start, tokens, blocks, *, sample):
        return self.forward_batch(
            (
                TokenForwardRow(
                    key,
                    req_index,
                    start,
                    tokens,
                    blocks,
                    sample,
                ),
            )
        )[0]

    @torch.inference_mode()
    def forward_batch(
        self, rows: tuple[TokenForwardRow, ...]
    ) -> tuple[int | None, ...]:
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer

        if not rows:
            return ()
        n = sum(len(row.tokens) for row in rows)
        if (
            any(not row.tokens for row in rows)
            or n > self.max_num_tokens
            or len(rows) > self.max_num_reqs
            or len({row.req_index for row in rows}) != len(rows)
        ):
            raise ValueError("token forward must fit the configured batch budgets")
        lengths = np.array([len(row.tokens) for row in rows], dtype=np.int32)
        starts = np.array([row.start for row in rows], dtype=np.int32)
        qsl = np.concatenate(
            (np.zeros(1, dtype=np.int32), np.cumsum(lengths, dtype=np.int32))
        )
        positions = np.concatenate(
            [
                np.arange(row.start, row.start + len(row.tokens), dtype=np.int64)
                for row in rows
            ]
        )
        plan = DraftStepPlan(
            rows=tuple(DraftStepRow(row.key, len(row.tokens), 0) for row in rows),
            req_indices=np.array([row.req_index for row in rows], dtype=np.int32),
            num_scheduled_tokens=lengths,
            query_start_loc=qsl,
            num_computed_tokens=starts,
            seq_lens=starts + lengths,
            positions=positions,
            slot_mapping=np.array(
                [
                    row.blocks[p // self.block_size] * self.block_size
                    + p % self.block_size
                    for row in rows
                    for p in range(row.start, row.start + len(row.tokens))
                ],
                dtype=np.int64,
            ),
            block_table_updates={
                row.req_index: BlockTableUpdate(0, list(row.blocks)) for row in rows
            },
            num_sampled=np.zeros(len(rows), dtype=np.int32),
            num_rejected=np.zeros(len(rows), dtype=np.int32),
            last_sampled=np.zeros(len(rows), dtype=np.int64),
            next_prefill_tokens=np.zeros((len(rows), 1), dtype=np.int32),
        )
        torch.accelerator.set_device_index(self.device)
        with set_current_vllm_config(self.vllm_config):
            self.prepare_block_tables(plan)
            batch = make_input_batch(
                plan,
                torch.tensor(
                    [token for row in rows for token in row.tokens], dtype=torch.int32
                ),
                self.input_buffers,
            )
            tables = self.block_tables.gather_block_tables(batch.idx_mapping, len(rows))
            slots = self.block_tables.compute_slot_mappings(
                batch.idx_mapping, batch.query_start_loc, batch.positions, n
            )
            metadata = self.model_state.prepare_attn(
                batch,
                CUDAGraphMode.NONE,
                tables,
                slots,
                self.attn_groups,
                self.kv_config,
            )
            hidden = self.speculator._run_model(
                batch.input_ids,
                batch.positions,
                metadata,
                build_slot_mappings_by_layer(slots, self.kv_config),
                None,
            )
            sample_rows = [i for i, row in enumerate(rows) if row.sample]
            result: list[int | None] = [None] * len(rows)
            if sample_rows:
                indices = torch.tensor(
                    sample_rows, dtype=torch.int64, device=self.device
                )
                last = (
                    batch.query_start_loc[1:].to(torch.int64).index_select(0, indices)
                    - 1
                )
                self.speculator.current_draft_step.zero_()
                sampled = self.speculator.sample_draft(
                    hidden.index_select(0, last),
                    batch.positions.index_select(0, last),
                    batch.idx_mapping.index_select(0, indices),
                    self.temperature,
                    self.seeds,
                    self.speculator.current_draft_step,
                    None,
                )
                for row, token in zip(sample_rows, sampled.cpu().tolist()):
                    result[row] = int(token)
            else:
                # Prefill/K0 must complete before ACK and shared buffer reuse.
                torch.accelerator.synchronize(self.device)
            return tuple(result)

    @torch.inference_mode()
    def warmup(self) -> None:
        spec = self.vllm_config.speculative_config
        assert spec is not None
        pool = DraftBlockPool(self.num_blocks, self.block_size)
        state = TokenStateMachine(
            self,
            pool,
            max_num_reqs=1,
            max_model_len=self.vllm_config.model_config.max_model_len,
            vocab_size=self.vllm_config.model_config.get_vocab_size(),
            max_k=spec.num_speculative_tokens,
        )
        key = SequenceKey("token-warmup", 0, 0)
        state.open(key)
        failure: BaseException | None = None
        try:
            # One actual prompt forward, followed by correction catch-up and
            # as many decode steps as the real context/KV planner can admit.
            state.prefill(key, 0, (0,), is_final=True)
            available = min(
                state.max_model_len, (self.num_blocks - 1) * self.block_size
            )
            if available >= 2:
                k = min(state.max_k, available - 1)
                state.advance(0, TokenAdvanceRow(key, 1, None, 0, 0, (0,)), k)
        except BaseException as error:
            failure = error
            raise
        finally:
            state.close((key,))
            if failure is None or not runtime_unusable(failure):
                try:
                    self.input_buffers.input_ids.zero_()
                    self.input_buffers.positions.zero_()
                    torch.accelerator.synchronize(self.device)
                except BaseException as cleanup_error:
                    if failure is None or runtime_unusable(cleanup_error):
                        raise
                    failure.add_note(f"token warmup cleanup failed: {cleanup_error}")


def make_token_gpu_adapter(
    config: VllmConfig,
    *,
    device: torch.device,
    kv_cache_bytes: int,
    devices: tuple[torch.device, ...] | None = None,
    outcome_cache: "TokenCacheOptions | None" = None,
) -> TokenOnlyDraftAdapter:
    spec = config.speculative_config
    assert spec is not None
    draft = spec.draft_model_config
    assert draft is not None
    if outcome_cache is not None:
        from vllm.v1.spec_decode.remote.runtime.token_schema import (
            supports_token_outcome_cache,
        )

        if not supports_token_outcome_cache(config, require_remote=False):
            raise ValueError("outcome cache requires ordinary full-attention Qwen3 TP1")
    executor: TokenGPUStepExecutor | TokenTPExecutor
    if spec.draft_parallel_config.tensor_parallel_size > 1:
        from vllm.v1.spec_decode.remote.runtime.tp_executor import (
            TokenTPExecutor as TPExecutor,
        )

        if devices is None:
            raise ValueError("draft TP2 requires two explicit CUDA devices")
        executor = TPExecutor(config, devices=devices, kv_cache_bytes=kv_cache_bytes)
    elif devices is not None:
        raise ValueError("multiple draft devices require draft TP2")
    else:
        executor = TokenGPUStepExecutor(
            config,
            device=device,
            kv_cache_bytes=kv_cache_bytes,
            branch_slots=0 if outcome_cache is None else outcome_cache.max_branches,
        )
    try:
        state = TokenStateMachine(
            executor,
            DraftBlockPool(executor.num_blocks, executor.block_size),
            max_num_reqs=config.scheduler_config.max_num_seqs,
            max_model_len=config.model_config.max_model_len,
            vocab_size=config.model_config.get_vocab_size(),
            max_k=spec.num_speculative_tokens,
        )
        cache = None
        if outcome_cache is not None:
            assert isinstance(executor, TokenGPUStepExecutor)
            from vllm.v1.kv_cache_interface import AttentionSpec
            from vllm.v1.spec_decode.remote.runtime.token_cache import TokenOutcomeCache

            if any(
                not isinstance(group.kv_cache_spec, AttentionSpec)
                or group.kv_cache_spec.dtype != torch.bfloat16
                for group in executor.kv_config.kv_cache_groups
            ):
                raise ValueError("loaded outcome cache requires bfloat16 KV")
            page_bytes = sum(
                group.kv_cache_spec.page_size_bytes * len(group.layer_names)
                for group in executor.kv_config.kv_cache_groups
            )
            blocks = outcome_cache.kv_budget_bytes // page_bytes
            if not 1 <= blocks <= executor.num_blocks - 1:
                raise ValueError(
                    "outcome cache KV budget must fit the physical draft arena"
                )
            cache = TokenOutcomeCache(
                state,
                live_slots=config.scheduler_config.max_num_seqs,
                max_branches=outcome_cache.max_branches,
                max_branch_blocks=blocks,
                copy_chunk_blocks=outcome_cache.copy_chunk_blocks,
                contract_id=(
                    executor.contract.target_fingerprint,
                    executor.contract.draft_fingerprint,
                    executor.contract.tokenizer_fingerprint,
                    "greedy",
                ),
            )
    except BaseException as error:
        try:
            executor.close()
        except BaseException as cleanup_error:
            error.add_note(f"token executor cleanup failed: {cleanup_error}")
        raise
    return TokenOnlyDraftAdapter(state, executor.contract, cache)
