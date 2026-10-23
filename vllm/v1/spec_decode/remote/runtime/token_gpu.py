# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-only native model/attention execution, without a target shell or engine."""

from dataclasses import replace

import numpy as np
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.spec_decode.remote.errors import runtime_unusable
from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.runtime.gpu import (
    GPUDraftStepExecutor,
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
    TokenStateMachine,
    TokenStepExecutor,
)


class TokenGPUStepExecutor(GPUDraftStepExecutor, TokenStepExecutor):
    def __init__(
        self, config: VllmConfig, *, device: torch.device, kv_cache_bytes: int
    ):
        self._token_contract = create_token_runtime_contract(config)
        spec = config.speculative_config
        assert spec is not None and spec.draft_model_config is not None
        if (
            config.parallel_config.tensor_parallel_size != 1
            or spec.draft_parallel_config.tensor_parallel_size != 1
        ):
            raise ValueError("token-only runtime requires target TP1 and draft TP1")
        if spec.draft_model_config.quantization is not None:
            raise ValueError(
                "token-only first runtime requires unquantized draft weights"
            )
        # Allocate attention/KV from the independent draft's dimensions. No
        # target modules, embeddings or output head are constructed here.
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
        super().__init__(resource, device=device, kv_cache_bytes=kv_cache_bytes)

    def _create_contract(self, config):
        return self._token_contract

    def _load_speculator(self, config, device, shared_weights_source):
        from vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator import (
            StandaloneARSpeculator,
        )

        speculator = StandaloneARSpeculator(config, device)
        speculator.load_standalone_model()
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

    @torch.inference_mode()
    def forward(self, key, req_index, start, tokens, blocks, *, sample):
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer

        if not tokens or len(tokens) > self.max_num_tokens:
            raise ValueError("token forward must fit the configured batch token budget")
        n = len(tokens)
        positions = np.arange(start, start + n, dtype=np.int64)
        plan = DraftStepPlan(
            rows=(DraftStepRow(key, n, 0),),
            req_indices=np.array([req_index], dtype=np.int32),
            num_scheduled_tokens=np.array([n], dtype=np.int32),
            query_start_loc=np.array([0, n], dtype=np.int32),
            num_computed_tokens=np.array([start], dtype=np.int32),
            seq_lens=np.array([start + n], dtype=np.int32),
            positions=positions,
            slot_mapping=np.array(
                [
                    blocks[p // self.block_size] * self.block_size + p % self.block_size
                    for p in positions
                ],
                dtype=np.int64,
            ),
            block_table_updates={req_index: BlockTableUpdate(0, list(blocks))},
            num_sampled=np.array([0], dtype=np.int32),
            num_rejected=np.array([0], dtype=np.int32),
            last_sampled=np.array([0], dtype=np.int64),
            next_prefill_tokens=np.zeros((1, 1), dtype=np.int32),
        )
        torch.accelerator.set_device_index(self.device)
        with set_current_vllm_config(self.vllm_config):
            self.prepare_block_tables(plan)
            batch = make_input_batch(
                plan, torch.tensor(tokens, dtype=torch.int32), self.input_buffers
            )
            tables = self.block_tables.gather_block_tables(batch.idx_mapping, 1)
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
            if sample:
                self.speculator.current_draft_step.zero_()
                sampled = self.speculator.sample_draft(
                    hidden[-1:],
                    batch.positions[-1:],
                    batch.idx_mapping,
                    self.temperature,
                    self.seeds,
                    self.speculator.current_draft_step,
                    None,
                )
                return int(sampled.cpu().item())
            # Prefill/K0 must complete before ACK and shared buffer reuse, too.
            torch.accelerator.synchronize(self.device)
            return None

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
    config: VllmConfig, *, device: torch.device, kv_cache_bytes: int
) -> TokenOnlyDraftAdapter:
    executor = TokenGPUStepExecutor(
        config, device=device, kv_cache_bytes=kv_cache_bytes
    )
    spec = config.speculative_config
    assert spec is not None
    state = TokenStateMachine(
        executor,
        DraftBlockPool(executor.num_blocks, executor.block_size),
        max_num_reqs=config.scheduler_config.max_num_seqs,
        max_model_len=config.model_config.max_model_len,
        vocab_size=config.model_config.get_vocab_size(),
        max_k=spec.num_speculative_tokens,
    )
    return TokenOnlyDraftAdapter(state, executor.contract)
