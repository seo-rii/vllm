# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordinary native recurrent execution for an independent token-only draft."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.spec_decode.remote.errors import CompletionUnknownError, runtime_unusable
from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.runtime.gpu import _resolve_local_rank, make_input_batch
from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
from vllm.v1.spec_decode.remote.runtime.recurrent import RecurrentTokenHistory
from vllm.v1.spec_decode.remote.runtime.step import DraftStepPlan, DraftStepRow
from vllm.v1.spec_decode.remote.runtime.token_state import (
    TokenAdvanceRow,
    TokenForwardRow,
    TokenStateMachine,
    TokenStepExecutor,
)

logger = init_logger(__name__)


def create_recurrent_executor_config(config: VllmConfig) -> VllmConfig:
    """Resolve the complete draft as an ordinary model with no speculative state."""
    from vllm.v1.spec_decode.remote.runtime.token_schema import (
        create_token_runtime_contract,
    )

    create_token_runtime_contract(config)
    return _create_recurrent_executor_config(config)


def _create_recurrent_executor_config(config: VllmConfig) -> VllmConfig:
    """Build ordinary execution resources after validating the token contract."""
    from vllm.config import AttentionConfig
    from vllm.config.mamba import MambaBackendEnum
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import get_draft_load_config

    spec = config.speculative_config
    if spec is None or spec.draft_model_config is None:
        raise ValueError("recurrent drafting requires a resolved draft model")
    if spec.draft_parallel_config is None or any(
        parallel.world_size != 1 or parallel.data_parallel_size != 1
        for parallel in (config.parallel_config, spec.draft_parallel_config)
    ):
        raise ValueError("recurrent drafting requires single-rank target and draft")
    draft = spec.draft_model_config
    if (
        draft.architecture != "NemotronHForCausalLM"
        or draft.hf_text_config.model_type != "nemotron_h"
    ):
        raise ValueError("recurrent drafting requires a full NemotronH model")
    scheduler = config.scheduler_config
    if config.model_config.max_model_len > scheduler.max_num_batched_tokens:
        raise ValueError("recurrent full-prefix replay must fit one token budget")
    resource = EngineArgs(
        model=draft.model,
        revision=draft.revision,
        tokenizer=draft.tokenizer,
        tokenizer_revision=draft.tokenizer_revision,
        tokenizer_mode="hf",
        trust_remote_code=draft.trust_remote_code,
        hf_overrides=draft.hf_overrides,
        model_impl=draft.model_impl,
        dtype=draft.dtype,
        seed=draft.seed,
        tensor_parallel_size=1,
        max_model_len=config.model_config.max_model_len,
        max_num_seqs=scheduler.max_num_seqs,
        max_num_batched_tokens=scheduler.max_num_batched_tokens,
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        attention_config=AttentionConfig(backend=spec.attention_backend),
        block_size=config.cache_config.block_size,
        kv_cache_dtype="bfloat16",
        mamba_cache_dtype="bfloat16",
        mamba_ssm_cache_dtype="float32",
        mamba_cache_mode="none",
        mamba_backend=MambaBackendEnum.TRITON,
        use_replayssm=False,
    ).create_engine_config()
    resource.load_config = get_draft_load_config(config)
    return resource


class RecurrentGPUStepExecutor(TokenStepExecutor):
    """Rebuild native recurrent state from the exact computed prefix per forward."""

    def __init__(
        self, config: VllmConfig, *, device: torch.device, kv_cache_bytes: int
    ) -> None:
        import vllm.envs as envs
        import vllm.ir
        from vllm.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
        from vllm.model_executor.model_loader import get_model
        from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
        from vllm.platforms import current_platform
        from vllm.utils.torch_utils import (
            set_random_seed,
            set_torch_threads_for_runtime,
        )
        from vllm.v1.attention.backends.utils import (
            get_supported_kv_cache_layouts,
            resolve_kv_cache_layout,
        )
        from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
        from vllm.v1.spec_decode.remote.runtime.recurrent_kv import (
            make_recurrent_kv_config,
        )
        from vllm.v1.spec_decode.remote.runtime.token_schema import (
            create_token_runtime_contract,
        )
        from vllm.v1.spec_decode.remote.runtime.weights import model_weight_bytes
        from vllm.v1.worker.block_table import get_block_table_width
        from vllm.v1.worker.gpu.attn_utils import (
            get_kv_cache_spec,
            init_attn_backend,
            init_kv_cache,
        )
        from vllm.v1.worker.gpu.block_table import BlockTables
        from vllm.v1.worker.gpu.input_batch import InputBuffers
        from vllm.v1.worker.gpu.model_states import init_model_state
        from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
        from vllm.v1.worker.workspace import init_workspace_manager

        self.contract = create_token_runtime_contract(config)
        spec = config.speculative_config
        assert spec is not None
        self.max_k = spec.num_speculative_tokens
        self.vllm_config = resource = _create_recurrent_executor_config(config)
        if kv_cache_bytes <= 0:
            raise ValueError("recurrent draft KV budget must be positive")
        if resource.speculative_config is not None or resource.num_speculative_tokens:
            raise ValueError("recurrent execution must use an ordinary model")
        self.device = device
        self.histories = RecurrentTokenHistory(
            max_num_reqs=resource.scheduler_config.max_num_seqs,
            max_model_len=resource.model_config.max_model_len,
            vocab_size=resource.model_config.get_vocab_size(),
        )
        self.completed_forwards = 0
        self.replayed_tokens = 0
        parallel = config.parallel_config
        local_rank = _resolve_local_rank(device, parallel.assigned_physical_gpu_ids)
        torch.accelerator.set_device_index(device)
        current_platform.check_if_supports_dtype(resource.model_config.dtype)
        current_platform.import_kernels()
        torch.set_float32_matmul_precision(envs.VLLM_FLOAT32_MATMUL_PRECISION)
        set_random_seed(resource.model_config.seed)
        set_torch_threads_for_runtime()
        init_workspace_manager(device, num_ubatches=1)
        resource.kernel_config.ir_op_priority.set_default()
        vllm.ir.set_default_torch_wrap(resource.compilation_config.ir_enable_torch_wrap)
        self._dist_directory = tempfile.TemporaryDirectory(
            prefix="vllm-recurrent-draft-"
        )
        with set_current_vllm_config(resource), torch.inference_mode():
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=local_rank,
                distributed_init_method=Path(
                    self._dist_directory.name, "rendezvous"
                ).as_uri(),
                backend="nccl",
            )
            initialize_model_parallel()
            model = get_model(vllm_config=resource)
            if (
                not isinstance(model, NemotronHForCausalLM)
                or type(model) is not NemotronHForCausalLM
            ):
                raise ValueError(
                    "recurrent draft requires the native full NemotronH model"
                )
            self.model = model.eval()
            specs = get_kv_cache_spec(resource)
            layers = get_layers_from_vllm_config(
                resource,
                AttentionLayerBase,  # type: ignore[type-abstract]
            )
            layouts = get_supported_kv_cache_layouts(
                layer.get_attn_backend() for layer in layers.values()
            )
            layout = resolve_kv_cache_layout(
                resource, [[layout.name for layout in layouts]], specs.values()
            )
            scheduler = resource.scheduler_config
            self.kv_config = make_recurrent_kv_config(
                resource, specs, kv_cache_bytes, scheduler.max_num_seqs, layout
            )
            self.num_blocks = self.kv_config.num_blocks
            resource.cache_config.num_gpu_blocks = self.num_blocks
            self.block_size = next(
                value.block_size
                for value in specs.values()
                if type(value) is FullAttentionSpec
            )
            self.input_buffers = InputBuffers(
                scheduler.max_num_seqs, scheduler.max_num_batched_tokens, device
            )
            groups = [group.kv_cache_spec for group in self.kv_config.kv_cache_groups]
            self.block_tables = BlockTables(
                block_sizes=[group.block_size for group in groups],
                max_num_reqs=scheduler.max_num_seqs,
                max_num_batched_tokens=scheduler.max_num_batched_tokens,
                max_num_blocks_per_group=[
                    1
                    if isinstance(group, MambaSpec)
                    else get_block_table_width(
                        math.ceil(
                            resource.model_config.max_model_len / group.block_size
                        ),
                        group.block_size,
                        token_alignment=group.block_table_token_alignment,
                    )
                    for group in groups
                ],
                device=device,
                slot_mapping_enabled=[group.uses_slot_mapping for group in groups],
                dcp_sharded=[group.dcp_sharded for group in groups],
            )
            self.attn_groups, _, kernel_sizes = init_attn_backend(
                self.kv_config, resource, device
            )
            model_state = init_model_state(resource, self.model, None, device)
            if (
                not isinstance(model_state, MambaHybridModelState)
                or type(model_state) is not MambaHybridModelState
            ):
                raise ValueError("recurrent draft requires native Mamba hybrid state")
            self.model_state = model_state
            self.kv_caches = init_kv_cache(
                resource.compilation_config.static_forward_context,
                self.kv_config,
                device,
                kernel_sizes,
                resource,
                block_tables=self.block_tables,
                attn_groups=(group for batch in self.attn_groups for group in batch),
            )
            self.model_state.initialize_kv_cache(self.kv_config, self.block_tables)
            self.num_computed_tokens = torch.zeros(
                scheduler.max_num_seqs, dtype=torch.int32, device=device
            )
            recurrent_layers = get_layers_from_vllm_config(resource, MambaMixer2)
            cache_layouts = []
            for group in self.kv_config.kv_cache_groups:
                tensor = next(
                    tensor
                    for tensor in self.kv_config.kv_cache_tensors
                    if group.layer_names[0] in tensor.layers
                )
                value = group.kv_cache_spec
                entry = dict(
                    kind=type(value).__name__,
                    layers=list(group.layer_names),
                    blocks=self.kv_config.num_blocks_of(tensor),
                )
                if isinstance(value, MambaSpec):
                    entry.update(
                        shapes=value.shapes,
                        dtypes=[str(dtype) for dtype in value.dtypes],
                        num_speculative_blocks=value.num_speculative_blocks,
                        cache_mode=value.mamba_cache_mode,
                    )
                cache_layouts.append(entry)
            self.loaded_identity = dict(
                executor=type(self).__name__,
                model=type(self.model).__name__,
                state=type(self.model_state).__name__,
                speculative_config=resource.speculative_config,
                num_speculative_tokens=resource.num_speculative_tokens,
                mamba_backend=resource.mamba_config.backend.value,
                replay_policy="full_computed_prefix",
                max_num_reqs=scheduler.max_num_seqs,
                row_capacity=self.max_num_reqs,
                max_model_len=resource.model_config.max_model_len,
                max_num_tokens=self.max_num_tokens,
                attention_blocks=self.num_blocks,
                block_size=self.block_size,
                kv_bytes=self.kv_config.kv_cache_tensors[0].size,
                kv_layouts=cache_layouts,
                native_states={
                    name: [
                        dict(shape=list(state.shape), dtype=str(state.dtype))
                        for state in layer.kv_cache
                    ]
                    for name, layer in recurrent_layers.items()
                },
            )
        draft_bytes, embedding_bytes, head_bytes = model_weight_bytes(self.model)
        logger.info(
            "Recurrent token-only draft loaded: %s", json.dumps(self.loaded_identity)
        )
        logger.info(
            "Recurrent draft memory bytes: draft_specific=%d embedding=%d lm_head=%d "
            "kv=%d allocated=%d; computed transcript bound=%d token IDs",
            draft_bytes,
            embedding_bytes,
            head_bytes,
            self.kv_config.kv_cache_tensors[0].size,
            torch.accelerator.memory_allocated(device),
            scheduler.max_num_seqs * resource.model_config.max_model_len,
        )

    @property
    def max_num_tokens(self) -> int:
        return self.input_buffers.max_num_tokens

    @property
    def max_num_reqs(self) -> int:
        # Keep round-deadline checks between complete prefix replays.
        return 1

    def close_sequences(self, keys: tuple[SequenceKey, ...]) -> None:
        self.histories.close_sequences(keys)

    def _prepare_block_tables(self, req_index: int, blocks: tuple[int, ...]) -> None:
        from vllm.v1.kv_cache_interface import MambaSpec

        previous = self.block_tables.num_blocks.np[:, req_index].copy()
        try:
            self.block_tables.append_block_ids(
                req_index,
                tuple(
                    [req_index + 1]
                    if isinstance(group.kv_cache_spec, MambaSpec)
                    else list(blocks)
                    for group in self.kv_config.kv_cache_groups
                ),
                overwrite=True,
            )
            self.block_tables.apply_staged_writes()
        except BaseException as error:
            try:
                for table in self.block_tables.block_tables:
                    table.clear_staged_writes()
                if not runtime_unusable(error):
                    torch.accelerator.synchronize(self.device)
                    self.block_tables.num_blocks.np[:, req_index] = previous
            except BaseException as completion:
                if runtime_unusable(error):
                    raise error from completion
                if runtime_unusable(completion):
                    raise completion from error
                raise CompletionUnknownError(
                    "recurrent block-table upload could not be retired"
                ) from completion
            raise

    @torch.inference_mode()
    def forward(self, key, req_index, start, tokens, blocks, *, sample):
        from vllm.forward_context import set_forward_context
        from vllm.v1.core.sched.output import NewRequestData
        from vllm.v1.kv_cache_interface import MambaSpec
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer

        update = self.histories.prepare(
            TokenForwardRow(key, req_index, start, tokens, blocks, sample)
        )
        prefix = update.tokens
        n = len(prefix)
        if n > self.max_num_tokens or len(blocks) * self.block_size < n:
            raise ValueError("recurrent replay exceeds its execution or KV budget")
        plan = DraftStepPlan(
            rows=(DraftStepRow(key, n, num_sampled=0),),
            req_indices=np.array([req_index], dtype=np.int32),
            num_scheduled_tokens=np.array([n], dtype=np.int32),
            query_start_loc=np.array([0, n], dtype=np.int32),
            num_computed_tokens=np.zeros(1, dtype=np.int32),
            seq_lens=np.array([n], dtype=np.int32),
            positions=np.arange(n, dtype=np.int64),
            slot_mapping=np.array(
                [
                    blocks[p // self.block_size] * self.block_size + p % self.block_size
                    for p in range(n)
                ],
                dtype=np.int64,
            ),
            block_table_updates={},
            num_sampled=np.zeros(1, dtype=np.int32),
            num_rejected=np.zeros(1, dtype=np.int32),
            last_sampled=np.zeros(1, dtype=np.int64),
            next_prefill_tokens=np.zeros((1, 1), dtype=np.int32),
        )
        submitted = False
        try:
            torch.accelerator.set_device_index(self.device)
            with set_current_vllm_config(self.vllm_config):
                self._prepare_block_tables(req_index, blocks)
                submitted = True
                if not update.previous.tokens:
                    self.model_state.add_request(
                        req_index,
                        NewRequestData(
                            req_id=str(key),
                            prompt_token_ids=list(prefix),
                            mm_features=[],
                            sampling_params=None,
                            pooling_params=None,
                            block_ids=tuple(
                                [req_index + 1]
                                if isinstance(group.kv_cache_spec, MambaSpec)
                                else list(blocks)
                                for group in self.kv_config.kv_cache_groups
                            ),
                            num_computed_tokens=0,
                            lora_request=None,
                            prefill_token_ids=list(prefix),
                        ),
                    )
                    self.model_state.apply_staged_writes()
                batch = make_input_batch(
                    plan, torch.tensor(prefix, dtype=torch.int32), self.input_buffers
                )
                tables = self.block_tables.gather_block_tables(batch.idx_mapping, 1)
                slots = self.block_tables.compute_slot_mappings(
                    batch.idx_mapping, batch.query_start_loc, batch.positions, n
                )
                self.model_state.preprocess_state(
                    batch, tables, self.kv_config, self.num_computed_tokens
                )
                metadata = self.model_state.prepare_attn(
                    batch,
                    CUDAGraphMode.NONE,
                    tables,
                    slots,
                    self.attn_groups,
                    self.kv_config,
                )
                with set_forward_context(
                    metadata,
                    self.vllm_config,
                    num_tokens=n,
                    slot_mapping=build_slot_mappings_by_layer(slots, self.kv_config),
                    is_padding=batch.is_padding,
                ):
                    hidden = self.model(
                        input_ids=batch.input_ids, positions=batch.positions
                    )
                self.model_state.postprocess_state(
                    batch.idx_mapping, 1 if sample else 0
                )
                if sample:
                    logits = self.model.compute_logits(hidden[-1:])
                    assert logits is not None
                    result = int(logits.argmax(dim=-1).cpu().item())
                else:
                    torch.accelerator.synchronize(self.device)
                    result = None
            self.histories.commit(update)
            self.completed_forwards += 1
            self.replayed_tokens += n
            return result
        except BaseException as error:
            if submitted and not runtime_unusable(error):
                try:
                    torch.accelerator.synchronize(self.device)
                except BaseException as completion:
                    if runtime_unusable(completion):
                        raise completion from error
                    raise CompletionUnknownError(
                        "recurrent forward completion could not be established"
                    ) from completion
            raise

    @torch.inference_mode()
    def warmup(self) -> None:
        if self.histories.num_sequences:
            raise ValueError("recurrent warmup must precede live requests")
        pool = DraftBlockPool(self.num_blocks, self.block_size)
        state = TokenStateMachine(
            self,
            pool,
            max_num_reqs=1,
            max_model_len=self.vllm_config.model_config.max_model_len,
            vocab_size=self.vllm_config.model_config.get_vocab_size(),
            max_k=self.max_k,
        )
        key = SequenceKey("recurrent-warmup", 0, 0)
        state.open(key)
        try:
            state.prefill(key, 0, (0,), is_final=True)
            available = min(
                state.max_model_len, (self.num_blocks - 1) * self.block_size
            )
            if available >= 2:
                k = min(self.max_k, available - 2)
                state.advance(0, TokenAdvanceRow(key, 1, None, 0, 0, (0,)), k)
                if available >= 3:
                    previous = state.sequences[key]
                    state.advance(
                        1,
                        TokenAdvanceRow(
                            key,
                            previous.committed,
                            previous.proposal_id,
                            len(previous.proposal_tokens),
                            0,
                            (0,),
                        ),
                        0,
                    )
        finally:
            state.close((key,))
