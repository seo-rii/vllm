# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-owner Gemma4 assistant consuming private, completed KV snapshots."""

import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from vllm.config import replace, set_current_vllm_config
from vllm.v1.spec_decode.remote.errors import cuda_context_fatal
from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.runtime.schema import checkpoint_revision
from vllm.v1.spec_decode.remote.runtime.target_kv_schema import (
    create_target_kv_contract,
)
from vllm.v1.spec_decode.remote.runtime.weights import selected_checkpoint_weights
from vllm.v1.spec_decode.remote.target_kv import (
    SnapshotCompletionError,
    SnapshotIdentity,
    SnapshotLayer,
    TargetKVSnapshot,
    import_target_kv_batch,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig


def materialize_gemma4_embedding(
    model_config: "ModelConfig", device: torch.device
) -> torch.nn.Module:
    """Load only the target embedding; the assistant keeps its own LM head."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    names = ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight")
    tensors = selected_checkpoint_weights(
        model_config.model,
        checkpoint_revision(model_config.model, model_config.revision),
        names,
    )
    if len(tensors) != 1:
        raise ValueError("Gemma4 target requires exactly one native embedding tensor")
    weight = next(iter(tensors.values()))
    shape = (model_config.get_vocab_size(), model_config.get_hidden_size())
    if tuple(weight.shape) != shape or weight.dtype != torch.bfloat16:
        raise ValueError(f"Gemma4 target embedding must be BF16 with shape {shape}")
    with device:
        embedding = VocabParallelEmbedding(
            *shape, params_dtype=torch.bfloat16, quant_config=None
        )
    embedding.weight_loader(embedding.weight, weight)
    return embedding


class Gemma4SnapshotExecutor:
    """One synchronous Q-only batch per call; target and draft KV stay immutable.

    Each imported snapshot is retained through the final CUDA completion fence.
    Context-fatal errors permanently fence this executor; process restart is
    required, and shutdown must not enqueue cleanup kernels in that context.
    """

    def __init__(self, config: "VllmConfig", *, device: torch.device) -> None:
        import vllm.envs as envs
        import vllm.ir
        from vllm.compilation.backends import set_model_tag
        from vllm.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.model_executor.model_loader import get_model
        from vllm.model_executor.model_loader.utils import get_draft_load_config
        from vllm.platforms import current_platform
        from vllm.utils.torch_utils import (
            set_random_seed,
            set_torch_threads_for_runtime,
        )
        from vllm.v1.spec_decode.remote.runtime.gpu import _resolve_local_rank
        from vllm.v1.worker.workspace import init_workspace_manager

        self.contract = create_target_kv_contract(config)
        self.device = device
        self.healthy = True
        self._lock = threading.Lock()
        self._quarantine: list[object] = []
        local_rank = _resolve_local_rank(
            device, config.parallel_config.assigned_physical_gpu_ids
        )
        torch.accelerator.set_device_index(device)
        current_platform.check_if_supports_dtype(torch.bfloat16)
        current_platform.import_kernels()
        torch.set_float32_matmul_precision(envs.VLLM_FLOAT32_MATMUL_PRECISION)
        set_random_seed(config.model_config.seed)
        set_torch_threads_for_runtime()
        init_workspace_manager(device, num_ubatches=1)
        config.kernel_config.ir_op_priority.set_default()
        vllm.ir.set_default_torch_wrap(config.compilation_config.ir_enable_torch_wrap)
        self._dist_directory = tempfile.TemporaryDirectory(prefix="vllm-gemma-draft-")
        speculative = config.speculative_config
        assert speculative is not None and speculative.draft_model_config is not None
        self.config = replace(config, model_config=speculative.draft_model_config)
        with set_current_vllm_config(self.config), torch.inference_mode(), device:
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
            with set_model_tag("eagle_head"):
                self.model = get_model(
                    vllm_config=self.config,
                    model_config=self.config.model_config,
                    load_config=get_draft_load_config(self.config),
                )
            self.model.model.embed_tokens = materialize_gemma4_embedding(
                config.model_config, device
            )
            self.model.eval()
            self._attentions = tuple(
                layer.self_attn.attn for layer in self.model.model.layers
            )
            if len(self._attentions) != len(self.contract.draft_source_names):
                raise ValueError("loaded assistant layer count differs from contract")
            self._empty_caches = tuple(attn.kv_cache for attn in self._attentions)
            for attn, source in zip(self._attentions, self.contract.draft_source_names):
                spec = self.contract.expected_specs[source]
                if (
                    attn.head_size != spec.head_size
                    or attn.num_kv_heads != spec.num_kv_heads
                    or attn.num_heads != self.contract.draft_num_attention_heads
                    or attn.attn_backend.get_name() != "TRITON_ATTN"
                    or attn.kv_sharing_target_layer_name is not None
                ):
                    raise ValueError("loaded assistant attention violates contract")
            capacity = self.contract.max_num_seqs
            self._input_ids = torch.zeros(capacity, dtype=torch.int32, device=device)
            self._positions = torch.zeros(capacity, dtype=torch.int64, device=device)
            self._hidden = torch.zeros(
                (capacity, self.contract.hidden_size),
                dtype=torch.bfloat16,
                device=device,
            )
            self._query_start = torch.arange(
                capacity + 1, dtype=torch.int32, device=device
            )
            self._seq_lens = torch.ones(capacity, dtype=torch.int32, device=device)
            self._slots = torch.full((capacity,), -1, dtype=torch.int64, device=device)
            # Native decode uses 16 split-softmax segments. Allocate against each
            # actual heterogeneous head dimension, not the model-wide default.
            self._scratch = tuple(
                (
                    torch.empty(
                        (
                            capacity,
                            attn.num_heads,
                            16,
                            1 << (attn.head_size - 1).bit_length(),
                        ),
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.empty(
                        (capacity, attn.num_heads, 16),
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.empty(
                        (capacity, attn.num_heads, 16),
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                for attn in self._attentions
            )
            torch.cuda.current_stream(device).synchronize()

    def execute(
        self,
        snapshot: TargetKVSnapshot,
        *,
        expected_identity: SnapshotIdentity,
        expected_materialized_length: int,
        recovery_token: int,
        hidden_state: torch.Tensor,
    ) -> int:
        """Return the native assistant argmax after all snapshot reads finish."""
        return self.execute_batch(
            (snapshot,),
            expected_identities=(expected_identity,),
            expected_materialized_lengths=(expected_materialized_length,),
            recovery_tokens=(recovery_token,),
            hidden_states=hidden_state,
        )[0]

    def execute_batch(
        self,
        snapshots: tuple[TargetKVSnapshot, ...],
        *,
        expected_identities: tuple[SnapshotIdentity, ...],
        expected_materialized_lengths: tuple[int, ...],
        recovery_tokens: tuple[int, ...],
        hidden_states: torch.Tensor,
    ) -> tuple[int, ...]:
        """Run one assistant forward over independently owned request snapshots."""
        from vllm.forward_context import BatchDescriptor, set_forward_context
        from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

        count = len(snapshots)
        if (
            not 1 <= count <= self.contract.max_num_seqs
            or len(recovery_tokens) != count
            or len(expected_identities) != count
            or len(expected_materialized_lengths) != count
        ):
            raise ValueError("target-KV query batch differs from the bounded contract")
        if any(not 0 <= token < self.contract.vocab_size for token in recovery_tokens):
            raise ValueError("recovery token is outside the target vocabulary")
        if hidden_states.shape != (count, self.contract.hidden_size) or (
            hidden_states.dtype != torch.bfloat16 or hidden_states.device.type != "cpu"
        ):
            raise ValueError("assistant requires CPU BF16 target hidden rows")
        if any(
            layer.pages.device.type != "cpu"
            for snapshot in snapshots
            for layer in snapshot.layers
        ):
            raise ValueError("inline target-KV runtime requires CPU snapshot pages")
        with self._lock, torch.inference_mode():
            if not self.healthy:
                raise RuntimeError("Gemma4 draft runtime requires process restart")
            device_index = self.device.index
            assert device_index is not None
            with torch.accelerator.device_index(device_index):
                imported = None
                complete = False
                try:
                    imported = import_target_kv_batch(
                        snapshots,
                        expected_identities=expected_identities,
                        expected_materialized_lengths=expected_materialized_lengths,
                        expected_specs=self.contract.expected_specs,
                        budget=self.contract.receiver_budget,
                        device=self.device,
                    )
                    by_name = {layer.layer_name: layer for layer in imported.layers}
                    input_ids = self._input_ids[:count]
                    positions = self._positions[:count]
                    seq_lens = self._seq_lens[:count]
                    hidden = self._hidden[:count]
                    slots = self._slots[:count]
                    input_ids.copy_(
                        torch.tensor(recovery_tokens, dtype=torch.int32, device="cpu")
                    )
                    seq_lens.copy_(
                        torch.tensor(
                            expected_materialized_lengths,
                            dtype=torch.int32,
                            device="cpu",
                        )
                    )
                    positions.copy_(seq_lens).sub_(1)
                    hidden.copy_(hidden_states)
                    metadata = {}
                    for attn, source, scratch in zip(
                        self._attentions,
                        self.contract.draft_source_names,
                        self._scratch,
                    ):
                        layer = by_name[source]
                        attn.kv_cache = layer.cache
                        metadata[attn.layer_name] = TritonAttentionMetadata(
                            num_actual_tokens=count,
                            max_query_len=1,
                            query_start_loc=self._query_start[: count + 1],
                            max_seq_len=max(expected_materialized_lengths),
                            seq_lens=seq_lens,
                            block_table=layer.block_table,
                            slot_mapping=slots,
                            seq_threshold_3D=128 // attn.num_kv_heads,
                            num_par_softmax_segments=16,
                            softmax_segm_output=scratch[0][:count],
                            softmax_segm_max=scratch[1][:count],
                            softmax_segm_expsum=scratch[2][:count],
                            causal=True,
                        )
                    with (
                        set_current_vllm_config(self.config),
                        set_forward_context(
                            metadata,
                            self.config,
                            num_tokens=count,
                            batch_descriptor=BatchDescriptor(num_tokens=count),
                            slot_mapping={name: slots for name in metadata},
                            skip_compiled=True,
                        ),
                    ):
                        draft_hidden, _ = self.model(
                            input_ids=input_ids,
                            positions=positions,
                            hidden_states=hidden,
                        )
                        logits = self.model.compute_logits(draft_hidden)
                        if logits.ndim != 2 or logits.shape[0] != count:
                            raise ValueError("assistant logits differ from query batch")
                        tokens = tuple(logits.argmax(dim=-1).cpu().tolist())
                    torch.cuda.current_stream(self.device).synchronize()
                    complete = True
                    return tokens
                except BaseException as error:
                    if cuda_context_fatal(error) or isinstance(
                        error, SnapshotCompletionError
                    ):
                        self.healthy = False
                    else:
                        try:
                            torch.cuda.current_stream(self.device).synchronize()
                            complete = True
                        except BaseException as fence_error:
                            self.healthy = False
                            raise SnapshotCompletionError(
                                "assistant snapshot consumer completion is unknown"
                            ) from fence_error
                    raise
                finally:
                    if complete:
                        for attn, empty in zip(self._attentions, self._empty_caches):
                            attn.kv_cache = empty
                    elif imported is not None:
                        # Keep ownership even if the failing forward did not bind
                        # every source. Host cancellation never proves completion.
                        self._quarantine.append(imported)

    def warmup(self) -> None:
        """Compile and synchronize real assistant forwards before listening."""
        from vllm.v1.kv_cache_interface import SlidingWindowSpec

        identity = SnapshotIdentity(SequenceKey("warmup", 0, 0), 1, 0)
        for length in dict.fromkeys((1, self.contract.snapshot_budget.max_model_len)):
            layers = []
            for name, spec in self.contract.expected_specs.items():
                start = (
                    max(0, length - spec.sliding_window)
                    if isinstance(spec, SlidingWindowSpec)
                    else 0
                )
                logical = tuple(
                    range(start // spec.block_size, (length - 1) // spec.block_size + 1)
                )
                layers.append(
                    SnapshotLayer(
                        name,
                        spec,
                        logical,
                        torch.zeros(
                            (
                                len(logical),
                                spec.num_kv_heads,
                                spec.block_size,
                                2 * spec.head_size,
                            ),
                            dtype=torch.bfloat16,
                        ),
                    )
                )
            snapshot = TargetKVSnapshot(identity, length, tuple(layers))
            self.contract.snapshot_budget.check(
                length, snapshot.num_pages, snapshot.nbytes
            )
            self.execute(
                snapshot,
                expected_identity=identity,
                expected_materialized_length=length,
                recovery_token=0,
                hidden_state=torch.zeros(
                    (1, self.contract.hidden_size), dtype=torch.bfloat16
                ),
            )
            count = self.contract.max_num_seqs
            if count > 1:
                identities = tuple(
                    SnapshotIdentity(SequenceKey("warmup", row, 0), 1, 0)
                    for row in range(count)
                )
                self.execute_batch(
                    tuple(
                        TargetKVSnapshot(row_identity, length, snapshot.layers)
                        for row_identity in identities
                    ),
                    expected_identities=identities,
                    expected_materialized_lengths=(length,) * count,
                    recovery_tokens=(0,) * count,
                    hidden_states=torch.zeros(
                        (count, self.contract.hidden_size), dtype=torch.bfloat16
                    ),
                )

    def close(self) -> None:
        """Release process-local rendezvous metadata, with no CUDA cleanup calls."""
        self._dist_directory.cleanup()
