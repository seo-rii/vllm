# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 adapter for a target-conditioned drafter on a dedicated device."""

import itertools
import time
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.v1.spec_decode.remote.capabilities import TargetFeatureKind
from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.runtime.schema import create_runtime_contract
from vllm.v1.spec_decode.remote.session import RemoteDraftSession, VerifierIdentity
from vllm.v1.spec_decode.remote.state import FeatureBatch, RemoteProposalBatch
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

logger = init_logger(__name__)


@dataclass
class _Request:
    key: SequenceKey
    num_computed: int = 0


class RemoteSpeculator(BaseSpeculator):
    """Keep all draft weights and KV on the server; TP rank zero owns its session."""

    supports_mm_inputs = False
    draft_logits = None

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self._validate_config(vllm_config)
        spec = vllm_config.speculative_config
        assert spec is not None and spec.remote_draft is not None
        assert spec.method is not None
        self.device = device
        self.num_speculative_steps = spec.num_speculative_tokens
        self._contract = create_runtime_contract(vllm_config)
        self._tp = get_tp_group()
        self._session: RemoteDraftSession | None = None
        self._requests: dict[str, _Request] = {}
        self._sequence_ids = itertools.count()
        self._batch_ids = itertools.count()
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
            self._session = RemoteDraftSession.from_config(spec.remote_draft, identity)
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
        error = None
        if self._session is not None:
            try:
                keys = tuple(
                    self._requests.pop(req_id).key
                    for req_id in req_ids
                    if req_id in self._requests
                )
                if keys:
                    self._session.close_sequences(keys)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        error = self._tp.broadcast_object(error, src=0)
        if error is not None:
            raise RuntimeError(f"remote drafter request cleanup failed: {error}")

    def close(self) -> None:
        if self._session is not None:
            session = self._session
            self._session = None
            try:
                session.close()
            except Exception:
                logger.exception("Failed to close remote drafter session")
            finally:
                self._requests.clear()
                logger.info("Remote drafter verifier statistics: %s", self._stats)

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
        if dummy_run or is_profile:
            return torch.full(
                (input_batch.num_reqs, self.num_speculative_steps),
                -1,
                dtype=torch.int64,
                device=self.device,
            )
        error = None
        tokens = None
        if self._tp.rank_in_group == 0:
            started = time.perf_counter()
            try:
                if torch.any(temperature[input_batch.idx_mapping] != 0).item():
                    raise ValueError(
                        "remote drafter currently requires greedy target sampling"
                    )
                tokens = self._propose(
                    input_batch,
                    last_hidden_states,
                    aux_hidden_states,
                    num_sampled,
                    num_rejected,
                    last_sampled,
                    next_prefill_tokens,
                ).tolist()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                self._stats["proposal_calls"] += 1
                self._stats["proposal_seconds"] += time.perf_counter() - started
        # This small CPU message shares failures and tokens together. A device
        # broadcast can initialize NCCL after KV profiling, exhausting the
        # target GPU memory that profiling has already assigned to its cache.
        error, tokens = self._tp.broadcast_object((error, tokens), src=0)
        if error is not None:
            raise RuntimeError(f"remote drafter proposal failed: {error}")
        assert tokens is not None
        return torch.tensor(tokens, dtype=torch.int64, device=self.device)

    def _propose(
        self,
        batch: InputBatch,
        hidden: torch.Tensor,
        aux: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
    ) -> torch.Tensor:
        session = self._session
        assert session is not None
        sampled = num_sampled.cpu().tolist()
        rejected = num_rejected.cpu().tolist()
        recovery = last_sampled[batch.idx_mapping].reshape(-1).cpu()
        features = self._features(batch, hidden, aux, next_prefill_tokens)
        starts = batch.query_start_loc_np
        positions = features[TargetFeatureKind.POSITIONS.value]
        chunks = []
        rows, keys, query_lens, accepted = [], [], [], []
        token_indices: list[int] = []
        session_started = time.perf_counter()
        for row, req_id in enumerate(batch.req_ids):
            start, end = int(starts[row]), int(starts[row + 1])
            request = self._requests.get(req_id)
            if request is None:
                if int(positions[start]) != 0 or not batch.is_prefilling_np[row]:
                    raise ValueError(
                        "new remote sequence requires prefill from position zero"
                    )
                request = _Request(session.open_sequence(next(self._sequence_ids)))
                self._requests[req_id] = request
            if int(positions[start]) != request.num_computed:
                raise ValueError(f"remote context discontinuity for request {req_id}")
            query_len = end - start
            if batch.is_prefilling_np[row]:
                is_final = int(positions[end - 1]) + 1 >= int(batch.prefill_len_np[row])
                if rejected[row] != 0 or sampled[row] != int(is_final):
                    raise ValueError("unexpected sampling counts during remote prefill")
                chunks.append(
                    (
                        request.key,
                        self._select(features, range(start, end), [row]),
                        is_final,
                    )
                )
                request.num_computed += query_len
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
        session.prefill_many(chunks)
        output = torch.full(
            (batch.num_reqs, self.num_speculative_steps), -1, dtype=torch.int64
        )
        if rows:
            proposal = RemoteProposalBatch(
                batch_id=next(self._batch_ids),
                keys=tuple(keys),
                accepted_counts=torch.tensor(accepted, dtype=torch.int32),
                recovery_tokens=recovery[rows],
                features=self._select(features, token_indices, rows),
                query_lens=torch.tensor(query_lens, dtype=torch.int32),
            )
            result = session.collect(session.dispatch(proposal)).output
            valid = (
                torch.arange(self.num_speculative_steps)[None, :]
                < result.valid_lengths[:, None]
            )
            output[rows] = torch.where(valid, result.token_ids, -1)
        self._stats["session_seconds"] += time.perf_counter() - session_started
        return output

    def _features(
        self,
        batch: InputBatch,
        hidden: torch.Tensor,
        aux: list[torch.Tensor] | None,
        next_prefill_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        source = {
            TargetFeatureKind.TOKEN_IDS.value: batch.input_ids[: batch.num_tokens],
            TargetFeatureKind.POSITIONS.value: batch.positions[: batch.num_tokens],
            TargetFeatureKind.LAST_HIDDEN_STATES.value: hidden[: batch.num_tokens],
            TargetFeatureKind.NEXT_PREFILL_TOKENS.value: next_prefill_tokens[
                :1, batch.idx_mapping
            ].T,
        }
        if aux is not None:
            source[TargetFeatureKind.AUX_HIDDEN_STATES.value] = torch.cat(
                [state[: batch.num_tokens] for state in aux], dim=-1
            )
        result = {}
        for slot in self._contract.schema.slots:
            tensor = source[slot.kind]
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
