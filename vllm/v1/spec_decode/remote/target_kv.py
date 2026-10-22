# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, synchronous target-KV snapshots for a single Q-only draft query.

This is a tensor primitive, not a remote backend or a wire protocol. Source
views use the native logical [blocks, heads, tokens, K+V] layout. Snapshots and
imports own separate storage; neither retains an alias to target KV.

The caller must fence target writes/reallocation while snapshotting and order
them on the current CUDA stream. Copies complete before either API returns.
An imported cache must remain alive until its consumer completes; consumers on
another CUDA stream must also call ``record_stream``. Cancellation is not a
completion event and does not authorize reuse of an export or imported cache.
Recoverable copy failures are fenced before propagation. A context-fatal CUDA
error or unknown copy completion requires abandoning the runtime; callers must
not recover by reusing source pages or serving a target-only continuation.
"""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    SlidingWindowSpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.spec_decode.remote.errors import (
    CompletionUnknownError,
    cuda_context_fatal,
)
from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.worker.block_table import get_block_table_width

AttentionSpec = FullAttentionSpec | SlidingWindowSpec


class SnapshotCompletionError(CompletionUnknownError):
    """Copy completion is unknown; the runtime must not reuse source state."""


@dataclass(frozen=True)
class SnapshotIdentity:
    key: SequenceKey
    session_epoch: int
    round_id: int

    def __post_init__(self) -> None:
        if (
            not self.key.verifier_instance_id
            or self.key.sequence_id < 0
            or self.key.generation < 0
            or self.session_epoch <= 0
            or self.round_id < 0
        ):
            raise ValueError("invalid target-KV snapshot identity")


@dataclass(frozen=True)
class SnapshotBudget:
    """Per-call limits; the caller also budgets concurrently retained snapshots."""

    max_pages: int
    max_bytes: int
    max_model_len: int

    def __post_init__(self) -> None:
        if min(self.max_pages, self.max_bytes, self.max_model_len) <= 0:
            raise ValueError("target-KV snapshot limits must be positive")

    def check(self, length: int, pages: int, nbytes: int) -> None:
        if not 1 <= length <= self.max_model_len:
            raise ValueError("target-KV context exceeds snapshot length limit")
        if pages > self.max_pages or nbytes > self.max_bytes:
            raise ValueError("target-KV snapshot exceeds page or byte budget")


@dataclass(frozen=True)
class DraftPosition:
    hidden_index: int
    position: int
    materialized_length: int


def select_draft_position(
    *,
    query_start: int,
    query_length: int,
    sequence_length: int,
    num_rejected: int,
    num_sampled: int,
) -> DraftPosition | None:
    """Select the native K1 assistant query; unfinished prefill has no query."""
    if (
        query_start < 0
        or query_length <= 0
        or sequence_length < query_length
        or not 0 <= num_rejected < query_length
        or num_sampled < 0
    ):
        raise ValueError("invalid target query or rejection counts")
    if num_sampled == 0:
        if num_rejected:
            raise ValueError("unfinished prefill cannot reject draft tokens")
        return None
    length = sequence_length - num_rejected
    return DraftPosition(
        query_start + query_length - num_rejected - 1, length - 1, length
    )


def gemma4_source_layers(
    target_layer_types: Sequence[str],
    draft_layer_types: Sequence[str],
    *,
    num_kv_shared_layers: int,
) -> tuple[int, ...]:
    """Map assistant layers to the last non-shared matching target layers."""
    supported = {"full_attention", "sliding_attention"}
    if (
        not target_layer_types
        or not draft_layer_types
        or not 0 <= num_kv_shared_layers < len(target_layer_types)
        or any(t not in supported for t in (*target_layer_types, *draft_layer_types))
    ):
        raise ValueError("unsupported Gemma4 target/draft attention layer types")
    owners = {
        kind: i
        for i, kind in enumerate(
            target_layer_types[: len(target_layer_types) - num_kv_shared_layers]
        )
    }
    if any(kind not in owners for kind in draft_layer_types):
        raise ValueError("Gemma4 assistant layer has no matching target KV owner")
    return tuple(owners[kind] for kind in draft_layer_types)


@dataclass(frozen=True)
class TargetKVSource:
    layer_name: str
    spec: AttentionSpec
    cache: torch.Tensor
    block_table: torch.Tensor
    layout: KVCacheLayout


@dataclass(frozen=True)
class SnapshotLayer:
    layer_name: str
    spec: AttentionSpec
    logical_pages: tuple[int, ...]
    pages: torch.Tensor


@dataclass(frozen=True)
class TargetKVSnapshot:
    identity: SnapshotIdentity
    materialized_length: int
    layers: tuple[SnapshotLayer, ...]

    @property
    def num_pages(self) -> int:
        return sum(layer.pages.shape[0] for layer in self.layers)

    @property
    def nbytes(self) -> int:
        return sum(
            layer.pages.numel() * layer.pages.element_size() for layer in self.layers
        )


@dataclass(frozen=True)
class ImportedLayer:
    layer_name: str
    spec: AttentionSpec
    cache: torch.Tensor
    block_table: torch.Tensor


@dataclass(frozen=True)
class ImportedTargetKV:
    identity: SnapshotIdentity
    materialized_length: int
    layers: tuple[ImportedLayer, ...]

    @property
    def num_pages(self) -> int:
        """Allocated pages, including one null page per layer."""
        return sum(layer.cache.shape[0] for layer in self.layers)

    @property
    def nbytes(self) -> int:
        return sum(
            layer.cache.numel() * layer.cache.element_size()
            + layer.block_table.numel() * layer.block_table.element_size()
            for layer in self.layers
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Protect receiver allocations used on another CUDA stream."""
        for layer in self.layers:
            layer.cache.record_stream(stream)
            layer.block_table.record_stream(stream)


def _validate_spec(spec: AttentionSpec) -> None:
    # Subclasses may represent MLA, sinks, R-SWA, or another state contract.
    if type(spec) not in (FullAttentionSpec, SlidingWindowSpec):
        raise ValueError("target-KV snapshots require plain full or sliding attention")
    if (
        spec.dtype != torch.bfloat16
        or spec.kv_quant_mode != KVQuantMode.NONE
        or min(spec.block_size, spec.num_kv_heads, spec.head_size) <= 0
        or spec.head_size_v != spec.head_size
        or spec.tokens_per_state != 1
        or spec.num_head_slots not in (None, spec.num_kv_heads)
        or spec.state_content_bytes is not None
        or spec.page_size_padded not in (None, spec.unpadded_page_size_bytes)
        or (spec.sliding_window is not None and spec.sliding_window <= 0)
        or (
            isinstance(spec, FullAttentionSpec)
            and (spec.non_causal or spec.attention_chunk_size is not None)
        )
    ):
        raise ValueError("target-KV snapshots require dense causal BF16 K/V")


def _range(spec: AttentionSpec, length: int) -> tuple[int, int, int]:
    start = 0 if spec.sliding_window is None else max(0, length - spec.sliding_window)
    return (
        start,
        start // spec.block_size,
        (length + spec.block_size - 1) // spec.block_size,
    )


def _page_shape(spec: AttentionSpec) -> tuple[int, int, int]:
    return spec.num_kv_heads, spec.block_size, 2 * spec.head_size


def _table_width(spec: AttentionSpec, num_blocks: int) -> int:
    return get_block_table_width(
        num_blocks,
        spec.block_size,
        token_alignment=spec.block_table_token_alignment,
    )


def _copy_live_tokens(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    logical_page: int,
    start: int,
    length: int,
    block_size: int,
) -> None:
    begin = max(0, start - logical_page * block_size)
    end = min(block_size, length - logical_page * block_size)
    destination[:, begin:end].copy_(source[:, begin:end])


def _complete_copies(device: torch.device) -> None:
    if device.type == "cuda":
        try:
            torch.cuda.current_stream(device).synchronize()
        except Exception as error:
            raise SnapshotCompletionError(
                "target-KV copy completion is unknown; abandon the runtime"
            ) from error


@contextmanager
def _completed_copy_phase(device: torch.device) -> Iterator[None]:
    try:
        yield
    except BaseException as error:
        if not cuda_context_fatal(error):
            _complete_copies(device)
        raise
    else:
        _complete_copies(device)


@torch.inference_mode()
def snapshot_target_kv(
    identity: SnapshotIdentity,
    materialized_length: int,
    sources: Sequence[TargetKVSource],
    *,
    budget: SnapshotBudget,
) -> TargetKVSnapshot:
    """Copy live pages after validating all sources and the total allocation.

    Source block tables must map absolute logical kernel pages to physical
    pages in the supplied cache view. Scheduler pages must be converted first.
    Entries outside a sliding window are ignored; missing live entries fail.
    Uncommitted tails and unused window prefixes are zero in the snapshot.
    """
    budget.check(materialized_length, 0, 0)
    if not sources or len({s.layer_name for s in sources}) != len(sources):
        raise ValueError("target-KV source layers must be nonempty and unique")
    plans = []
    total_pages = total_bytes = 0
    device = sources[0].cache.device
    for source in sources:
        spec, cache = source.spec, source.cache
        _validate_spec(spec)
        if (
            not source.layer_name
            or cache.device != device
            or device.type not in ("cpu", "cuda")
            or cache.dtype != spec.dtype
            or cache.ndim != 4
            or tuple(cache.shape[1:]) != _page_shape(spec)
        ):
            raise ValueError(
                "target-KV cache shape, dtype, or device differs from spec"
            )
        physical = (
            cache
            if source.layout == KVCacheLayout.LBHNC
            else cache.transpose(1, 2)
            if source.layout == KVCacheLayout.LBNHC
            else None
        )
        if physical is None or not physical.is_contiguous():
            raise ValueError("unsupported target-KV cache layout or strides")
        start, first, end = _range(spec, materialized_length)
        total_pages += end - first
        total_bytes += (end - first) * spec.unpadded_page_size_bytes
        budget.check(materialized_length, total_pages, total_bytes)
        table = source.block_table
        if (
            table.ndim != 1
            or table.dtype not in (torch.int32, torch.int64)
            or table.device.type not in ("cpu", "cuda")
            or (table.is_cuda and table.device != device)
            or table.numel() < end
        ):
            raise ValueError("invalid absolute target-KV block table")
        blocks = table[first:end].tolist()
        if any(block <= 0 or block >= cache.shape[0] for block in blocks) or len(
            set(blocks)
        ) != len(blocks):
            raise ValueError("missing, aliased, or out-of-range live target-KV page")
        plans.append((source, start, first, blocks))

    layers = []
    for source, start, first, blocks in plans:
        pages = torch.zeros(
            (len(blocks), *_page_shape(source.spec)),
            dtype=source.spec.dtype,
            device=device,
        )
        layers.append(
            SnapshotLayer(
                source.layer_name,
                source.spec,
                tuple(range(first, first + len(blocks))),
                pages,
            )
        )
    # Allocate every destination before issuing any reads of live target pages.
    with _completed_copy_phase(device):
        for (source, start, first, blocks), layer in zip(plans, layers):
            if device.type == "cuda":
                source.cache.record_stream(torch.cuda.current_stream(device))
            for i, block in enumerate(blocks):
                _copy_live_tokens(
                    layer.pages[i],
                    source.cache[block],
                    logical_page=first + i,
                    start=start,
                    length=materialized_length,
                    block_size=source.spec.block_size,
                )
    return TargetKVSnapshot(identity, materialized_length, tuple(layers))


@torch.inference_mode()
def import_target_kv(
    snapshot: TargetKVSnapshot,
    *,
    expected_identity: SnapshotIdentity,
    expected_materialized_length: int,
    expected_specs: Mapping[str, AttentionSpec],
    budget: SnapshotBudget,
    device: torch.device,
) -> ImportedTargetKV:
    """Validate one expected round and build private receiver KV and tables.

    The caller owns the outstanding-round fence: pass its current exact
    identity and materialized length, never values taken from the incoming
    snapshot itself.
    Every import allocates independent storage, including sibling imports.
    """
    if snapshot.identity != expected_identity:
        raise ValueError("stale or mismatched target-KV snapshot identity")
    length = snapshot.materialized_length
    if length != expected_materialized_length:
        raise ValueError("target-KV snapshot length differs from the expected query")
    budget.check(length, 0, 0)
    names = [layer.layer_name for layer in snapshot.layers]
    if not names or len(set(names)) != len(names) or set(names) != set(expected_specs):
        raise ValueError("target-KV snapshot layers differ from the expected contract")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("target-KV receiver requires a CPU or CUDA device")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.accelerator.current_device_index())
    total_pages = total_bytes = 0
    for layer in snapshot.layers:
        spec = layer.spec
        _validate_spec(spec)
        start, first, end = _range(spec, length)
        if spec != expected_specs[layer.layer_name]:
            raise ValueError("target-KV snapshot attention spec differs from contract")
        total_pages += end - first + 1
        total_bytes += (end - first + 1) * spec.unpadded_page_size_bytes + _table_width(
            spec, end
        ) * 4
        budget.check(length, total_pages, total_bytes)
        if (
            layer.logical_pages != tuple(range(first, end))
            or layer.pages.dtype != spec.dtype
            or tuple(layer.pages.shape) != (end - first, *_page_shape(spec))
            or not layer.pages.is_contiguous()
            or layer.pages.device.type not in ("cpu", "cuda")
        ):
            raise ValueError("invalid target-KV snapshot pages or logical indices")
        if layer.pages.is_cuda and layer.pages.device != device:
            raise ValueError("CUDA target-KV imports require the same CUDA device")

    layers = []
    for layer in snapshot.layers:
        spec = layer.spec
        start, first, end = _range(spec, length)
        cache = torch.zeros(
            (end - first + 1, *_page_shape(spec)), dtype=spec.dtype, device=device
        )
        block_table = torch.zeros(
            _table_width(spec, end), dtype=torch.int32, device=device
        )
        layers.append(ImportedLayer(layer.layer_name, spec, cache, block_table))
    with _completed_copy_phase(device):
        for source, receiver in zip(snapshot.layers, layers):
            start, _, _ = _range(source.spec, length)
            for i, logical in enumerate(source.logical_pages):
                receiver.block_table[logical].fill_(i + 1)
                _copy_live_tokens(
                    receiver.cache[i + 1],
                    source.pages[i],
                    logical_page=logical,
                    start=start,
                    length=length,
                    block_size=source.spec.block_size,
                )
    return ImportedTargetKV(snapshot.identity, length, tuple(layers))
