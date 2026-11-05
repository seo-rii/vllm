# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared real configuration and stateful external draft executor fixtures."""

import inspect
import sys
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, Qwen3Config

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.v1.spec_decode.remote.runtime.adapter import DraftStepExecutor
from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
from vllm.v1.spec_decode.remote.runtime.step import DraftStepPlan, DraftStepPlanner

K = 3
VOCAB = 1000
BLOCK_SIZE = 2
STRIDE = 1 << 20


@contextmanager
def allocation_failures(faults, *, use_monitoring=True):
    """Inject line faults; install in the executing thread on Python 3.11."""
    pending = {}
    for method, fragment, error in faults:
        lines, first = inspect.getsourcelines(method)
        matches = [first + i for i, line in enumerate(lines) if fragment in line]
        assert len(matches) == 1, (method.__qualname__, fragment, matches)
        pending[method.__code__, matches[0]] = error
    codes = {code for code, _ in pending}
    triggered = []

    def fail(code, line):
        error = pending.pop((code, line), None)
        if error is not None:
            triggered.append(error)
            raise error

    monitoring = getattr(sys, "monitoring", None) if use_monitoring else None
    if monitoring is None:
        previous_trace, previous_profile = sys.gettrace(), sys.getprofile()

        def trace(frame, event, arg):
            delegate = previous_trace(frame, event, arg) if previous_trace else None
            if frame.f_code not in codes:
                return delegate

            def lines(frame, event, arg):
                nonlocal delegate
                if delegate is not None:
                    delegate = delegate(frame, event, arg)
                if event == "line":
                    fail(frame.f_code, frame.f_lineno)
                return lines

            return lines

        def profile(frame, event, arg):
            if previous_profile is not None:
                previous_profile(frame, event, arg)
            # Python disables tracing when a callback raises. Re-arm at the
            # next selected call so a rollback can fail independently.
            if (
                event == "call"
                and frame.f_code in codes
                and sys.gettrace() is not trace
            ):
                sys.settrace(trace)
                frame.f_trace = trace(frame, event, arg)

        sys.setprofile(profile)
        sys.settrace(trace)
        try:
            yield triggered
        finally:
            sys.settrace(previous_trace)
            sys.setprofile(previous_profile)
        return

    tool_id = next(index for index in range(6) if monitoring.get_tool(index) is None)
    monitoring.use_tool_id(tool_id, "remote-allocation-faults")
    try:
        monitoring.register_callback(tool_id, monitoring.events.LINE, fail)
        for code in codes:
            monitoring.set_local_events(tool_id, code, monitoring.events.LINE)
        yield triggered
    finally:
        for code in codes:
            monitoring.set_local_events(tool_id, code, 0)
        monitoring.register_callback(tool_id, monitoring.events.LINE, None)
        monitoring.free_tool_id(tool_id)


@contextmanager
def cpu_accelerator_cleanup():
    """CPU model loading has no accelerator allocator to empty."""
    with pytest.MonkeyPatch.context() as patch:
        if torch.accelerator.current_accelerator() is None:
            patch.setattr(torch.accelerator, "empty_cache", lambda: None)
        yield


@pytest.fixture
def cpu_parallel_state(tmp_path_factory):
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    rendezvous = tmp_path_factory.mktemp("remote-dist") / "store"
    with cpu_accelerator_cleanup(), set_current_vllm_config(make_config()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{rendezvous}",
            backend="gloo",
        )
        try:
            initialize_model_parallel(backend="gloo")
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


@lru_cache(maxsize=1)
def _local_config():
    directory = TemporaryDirectory(prefix="vllm-remote-config-")
    root = Path(directory.name)
    target, draft = root / "target", root / "draft"
    Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=VOCAB,
        max_position_embeddings=64,
    ).save_pretrained(target)
    LlamaConfig(
        architectures=["Eagle3LlamaForCausalLM"],
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=VOCAB,
        max_position_embeddings=64,
        eagle_config={"use_aux_hidden_state": False},
    ).save_pretrained(draft)
    for path in (target, draft):
        save_file(
            {"model.embed_tokens.weight": torch.ones(2, 2)}, path / "model.safetensors"
        )
    (target / "tokenizer.json").write_text('{"model":{"vocab":{"a":0}}}')
    model = ModelConfig(
        model=str(target), dtype="float32", max_model_len=64, skip_tokenizer_init=True
    )
    parallel = ParallelConfig()
    spec = SpeculativeConfig(
        model=str(draft),
        method="eagle3",
        num_speculative_tokens=K,
        target_model_config=model,
        target_parallel_config=parallel,
    )
    config = VllmConfig(
        model_config=model,
        parallel_config=parallel,
        speculative_config=spec,
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=64,
            max_model_len=64,
            async_scheduling=False,
            is_encoder_decoder=False,
        ),
        cache_config=CacheConfig(enable_prefix_caching=False),
        device_config=DeviceConfig("cpu"),
    )
    return directory, config


def make_config(*, num_speculative_tokens=K):
    config = deepcopy(_local_config()[1])
    config.speculative_config.num_speculative_tokens = num_speculative_tokens
    return config


def make_planner(num_blocks=64, max_num_reqs=4, max_model_len=64):
    return DraftStepPlanner(
        DraftBlockPool(num_blocks, BLOCK_SIZE),
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        num_speculative_tokens=K,
    )


def digest_row(prefix, next_token) -> list[int]:
    digest = sum((token + 1) * (position + 1) for position, token in enumerate(prefix))
    return [(digest + next_token + j) % VOCAB for j in range(K)]


class KVDigestExecutor(DraftStepExecutor):
    """Deterministic external CPU executor with actual per-request KV storage.

    Writes (position, target token) into every token's KV slot and drafts
    from a digest of the committed prefix read back through its own block
    tables, so any position, slot, block table, or committed-length error
    in the runtime changes the drafts.
    """

    def __init__(self, pool: DraftBlockPool, max_num_tokens=None):
        self.pool = pool
        self._max_num_tokens = max_num_tokens
        self.kv = np.full(pool.num_blocks * pool.block_size, -1, dtype=np.int64)
        self.block_tables: dict[int, list[int]] = defaultdict(list)
        self.plans: list[DraftStepPlan] = []
        self.effective_ks: list[int] = []

    @property
    def max_num_tokens(self):
        return self._max_num_tokens

    def execute(self, plan, features, *, num_speculative_tokens=None):
        effective_k = K if num_speculative_tokens is None else num_speculative_tokens
        self.effective_ks.append(effective_k)
        if self.max_num_tokens is not None and plan.num_tokens > self.max_num_tokens:
            raise ValueError("draft step exceeds the configured feature-token budget")
        self.plans.append(plan)
        for req_index, update in plan.block_table_updates.items():
            table = self.block_tables[req_index][: update.start]
            self.block_tables[req_index] = table + update.block_ids
        tokens = features[0].numpy().astype(np.int64)
        assert tokens.shape == (plan.num_tokens,)
        self.kv[plan.slot_mapping] = plan.positions * STRIDE + tokens
        drafts = []
        for i in range(plan.num_reqs):
            committed = int(plan.seq_lens[i] - plan.num_rejected[i])
            table = np.array(self.block_tables[int(plan.req_indices[i])])
            positions = np.arange(committed)
            slots = table[positions // BLOCK_SIZE] * BLOCK_SIZE + positions % BLOCK_SIZE
            values = self.kv[slots]
            assert (values // STRIDE == positions).all()
            next_token = (
                plan.last_sampled[i]
                if plan.num_sampled[i]
                else plan.next_prefill_tokens[i][0]
            )
            drafts.append(
                digest_row((values % STRIDE).tolist(), int(next_token))[:effective_k]
                + [0] * (K - effective_k)
            )
        return torch.tensor(drafts)
