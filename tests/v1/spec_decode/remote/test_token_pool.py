# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sticky routing and failure isolation through actual v3 sockets and CPU KV."""

import threading
import time
from contextlib import ExitStack

import pytest
from msgspec.structs import replace

from tests.v1.spec_decode.remote.test_token_runtime import KVOracle
from vllm.v1.spec_decode.remote.capabilities import SpeculatorPlacementCapabilities
from vllm.v1.spec_decode.remote.protocol import (
    FeatureSlot,
    RemoteServerLimits,
    TargetFeatureSchema,
)
from vllm.v1.spec_decode.remote.runtime.kv import DraftBlockPool
from vllm.v1.spec_decode.remote.runtime.schema import RuntimeContract
from vllm.v1.spec_decode.remote.runtime.token_adapter import TokenOnlyDraftAdapter
from vllm.v1.spec_decode.remote.runtime.token_state import (
    TokenAdvanceRow,
    TokenStateMachine,
)
from vllm.v1.spec_decode.remote.token_pool import TokenDraftPool
from vllm.v1.spec_decode.remote.token_protocol import TokenHello
from vllm.v1.spec_decode.remote.token_server import TokenDraftServer


class GatedOracle(KVOracle):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = None

    def forward(self, *args, **kwargs):
        if self.release is not None:
            self.entered.set()
            assert self.release.wait(5), "test must release the external executor"
        return super().forward(*args, **kwargs)


@pytest.fixture
def pool_runtime():
    servers, states, oracles = [], [], []
    contract = RuntimeContract(
        TargetFeatureSchema(3, (FeatureSlot("token_ids", "int32", ()),)),
        SpeculatorPlacementCapabilities(
            state_dependency="own_kv",
            required_features=("token_ids",),
            standalone_weights="complete",
        ),
        "target",
        "tokenizer",
        "draft",
    )
    with ExitStack() as cleanup:
        for _ in range(2):
            oracle = GatedOracle()
            state = TokenStateMachine(
                oracle,
                DraftBlockPool(33, 4),
                max_num_reqs=4,
                max_model_len=64,
                vocab_size=128,
                max_k=3,
            )
            server = TokenDraftServer(
                TokenOnlyDraftAdapter(state, contract),
                limits=RemoteServerLimits(4, 64, 4, 64),
            )
            server.start("tcp://127.0.0.1:0")
            cleanup.callback(server.stop)
            servers.append(server)
            states.append(state)
            oracles.append(oracle)
        hello = TokenHello("v-a", "target", "draft", "tokenizer", 3, 128, 4, 64, 64)
        endpoints = []
        for server in servers:
            assert server.endpoint is not None
            endpoints.append(server.endpoint)

        def client(identity="v-a", policy="error"):
            pool = TokenDraftPool(
                tuple(endpoints),
                replace(hello, verifier_instance_id=identity),
                failure_policy=policy,
                request_timeout_ms=5000,
            )
            cleanup.callback(pool.close)
            pool.connect()
            return pool

        try:
            yield client, servers, states, oracles
        finally:
            for oracle in oracles:
                if oracle.release is not None:
                    oracle.release.set()


def open_prompt(pool, sequence_id, prompt=(1, 2)):
    key = pool.open_sequence(sequence_id, deadline=pool.deadline())
    pool.prefill(key, prompt, is_final=True, deadline=pool.deadline())
    return key


def test_two_verifiers_share_two_workers_without_namespace_or_row_mixups(pool_runtime):
    client, _, states, _ = pool_runtime
    first, second = client(), client("v-b")
    a = [open_prompt(first, i, (1, i + 2)) for i in range(2)]
    b = [open_prompt(second, i, (7, i + 8)) for i in range(2)]
    for state in states:
        assert {key.verifier_instance_id for key in state.sequences} == {"v-a", "v-b"}
    for pool, keys, prefixes in (
        (first, a[::-1], [(1, 3), (1, 2)]),
        (second, b, [(7, 8), (7, 9)]),
    ):
        rows = tuple(TokenAdvanceRow(key, 2, None, 0, 0, (11,)) for key in keys)
        output = pool.collect(pool.dispatch(0, rows, 3, deadline=pool.deadline()))
        assert output.valid_lengths == (3, 3)
        assert tuple(tokens[0] for tokens in output.token_ids) == tuple(
            KVOracle.predict([*prefix, 11]) for prefix in prefixes
        )
        # K0 still applies the authoritative correction, then K can grow again.
        rows = tuple(TokenAdvanceRow(key, 3, 0, 3, 0, (12,)) for key in keys)
        assert pool.collect(
            pool.dispatch(1, rows, 0, deadline=pool.deadline())
        ).valid_lengths == (0, 0)
        rows = tuple(TokenAdvanceRow(key, 4, 1, 0, 0, (13,)) for key in keys)
        assert pool.collect(
            pool.dispatch(2, rows, 1, deadline=pool.deadline())
        ).valid_lengths == (1, 1)
        pool.close_sequences(tuple(keys))
    assert all(
        not state.sequences and state.block_pool.num_free_blocks == 32
        for state in states
    )


def test_both_workers_make_progress_before_collect_and_unrelated_close_is_deferred(
    pool_runtime,
):
    client, _, states, oracles = pool_runtime
    pool = client()
    keys = [open_prompt(pool, i) for i in range(3)]
    release = threading.Event()
    for oracle in oracles:
        oracle.release = release
    rows = tuple(TokenAdvanceRow(key, 2, None, 0, 0, (3,)) for key in keys[:2])
    handle = pool.dispatch(0, rows, 2, deadline=pool.deadline())
    try:
        assert all(oracle.entered.wait(3) for oracle in oracles)
        pool.close_sequences(())
        pool.close_sequences((keys[2],))
        assert not pool.active(keys[2])
        assert any(keys[2] in state.sequences for state in states)
    finally:
        release.set()
    assert pool.collect(handle).valid_lengths == (2, 2)
    assert all(keys[2] not in state.sequences for state in states)
    with pytest.raises(RuntimeError, match="stale"):
        pool.collect(handle)


def test_worker_loss_falls_back_only_its_sticky_rows_and_new_requests_use_survivor(
    pool_runtime,
):
    client, servers, states, _ = pool_runtime
    pool = client(policy="target_only")
    keys = [open_prompt(pool, i) for i in range(2)]
    servers[0].stop()
    rows = tuple(TokenAdvanceRow(key, 2, None, 0, 0, (3,)) for key in keys)
    result = pool.collect(pool.dispatch(0, rows, 3, deadline=pool.deadline()))
    assert result.valid_lengths == (0, 3)
    assert not pool.active(keys[0]) and pool.active(keys[1])
    assert keys[0] not in states[1].sequences
    fresh = open_prompt(pool, 2, (4,))
    assert fresh in states[1].sequences and pool.active(fresh)
    pool.close_sequences(tuple(keys) + (fresh,))


def test_pool_rejects_foreign_cancelled_and_concurrent_handles(pool_runtime):
    client, _, _, _ = pool_runtime
    pool, sibling = client(), client("v-b")
    key = open_prompt(pool, 0)
    row = TokenAdvanceRow(key, 2, None, 0, 0, (3,))
    handle = pool.dispatch(0, (row,), 1, deadline=pool.deadline())
    with pytest.raises(RuntimeError, match="busy"):
        pool.dispatch(1, (row,), 1, deadline=pool.deadline())
    with pytest.raises(RuntimeError, match="foreign"):
        sibling.collect(handle)
    pool.cancel(handle)
    assert not pool.active(key)
    with pytest.raises(RuntimeError, match="stale"):
        pool.collect(handle)


def test_reused_sequence_id_on_another_worker_has_a_new_generation(pool_runtime):
    client, _, states, _ = pool_runtime
    pool = client()
    first = open_prompt(pool, 0)
    pool.close_sequences((first,))
    second = open_prompt(pool, 0)
    assert first != second
    assert second in states[1].sequences
    pool.close_sequences((first,))
    assert pool.active(second)


@pytest.mark.parametrize("finish", ["cancel", "close"])
def test_cancel_retires_deferred_request_on_an_unaffected_worker(pool_runtime, finish):
    client, _, states, _ = pool_runtime
    pool = client()
    first, other = (open_prompt(pool, i) for i in range(2))
    row = TokenAdvanceRow(first, 2, None, 0, 0, (3,))
    handle = pool.dispatch(0, (row,), 1, deadline=pool.deadline())
    pool.close_sequences((other,))
    if finish == "cancel":
        pool.cancel(handle)
    else:
        pool.close_sequences((first, other))
    assert other not in pool.records
    assert other not in states[1].sequences
    assert states[1].block_pool.num_free_blocks == 32


def test_native_speculative_config_preserves_all_pool_endpoints(tmp_path):
    from tests.v1.spec_decode.remote.test_token_runtime import token_config
    from vllm.config import SpeculativeConfig

    config = token_config(tmp_path)
    endpoints = ("unix:///draft0.sock", "unix:///draft1.sock")
    spec = SpeculativeConfig(
        model=config.speculative_config.model,
        method="draft_model",
        num_speculative_tokens=3,
        target_model_config=config.model_config,
        target_parallel_config=config.parallel_config,
        execution={"placement": "remote", "endpoints": endpoints},
    )
    assert spec.use_unified_draft_backend() and spec.use_remote_draft()
    assert spec.get_remote_draft_endpoints() == endpoints


def test_recovered_worker_rejoins_without_reviving_its_old_sequences(pool_runtime):
    client, servers, states, _ = pool_runtime
    pool = client(policy="target_only")
    keys = [open_prompt(pool, i) for i in range(2)]
    endpoint = servers[0].endpoint
    assert endpoint is not None
    servers[0].stop()
    result = pool.collect(
        pool.dispatch(
            0,
            tuple(TokenAdvanceRow(key, 2, None, 0, 0, (3,)) for key in keys),
            1,
            deadline=pool.deadline(),
        )
    )
    assert result.valid_lengths == (0, 1)
    recovered = TokenStateMachine(
        GatedOracle(),
        DraftBlockPool(33, 4),
        max_num_reqs=4,
        max_model_len=64,
        vocab_size=128,
        max_k=3,
    )
    server = TokenDraftServer(
        TokenOnlyDraftAdapter(recovered, servers[0].adapter.contract),
        limits=servers[0].limits,
    )
    try:
        server.start(endpoint)
        until = time.monotonic() + 3
        sequence_id = 2
        while True:
            fresh = open_prompt(pool, sequence_id)
            rejoined = fresh in recovered.sequences
            pool.close_sequences((fresh,))
            if rejoined or time.monotonic() >= until:
                break
            sequence_id += 1
            time.sleep(0.01)
        assert rejoined
        assert not pool.active(keys[0])
        assert keys[0] not in recovered.sequences
        assert keys[1] in states[1].sequences and pool.active(keys[1])
        pool.close_sequences(tuple(keys))
    finally:
        server.stop()
