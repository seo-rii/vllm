# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-verifier own-KV batches through the real v2 socket and state machine."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace

import pytest

from tests.v1.spec_decode.remote.runtime_fixtures import (
    KVDigestExecutor,
    digest_row,
    make_planner,
)
from tests.v1.spec_decode.remote.test_runtime import (
    CAPABILITIES,
    IDENTITY,
    SCHEMA,
    chunk,
    round_,
)
from tests.v1.spec_decode.remote.test_token_pool_server import until
from vllm.v1.spec_decode.remote.conditioned_server import ConditionedDraftServer
from vllm.v1.spec_decode.remote.protocol import RemoteServerLimits, SpeculatorStatusCode
from vllm.v1.spec_decode.remote.runtime.adapter import TargetConditionedDraftAdapter
from vllm.v1.spec_decode.remote.runtime.work_queue import DraftWork
from vllm.v1.spec_decode.remote.session import RemoteDraftSession
from vllm.v1.spec_decode.remote.state import SequenceState


class OwnerDigestExecutor(KVDigestExecutor):
    def __init__(self, pool):
        super().__init__(pool, max_num_tokens=8)
        self.threads: set[int] = set()

    def warmup(self):
        self.threads.add(threading.get_ident())

    def execute(self, *args, **kwargs):
        self.threads.add(threading.get_ident())
        return super().execute(*args, **kwargs)


@contextmanager
def runtime(tmp_path, *, executor_type=OwnerDigestExecutor, **kwargs):
    planner = make_planner(num_blocks=32)
    executor = executor_type(planner.block_pool)
    adapter = TargetConditionedDraftAdapter(
        executor,
        planner,
        capabilities=CAPABILITIES,
        feature_schema=SCHEMA,
        methods={"mtp"},
    )
    server = ConditionedDraftServer(
        adapter,
        limits=RemoteServerLimits(4, 64, 4, 64),
        batch_wait_ms=0,
        **kwargs,
    )
    endpoint = server.start(f"unix://{tmp_path}/conditioned.sock")
    sessions = []

    def client(identity, *, timeout_ms=3000):
        session = RemoteDraftSession(
            endpoint,
            replace(IDENTITY, verifier_instance_id=identity),
            request_timeout_ms=timeout_ms,
        )
        session.connect()
        sessions.append(session)
        return session

    try:
        yield server, executor, client
    finally:
        for session in sessions:
            session.close()
        server.stop()
        if server.healthy:
            assert planner.num_open == len(server.registry) == 0
            assert (
                planner.block_pool.num_free_blocks == planner.block_pool.num_blocks - 1
            )
            assert server.work_queue.depth == server.work_queue.pending_bytes == 0


@pytest.mark.parametrize("budgets", [(3, 3), (0, 3), (1, 1)])
def test_conditioned_clients_share_forward_without_sharing_kv_or_k(tmp_path, budgets):
    with runtime(tmp_path) as (server, oracle, client):
        a, b = client("a"), client("b")
        ka, kb = a.open_sequence(0), b.open_sequence(0)
        a.prefill(ka, chunk([11], lookahead=12), is_final=False)
        a.prefill(ka, chunk([12, 13]), is_final=True)
        b.prefill(kb, chunk([21, 22]), is_final=True)
        entered, release = threading.Event(), threading.Event()

        def hold():
            entered.set()
            assert release.wait(3)

        held = server.work_queue.submit(DraftWork("gate", hold))
        assert entered.wait(3)
        with ThreadPoolExecutor(max_workers=2) as threads:
            pa = threads.submit(
                round_,
                a,
                0,
                [ka],
                [0],
                [],
                [0],
                [30],
                num_speculative_tokens=budgets[0],
            )
            pb = threads.submit(
                round_,
                b,
                7,
                [kb],
                [0],
                [],
                [0],
                [40],
                num_speculative_tokens=budgets[1],
            )
            try:
                until(lambda: server.work_queue.depth == 3)
            finally:
                release.set()
            held.wait()
            ra, rb = pa.result(3), pb.result(3)
        for result, prefix, recovery, k in (
            (ra, [11, 12, 13], 30, budgets[0]),
            (rb, [21, 22], 40, budgets[1]),
        ):
            assert result.row_statuses == (SpeculatorStatusCode.OK,)
            assert result.output.valid_lengths.tolist() == [k]
            assert result.output.token_ids.tolist() == [
                digest_row(prefix, recovery)[:k] + [0] * (3 - k)
            ]
        assert [plan.num_reqs for plan in oracle.plans] == (
            [1, 2] if budgets[0] == budgets[1] else [1, 1, 1]
        )
        assert len(oracle.threads) == 1 and threading.get_ident() not in oracle.threads
        # Reordered, independent next rounds use each verifier's committed prefix.
        used = ra.output.token_ids[0, : budgets[0]].tolist()
        later = round_(a, 1, [ka], [len(used) + 1], [30, *used], [len(used)], [31])
        assert later.output.token_ids.tolist() == [
            digest_row([11, 12, 13, 30, *used], 31)
        ]
        a.close_sequences((ka,))
        assert b.sequence_state(kb) is SequenceState.READY


def test_conditioned_verifier_slot_credit_preserves_other_clients(tmp_path):
    with runtime(tmp_path, max_sequences_per_verifier=1) as (server, _, client):
        a, b = client("a"), client("b")
        ka = a.open_sequence(0)
        rejected = a.open_sequence(1)
        assert a.sequence_state(rejected) is SequenceState.TARGET_ONLY
        kb = b.open_sequence(0)
        b.prefill(kb, chunk([21]), is_final=True)
        assert round_(b, 0, [kb], [0], [], [0], [22]).output.valid_lengths.tolist() == [
            3
        ]
        assert len(server.registry) == 2
        a.close_sequences((ka, rejected))
        replacement = a.open_sequence(2)
        a.prefill(replacement, chunk([11]), is_final=True)
        assert round_(
            a, 0, [replacement], [0], [], [0], [12]
        ).output.token_ids.tolist() == [digest_row([11], 12)]


def test_conditioned_queued_expiry_retires_only_the_expired_verifier(tmp_path):
    with runtime(tmp_path) as (server, oracle, client):
        a, b = client("a", timeout_ms=100), client("b")
        ka, kb = a.open_sequence(0), b.open_sequence(0)
        a.prefill(ka, chunk([11]), is_final=True)
        b.prefill(kb, chunk([21]), is_final=True)
        entered, release = threading.Event(), threading.Event()

        def hold():
            entered.set()
            assert release.wait(3)

        held = server.work_queue.submit(DraftWork("gate", hold))
        assert entered.wait(3)
        with ThreadPoolExecutor(max_workers=2) as threads:
            pa = threads.submit(round_, a, 0, [ka], [0], [], [0], [12])
            until(lambda: server.work_queue.depth == 2)
            expires = time.monotonic() + 0.1
            pb = threads.submit(round_, b, 0, [kb], [0], [], [0], [22])
            try:
                assert pa.result(3).output.valid_lengths.tolist() == [0]
                until(lambda: time.monotonic() >= expires)
            finally:
                release.set()
            held.wait()
            assert pb.result(3).output.token_ids.tolist() == [digest_row([21], 22)]
        assert all(row.key != ka for plan in oracle.plans for row in plan.rows)
        assert b.sequence_state(kb) is SequenceState.READY
        assert server.healthy


class FatalDigestExecutor(OwnerDigestExecutor):
    def execute(self, plan, *args, **kwargs):
        self.threads.add(threading.get_ident())
        self.plans.append(plan)
        raise RuntimeError("CUDA error: device-side assert triggered")


def test_conditioned_fatal_forward_fences_later_k_group(tmp_path):
    with runtime(tmp_path, executor_type=FatalDigestExecutor) as (
        server,
        oracle,
        client,
    ):
        a, b = client("a"), client("b")
        ka, kb = a.open_sequence(0), b.open_sequence(0)
        a.prefill(ka, chunk([11]), is_final=True)
        b.prefill(kb, chunk([21]), is_final=True)
        entered, release = threading.Event(), threading.Event()

        def hold():
            entered.set()
            assert release.wait(3)

        held = server.work_queue.submit(DraftWork("gate", hold))
        assert entered.wait(3)
        with ThreadPoolExecutor(max_workers=2) as threads:
            pa = threads.submit(
                round_, a, 0, [ka], [0], [], [0], [12], num_speculative_tokens=1
            )
            until(lambda: server.work_queue.depth == 2)
            pb = threads.submit(round_, b, 0, [kb], [0], [], [0], [22])
            try:
                until(lambda: server.work_queue.depth == 3)
            finally:
                release.set()
            held.wait()
            assert pa.result(3).output.valid_lengths.tolist() == [0]
            assert pb.result(3).output.valid_lengths.tolist() == [0]
        assert not server.healthy
        assert len(oracle.plans) == 1
        assert tuple(row.key for row in oracle.plans[0].rows) == (ka,)
