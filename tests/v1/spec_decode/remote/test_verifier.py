# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 request/chunk translation against a real socket and stateful CPU KV.

These tests exercise the public propose/lifecycle boundary without CUDA. The
KV digest makes missing, duplicated and rejected context observable in drafts.
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.spec_decode.remote.test_runtime import (
    VOCAB,
    K,
    KVDigestExecutor,
    digest_row,
    make_planner,
)
from vllm.config.speculative import RemoteDraftConfig
from vllm.v1.spec_decode.remote.capabilities import SpeculatorPlacementCapabilities
from vllm.v1.spec_decode.remote.protocol import (
    FeatureSlot,
    SequenceAck,
    SpeculatorStatusCode,
    TargetFeatureSchema,
)
from vllm.v1.spec_decode.remote.runtime.adapter import TargetConditionedDraftAdapter
from vllm.v1.spec_decode.remote.runtime.schema import RuntimeContract
from vllm.v1.spec_decode.remote.server import AdapterError, RemoteDraftServer
from vllm.v1.spec_decode.remote.transport import ConnectionClosed
from vllm.v1.worker.gpu.spec_decode import remote

SCHEMA = TargetFeatureSchema(
    1,
    (
        FeatureSlot("token_ids", "int32", ()),
        FeatureSlot("positions", "int64", ()),
        FeatureSlot("last_hidden_states", "float32", (4,)),
        FeatureSlot(
            "next_prefill_tokens", "int64", (1,), optional=True, per_sequence=True
        ),
    ),
)
CONTRACT = RuntimeContract(
    SCHEMA,
    SpeculatorPlacementCapabilities(
        state_dependency="own_kv",
        required_features=("token_ids", "positions", "last_hidden_states"),
        num_prefill_lookahead=1,
        standalone_weights="complete",
    ),
    "target",
    "tokenizer",
    "draft",
)


def test_ipc_features_do_not_materialize_hidden_states_on_host():
    instance = object.__new__(remote.RemoteSpeculator)
    instance.device = torch.device("meta")
    instance._contract = CONTRACT
    instance._session = SimpleNamespace(selected_transport="cuda_ipc")
    batch = inputs(["a"], [[1, 2]], [[0, 1]], [2])
    hidden = torch.empty(2, 4, device="meta")
    # A meta tensor cannot be copied to CPU. This catches an accidental host
    # materialization without requiring a GPU in the protocol test suite.
    features = instance._features(batch, hidden, None, torch.tensor([[3]]))
    assert features["last_hidden_states"].device.type == "meta"
    assert features["token_ids"].tolist() == [1, 2]
    assert features["positions"].tolist() == [0, 1]
    assert features["next_prefill_tokens"].tolist() == [[3]]


class TPGroup:
    def __init__(self):
        self.rank_in_group = 0
        self.payload = None

    def broadcast_object(self, value, src):
        if self.rank_in_group == src:
            self.payload = value
        return self.payload

    def broadcast(self, value, src):
        pytest.fail("remote drafts must not initialize a GPU tensor collective")


def config(endpoint):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            async_scheduling=False, max_num_seqs=4, max_num_batched_tokens=64
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        model_config=SimpleNamespace(
            is_multimodal_model=False, max_model_len=64, get_vocab_size=lambda: VOCAB
        ),
        lora_config=None,
        watermark_config=None,
        speculative_config=SimpleNamespace(
            method="eagle3",
            num_speculative_tokens=K,
            enable_adaptive_verification=False,
            remote_draft=RemoteDraftConfig(
                endpoint=endpoint, failure_policy="error", request_timeout_ms=5000
            ),
        ),
    )


@pytest.fixture
def verifier(monkeypatch, request):
    planner = make_planner()
    executor = KVDigestExecutor(planner.block_pool)
    adapter = TargetConditionedDraftAdapter(
        executor,
        planner,
        capabilities=CONTRACT.capabilities,
        feature_schema=SCHEMA,
        methods={"eagle3"},
        draft_checkpoint_fingerprint=CONTRACT.draft_fingerprint,
    )
    server = RemoteDraftServer(
        adapter,
        target_fingerprint=CONTRACT.target_fingerprint,
        tokenizer_fingerprint=CONTRACT.tokenizer_fingerprint,
    )
    server.start("tcp://127.0.0.1:0")
    group = TPGroup()
    monkeypatch.setattr(remote, "get_tp_group", lambda: group)
    monkeypatch.setattr(remote, "create_runtime_contract", lambda _: CONTRACT)
    settings = config(server.endpoint)
    settings.speculative_config.remote_draft.failure_policy = getattr(
        request, "param", "error"
    )
    instance = remote.RemoteSpeculator(settings, torch.device("cpu"))
    yield instance, executor, settings, group
    instance.close()
    server.stop()


def inputs(ids, tokens, positions, prefill, slots=None):
    count = len(ids)
    slots = slots if slots is not None else list(range(count))
    lens = [len(row) for row in tokens]
    return SimpleNamespace(
        req_ids=ids,
        num_reqs=count,
        num_tokens=sum(lens),
        idx_mapping=torch.tensor(slots),
        query_start_loc_np=np.cumsum([0, *lens]),
        input_ids=torch.tensor(sum(tokens, []), dtype=torch.int32),
        positions=torch.tensor(sum(positions, []), dtype=torch.int64),
        is_prefilling_np=np.array([length > 0 for length in prefill]),
        prefill_len_np=np.array(prefill),
    )


def propose(verifier, batch, sampled, rejected, recovery, lookahead=None, **kwargs):
    last_sampled = torch.zeros(4, 1, dtype=torch.int64)
    last_sampled[batch.idx_mapping, 0] = torch.tensor(recovery)
    next_prefill = torch.zeros(1, 4, dtype=torch.int32)
    if lookahead is not None:
        next_prefill[0, batch.idx_mapping] = torch.tensor(lookahead, dtype=torch.int32)
    return verifier.propose(
        batch,
        {},
        {},
        torch.zeros(batch.num_tokens, 4),
        kwargs.pop("aux_hidden_states", None),
        torch.tensor(sampled),
        torch.tensor(rejected),
        last_sampled,
        next_prefill,
        kwargs.pop("temperature", torch.zeros(4)),
        torch.zeros(4, dtype=torch.int64),
        **kwargs,
    )


def test_chunks_rejections_and_request_reordering_preserve_verified_context(verifier):
    instance, executor, _, _ = verifier
    batch = inputs(["a", "b"], [[11, 12], [21, 22]], [[0, 1], [0, 1]], [4, 2], [3, 1])
    drafts = propose(instance, batch, [0, 1], [0, 0], [0, 30], [13, 0])
    b_drafts = digest_row([21, 22], 30)
    assert drafts.tolist() == [[-1] * K, b_drafts]
    assert executor.plans[0].next_prefill_tokens.tolist() == [[13]]

    # Finish a's prompt while b rejects its last two proposals. The final
    # prompt must be buffered then consumed once by a zero-query round.
    batch = inputs(
        ["b", "a"],
        [[30, *b_drafts], [13, 14]],
        [[2, 3, 4, 5], [2, 3]],
        [0, 4],
        [1, 3],
    )
    drafts = propose(instance, batch, [2, 1], [2, 0], [31, 40])
    b_prefix = [21, 22, 30, b_drafts[0]]
    assert drafts.tolist() == [
        digest_row(b_prefix, 31),
        digest_row([11, 12, 13, 14], 40),
    ]
    assert executor.plans[-1].num_computed_tokens.tolist() == [2, 2]
    assert executor.plans[-1].num_rejected.tolist() == [2, 0]

    batch = inputs(["b"], [[31, *drafts[0].tolist()]], [[4, 5, 6, 7]], [0], [1])
    assert propose(instance, batch, [1], [3], [32]).tolist() == [
        digest_row([*b_prefix, 31], 32)
    ]


def test_close_releases_kv_and_reused_request_id_starts_from_zero(verifier):
    instance, executor, _, _ = verifier
    batch = inputs(["request"], [[1, 2]], [[0, 1]], [2], [3])
    propose(instance, batch, [1], [0], [3])
    old_key = executor.plans[-1].rows[0].key
    instance.close_requests({"request", "already-gone"})
    batch = inputs(["request"], [[7]], [[0]], [1], [3])
    assert propose(instance, batch, [1], [0], [8]).tolist() == [digest_row([7], 8)]
    plan = executor.plans[-1]
    assert plan.rows[0].key != old_key
    assert plan.num_computed_tokens.tolist() == [0]
    assert list(plan.block_table_updates.values())[0].start == 0


@pytest.mark.parametrize("flag", ["dummy_run", "is_profile"])
def test_dummy_and_profile_do_not_contact_or_advance_server(
    verifier, monkeypatch, flag
):
    instance, executor, _, _ = verifier

    def reject_network(*args, **kwargs):
        pytest.fail("dummy/profile run contacted remote session")

    monkeypatch.setattr(instance._session, "open_sequence", reject_network)
    monkeypatch.setattr(instance._session, "prefill_many", reject_network)
    monkeypatch.setattr(instance._session, "dispatch", reject_network)
    batch = inputs(["a"], [[1]], [[0]], [1])
    assert propose(instance, batch, [1], [0], [2], **{flag: True}).tolist() == [
        [-1] * K
    ]
    assert executor.plans == []


@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.parametrize("transport", ["inline", "cuda_ipc"])
@pytest.mark.parametrize("mixed", [False, True])
def test_profile_budgets_feature_copies_without_mutating_remote_state(
    verifier, monkeypatch, auxiliary, transport, mixed
):
    instance, executor, _, group = verifier
    instance._session._selected_transport = transport
    aux = [torch.zeros(3, 4), torch.ones(3, 4)] if auxiliary else None
    if auxiliary:
        monkeypatch.setattr(
            instance,
            "_contract",
            SimpleNamespace(
                schema=TargetFeatureSchema(
                    1,
                    (
                        *SCHEMA.slots[:2],
                        FeatureSlot("aux_hidden_states", "float32", (8,)),
                        *SCHEMA.slots[3:],
                    ),
                )
            ),
        )
    feature_calls, selected = [], []
    prepare = instance._features
    select = instance._select

    def prepare_features(*args, **kwargs):
        result = prepare(*args, **kwargs)
        feature_calls.append(kwargs)
        return result

    def select_features(features, tokens, rows):
        result = select(features, tokens, rows)
        source, copy = (
            features[instance._contract.schema.slots[2].kind],
            result.slots[2],
        )
        assert source.untyped_storage().data_ptr() != copy.untyped_storage().data_ptr()
        assert torch.equal(source, copy)
        selected.append(tuple(copy.shape))
        return result

    def reject_network(*args, **kwargs):
        pytest.fail("profiling contacted the remote session")

    monkeypatch.setattr(instance, "_features", prepare_features)
    monkeypatch.setattr(instance, "_select", select_features)
    monkeypatch.setattr(instance._session, "open_sequence", reject_network)
    monkeypatch.setattr(instance._session, "prefill_many", reject_network)
    monkeypatch.setattr(instance._session, "dispatch", reject_network)
    batch = inputs(
        ["a", "b"], [[1, 2], [3]], [[0, 1], [2 if mixed else 0]], [2, 0 if mixed else 1]
    )
    before_stats = dict(instance._stats)
    output = propose(
        instance,
        batch,
        [1, 1],
        [0, 0],
        [4, 5],
        aux_hidden_states=aux,
        is_profile=True,
        dummy_run=True,
    )
    assert output.tolist() == [[-1] * K, [-1] * K]
    assert feature_calls == [{"token_rows": [0, 1, 2], "sequence_rows": [0, 1]}]
    assert selected == [(3, 8 if auxiliary else 4)]
    assert instance._stats == before_stats
    assert instance._requests == {}
    assert executor.plans == []
    # Only rank zero owns and profiles the transferred features.
    group.rank_in_group = 1
    propose(instance, batch, [1, 1], [0, 0], [4, 5], is_profile=True)
    assert len(feature_calls) == 1
    group.rank_in_group = 0


@pytest.mark.parametrize("verifier", ["target_only"], indirect=True)
def test_profile_allocation_failure_is_not_hidden_by_target_only(verifier, monkeypatch):
    instance, _, _, _ = verifier

    def fail_features(*args, **kwargs):
        raise torch.OutOfMemoryError("profile feature allocation")

    monkeypatch.setattr(instance, "_features", fail_features)
    batch = inputs(["a"], [[1]], [[0]], [1])
    with pytest.raises(torch.OutOfMemoryError, match="profile feature allocation"):
        propose(instance, batch, [1], [0], [2], is_profile=True)
    assert instance._requests == {}


@pytest.mark.parametrize("effective_k", [0, 1, K])
def test_effective_k_reaches_executor_and_preserves_context(verifier, effective_k):
    instance, executor, _, _ = verifier
    batch = inputs(["a"], [[11, 12]], [[0, 1]], [2])
    output = propose(
        instance, batch, [1], [0], [13], num_speculative_tokens=effective_k
    ).tolist()[0]
    expected = digest_row([11, 12], 13)[:effective_k]
    assert output == expected + [-1] * (K - effective_k)
    assert executor.effective_ks == [effective_k]
    assert instance._requests["a"].num_computed == 2
    verified = [13, *expected]
    batch = inputs(["a"], [verified], [list(range(2, 2 + len(verified)))], [0])
    assert propose(instance, batch, [len(verified)], [0], [14]).tolist() == [
        digest_row([11, 12, *verified], 14)
    ]
    assert executor.effective_ks == [effective_k, K]


@pytest.mark.parametrize("effective_k", [-1, K + 1])
def test_invalid_effective_k_does_not_open_or_advance_requests(verifier, effective_k):
    instance, executor, _, _ = verifier
    batch = inputs(["a"], [[1]], [[0]], [1])
    with pytest.raises(RuntimeError, match="draft budget"):
        propose(instance, batch, [1], [0], [2], num_speculative_tokens=effective_k)
    assert instance._requests == {}
    assert executor.plans == []


def test_nonzero_tp_rank_receives_cpu_tokens_without_gpu_collective(verifier):
    instance, executor, settings, group = verifier
    group.rank_in_group = 1
    peer = remote.RemoteSpeculator(settings, torch.device("cpu"))
    assert peer._session is None
    group.rank_in_group = 0
    batch = inputs(["a"], [[1]], [[0]], [1])
    expected = propose(instance, batch, [1], [0], [2])
    assert group.payload == (None, expected.tolist())
    plan_count = len(executor.plans)
    group.rank_in_group = 1
    actual = propose(peer, batch, [1], [0], [2])
    assert torch.equal(actual, expected)
    assert len(executor.plans) == plan_count
    group.rank_in_group = 0


def test_error_is_shared_instead_of_tokens_with_tp_peer(verifier):
    instance, _, settings, group = verifier
    group.rank_in_group = 1
    peer = remote.RemoteSpeculator(settings, torch.device("cpu"))
    group.rank_in_group = 0
    batch = inputs(["a"], [[1]], [[0]], [1])
    with pytest.raises(RuntimeError, match="greedy target sampling"):
        propose(instance, batch, [1], [0], [2], temperature=torch.ones(4))
    group.rank_in_group = 1
    with pytest.raises(RuntimeError, match="greedy target sampling"):
        propose(peer, batch, [1], [0], [2])
    assert group.payload[1] is None
    group.rank_in_group = 0


def test_request_close_error_is_shared_with_tp_peer_without_session(
    verifier, monkeypatch
):
    instance, _, settings, group = verifier
    group.rank_in_group = 1
    peer = remote.RemoteSpeculator(settings, torch.device("cpu"))
    group.rank_in_group = 0
    batch = inputs(["a"], [[1]], [[0]], [1])
    propose(instance, batch, [1], [0], [2])

    def fail(keys):
        raise RuntimeError("close failed")

    monkeypatch.setattr(instance._session, "close_sequences", fail)
    with pytest.raises(RuntimeError, match="request cleanup failed.*close failed"):
        instance.close_requests({"a"})
    group.rank_in_group = 1
    with pytest.raises(RuntimeError, match="request cleanup failed.*close failed"):
        peer.close_requests({"a"})
    group.rank_in_group = 0


def test_shutdown_error_does_not_mask_an_engine_error(verifier, monkeypatch):
    instance, _, _, _ = verifier
    session = instance._session
    close = session.close

    def fail():
        close()
        raise OSError("socket close failed")

    monkeypatch.setattr(session, "close", fail)
    instance.close()
    instance.close()
    assert instance._session is None


@pytest.mark.parametrize("existing,computed", [(True, 0), (False, 4)])
def test_runner_rejects_streaming_and_missing_prefix_before_removing_state(
    existing, computed
):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(use_remote_draft=lambda: True),
        req_states=SimpleNamespace(req_id_to_index={"a": 0} if existing else {}),
    )
    scheduler = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="a", prefill_token_ids=[1, 2], num_computed_tokens=computed
            )
        ]
    )
    with pytest.raises(ValueError, match="streaming input updates"):
        GPUModelRunner.add_requests(runner, scheduler)


def test_invalid_verification_counts_fail_before_advancing_context(verifier):
    instance, executor, _, _ = verifier
    batch = inputs(["a"], [[1]], [[0]], [1])
    drafts = propose(instance, batch, [1], [0], [2]).tolist()[0]
    plan_count = len(executor.plans)
    batch = inputs(["a"], [[2, *drafts]], [[1, 2, 3, 4]], [0])
    with pytest.raises(RuntimeError, match="counts do not cover"):
        propose(instance, batch, [2], [0], [3])
    assert len(executor.plans) == plan_count


@pytest.mark.parametrize("verifier", ["target_only"], indirect=True)
def test_failed_remote_row_stays_target_only_without_reusing_stale_kv(
    verifier, monkeypatch
):
    instance, executor, _, _ = verifier
    calls = []

    def fail(plan, features, *, num_speculative_tokens=None):
        calls.append(plan)
        raise AdapterError(SpeculatorStatusCode.OUT_OF_MEMORY, "draft KV exhausted")

    monkeypatch.setattr(executor, "execute", fail)
    batch = inputs(["a"], [[1]], [[0]], [1])
    assert propose(instance, batch, [1], [0], [2]).tolist() == [[-1] * K]
    batch = inputs(["a"], [[2]], [[1]], [0])
    assert propose(instance, batch, [1], [0], [3]).tolist() == [[-1] * K]
    assert len(calls) == 1


@pytest.mark.parametrize("verifier", ["target_only", "error"], indirect=True)
def test_partial_prefill_ack_disconnect_preserves_fallback_and_shares_policy(
    verifier, monkeypatch
):
    instance, executor, settings, group = verifier
    group.rank_in_group = 1
    peer = remote.RemoteSpeculator(settings, torch.device("cpu"))
    group.rank_in_group = 0
    session = instance._session
    prefill_many = session.prefill_many
    acknowledged: list[SequenceAck] = []

    def disconnect_after_first_ack(chunks):
        await_ack = session._await

        def receive(*args, **kwargs):
            if acknowledged:
                raise ConnectionClosed("connection lost after first prefill ACK")
            envelope, ack = await_ack(*args, **kwargs)
            assert isinstance(ack, SequenceAck)
            assert ack.status is SpeculatorStatusCode.OK
            acknowledged.append(ack)
            return envelope, ack

        with monkeypatch.context() as patch:
            patch.setattr(session, "_await", receive)
            return prefill_many(chunks)

    monkeypatch.setattr(session, "prefill_many", disconnect_after_first_ack)
    batch = inputs(["a", "b"], [[1, 2], [11, 12]], [[0, 1], [0, 1]], [2, 2])
    policy = settings.speculative_config.remote_draft.failure_policy
    error = "RemoteDraftError: prefill failed for"
    if policy == "error":
        with pytest.raises(RuntimeError, match=error):
            propose(instance, batch, [1, 1], [0, 0], [3, 13])
    else:
        assert propose(instance, batch, [1, 1], [0, 0], [3, 13]).tolist() == [
            [-1] * K,
            [-1] * K,
        ]
    assert len(acknowledged) == 1
    assert not session.connected
    assert all(
        session.sequence_state(instance._requests[req_id].key)
        is remote.SequenceState.INVALID
        for req_id in batch.req_ids
    )
    assert executor.plans == []

    group.rank_in_group = 1
    try:
        if policy == "error":
            with pytest.raises(RuntimeError, match=error):
                propose(peer, batch, [1, 1], [0, 0], [3, 13])
        else:
            assert propose(peer, batch, [1, 1], [0, 0], [3, 13]).tolist() == [
                [-1] * K,
                [-1] * K,
            ]
    finally:
        group.rank_in_group = 0

    def reject_features(*args, **kwargs):
        pytest.fail("disconnected own-KV requests resumed remote decoding")

    monkeypatch.setattr(instance, "_features", reject_features)
    batch = inputs(["a", "b"], [[3], [13]], [[2], [2]], [0, 0])
    assert propose(instance, batch, [1, 1], [0, 0], [4, 14]).tolist() == [
        [-1] * K,
        [-1] * K,
    ]


@pytest.mark.parametrize("verifier", ["target_only"], indirect=True)
def test_target_only_round_skips_feature_preparation_and_still_closes(
    verifier, monkeypatch
):
    instance, executor, _, _ = verifier

    def fail(plan, features, *, num_speculative_tokens=None):
        raise AdapterError(SpeculatorStatusCode.OUT_OF_MEMORY, "draft KV exhausted")

    monkeypatch.setattr(executor, "execute", fail)
    batch = inputs(["a"], [[1]], [[0]], [1])
    propose(instance, batch, [1], [0], [2])

    def reject_features(*args, **kwargs):
        pytest.fail("target-only round prepared remote features")

    monkeypatch.setattr(instance, "_features", reject_features)
    batch = inputs(["a"], [[2]], [[1]], [0])
    assert propose(instance, batch, [1], [0], [3]).tolist() == [[-1] * K]
    key = instance._requests["a"].key
    instance.close_requests({"a"})
    assert instance._requests == {}
    assert instance._session.sequence_state(key) is None


@pytest.mark.parametrize("verifier", ["target_only"], indirect=True)
@pytest.mark.parametrize("reconnect", [False, True])
def test_disconnected_new_request_attempts_reconnect_before_feature_preparation(
    verifier, monkeypatch, reconnect
):
    instance, _, _, _ = verifier
    session = instance._session
    session._invalidate("test connection loss")
    attempts = []
    maybe_reconnect = session.maybe_reconnect

    def unavailable():
        attempts.append(True)
        return maybe_reconnect() if reconnect else False

    def reject_features(*args, **kwargs):
        pytest.fail("disconnected request prepared remote features")

    monkeypatch.setattr(session, "maybe_reconnect", unavailable)
    if not reconnect:
        monkeypatch.setattr(instance, "_features", reject_features)
    batch = inputs(["new"], [[1, 2]], [[0, 1]], [2])
    expected = digest_row([1, 2], 3) if reconnect else [-1] * K
    assert propose(instance, batch, [1], [0], [3]).tolist() == [expected]
    assert attempts
    key = instance._requests["new"].key
    instance.close_requests({"new"})
    assert session.sequence_state(key) is None


@pytest.mark.parametrize("verifier", ["target_only"], indirect=True)
def test_mixed_target_only_batch_transfers_only_active_ragged_features(
    verifier, monkeypatch
):
    instance, executor, _, _ = verifier
    execute = executor.execute

    def fail(plan, features, *, num_speculative_tokens=None):
        raise AdapterError(SpeculatorStatusCode.OUT_OF_MEMORY, "draft KV exhausted")

    monkeypatch.setattr(executor, "execute", fail)
    propose(instance, inputs(["dead"], [[1]], [[0]], [1]), [1], [0], [2])
    monkeypatch.setattr(executor, "execute", execute)
    transferred = []
    tensor_to = torch.Tensor.to

    def track_hidden(tensor, *args, **kwargs):
        if tensor.ndim == 2 and tensor.dtype == torch.float32:
            transferred.append(tensor[:, 0].tolist())
        return tensor_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", track_hidden)
    batch = inputs(
        ["b", "dead", "c"],
        [[11, 12], [2], [21, 22, 23]],
        [[0, 1], [1], [0, 1, 2]],
        [4, 0, 3],
        [3, 0, 1],
    )
    result = instance._propose(
        batch,
        torch.arange(6, dtype=torch.float32)[:, None].expand(-1, 4),
        None,
        torch.tensor([0, 1, 1]),
        torch.tensor([0, 0, 0]),
        torch.tensor([[3], [30], [0], [0]]),
        torch.tensor([[0, 0, 0, 13]]),
    )
    assert result.tolist() == [[-1] * K, [-1] * K, digest_row([21, 22, 23], 30)]
    assert transferred == [[0, 1, 3, 4, 5]]
    assert executor.plans[-2].next_prefill_tokens.tolist() == [[13]]


def test_auxiliary_features_are_selected_before_concatenation(monkeypatch):
    instance = object.__new__(remote.RemoteSpeculator)
    instance.device = torch.device("cpu")
    instance._session = SimpleNamespace(selected_transport="inline")
    instance._contract = SimpleNamespace(
        schema=TargetFeatureSchema(
            1, (FeatureSlot("aux_hidden_states", "float32", (8,)),)
        )
    )
    batch = inputs(["a", "b"], [[1, 2], [3]], [[0, 1], [0]], [2, 1])
    concatenated = []
    tensor_cat = torch.cat

    def track_aux(tensors, **kwargs):
        concatenated.append([tensor[:, 0].tolist() for tensor in tensors])
        return tensor_cat(tensors, **kwargs)

    monkeypatch.setattr(torch, "cat", track_aux)
    state = torch.arange(3, dtype=torch.float32)[:, None].expand(-1, 4)
    features = instance._features(
        batch,
        state,
        [state, state + 10],
        torch.zeros(1, 2, dtype=torch.int64),
        token_rows=[2],
        sequence_rows=[1],
    )
    assert concatenated == [[[2], [12]]]
    assert features["aux_hidden_states"].shape == (1, 8)


@pytest.mark.parametrize("verifier", ["target_only", "error"], indirect=True)
@pytest.mark.parametrize("stage", ["aux_cat", "prefill_select", "decode_select"])
def test_remote_feature_oom_latches_affected_rows_and_shares_policy(
    verifier, monkeypatch, stage
):
    instance, executor, settings, group = verifier
    group.rank_in_group = 1
    peer = remote.RemoteSpeculator(settings, torch.device("cpu"))
    group.rank_in_group = 0
    policy = settings.speculative_config.remote_draft.failure_policy
    aux = None
    if stage == "decode_select":
        first = inputs(["a"], [[1]], [[0]], [1])
        drafts = propose(instance, first, [1], [0], [2])[0].tolist()
        batch = inputs(["a"], [[2, *drafts]], [[1, 2, 3, 4]], [0])
        sampled, rejected, recovery = [1], [3], [3]
        expected_computed = [2]
    elif stage == "prefill_select":
        batch = inputs(["a", "b"], [[1], [3]], [[0], [0]], [4, 4])
        sampled, rejected, recovery = [0, 0], [0, 0], [0, 0]
        expected_computed = [1, 0]
    else:
        batch = inputs(["a"], [[1]], [[0]], [1])
        sampled, rejected, recovery = [1], [0], [2]
        expected_computed = [0]
        aux = [torch.zeros(1, 4), torch.zeros(1, 4)]
    plan_count = len(executor.plans)
    original_select = instance._select
    select_calls = 0

    def fail_cat(*args, **kwargs):
        raise torch.OutOfMemoryError("remote feature allocation failed")

    def select_with_allocation_failure(features, token_rows, sequence_rows):
        nonlocal select_calls
        select_calls += 1
        fail_on_call = 2 if stage == "prefill_select" else 1
        if select_calls != fail_on_call:
            return original_select(features, token_rows, sequence_rows)
        getitem = torch.Tensor.__getitem__

        def fail_feature_index(tensor, index):
            if any(tensor is feature for feature in features.values()):
                raise torch.OutOfMemoryError("remote feature allocation failed")
            return getitem(tensor, index)

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "__getitem__", fail_feature_index)
            return original_select(features, token_rows, sequence_rows)

    with monkeypatch.context() as patch:
        if stage == "aux_cat":
            # Failure happens in the real _features before any tensor is sent;
            # the real session/server keep their negotiated contract unchanged.
            patch.setattr(
                instance,
                "_contract",
                SimpleNamespace(
                    schema=TargetFeatureSchema(
                        1,
                        (
                            *SCHEMA.slots,
                            FeatureSlot("aux_hidden_states", "float32", (8,)),
                        ),
                    )
                ),
            )
            patch.setattr(torch, "cat", fail_cat)
        else:
            patch.setattr(instance, "_select", select_with_allocation_failure)
        if policy == "error":
            with pytest.raises(RuntimeError, match="remote feature allocation failed"):
                propose(
                    instance, batch, sampled, rejected, recovery, aux_hidden_states=aux
                )
        else:
            assert propose(
                instance, batch, sampled, rejected, recovery, aux_hidden_states=aux
            ).tolist() == [[-1] * K for _ in batch.req_ids]
    assert len(executor.plans) == plan_count
    assert [
        instance._requests[req_id].num_computed for req_id in batch.req_ids
    ] == expected_computed
    assert all(
        instance._session.sequence_state(instance._requests[req_id].key)
        is remote.SequenceState.TARGET_ONLY
        for req_id in batch.req_ids
    )
    group.rank_in_group = 1
    try:
        if policy == "error":
            with pytest.raises(RuntimeError, match="remote feature allocation failed"):
                propose(peer, batch, sampled, rejected, recovery)
        else:
            assert propose(peer, batch, sampled, rejected, recovery).tolist() == [
                [-1] * K for _ in batch.req_ids
            ]
    finally:
        group.rank_in_group = 0

    # Do not resume an own-KV sequence after skipping its verified context.
    # A separate request can still use the healthy connection and clean up.
    with monkeypatch.context() as patch:

        def reject_features(*args, **kwargs):
            pytest.fail("an allocation-failed request resumed remote decoding")

        patch.setattr(instance, "_features", reject_features)
        next_batch = inputs(
            batch.req_ids,
            [[7]] * batch.num_reqs,
            [[8]] * batch.num_reqs,
            [0] * batch.num_reqs,
        )
        assert propose(
            instance,
            next_batch,
            [1] * batch.num_reqs,
            [0] * batch.num_reqs,
            [8] * batch.num_reqs,
        ).tolist() == [[-1] * K for _ in batch.req_ids]
    fresh = inputs(["fresh"], [[11]], [[0]], [1])
    assert propose(instance, fresh, [1], [0], [12]).tolist() == [digest_row([11], 12)]
    instance.close_requests({*batch.req_ids, "fresh"})
    assert instance._requests == {}


@pytest.mark.parametrize("verifier", ["target_only"], indirect=True)
def test_feature_failure_retires_remote_kv_without_waiting_for_an_ack(
    verifier, monkeypatch
):
    instance, executor, _, _ = verifier
    session = instance._session
    free_blocks = executor.pool.num_free_blocks
    propose(instance, inputs(["a"], [[1, 2]], [[0, 1]], [2]), [1], [0], [3])
    key = instance._requests["a"].key
    assert executor.pool.num_free_blocks < free_blocks
    released = threading.Event()
    release = executor.pool.release

    def release_blocks(req_index):
        release(req_index)
        released.set()

    def fail_features(*args, **kwargs):
        raise torch.OutOfMemoryError("remote feature allocation failed")

    with monkeypatch.context() as patch:
        patch.setattr(executor.pool, "release", release_blocks)
        patch.setattr(instance, "_features", fail_features)
        patch.setattr(
            session,
            "_await",
            lambda *args, **kwargs: pytest.fail("cleanup waited for ACK"),
        )
        batch = inputs(["a"], [[3]], [[2]], [0])
        assert propose(instance, batch, [1], [0], [4]).tolist() == [[-1] * K]
        assert released.wait(2), "terminal fallback retained remote draft KV"
    assert executor.pool.num_free_blocks == free_blocks
    assert session.sequence_state(key) is remote.SequenceState.TARGET_ONLY
    assert session.connected
    assert instance._requests["a"].key == key

    # The late close ACK must not interfere with a healthy new request.
    assert propose(
        instance, inputs(["b"], [[11]], [[0]], [1]), [1], [0], [12]
    ).tolist() == [digest_row([11], 12)]


@pytest.mark.parametrize("verifier", ["target_only"], indirect=True)
def test_remote_feature_cuda_context_error_is_shared_and_not_fallback(
    verifier, monkeypatch
):
    instance, _, _, group = verifier

    def fail_features(*args, **kwargs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(instance, "_features", fail_features)
    batch = inputs(["a"], [[1]], [[0]], [1])
    with pytest.raises(RuntimeError, match="illegal memory access"):
        propose(instance, batch, [1], [0], [2])
    assert group.payload[1] is None


@pytest.mark.parametrize("option", ["prefix", "async", "tp4", "pp2", "embeddings"])
def test_unsupported_placement_fails_before_connecting(option, monkeypatch):
    settings = config("tcp://127.0.0.1:1")
    if option == "prefix":
        settings.cache_config.enable_prefix_caching = True
    elif option == "async":
        settings.scheduler_config.async_scheduling = True
    elif option == "tp4":
        settings.parallel_config.tensor_parallel_size = 4
    elif option == "pp2":
        settings.parallel_config.pipeline_parallel_size = 2
    else:
        settings.model_config.enable_prompt_embeds = True
    with pytest.raises(ValueError, match="remote drafter"):
        remote.RemoteSpeculator(settings, torch.device("cpu"))
