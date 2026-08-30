# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verifier session against the fake draft server: negotiation, proposal
rounds, failure latching, stale-response discard, and shutdown recovery."""

import dataclasses

import pytest
import torch

from vllm.v1.spec_decode.remote.capabilities import (
    SpeculatorPlacementCapabilities,
    TargetFeatureKind,
)
from vllm.v1.spec_decode.remote.protocol import SequenceKey, SpeculatorStatusCode
from vllm.v1.spec_decode.remote.server import FakeDraftAdapter, RemoteDraftServer
from vllm.v1.spec_decode.remote.session import (
    RemoteDraftError,
    RemoteDraftSession,
    RemoteProposalHandle,
    VerifierIdentity,
    frame_to_tensor,
    tensor_to_frame,
)
from vllm.v1.spec_decode.remote.state import (
    FeatureBatch,
    HandleState,
    InvalidStateTransition,
    RemoteProposalBatch,
    SequenceState,
)

VOCAB = 50
NUM_SPEC_TOKENS = 3
OK = SpeculatorStatusCode.OK
IDENTITY = VerifierIdentity(
    verifier_instance_id="verifier-a",
    target_fingerprint="tf",
    tokenizer_fingerprint="kf",
    method="eagle3",
    num_speculative_tokens=NUM_SPEC_TOKENS,
    parallel_drafting=False,
    provided_features=(TargetFeatureKind.TOKEN_IDS.value,),
)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds: float):
        self.now += seconds


def tokens(values, dtype=torch.int32):
    return torch.tensor(values, dtype=dtype)


def features(values):
    return FeatureBatch(schema_id=1, slots=(tokens(values),))


def batch(batch_id, keys, recovery, accepted=None):
    accepted = accepted or [0] * len(keys)
    return RemoteProposalBatch(
        batch_id=batch_id,
        keys=tuple(keys),
        accepted_counts=tokens(accepted),
        recovery_tokens=tokens(recovery, torch.int64),
        features=features(recovery),
    )


def expected_row(last_token: int):
    return [(last_token + j + 1) % VOCAB for j in range(NUM_SPEC_TOKENS)]


@pytest.fixture
def adapter():
    return FakeDraftAdapter(vocab_size=VOCAB)


@pytest.fixture
def server(adapter):
    server = RemoteDraftServer(adapter, server_id="srv")
    server.start("tcp://127.0.0.1:0")
    yield server
    adapter.release()
    server.stop()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def make_session(server, clock):
    sessions = []

    def factory(identity=IDENTITY, *, connect=True, **kwargs):
        kwargs.setdefault("request_timeout_ms", 100)
        session = RemoteDraftSession(server.endpoint, identity, clock=clock, **kwargs)
        sessions.append(session)
        if connect:
            session.connect()
        return session

    yield factory
    for session in sessions:
        session.close()


@pytest.fixture
def session(make_session):
    return make_session()


def ready(session, sequence_id, prompt=(1, 2, 3)):
    key = session.open_sequence(sequence_id)
    session.prefill(key, features(list(prompt)), is_final=True)
    assert session.sequence_state(key) is SequenceState.READY
    return key


def round_trip(session, batch_):
    return session.collect(session.dispatch(batch_))


# ----------------------------------------------------------------------
# Handshake
# ----------------------------------------------------------------------


def test_handshake_negotiates_inline_transport_and_schema(session, adapter):
    assert session.connected
    assert session.selected_transport == "inline"
    assert session.session_epoch >= 1
    assert session.feature_schema == adapter.feature_schema()
    assert session.capabilities == adapter.capabilities()
    assert session.limits.max_batch_size >= 1


def test_handshake_rejects_incompatible_capabilities(clock):
    adapter = FakeDraftAdapter(
        vocab_size=VOCAB,
        capabilities=SpeculatorPlacementCapabilities(
            state_dependency="target_kv",
            required_features=(TargetFeatureKind.TOKEN_IDS.value,),
            standalone_weights="complete",
        ),
    )
    server = RemoteDraftServer(adapter)
    server.start("tcp://127.0.0.1:0")
    try:
        session = RemoteDraftSession(server.endpoint, IDENTITY, clock=clock)
        with pytest.raises(RemoteDraftError, match="target-owned state"):
            session.connect()
        assert not session.connected
    finally:
        server.stop()


def test_handshake_rejects_missing_provided_feature(make_session):
    identity = dataclasses.replace(IDENTITY, provided_features=())
    with pytest.raises(RemoteDraftError, match="cannot provide"):
        make_session(identity)


def test_handshake_surfaces_server_rejection(make_session):
    identity = dataclasses.replace(IDENTITY, method="ngram")
    with pytest.raises(RemoteDraftError, match="method 'ngram'"):
        make_session(identity)


def test_handshake_timeout_is_reported():
    # Nothing answers HELLO on a bare listener, so the handshake must give
    # up at the startup deadline instead of hanging (real clock here: the
    # deadline loop is what is under test).
    from vllm.v1.spec_decode.remote.transport import listen

    listener = listen("tcp://127.0.0.1:0")
    try:
        session = RemoteDraftSession(
            listener.endpoint, IDENTITY, startup_timeout_ms=200
        )
        with pytest.raises(RemoteDraftError, match="timed out"):
            session.connect()
    finally:
        listener.close()


def test_explicit_transport_is_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        RemoteDraftSession("tcp://127.0.0.1:1", IDENTITY, transport="zmq")


# ----------------------------------------------------------------------
# Proposal rounds
# ----------------------------------------------------------------------


def test_round_trip_matches_fake_formula(session, adapter):
    a = session.open_sequence(1)
    session.prefill(a, features([1, 2]), is_final=False)
    session.prefill(a, features([3]), is_final=True)
    b = ready(session, 2, prompt=(9,))
    assert adapter.prefix(a) == [1, 2, 3]

    result = round_trip(session, batch(0, [a, b], recovery=[5, 47], accepted=[0, 1]))
    assert result.row_statuses == (OK, OK)
    assert result.output.token_ids.tolist() == [expected_row(5), expected_row(47)]
    assert result.output.valid_lengths.tolist() == [NUM_SPEC_TOKENS] * 2
    assert adapter.prefix(a) == [1, 2, 3, 5]
    assert session.sequence_state(a) is SequenceState.READY

    result = round_trip(session, batch(1, [b], recovery=[48]))
    assert result.output.token_ids.tolist() == [expected_row(48)]
    assert adapter.proposed_batches == [0, 1]


def test_handle_records_dispatch_metadata(session):
    a = ready(session, 1)
    handle = session.dispatch(batch(0, [a], recovery=[1]))
    assert isinstance(handle, RemoteProposalHandle)
    assert handle.state is HandleState.DISPATCHED
    assert handle.active_rows == (0,)
    assert handle.session_epoch == session.session_epoch
    assert session.sequence_state(a) is SequenceState.IN_FLIGHT
    session.collect(handle)
    assert handle.state is HandleState.COLLECTED


def test_unready_rows_are_skipped_locally(session, adapter):
    a = ready(session, 1)
    opened = session.open_sequence(2)
    unknown = SequenceKey("verifier-a", 99, 0)
    stale = SequenceKey(a.verifier_instance_id, a.sequence_id, a.generation + 1)

    handle = session.dispatch(
        batch(0, [a, opened, unknown, stale], recovery=[4, 5, 6, 7])
    )
    assert handle.active_rows == (0,)
    result = session.collect(handle)
    assert result.row_statuses == (
        OK,
        SpeculatorStatusCode.SEQUENCE_DESYNC,
        SpeculatorStatusCode.SEQUENCE_DESYNC,
        SpeculatorStatusCode.STALE_GENERATION,
    )
    assert result.output.valid_lengths.tolist() == [NUM_SPEC_TOKENS, 0, 0, 0]
    assert result.output.token_ids[0].tolist() == expected_row(4)
    assert adapter.proposed_batches == [0]


def test_all_rows_skipped_completes_without_wire_traffic(session, adapter):
    handle = session.dispatch(batch(0, [SequenceKey("verifier-a", 5, 0)], recovery=[1]))
    assert handle.state is HandleState.COMPLETED
    result = session.collect(handle)
    assert result.output.valid_lengths.tolist() == [0]
    assert adapter.proposed_batches == []


def test_collect_is_one_shot(session):
    a = ready(session, 1)
    handle = session.dispatch(batch(0, [a], recovery=[1]))
    session.collect(handle)
    with pytest.raises(InvalidStateTransition, match="collected"):
        session.collect(handle)


def test_batch_ids_must_increase(session):
    a = ready(session, 1)
    session.collect(session.dispatch(batch(3, [a], recovery=[1])))
    with pytest.raises(ValueError, match="not newer"):
        session.dispatch(batch(3, [a], recovery=[1]))


def test_dispatch_validates_schema_and_shapes(session):
    a = ready(session, 1)
    wrong_schema = RemoteProposalBatch(
        batch_id=0,
        keys=(a,),
        accepted_counts=tokens([0]),
        recovery_tokens=tokens([1]),
        features=FeatureBatch(schema_id=2, slots=(tokens([1]),)),
    )
    with pytest.raises(ValueError, match="schema"):
        session.dispatch(wrong_schema)
    with pytest.raises(ValueError, match="accepted_counts"):
        session.dispatch(
            RemoteProposalBatch(1, (a,), tokens([0, 0]), tokens([1]), features([1]))
        )
    with pytest.raises(ValueError, match="required feature slot"):
        session.dispatch(
            RemoteProposalBatch(
                2, (a,), tokens([0]), tokens([1]), FeatureBatch(1, (None,))
            )
        )


# ----------------------------------------------------------------------
# Failure semantics
# ----------------------------------------------------------------------


def test_timeout_latches_target_only_and_purges_stale_result(session, adapter, clock):
    a = ready(session, 1)
    b = ready(session, 2)
    adapter.hold_batches.add(0)
    handle = session.dispatch(batch(0, [a, b], recovery=[1, 2]))
    clock.advance(10.0)
    result = session.collect(handle)
    assert handle.state is HandleState.COLLECTED
    assert result.row_statuses == (
        SpeculatorStatusCode.TIMEOUT,
        SpeculatorStatusCode.TIMEOUT,
    )
    assert result.output.valid_lengths.tolist() == [0, 0]
    assert session.sequence_state(a) is SequenceState.TARGET_ONLY

    # The late result must never surface: the next round only sees the
    # fresh sequence and the stale frames are dropped.
    adapter.release()
    c = ready(session, 3)
    result = round_trip(session, batch(1, [a, c], recovery=[8, 9]))
    assert result.row_statuses == (SpeculatorStatusCode.TIMEOUT, OK)
    assert result.output.token_ids[1].tolist() == expected_row(9)
    assert result.output.valid_lengths.tolist() == [0, NUM_SPEC_TOKENS]
    assert session.pending_frames == 0
    assert adapter.proposed_batches == [0, 1]

    # Latched rows ignore prefill and stay latched until closed.
    session.prefill(a, features([1]), is_final=True)
    assert session.sequence_state(a) is SequenceState.TARGET_ONLY
    session.close_sequences((a,))
    assert session.sequence_state(a) is None


def test_failure_policy_error_raises_after_latching(make_session, adapter, clock):
    session = make_session(failure_policy="error")
    a = ready(session, 1)
    adapter.hold_batches.add(0)
    handle = session.dispatch(batch(0, [a], recovery=[1]))
    clock.advance(10.0)
    with pytest.raises(RemoteDraftError, match="TIMEOUT"):
        session.collect(handle)
    assert session.sequence_state(a) is SequenceState.TARGET_ONLY
    adapter.release()


def test_stale_generation_desyncs_then_reopen_recovers(session, server, adapter):
    a = ready(session, 1)
    newer = SequenceKey(a.verifier_instance_id, a.sequence_id, a.generation + 1)
    with server.lock:
        server.registry.open(newer)
        adapter.open_sequence(newer)

    result = round_trip(session, batch(0, [a], recovery=[1]))
    assert result.row_statuses == (SpeculatorStatusCode.STALE_GENERATION,)
    assert session.sequence_state(a) is SequenceState.DESYNCED

    reopened = session.open_sequence(1)
    assert reopened.generation == a.generation + 1
    session.prefill(reopened, features([4, 5]), is_final=True)
    result = round_trip(session, batch(1, [reopened], recovery=[6]))
    assert result.row_statuses == (OK,)
    assert adapter.prefix(reopened) == [4, 5, 6]


def test_server_shutdown_degrades_to_target_only(session, server, adapter):
    a = ready(session, 1)
    adapter.release()
    server.stop()

    handle = session.dispatch(batch(0, [a], recovery=[1]))
    result = session.collect(handle)
    assert result.output.valid_lengths.tolist() == [0]
    assert result.row_statuses[0] is not OK
    assert not session.connected
    assert session.sequence_state(a) is SequenceState.INVALID

    later = session.open_sequence(2)
    assert session.sequence_state(later) is SequenceState.INVALID
    session.prefill(later, features([1]), is_final=True)
    result = round_trip(session, batch(1, [a, later], recovery=[1, 2]))
    assert result.output.valid_lengths.tolist() == [0, 0]
    session.close_sequences((a, later))
    assert session.sequence_state(later) is None


def test_server_shutdown_raises_under_error_policy(make_session, server, adapter):
    session = make_session(failure_policy="error")
    a = ready(session, 1)
    adapter.release()
    server.stop()
    handle = session.dispatch(batch(0, [a], recovery=[1]))
    with pytest.raises(RemoteDraftError):
        session.collect(handle)


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


def test_open_twice_is_an_error(session):
    session.open_sequence(1)
    with pytest.raises(RemoteDraftError, match="already open"):
        session.open_sequence(1)


def test_reopened_sequence_id_gets_a_new_generation(session):
    first = ready(session, 1)
    session.close_sequences((first,))
    second = session.open_sequence(1)
    assert second.generation == first.generation + 1


def test_close_is_idempotent(session, server, adapter):
    a = ready(session, 1)
    b = ready(session, 2)
    stale = SequenceKey(a.verifier_instance_id, a.sequence_id, 99)
    unknown = SequenceKey("verifier-a", 42, 0)
    session.close_sequences((a, stale, unknown))
    session.close_sequences((a, b))
    session.close_sequences((b,))
    assert session.sequence_state(a) is None
    assert session.sequence_state(b) is None
    assert len(server.registry) == 0
    assert adapter.prefix(a) is None


def test_prefill_rejects_stale_key_and_ready_sequence(session):
    a = ready(session, 1)
    with pytest.raises(RemoteDraftError, match="cannot accept prefill"):
        session.prefill(a, features([1]), is_final=True)
    stale = SequenceKey(a.verifier_instance_id, a.sequence_id, 7)
    with pytest.raises(ValueError, match="not current"):
        session.prefill(stale, features([1]), is_final=True)


def test_ping_round_trip(session):
    ready(session, 1)
    pong = session.ping()
    assert pong.active_sequences == 1


def test_from_config_reads_placement_fields(server, clock):
    class Config:
        endpoint = server.endpoint
        transport = "auto"
        failure_policy = "error"
        request_timeout_ms = 250
        startup_timeout_ms = 1000

    session = RemoteDraftSession.from_config(Config(), IDENTITY, clock=clock)
    session.connect()
    try:
        assert session.connected
        assert session._failure_policy == "error"
        assert session._request_timeout_s == 0.25
    finally:
        session.close()


# ----------------------------------------------------------------------
# Frame helpers
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "tensor",
    [
        torch.arange(6, dtype=torch.int32).reshape(2, 3),
        torch.tensor([1.5, -2.0], dtype=torch.bfloat16),
        torch.empty(0, 4, dtype=torch.int64),
        torch.tensor(7, dtype=torch.int64),
    ],
    ids=["int32-2d", "bfloat16", "empty", "scalar"],
)
def test_tensor_frame_roundtrip(tensor):
    frame = tensor_to_frame(11, tensor)
    assert frame.slot == 11
    restored = frame_to_tensor(frame)
    assert restored.dtype == tensor.dtype
    assert restored.shape == tensor.shape
    assert torch.equal(restored, tensor)
