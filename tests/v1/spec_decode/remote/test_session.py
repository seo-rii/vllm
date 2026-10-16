# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verifier session against the fake draft server: negotiation, proposal
rounds, failure latching, stale-response discard, and shutdown recovery."""

import contextlib
import dataclasses
import threading
import time
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.spec_decode.remote import session as session_module
from vllm.v1.spec_decode.remote.capabilities import (
    SpeculatorPlacementCapabilities,
    TargetFeatureKind,
)
from vllm.v1.spec_decode.remote.protocol import (
    ErrorReply,
    FeatureSlot,
    HelloAck,
    RemoteServerLimits,
    SequenceAck,
    SequenceKey,
    SpeculatorStatusCode,
    TargetFeatureSchema,
    decode_envelope,
    decode_payload,
    encode_message,
)
from vllm.v1.spec_decode.remote.server import (
    DEFAULT_LIMITS,
    FakeDraftAdapter,
    RemoteDraftServer,
)
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
from vllm.v1.spec_decode.remote.transport import TransportError, listen

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
    max_num_seqs=8,
    max_num_batched_tokens=64,
    max_model_len=64,
    vocab_size=VOCAB,
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


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("max_num_seqs", 300, "max_batch_size=256"),
        ("max_num_batched_tokens", 20000, "max_feature_tokens=16384"),
        ("max_model_len", 40000, "max_model_len=32768"),
    ],
)
def test_handshake_rejects_limits_below_verifier_config(
    make_session, field, value, reason
):
    # Both sides check: the server refuses the HELLO, and a server that
    # does not check is still rejected by the verifier.
    identity = dataclasses.replace(IDENTITY, **{field: value})
    with pytest.raises(RemoteDraftError, match=reason):
        make_session(identity)
    session = make_session(connect=False, identity=identity)
    ack = HelloAck(
        server_id="srv",
        session_id="s",
        session_epoch=1,
        selected_transport="inline",
        capabilities=FakeDraftAdapter().capabilities(),
        feature_schema=FakeDraftAdapter().feature_schema(),
        limits=DEFAULT_LIMITS,
    )
    assert any(
        reason in r for r in session._handshake_rejections(session._hello(), ack)
    )


def test_explicit_inline_transport_connects(make_session):
    session = make_session(transport="inline")
    assert session.selected_transport == "inline"


def test_unimplemented_transport_is_rejected():
    with pytest.raises(NotImplementedError):
        RemoteDraftSession("tcp://127.0.0.1:1", IDENTITY, transport="zmq")


def test_cuda_ipc_requires_unix_socket_before_connecting():
    with pytest.raises(ValueError, match="UNIX socket"):
        RemoteDraftSession("tcp://127.0.0.1:1", IDENTITY, transport="cuda_ipc")


def test_ipc_prefill_releases_leases_only_after_all_acknowledgements(
    session, monkeypatch
):
    keys = (session.open_sequence(1), session.open_sequence(2))
    exporter = Mock()
    leases = [Mock(), Mock()]
    session._ipc_exporter = exporter
    session._ipc_pending.extend(leases)
    original = session._await
    seen = 0

    def await_ack(*args, **kwargs):
        nonlocal seen
        exporter.release.assert_not_called()
        reply = original(*args, **kwargs)
        seen += 1
        return reply

    monkeypatch.setattr(session, "_await", await_ack)
    session.prefill_many([(key, features([1]), True) for key in keys])
    assert seen == 2
    assert [call.args[0] for call in exporter.release.call_args_list] == leases
    assert not session._ipc_pending


def test_ipc_prefill_timeout_quarantines_instead_of_reusing_lease(session, monkeypatch):
    from vllm.v1.spec_decode.remote.transport import TransportTimeout

    key = session.open_sequence(1)
    exporter = Mock()
    session._ipc_exporter = exporter
    session._ipc_pending.append(Mock())
    monkeypatch.setattr(session, "_await", Mock(side_effect=TransportTimeout("late")))
    session.prefill(key, features([1]), is_final=True)
    exporter.release.assert_not_called()
    exporter.close.assert_called_once_with()
    assert not session.connected
    assert not session.maybe_reconnect()


def test_ipc_decode_releases_only_after_proposal_response(session, monkeypatch):
    key = ready(session, 1)
    exporter = Mock()
    lease = Mock()
    session._ipc_exporter = exporter
    session._ipc_pending.append(lease)
    handle = session.dispatch(batch(0, [key], recovery=[4]))
    exporter.release.assert_not_called()
    result = session.collect(handle)
    exporter.release.assert_called_once_with(lease)
    assert result.output.valid_lengths.tolist() == [NUM_SPEC_TOKENS]


def test_ipc_decode_timeout_invalidates_and_quarantines(session, monkeypatch):
    from vllm.v1.spec_decode.remote.transport import TransportTimeout

    key = ready(session, 1)
    exporter = Mock()
    session._ipc_exporter = exporter
    session._ipc_pending.append(Mock())
    handle = session.dispatch(batch(0, [key], recovery=[4]))
    monkeypatch.setattr(session, "_await", Mock(side_effect=TransportTimeout("late")))
    result = session.collect(handle)
    exporter.release.assert_not_called()
    exporter.close.assert_called_once_with()
    assert result.output.valid_lengths.tolist() == [0]
    assert not session.connected
    assert not session.maybe_reconnect()


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


class PaddedAdapter(FakeDraftAdapter):
    """Proposes one valid token per row and pads the rest with junk."""

    def __init__(self, padding: int, first: int):
        super().__init__(vocab_size=VOCAB)
        self.padding = padding
        self.first = first

    def advance_and_propose(self, batch_):
        result = super().advance_and_propose(batch_)
        rows = len(batch_.keys)
        row = [self.first] + [self.padding] * (NUM_SPEC_TOKENS - 1)
        return dataclasses.replace(
            result,
            token_ids=tensor_to_frame(0, torch.tensor([row] * rows)),
            valid_lengths=tensor_to_frame(0, torch.ones(rows, dtype=torch.int32)),
        )


@pytest.mark.parametrize(
    ("padding", "first", "status", "expected"),
    [
        (-1, 7, OK, [7, 0, 0]),
        (7, VOCAB, SpeculatorStatusCode.INTERNAL_ERROR, [0, 0, 0]),
        (7, -3, SpeculatorStatusCode.INTERNAL_ERROR, [0, 0, 0]),
    ],
    ids=["junk-padding-is-zeroed", "too-large", "negative"],
)
def test_draft_tokens_are_checked_against_the_target_vocab(
    clock, padding, first, status, expected
):
    adapter = PaddedAdapter(padding, first)
    server = RemoteDraftServer(adapter)
    server.start("tcp://127.0.0.1:0")
    session = RemoteDraftSession(server.endpoint, IDENTITY, clock=clock)
    try:
        session.connect()
        a = ready(session, 1)
        result = round_trip(session, batch(0, [a], recovery=[4]))
        assert result.row_statuses == (status,)
        assert result.output.token_ids.tolist() == [expected]
        if status is OK:
            assert result.output.valid_lengths.tolist() == [1]
        else:
            assert result.output.valid_lengths.tolist() == [0]
            assert session.sequence_state(a) is SequenceState.TARGET_ONLY
    finally:
        session.close()
        server.stop()


def test_handle_records_dispatch_metadata(session):
    a = ready(session, 1)
    handle = session.dispatch(batch(0, [a], recovery=[1]))
    assert isinstance(handle, RemoteProposalHandle)
    assert handle.state is HandleState.DISPATCHED
    assert handle.active_rows == (0,)
    assert handle.session_epoch == session.session_epoch
    assert handle.session_id == session.session_id
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


def test_duplicate_keys_are_rejected_without_consuming_the_batch(session, adapter):
    a = ready(session, 1)
    with pytest.raises(ValueError, match="duplicate sequence keys"):
        session.dispatch(batch(0, [a, a], recovery=[4, 4]))

    assert session.sequence_state(a) is SequenceState.READY
    assert adapter.proposed_batches == []
    assert adapter.prefix(a) == [1, 2, 3]
    result = round_trip(session, batch(0, [a], recovery=[4]))
    assert result.row_statuses == (OK,)
    assert adapter.prefix(a) == [1, 2, 3, 4]


def test_only_one_proposal_may_be_in_flight(session):
    a = ready(session, 1)
    b = ready(session, 2)
    handle = session.dispatch(batch(0, [a], recovery=[1]))
    with pytest.raises(RemoteDraftError, match="in flight"):
        session.dispatch(batch(1, [b], recovery=[2]))
    session.collect(handle)
    result = round_trip(session, batch(1, [b], recovery=[2]))
    assert result.row_statuses == (OK,)


def test_other_rpcs_cannot_consume_an_inflight_proposal(session):
    a = ready(session, 1)
    b = session.open_sequence(2)
    handle = session.dispatch(batch(0, [a], recovery=[1]))

    with pytest.raises(RemoteDraftError, match="still in flight"):
        session.ping()
    with pytest.raises(RemoteDraftError, match="still in flight"):
        session.open_sequence(3)
    with pytest.raises(RemoteDraftError, match="still in flight"):
        session.prefill(b, features([2]), is_final=True)
    with pytest.raises(RemoteDraftError, match="still in flight"):
        session.close_sequences((b,))

    assert session.sequence_state(b) is SequenceState.PREFILLING
    assert session.sequence_state(SequenceKey("verifier-a", 3, 0)) is None
    assert session.collect(handle).row_statuses == (OK,)
    assert session.pending_frames == 0


def ragged_batch(batch_id, keys, query_lens, token_ids, *extra_slots):
    return RemoteProposalBatch(
        batch_id=batch_id,
        keys=tuple(keys),
        accepted_counts=tokens([0] * len(keys)),
        recovery_tokens=tokens(list(range(len(keys))), torch.int64),
        features=FeatureBatch(1, (tokens(token_ids), *extra_slots)),
        query_lens=tokens(query_lens),
    )


def test_ragged_token_features_follow_their_rows(session, adapter):
    # Token-level features carry each row's target tokens back to back
    # (MRV2's [num_tokens, ...] layout); skipping an unready row must drop
    # exactly its segment.
    a = ready(session, 1)
    skipped = session.open_sequence(2)
    b = ready(session, 3)
    result = round_trip(
        session,
        ragged_batch(0, [a, skipped, b], [2, 3, 1], [10, 11, 20, 21, 22, 30]),
    )
    assert result.row_statuses[0] is OK
    assert result.row_statuses[2] is OK
    assert adapter.round_tokens == {a: [10, 11], b: [30]}


@pytest.mark.parametrize(
    ("query_lens", "token_ids", "match"),
    [
        ([2], [1, 2], "query_lens must be"),
        ([-1, 3], [1, 2], "non-negative"),
        ([1, 1], [1, 2, 3], "expected 2 rows"),
    ],
)
def test_query_lens_must_describe_token_features(
    session, adapter, query_lens, token_ids, match
):
    a = ready(session, 1)
    b = ready(session, 2)
    with pytest.raises(ValueError, match=match):
        session.dispatch(ragged_batch(0, [a, b], query_lens, token_ids))
    assert adapter.proposed_batches == []
    # A rejected batch does not consume its id.
    assert round_trip(session, batch(0, [a], recovery=[1])).row_statuses == (OK,)


def test_per_sequence_slots_hold_one_row_per_key(clock):
    schema = TargetFeatureSchema(
        schema_id=1,
        slots=(
            FeatureSlot(kind="token_ids", dtype="int32", trailing_shape=()),
            FeatureSlot(
                kind="temperature",
                dtype="float32",
                trailing_shape=(),
                per_sequence=True,
            ),
        ),
    )
    adapter = FakeDraftAdapter(vocab_size=VOCAB, feature_schema=schema)
    server = RemoteDraftServer(adapter)
    server.start("tcp://127.0.0.1:0")
    identity = dataclasses.replace(
        IDENTITY, provided_features=("token_ids", "temperature")
    )
    session = RemoteDraftSession(server.endpoint, identity, clock=clock)
    temps = torch.tensor([0.5, 0.0, 1.0])
    try:
        session.connect()
        a = session.open_sequence(1)
        with pytest.raises(ValueError, match="temperature"):
            session.prefill(
                a, FeatureBatch(1, (tokens([1, 2]), temps[:2])), is_final=True
            )
        session.prefill(a, FeatureBatch(1, (tokens([1, 2]), temps[:1])), is_final=True)
        skipped = session.open_sequence(2)
        b = session.open_sequence(3)
        session.prefill(b, FeatureBatch(1, (tokens([4]), temps[:1])), is_final=True)

        with pytest.raises(ValueError, match="temperature"):
            session.dispatch(
                ragged_batch(0, [a, skipped, b], [1, 1, 2], [7, 8, 9, 9], temps[:2])
            )
        result = round_trip(
            session,
            ragged_batch(0, [a, skipped, b], [1, 1, 2], [7, 8, 9, 9], temps),
        )
        assert result.row_statuses[0] is OK
        assert result.row_statuses[2] is OK
        assert adapter.round_tokens == {a: [7], b: [9, 9]}
    finally:
        session.close()
        adapter.release()
        server.stop()


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
    with pytest.raises(ValueError, match="recovery_tokens"):
        session.dispatch(
            RemoteProposalBatch(
                2,
                (a,),
                tokens([0]),
                tokens([[1]]),
                features([1]),
            )
        )
    with pytest.raises(ValueError, match="required feature slot"):
        session.dispatch(
            RemoteProposalBatch(
                3, (a,), tokens([0]), tokens([1]), FeatureBatch(1, (None,))
            )
        )


# ----------------------------------------------------------------------
# Failure semantics
# ----------------------------------------------------------------------


def test_queue_full_latches_target_only_without_skipping_recovery(
    session, adapter, monkeypatch
):
    a = ready(session, 1)
    propose = adapter.advance_and_propose

    def busy_once(batch_):
        if batch_.batch_id == 0:
            result = propose(dataclasses.replace(batch_, active_rows=()))
            return dataclasses.replace(
                result,
                valid_lengths=tensor_to_frame(0, tokens([0])),
                row_statuses=(SpeculatorStatusCode.QUEUE_FULL,),
            )
        return propose(batch_)

    monkeypatch.setattr(adapter, "advance_and_propose", busy_once)
    result = round_trip(session, batch(0, [a], recovery=[4]))
    assert result.row_statuses == (SpeculatorStatusCode.QUEUE_FULL,)
    assert result.output.valid_lengths.tolist() == [0]
    assert session.sequence_state(a) is SequenceState.TARGET_ONLY

    result = round_trip(session, batch(1, [a], recovery=[5]))
    assert result.row_statuses == (SpeculatorStatusCode.QUEUE_FULL,)
    assert result.output.valid_lengths.tolist() == [0]
    assert adapter.proposed_batches == [0]
    assert adapter.prefix(a) == [1, 2, 3]


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


def test_dispatch_expired_during_encoding_sends_no_proposal(
    session, adapter, clock, monkeypatch
):
    # Nothing was written, so the round times out without tearing down the
    # connection the other requests still use.
    a = ready(session, 1)
    b = ready(session, 2)

    def slow_encode(slot, tensor):
        frame = tensor_to_frame(slot, tensor)
        clock.advance(1.0)
        return frame

    monkeypatch.setattr(session_module, "tensor_to_frame", slow_encode)
    result = round_trip(session, batch(0, [a], recovery=[4]))
    assert result.row_statuses == (SpeculatorStatusCode.TIMEOUT,)
    assert result.output.valid_lengths.tolist() == [0]
    assert session.connected
    assert adapter.proposed_batches == []

    monkeypatch.undo()
    assert round_trip(session, batch(1, [b], recovery=[5])).row_statuses == (OK,)


def test_large_writes_get_a_size_proportional_budget(session, monkeypatch):
    a = session.open_sequence(1)
    monkeypatch.setattr(session_module, "_MIN_WRITE_BYTES_PER_S", 1000)
    budgets = []
    send_data = session._conn.send_data

    def record(frame, *, deadline=None):
        budgets.append(deadline - time.monotonic())
        send_data(frame, deadline=deadline)

    monkeypatch.setattr(session._conn, "send_data", record)
    # 1000 int32 tokens at 1000 B/s is a 4 s write budget, far beyond the
    # 100 ms request timeout that would otherwise cut the copy short.
    session.prefill(a, features(list(range(1000))), is_final=True)
    assert budgets and budgets[0] > 3.0
    assert session.sequence_state(a) is SequenceState.READY


def test_collect_timeout_does_not_write_after_deadline(
    session, adapter, clock, monkeypatch
):
    a = ready(session, 1)
    adapter.hold_batches.add(0)
    handle = session.dispatch(batch(0, [a], recovery=[4]))
    clock.advance(1.0)

    def reject_late_write(*args, **kwargs):
        pytest.fail("fallback must not wait on a cancellation write")

    monkeypatch.setattr(session._conn, "send_control", reject_late_write)
    result = session.collect(handle)
    assert result.row_statuses == (SpeculatorStatusCode.TIMEOUT,)
    assert result.output.valid_lengths.tolist() == [0]
    assert session.connected
    adapter.release()


def test_latched_rows_report_their_failure_once(session, adapter, clock, monkeypatch):
    a = ready(session, 1)
    b = ready(session, 2)
    adapter.hold_batches.add(0)
    handle = session.dispatch(batch(0, [a], recovery=[1]))
    clock.advance(10.0)
    assert session.collect(handle).row_statuses == (SpeculatorStatusCode.TIMEOUT,)
    adapter.release()

    warning = Mock()
    monkeypatch.setattr(session_module.logger, "warning", warning)
    for batch_id in (1, 2, 3):
        result = round_trip(session, batch(batch_id, [a, b], recovery=[1, 2]))
        assert result.row_statuses == (SpeculatorStatusCode.TIMEOUT, OK)
    warning.assert_not_called()


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


def limited_server(**limits):
    adapter = FakeDraftAdapter(vocab_size=VOCAB)
    fields = dict(
        max_batch_size=8, max_feature_tokens=64, max_sequences=64, max_model_len=64
    )
    fields.update(limits)
    server = RemoteDraftServer(adapter, limits=RemoteServerLimits(**fields))
    server.start("tcp://127.0.0.1:0")
    return server, adapter


@pytest.mark.parametrize("policy", ["target_only", "error"])
def test_rows_beyond_max_batch_size_follow_failure_policy(clock, policy):
    # Reachable only when the runner sends more rows than it declared; the
    # overflow must fall back to target-only rather than kill the engine,
    # unless the error policy asks for failures to surface.
    server, adapter = limited_server(max_batch_size=2)
    identity = dataclasses.replace(IDENTITY, max_num_seqs=2)
    session = RemoteDraftSession(
        server.endpoint, identity, failure_policy=policy, clock=clock
    )
    try:
        session.connect()
        keys = [ready(session, i) for i in range(3)]
        handle = session.dispatch(batch(0, keys, recovery=[1, 2, 3]))
        if policy == "error":
            with pytest.raises(RemoteDraftError, match="QUEUE_FULL"):
                session.collect(handle)
        else:
            result = session.collect(handle)
            assert result.row_statuses == (OK, OK, SpeculatorStatusCode.QUEUE_FULL)
            assert result.output.valid_lengths.tolist() == [NUM_SPEC_TOKENS] * 2 + [0]
        assert session.sequence_state(keys[2]) is SequenceState.TARGET_ONLY
    finally:
        session.close()
        server.stop()


def test_rows_beyond_max_feature_tokens_degrade_instead_of_raising(clock):
    server, adapter = limited_server(max_feature_tokens=4)
    identity = dataclasses.replace(IDENTITY, max_num_batched_tokens=4)
    session = RemoteDraftSession(server.endpoint, identity, clock=clock)
    try:
        session.connect()
        keys = [ready(session, i, prompt=(1,)) for i in range(3)]
        result = round_trip(
            session, ragged_batch(0, keys, [3, 2, 1], [1, 2, 3, 4, 5, 6])
        )
        assert result.row_statuses == (OK, SpeculatorStatusCode.QUEUE_FULL, OK)
        assert adapter.round_tokens == {keys[0]: [1, 2, 3], keys[2]: [6]}
        assert session.sequence_state(keys[1]) is SequenceState.TARGET_ONLY
    finally:
        session.close()
        server.stop()


@pytest.mark.parametrize("policy", ["target_only", "error"])
def test_prefill_beyond_server_limits_follows_failure_policy(clock, policy):
    server, adapter = limited_server(max_feature_tokens=4, max_model_len=6)
    identity = dataclasses.replace(IDENTITY, max_num_batched_tokens=4, max_model_len=6)
    session = RemoteDraftSession(
        server.endpoint, identity, failure_policy=policy, clock=clock
    )
    try:
        session.connect()
        too_big = session.open_sequence(1)
        too_long = session.open_sequence(2)
        session.prefill(too_long, features([1, 2, 3, 4]), is_final=False)
        if policy == "error":
            with pytest.raises(RemoteDraftError, match="max_feature_tokens"):
                session.prefill(too_big, features([1] * 5), is_final=True)
            with pytest.raises(RemoteDraftError, match="max_model_len"):
                session.prefill(too_long, features([5, 6, 7]), is_final=True)
        else:
            session.prefill(too_big, features([1] * 5), is_final=True)
            session.prefill(too_long, features([5, 6, 7]), is_final=True)
        assert session.sequence_state(too_big) is SequenceState.TARGET_ONLY
        assert session.sequence_state(too_long) is SequenceState.TARGET_ONLY
        assert adapter.prefix(too_long) == [1, 2, 3, 4]
    finally:
        session.close()
        server.stop()


def test_stale_generation_desyncs_then_reopen_recovers(session, server, adapter):
    a = ready(session, 1)
    newer = SequenceKey(a.verifier_instance_id, a.sequence_id, a.generation + 1)
    with server.lock:
        server.registry.open(newer, owner=session.session_id)
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


def test_open_queue_full_latches_target_only(clock):
    adapter = FakeDraftAdapter(vocab_size=VOCAB)
    server = RemoteDraftServer(
        adapter,
        limits=RemoteServerLimits(
            max_batch_size=8,
            max_feature_tokens=64,
            max_sequences=1,
            max_model_len=64,
        ),
    )
    server.start("tcp://127.0.0.1:0")
    session = RemoteDraftSession(server.endpoint, IDENTITY, clock=clock)
    strict = RemoteDraftSession(
        server.endpoint, IDENTITY, failure_policy="error", clock=clock
    )
    try:
        session.connect()
        a = ready(session, 1)
        full = session.open_sequence(2)
        assert session.sequence_state(full) is SequenceState.TARGET_ONLY
        # The rejected sequence must degrade quietly, not wedge in OPENING:
        # prefill is a no-op and rounds skip the row with valid_length=0.
        session.prefill(full, features([1]), is_final=True)
        result = round_trip(session, batch(0, [a, full], recovery=[3, 4]))
        assert result.row_statuses == (OK, SpeculatorStatusCode.QUEUE_FULL)
        assert result.output.valid_lengths.tolist() == [NUM_SPEC_TOKENS, 0]

        strict.connect()
        with pytest.raises(RemoteDraftError, match="QUEUE_FULL"):
            strict.open_sequence(9)
    finally:
        session.close()
        strict.close()
        adapter.release()
        server.stop()


# ----------------------------------------------------------------------
# Reconnect
# ----------------------------------------------------------------------


def test_reconnect_completes_rounds_with_fresh_wire_state(session):
    a = ready(session, 1)
    assert round_trip(session, batch(0, [a], recovery=[5])).row_statuses == (OK,)
    old_epoch = session.session_epoch
    session.close()
    assert not session.connected

    session.connect()
    assert session.connected
    assert session.session_epoch != old_epoch
    b = session.open_sequence(1)
    assert b.generation == a.generation + 1
    session.prefill(b, features([1, 2]), is_final=True)
    # batch_id 0 repeats and the new server session numbers its responses
    # from 0 again; both must be accepted because that state is scoped to
    # the connection, or every post-reconnect round would look stale.
    result = round_trip(session, batch(0, [b], recovery=[7]))
    assert result.row_statuses == (OK,)
    assert result.output.token_ids.tolist() == [expected_row(7)]
    assert session.pending_frames == 0


def test_lost_connection_reconnects_with_backoff(make_session, server, clock):
    session = make_session()
    endpoint = server.endpoint
    old = ready(session, 1)
    server.stop()
    round_trip(session, batch(0, [old], recovery=[1]))
    assert not session.connected

    # No server yet: one attempt, then quiet until the backoff elapses.
    connects = Mock(wraps=session_module.transport_connect)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(session_module, "transport_connect", connects)
        assert session.sequence_state(session.open_sequence(2)) is SequenceState.INVALID
        assert session.sequence_state(session.open_sequence(3)) is SequenceState.INVALID
        assert connects.call_count == 1

        restarted = RemoteDraftServer(FakeDraftAdapter(vocab_size=VOCAB))
        restarted.start(endpoint)
        try:
            clock.advance(1.0)
            fresh = ready(session, 4)
            assert connects.call_count == 2
            assert session.connected
            result = round_trip(session, batch(0, [old, fresh], recovery=[1, 5]))
            assert result.row_statuses[1] is OK
            assert result.output.token_ids[1].tolist() == expected_row(5)
            assert session.sequence_state(old) is SequenceState.INVALID
        finally:
            session.close()
            restarted.stop()


def test_open_after_a_silent_restart_uses_the_new_server(make_session, server):
    # The session only learns that the old connection is dead when the
    # open fails on it; that request must still get the new connection.
    session = make_session()
    endpoint = server.endpoint
    ready(session, 1)
    server.stop()
    restarted = RemoteDraftServer(FakeDraftAdapter(vocab_size=VOCAB))
    restarted.start(endpoint)
    try:
        key = session.open_sequence(2)
        assert session.connected
        assert key.generation == 1
        assert session.sequence_state(key) is SequenceState.PREFILLING
    finally:
        session.close()
        restarted.stop()


def test_error_reply_to_an_earlier_request_is_ignored(clock):
    listener = listen("tcp://127.0.0.1:0")
    adapter = FakeDraftAdapter(vocab_size=VOCAB)
    ack = HelloAck(
        server_id="scripted",
        session_id="s",
        session_epoch=1,
        selected_transport="inline",
        capabilities=adapter.capabilities(),
        feature_schema=adapter.feature_schema(),
        limits=DEFAULT_LIMITS,
    )

    def serve():
        conn = listener.accept(timeout=5.0)
        with contextlib.suppress(TransportError):
            hello = decode_envelope(conn.recv(timeout=5.0).body)
            conn.send_control(
                encode_message(ack, session_id="s", request_id=hello.request_id)
            )
            envelope = decode_envelope(conn.recv(timeout=5.0).body)
            stale = ErrorReply(SpeculatorStatusCode.PROTOCOL_ERROR, "abandoned")
            conn.send_control(
                encode_message(stale, session_id="s", request_id=hello.request_id)
            )
            reply = SequenceAck(key=decode_payload(envelope).key)
            conn.send_control(
                encode_message(reply, session_id="s", request_id=envelope.request_id)
            )
            conn.recv(timeout=5.0)
        conn.close()

    thread = threading.Thread(target=serve)
    thread.start()
    session = RemoteDraftSession(listener.endpoint, IDENTITY, clock=clock)
    try:
        session.connect()
        key = session.open_sequence(1)
        assert session.sequence_state(key) is SequenceState.PREFILLING
    finally:
        session.close()
        thread.join()
        listener.close()


def test_reconnect_fences_handles_from_the_old_connection(session):
    a = ready(session, 1)
    handle = session.dispatch(batch(0, [a], recovery=[1]))
    old_session_id = handle.session_id
    session.close()
    session.connect()
    assert session.session_id != old_session_id
    # Simulate a server restart that reuses an epoch. The random session ID
    # remains authoritative even when an integer epoch happens to collide.
    handle.session_epoch = session.session_epoch
    with pytest.raises(RemoteDraftError, match="belongs to session"):
        session.collect(handle)


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


def test_open_twice_is_an_error(session):
    session.open_sequence(1)
    with pytest.raises(RemoteDraftError, match="already open"):
        session.open_sequence(1)


def test_acknowledged_close_forgets_the_sequence(session, adapter):
    # Generations exist to fence server state that may still be around; once
    # the server acknowledged the close there is none, and remembering every
    # sequence id ever opened would grow without bound.
    for sequence_id in range(50):
        session.close_sequences((ready(session, sequence_id),))
    assert session._generations == {}
    reopened = ready(session, 1)
    assert reopened.generation == 0
    result = round_trip(session, batch(0, [reopened], recovery=[4]))
    assert result.row_statuses == (OK,)


def test_unacknowledged_close_keeps_bumping_the_generation(session):
    first = ready(session, 1)
    session.close()
    session.close_sequences((first,))
    session.connect()
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


def test_prefill_many_writes_every_chunk_before_waiting(
    session, server, adapter, monkeypatch
):
    a = session.open_sequence(1)
    b = session.open_sequence(2)
    c = session.open_sequence(3)
    # The server holds a newer generation of b, so only b's chunk fails.
    newer = SequenceKey(b.verifier_instance_id, b.sequence_id, b.generation + 1)
    with server.lock:
        server.registry.open(newer, owner=session.session_id)
        adapter.open_sequence(newer)

    events: list[str] = []
    send, wait = session._send, session._await

    def record_send(*args, **kwargs):
        events.append("send")
        return send(*args, **kwargs)

    def record_wait(*args, **kwargs):
        events.append("wait")
        return wait(*args, **kwargs)

    monkeypatch.setattr(session, "_send", record_send)
    monkeypatch.setattr(session, "_await", record_wait)
    session.prefill_many(
        [
            (a, features([1, 2]), True),
            (b, features([3]), True),
            (c, features([4, 5]), False),
        ]
    )
    assert events == ["send"] * 3 + ["wait"] * 3
    assert session.sequence_state(a) is SequenceState.READY
    assert session.sequence_state(b) is SequenceState.DESYNCED
    assert session.sequence_state(c) is SequenceState.PREFILLING
    assert adapter.prefix(a) == [1, 2]
    assert adapter.prefix(c) == [4, 5]

    with pytest.raises(ValueError, match="one prefill chunk per sequence"):
        session.prefill_many([(c, features([6]), True), (c, features([7]), True)])


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
