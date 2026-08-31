# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone draft server driven over the raw wire protocol: handshake
gating, idempotent sequence commands, batch ordering, and error replies."""

import time
import zlib

import pytest

from vllm.v1.spec_decode.remote.capabilities import TargetFeatureKind
from vllm.v1.spec_decode.remote.protocol import (
    AdvanceAndPropose,
    CancelBatch,
    CloseSequence,
    ErrorReply,
    Hello,
    HelloAck,
    OpenSequence,
    Ping,
    Pong,
    PrefillChunk,
    ProposalResponse,
    RemoteServerLimits,
    SequenceAck,
    SequenceKey,
    SpeculatorStatusCode,
    decode_envelope,
    decode_payload,
    encode_message,
)
from vllm.v1.spec_decode.remote.server import (
    DEFAULT_LIMITS,
    FakeDraftAdapter,
    RemoteDraftServer,
    SequenceRegistry,
    StandaloneProposalBatch,
    StandaloneProposalResult,
)
from vllm.v1.spec_decode.remote.transport import (
    FRAME_DATA,
    ConnectionClosed,
    DataFrame,
    connect,
    decode_data_frame,
    frame_to_ints,
    ints_to_frame,
)

VERIFIER = "verifier-a"
NUM_SPEC_TOKENS = 4
VOCAB = 50
OK = SpeculatorStatusCode.OK


def key(sequence_id: int, generation: int = 0, verifier: str = VERIFIER):
    return SequenceKey(
        verifier_instance_id=verifier,
        sequence_id=sequence_id,
        generation=generation,
    )


def checksum(*frames) -> int:
    value = 0
    for frame in frames:
        value = zlib.crc32(frame.payload, value)
    return value & 0x7FFFFFFF


class RawClient:
    """Minimal verifier that speaks the protocol directly."""

    def __init__(self, endpoint: str):
        self.conn = connect(endpoint, timeout=5.0)
        self.session_id = ""
        self.request_id = 0
        self.frames: dict[int, DataFrame] = {}
        self.next_slot = 100

    def send(self, message, *, frames=()):
        for frame in frames:
            self.conn.send_data(frame)
        self.request_id += 1
        self.conn.send_control(
            encode_message(
                message, session_id=self.session_id, request_id=self.request_id
            )
        )
        return self.request_id

    def recv(self, timeout: float = 5.0):
        while True:
            frame = self.conn.recv(timeout=timeout)
            if frame.kind == FRAME_DATA:
                data = decode_data_frame(frame.body)
                self.frames[data.slot] = data
                continue
            envelope = decode_envelope(frame.body)
            return envelope, decode_payload(envelope)

    def call(self, message, *, frames=()):
        request_id = self.send(message, frames=frames)
        envelope, reply = self.recv()
        assert envelope.request_id == request_id
        return reply

    def hello(self, **overrides) -> HelloAck:
        fields = dict(
            verifier_instance_id=VERIFIER,
            target_fingerprint="tf",
            tokenizer_fingerprint="kf",
            method="eagle3",
            num_speculative_tokens=NUM_SPEC_TOKENS,
            parallel_drafting=False,
            supported_transports=("inline",),
        )
        fields.update(overrides)
        reply = self.call(Hello(**fields))
        if isinstance(reply, HelloAck):
            self.session_id = reply.session_id
        return reply

    def open(self, k: SequenceKey) -> SequenceAck:
        return self.call(OpenSequence(key=k))

    def prefill(
        self,
        k: SequenceKey,
        tokens,
        *,
        offset: int,
        is_final=True,
        checksum_override=None,
    ) -> SequenceAck:
        slot = self.reserve(1)
        frame = ints_to_frame(slot, tokens)
        chunk = PrefillChunk(
            key=k,
            offset=offset,
            num_tokens=len(tokens),
            is_final=is_final,
            feature_slot=slot,
            checksum=checksum(frame)
            if checksum_override is None
            else checksum_override,
        )
        return self.call(chunk, frames=[frame])

    def ready(self, k: SequenceKey, tokens=(1, 2, 3)) -> None:
        assert self.open(k).status is OK
        assert self.prefill(k, list(tokens), offset=0).status is OK

    def propose(self, batch_id: int, keys, recovery, accepted=None):
        accepted = accepted or [0] * len(keys)
        base = self.reserve(3)
        frames = [
            ints_to_frame(base, accepted),
            ints_to_frame(base + 1, recovery),
            ints_to_frame(base + 2, recovery),
        ]
        message = AdvanceAndPropose(
            batch_id=batch_id,
            keys=tuple(keys),
            accepted_counts_slot=base,
            recovery_tokens_slot=base + 1,
            feature_slot=base + 2,
        )
        return self.call(message, frames=frames)

    def propose_with_frames(self, batch_id, keys, accepted, recovery, feature):
        base = self.reserve(3)
        frames = [
            DataFrame(base, accepted.dtype, accepted.shape, accepted.payload),
            DataFrame(base + 1, recovery.dtype, recovery.shape, recovery.payload),
            DataFrame(base + 2, feature.dtype, feature.shape, feature.payload),
        ]
        return self.call(
            AdvanceAndPropose(
                batch_id=batch_id,
                keys=tuple(keys),
                accepted_counts_slot=base,
                recovery_tokens_slot=base + 1,
                feature_slot=base + 2,
            ),
            frames=frames,
        )

    def result(self, response: ProposalResponse):
        base = response.result_slot
        tokens = self.frames.pop(base)
        valid = self.frames.pop(base + 1)
        statuses = self.frames.pop(base + 2)
        rows = tokens.shape[0]
        flat = frame_to_ints(tokens)
        return (
            [
                flat[i * NUM_SPEC_TOKENS : (i + 1) * NUM_SPEC_TOKENS]
                for i in range(rows)
            ],
            frame_to_ints(valid),
            [SpeculatorStatusCode(s) for s in frame_to_ints(statuses)],
        )

    def reserve(self, count: int) -> int:
        slot = self.next_slot
        self.next_slot += count
        return slot

    def close(self):
        self.conn.close()


def expected_row(last_token: int):
    return [(last_token + j + 1) % VOCAB for j in range(NUM_SPEC_TOKENS)]


class MalformedResultAdapter(FakeDraftAdapter):
    def __init__(self, mode: str):
        super().__init__(vocab_size=VOCAB)
        self.mode = mode

    def advance_and_propose(
        self, batch: StandaloneProposalBatch
    ) -> StandaloneProposalResult:
        result = super().advance_and_propose(batch)
        if self.mode == "short_payload":
            valid_lengths = DataFrame(0, "int32", (len(batch.keys),), b"")
        else:
            valid_lengths = ints_to_frame(
                0, [batch.num_speculative_tokens + 1] * len(batch.keys)
            )
        return StandaloneProposalResult(
            result.token_ids, valid_lengths, result.row_statuses
        )


@pytest.fixture
def adapter():
    return FakeDraftAdapter(vocab_size=VOCAB)


@pytest.fixture
def server(adapter):
    server = RemoteDraftServer(
        adapter, server_id="srv", target_fingerprint="tf", tokenizer_fingerprint="kf"
    )
    server.start("tcp://127.0.0.1:0")
    yield server
    adapter.release()
    server.stop()


@pytest.fixture
def client(server):
    client = RawClient(server.endpoint)
    yield client
    client.close()


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# ----------------------------------------------------------------------
# Handshake
# ----------------------------------------------------------------------


def test_hello_ack_reports_adapter_capabilities(server, adapter, client):
    ack = client.hello()
    assert isinstance(ack, HelloAck)
    assert ack.server_id == "srv"
    assert ack.session_epoch >= 1
    assert ack.selected_transport == "inline"
    assert ack.capabilities == adapter.capabilities()
    assert ack.feature_schema == adapter.feature_schema()
    assert ack.limits == DEFAULT_LIMITS
    assert ack.feature_schema.slots[0].kind == TargetFeatureKind.TOKEN_IDS.value


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"target_fingerprint": "other"}, "target fingerprint"),
        ({"tokenizer_fingerprint": "other"}, "tokenizer fingerprint"),
        ({"method": "ngram"}, "method 'ngram'"),
        ({"supported_transports": ("cuda_ipc",)}, "no common transport"),
    ],
)
def test_hello_rejections_close_the_connection(client, overrides, reason):
    reply = client.hello(**overrides)
    assert isinstance(reply, ErrorReply)
    assert reply.status is SpeculatorStatusCode.UNSUPPORTED_FEATURE
    assert reason in reply.detail
    with pytest.raises(ConnectionClosed):
        client.recv(timeout=5.0)


def test_first_message_must_be_hello(client):
    reply = client.call(Ping(nonce=1))
    assert isinstance(reply, ErrorReply)
    assert reply.status is SpeculatorStatusCode.PROTOCOL_ERROR
    assert "expected hello" in reply.detail


def test_hello_rejects_missing_or_mismatched_draft_fingerprint():
    # The check must be fail-closed: when the server declares a draft
    # checkpoint fingerprint, an empty client value is a mismatch too.
    adapter = FakeDraftAdapter(vocab_size=VOCAB, draft_checkpoint_fingerprint="d1")
    server = RemoteDraftServer(adapter)
    server.start("tcp://127.0.0.1:0")
    try:
        for offered in ("", "other"):
            client = RawClient(server.endpoint)
            reply = client.hello(draft_checkpoint_fingerprint=offered)
            assert isinstance(reply, ErrorReply)
            assert "draft checkpoint fingerprint" in reply.detail
            client.close()
        client = RawClient(server.endpoint)
        assert isinstance(client.hello(draft_checkpoint_fingerprint="d1"), HelloAck)
        client.close()
    finally:
        adapter.release()
        server.stop()


def test_each_session_gets_a_fresh_epoch(server):
    first = RawClient(server.endpoint)
    second = RawClient(server.endpoint)
    try:
        epochs = {first.hello().session_epoch, second.hello().session_epoch}
        assert len(epochs) == 2
    finally:
        first.close()
        second.close()


# ----------------------------------------------------------------------
# Sequence commands
# ----------------------------------------------------------------------


def test_open_is_idempotent_and_generation_aware(client, adapter, server):
    client.hello()
    assert client.open(key(1)).status is OK
    assert client.open(key(1)).status is OK
    assert adapter.prefix(key(1)) == []

    assert client.open(key(1, generation=1)).status is OK
    assert adapter.prefix(key(1)) is None
    assert adapter.prefix(key(1, generation=1)) == []
    assert len(server.registry) == 1

    stale = client.open(key(1))
    assert stale.status is SpeculatorStatusCode.STALE_GENERATION


def test_open_rejects_foreign_verifier_key(client):
    client.hello()
    ack = client.open(key(1, verifier="someone-else"))
    assert ack.status is SpeculatorStatusCode.PROTOCOL_ERROR


def test_prefill_idempotency_rules(client, adapter):
    client.hello()
    k = key(2)
    client.open(k)
    assert client.prefill(k, [1, 2, 3], offset=0, is_final=False).status is OK
    assert adapter.prefix(k) == [1, 2, 3]

    duplicate = client.prefill(k, [1, 2, 3], offset=0, is_final=False)
    assert duplicate.status is OK
    assert adapter.prefix(k) == [1, 2, 3]

    changed_finality = client.prefill(k, [1, 2, 3], offset=0, is_final=True)
    assert changed_finality.status is SpeculatorStatusCode.SEQUENCE_DESYNC
    assert "finality" in changed_finality.detail

    mismatch = client.prefill(k, [9, 9, 9], offset=0, is_final=False)
    assert mismatch.status is SpeculatorStatusCode.SEQUENCE_DESYNC
    assert "different content" in mismatch.detail

    gap = client.prefill(k, [4], offset=5, is_final=False)
    assert gap.status is SpeculatorStatusCode.SEQUENCE_DESYNC
    assert "expected offset 3" in gap.detail

    bad_header = client.prefill(k, [4], offset=3, checksum_override=1)
    assert bad_header.status is SpeculatorStatusCode.SEQUENCE_DESYNC
    assert "checksum" in bad_header.detail

    assert client.prefill(k, [4], offset=3, is_final=True).status is OK
    assert adapter.prefix(k) == [1, 2, 3, 4]
    late = client.prefill(k, [5], offset=4, is_final=True)
    assert late.status is SpeculatorStatusCode.SEQUENCE_DESYNC
    assert "after final" in late.detail


def test_prefill_unknown_sequence_or_missing_frame(client):
    client.hello()
    unknown = client.prefill(key(3), [1], offset=0)
    assert unknown.status is SpeculatorStatusCode.SEQUENCE_DESYNC
    assert "unknown sequence" in unknown.detail

    client.open(key(3))
    missing = client.call(
        PrefillChunk(
            key=key(3),
            offset=0,
            num_tokens=1,
            is_final=True,
            feature_slot=999,
            checksum=0,
        )
    )
    assert missing.status is SpeculatorStatusCode.PROTOCOL_ERROR
    assert "missing" in missing.detail


def test_close_is_idempotent_and_keeps_newer_generation(client, adapter, server):
    client.hello()
    client.ready(key(4))
    for _ in range(2):
        ack = client.call(CloseSequence(keys=(key(4),)))
        assert ack.status is OK
    assert adapter.prefix(key(4)) is None
    assert len(server.registry) == 0

    client.ready(key(4, generation=1))
    assert client.call(CloseSequence(keys=(key(4),))).status is OK
    assert adapter.prefix(key(4, generation=1)) == [1, 2, 3]
    assert len(server.registry) == 1


def test_close_acks_every_key(client):
    client.hello()
    client.ready(key(5))
    client.ready(key(6))
    request_id = client.send(CloseSequence(keys=(key(5), key(6), key(7))))
    acked = set()
    for _ in range(3):
        envelope, ack = client.recv()
        assert envelope.request_id == request_id
        assert isinstance(ack, SequenceAck) and ack.status is OK
        acked.add(ack.key)
    assert acked == {key(5), key(6), key(7)}


def test_max_sequences_is_global_and_allows_identity_replacement():
    adapter = FakeDraftAdapter(vocab_size=VOCAB)
    server = RemoteDraftServer(
        adapter,
        limits=RemoteServerLimits(
            max_batch_size=4,
            max_feature_tokens=16,
            max_sequences=1,
            max_model_len=32,
        ),
    )
    server.start("tcp://127.0.0.1:0")
    first = RawClient(server.endpoint)
    replacement = RawClient(server.endpoint)
    other = RawClient(server.endpoint)
    try:
        first.hello()
        assert first.open(key(8)).status is OK

        replacement.hello()
        assert replacement.open(key(8, generation=1)).status is OK
        assert first.open(key(8)).status is SpeculatorStatusCode.STALE_GENERATION
        assert first.open(key(9)).status is SpeculatorStatusCode.QUEUE_FULL

        other.hello(verifier_instance_id="other")
        assert (
            other.open(key(8, verifier="other")).status
            is SpeculatorStatusCode.QUEUE_FULL
        )
        assert len(server.registry) == 1
    finally:
        first.close()
        replacement.close()
        other.close()
        adapter.release()
        server.stop()


# ----------------------------------------------------------------------
# Proposal rounds
# ----------------------------------------------------------------------


def test_proposal_follows_fake_formula(client, adapter):
    client.hello()
    client.ready(key(10), tokens=[1, 2])
    client.ready(key(11), tokens=[3])
    response = client.propose(0, [key(10), key(11)], recovery=[5, 47], accepted=[0, 2])
    assert isinstance(response, ProposalResponse)
    assert response.status is OK and response.sequence_number == 0
    tokens, valid, statuses = client.result(response)
    assert tokens == [expected_row(5), expected_row(47)]
    assert valid == [NUM_SPEC_TOKENS, NUM_SPEC_TOKENS]
    assert statuses == [OK, OK]
    assert adapter.prefix(key(10)) == [1, 2, 5]
    assert adapter.prefix(key(11)) == [3, 47]


def test_proposal_marks_unready_and_stale_rows(client):
    client.hello()
    client.ready(key(20))
    client.open(key(21))
    client.ready(key(22, generation=1))
    response = client.propose(
        0, [key(20), key(21), key(22, generation=0)], recovery=[7, 8, 9]
    )
    assert response.status is OK
    tokens, valid, statuses = client.result(response)
    assert statuses == [
        OK,
        SpeculatorStatusCode.SEQUENCE_DESYNC,
        SpeculatorStatusCode.STALE_GENERATION,
    ]
    assert valid == [NUM_SPEC_TOKENS, 0, 0]
    assert tokens[0] == expected_row(7)


def test_batch_ids_must_increase_and_sequence_numbers_do(client):
    client.hello()
    client.ready(key(30))
    first = client.propose(5, [key(30)], recovery=[1])
    repeat = client.propose(5, [key(30)], recovery=[1])
    older = client.propose(4, [key(30)], recovery=[1])
    newer = client.propose(6, [key(30)], recovery=[1])
    assert first.status is OK
    assert repeat.status is SpeculatorStatusCode.ROUND_MISMATCH
    assert repeat.result_slot is None
    assert older.status is SpeculatorStatusCode.ROUND_MISMATCH
    assert newer.status is OK
    assert [r.sequence_number for r in (first, repeat, older, newer)] == [0, 1, 2, 3]


def test_proposal_without_input_frames_is_protocol_error(client):
    client.hello()
    client.ready(key(40))
    response = client.call(
        AdvanceAndPropose(
            batch_id=0,
            keys=(key(40),),
            accepted_counts_slot=900,
            recovery_tokens_slot=901,
            feature_slot=902,
        )
    )
    assert response.status is SpeculatorStatusCode.PROTOCOL_ERROR


@pytest.mark.parametrize(
    ("accepted", "recovery"),
    [
        (
            ints_to_frame(0, [0], shape=(1, 1)),
            ints_to_frame(0, [1]),
        ),
        (
            ints_to_frame(0, [0]),
            ints_to_frame(0, [1], shape=(1, 1)),
        ),
        (
            ints_to_frame(0, [0]),
            DataFrame(0, "float32", (1,), b"\x00" * 4),
        ),
        (
            ints_to_frame(0, [0]),
            DataFrame(0, "int64", (1,), b""),
        ),
    ],
    ids=["accepted-shape", "recovery-shape", "recovery-dtype", "recovery-bytes"],
)
def test_proposal_rejects_malformed_integer_frames(client, adapter, accepted, recovery):
    client.hello()
    k = key(42)
    client.ready(k)
    response = client.propose_with_frames(
        0, [k], accepted, recovery, ints_to_frame(0, [1])
    )
    assert response.status is SpeculatorStatusCode.PROTOCOL_ERROR
    assert adapter.proposed_batches == []


@pytest.mark.parametrize("mode", ["short_payload", "out_of_range"])
def test_malformed_adapter_result_discards_sequence_state(mode):
    adapter = MalformedResultAdapter(mode)
    server = RemoteDraftServer(adapter)
    server.start("tcp://127.0.0.1:0")
    client = RawClient(server.endpoint)
    k = key(43)
    try:
        client.hello()
        client.ready(k)
        response = client.propose(0, [k], recovery=[2])
        assert response.status is SpeculatorStatusCode.INTERNAL_ERROR
        assert response.result_slot is None
        assert server.registry.get(k) is None
        assert adapter.prefix(k) is None
    finally:
        client.close()
        adapter.release()
        server.stop()


def test_rejected_batch_frames_are_not_read_by_a_later_round(client):
    # A rejected batch must not leave its frames behind to be misread by a
    # later round that reuses the same slot numbers.
    client.hello()
    client.ready(key(41))
    client.propose(3, [key(41)], recovery=[1])
    client.next_slot -= 3
    rejected = client.propose(3, [key(41)], recovery=[30])
    assert rejected.status is SpeculatorStatusCode.ROUND_MISMATCH
    client.next_slot -= 3
    fresh = client.propose(4, [key(41)], recovery=[20])
    assert fresh.status is OK
    tokens, _, _ = client.result(fresh)
    assert tokens == [expected_row(20)]


def test_accepted_counts_out_of_range_rejected(client, adapter):
    # Out-of-range accepted counts would corrupt prefix accounting via
    # ``prefix_length += accepted + 1``; the server must not trust them.
    client.hello()
    client.ready(key(80))
    negative = client.propose(0, [key(80)], recovery=[1], accepted=[-1])
    too_many = client.propose(
        1, [key(80)], recovery=[1], accepted=[NUM_SPEC_TOKENS + 1]
    )
    assert negative.status is SpeculatorStatusCode.PROTOCOL_ERROR
    assert too_many.status is SpeculatorStatusCode.PROTOCOL_ERROR
    good = client.propose(2, [key(80)], recovery=[5])
    assert good.status is OK
    assert adapter.proposed_batches == [2]
    assert adapter.prefix(key(80)) == [1, 2, 3, 5]


def test_prefill_and_rounds_respect_advertised_limits():
    adapter = FakeDraftAdapter(vocab_size=VOCAB)
    server = RemoteDraftServer(
        adapter,
        limits=RemoteServerLimits(
            max_batch_size=4,
            max_feature_tokens=4,
            max_sequences=8,
            max_model_len=6,
        ),
    )
    server.start("tcp://127.0.0.1:0")
    client = RawClient(server.endpoint)
    try:
        client.hello()
        k = key(90)
        assert client.open(k).status is OK
        too_long = client.prefill(k, [1] * 5, offset=0, is_final=False)
        assert too_long.status is SpeculatorStatusCode.PROTOCOL_ERROR
        assert "max_feature_tokens" in too_long.detail
        assert client.prefill(k, [1, 2, 3, 4], offset=0, is_final=False).status is OK
        overflow = client.prefill(k, [5, 6, 7], offset=4, is_final=True)
        assert overflow.status is SpeculatorStatusCode.OUT_OF_MEMORY
        assert "max_model_len" in overflow.detail
        assert client.prefill(k, [5, 6], offset=4, is_final=True).status is OK

        # The prefix sits at max_model_len: even accepted=0 grows it by the
        # bonus token, so the row must come back OUT_OF_MEMORY untouched.
        response = client.propose(0, [k], recovery=[9])
        assert response.status is OK
        _, valid, statuses = client.result(response)
        assert statuses == [SpeculatorStatusCode.OUT_OF_MEMORY]
        assert valid == [0]
        assert adapter.proposed_batches == []
        assert adapter.prefix(k) == [1, 2, 3, 4, 5, 6]
    finally:
        client.close()
        adapter.release()
        server.stop()


def test_cancel_is_accepted_without_reply(client):
    client.hello()
    client.send(CancelBatch(batch_id=123))
    assert client.call(Ping(nonce=4)).nonce == 4


# ----------------------------------------------------------------------
# Session-level errors and teardown
# ----------------------------------------------------------------------


def test_malformed_control_frame_gets_error_reply_and_keeps_session(client):
    client.hello()
    client.conn.send_control(b"\xc1garbage")
    _, reply = client.recv()
    assert isinstance(reply, ErrorReply)
    assert reply.status is SpeculatorStatusCode.PROTOCOL_ERROR
    assert client.call(Ping(nonce=2)).nonce == 2


def test_session_id_mismatch_rejected(client):
    client.hello()
    client.session_id = "bogus"
    reply = client.call(Ping())
    assert isinstance(reply, ErrorReply)
    assert "session id" in reply.detail


def test_unexpected_message_type_rejected(client):
    client.hello()
    reply = client.call(Pong(nonce=1))
    assert isinstance(reply, ErrorReply)
    assert "unexpected pong" in reply.detail


def test_ping_reports_active_sequences(client):
    client.hello()
    client.ready(key(50))
    client.ready(key(51))
    pong = client.call(Ping(nonce=77))
    assert isinstance(pong, Pong)
    assert pong.nonce == 77
    assert pong.active_sequences == 2


def test_disconnect_releases_sequences(server, adapter, client):
    client.hello()
    client.ready(key(60))
    client.ready(key(61))
    assert len(server.registry) == 2
    client.close()
    assert wait_until(lambda: len(server.registry) == 0)
    assert adapter.prefix(key(60)) is None


def test_late_disconnect_only_releases_entries_the_session_still_owns(server, adapter):
    # A reconnected verifier re-opens its sequences under new generations;
    # when the old connection finally goes away, its cleanup must not take
    # the replacement entries with it.
    first = RawClient(server.endpoint)
    second = RawClient(server.endpoint)
    try:
        first.hello()
        first.ready(key(70))
        first.ready(key(71))
        second.hello()
        second.ready(key(70, generation=1), tokens=(7, 8))
        assert len(server.registry) == 2

        first.close()
        assert wait_until(lambda: len(server.registry) == 1)
        response = second.propose(0, [key(70, generation=1)], recovery=[9])
        assert response.status is OK
        _, _, statuses = second.result(response)
        assert statuses == [OK]
        assert adapter.prefix(key(70, generation=1)) == [7, 8, 9]
        assert adapter.prefix(key(71)) is None
    finally:
        first.close()
        second.close()


def test_live_sessions_cannot_mutate_each_others_sequences(server, adapter):
    first = RawClient(server.endpoint)
    second = RawClient(server.endpoint)
    k = key(72)
    try:
        first.hello()
        first.ready(k)
        second.hello()

        assert second.open(k).status is SpeculatorStatusCode.SEQUENCE_DESYNC
        assert (
            second.prefill(k, [9], offset=3).status
            is SpeculatorStatusCode.SEQUENCE_DESYNC
        )
        response = second.propose(0, [k], recovery=[9])
        _, valid, statuses = second.result(response)
        assert valid == [0]
        assert statuses == [SpeculatorStatusCode.SEQUENCE_DESYNC]
        assert second.call(CloseSequence(keys=(k,))).status is OK

        response = first.propose(0, [k], recovery=[4])
        _, _, statuses = first.result(response)
        assert statuses == [OK]
        assert adapter.prefix(k) == [1, 2, 3, 4]
    finally:
        first.close()
        second.close()


def test_stop_disconnects_clients(server, client):
    client.hello()
    server.stop()
    with pytest.raises(ConnectionClosed):
        client.recv(timeout=5.0)


# ----------------------------------------------------------------------
# Registry unit rules
# ----------------------------------------------------------------------


def test_registry_open_generations():
    registry = SequenceRegistry()
    assert registry.open(key(1), owner="s1").created
    assert not registry.open(key(1), owner="s1").created
    outcome = registry.open(key(1, generation=2), owner="s1")
    assert outcome.created and outcome.replaced == key(1)
    assert registry.open(key(1, generation=1), owner="s1").status is (
        SpeculatorStatusCode.STALE_GENERATION
    )
    assert registry.get(key(1)) is None
    assert registry.get(key(1, generation=2)) is not None


def test_registry_prefill_and_rounds():
    registry = SequenceRegistry()
    registry.open(key(1), owner="s1")
    assert (
        registry.begin_round(key(1), 0, owner="s1")[0]
        is SpeculatorStatusCode.SEQUENCE_DESYNC
    )
    assert registry.prefill(
        key(1),
        owner="s1",
        offset=0,
        num_tokens=3,
        checksum=9,
        is_final=False,
    ) == (OK, "", True)
    assert (
        registry.prefill(
            key(1),
            owner="s1",
            offset=0,
            num_tokens=3,
            checksum=9,
            is_final=False,
        )[2]
        is False
    )
    assert (
        registry.prefill(
            key(1),
            owner="s1",
            offset=3,
            num_tokens=1,
            checksum=1,
            is_final=True,
        )[0]
        is OK
    )
    assert registry.begin_round(key(1), 0, owner="s1") == (OK, "")
    assert (
        registry.begin_round(key(1), 0, owner="s1")[0]
        is SpeculatorStatusCode.ROUND_MISMATCH
    )
    registry.advance(key(1), 2, owner="s1")
    assert registry.get(key(1)).prefix_length == 4 + 3


def test_registry_close_all_releases_only_owned_entries():
    registry = SequenceRegistry()
    registry.open(key(1), owner="s1")
    registry.open(key(2), owner="s1")
    registry.open(key(3, verifier="other"), owner="s2")
    assert set(registry.close_all("s1")) == {key(1), key(2)}
    assert len(registry) == 1
    assert registry.close_all("s1") == ()
    assert registry.close(key(3, verifier="other"), owner="s2") == key(
        3, verifier="other"
    )
    assert registry.close(key(3, verifier="other"), owner="s2") is None


def test_registry_new_generation_transfers_ownership():
    registry = SequenceRegistry()
    registry.open(key(1), owner="old")
    assert registry.open(key(1, generation=1), owner="new").created
    assert registry.close_all("old") == ()
    same_generation = registry.open(key(1, generation=1), owner="newer")
    assert same_generation.status is SpeculatorStatusCode.SEQUENCE_DESYNC
    assert registry.close_all("newer") == ()
    assert set(registry.close_all("new")) == {key(1, generation=1)}
    assert len(registry) == 0


def test_registry_rejects_other_owner_and_future_close():
    registry = SequenceRegistry()
    current = key(1, generation=1)
    registry.open(current, owner="s1")
    outcome = registry.open(current, owner="s2")
    assert outcome.status is SpeculatorStatusCode.SEQUENCE_DESYNC
    assert registry.close(key(1, generation=2), owner="s1") is None
    assert registry.close(current, owner="s2") is None
    assert registry.get(current) is not None


def test_entrypoints_alias_exposes_server_main():
    """`python -m vllm.entrypoints.remote_draft_server` is the documented CLI."""
    from vllm.entrypoints.remote_draft_server import main as alias_main
    from vllm.v1.spec_decode.remote.server import main as server_main

    assert alias_main is server_main
