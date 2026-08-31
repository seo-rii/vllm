# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone speculator server: sessions, sequence registry, and the
adapter contract a concrete drafter implements.

The server is torch-free so protocol and registry logic can be tested
without a GPU; adapters receive and return data-plane frames and do their
own device handling. Requests on one connection are processed serially,
which yields the initial invariants directly: no two proposals run on the
same sequence at once, batch IDs increase monotonically per session, and
a response is only sent after its result frames.
"""

import argparse
import contextlib
import threading
import time
import uuid
import zlib
from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, NamedTuple

import msgspec

from vllm.logger import init_logger
from vllm.v1.spec_decode.remote.capabilities import (
    REMOTE_DRAFT_SUPPORTED_METHODS,
    SpeculatorPlacementCapabilities,
    TargetFeatureKind,
)
from vllm.v1.spec_decode.remote.protocol import (
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    AdvanceAndPropose,
    CancelBatch,
    CloseSequence,
    ErrorReply,
    FeatureSlot,
    Hello,
    HelloAck,
    OpenSequence,
    Ping,
    Pong,
    PrefillChunk,
    ProposalResponse,
    ProtocolError,
    RemoteServerLimits,
    SequenceAck,
    SequenceKey,
    SpeculatorStatusCode,
    TargetFeatureSchema,
    decode_envelope,
    decode_payload,
    encode_message,
)
from vllm.v1.spec_decode.remote.state import SequenceState
from vllm.v1.spec_decode.remote.transport import (
    FRAME_CONTROL,
    INLINE_TRANSPORT,
    ConnectionClosed,
    DataFrame,
    FramedConnection,
    Listener,
    TransportError,
    TransportTimeout,
    decode_data_frame,
    frame_to_ints,
    ints_to_frame,
    listen,
)

logger = init_logger(__name__)

DEFAULT_LIMITS = RemoteServerLimits(
    max_batch_size=256,
    max_feature_tokens=16384,
    max_sequences=1024,
    max_model_len=32768,
)
SUPPORTED_TRANSPORTS = (INLINE_TRANSPORT,)

_ACCEPT_POLL_S = 0.2
_RESULT_FRAMES = 3


# ----------------------------------------------------------------------
# Adapter contract
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PrefillInputs:
    """One verified prompt chunk in negotiated schema order."""

    offset: int
    num_tokens: int
    is_final: bool
    features: tuple[DataFrame | None, ...]


@dataclass(frozen=True)
class StandaloneProposalBatch:
    """One advance-and-propose round as seen by an adapter.

    Rows not in ``active_rows`` were rejected by the registry (stale,
    desynced, not prefilled); the adapter must still return a row for them
    but must not advance their state.
    """

    batch_id: int
    keys: tuple[SequenceKey, ...]
    active_rows: tuple[int, ...]
    num_speculative_tokens: int
    accepted_counts: DataFrame
    recovery_tokens: DataFrame
    features: tuple[DataFrame | None, ...]


@dataclass(frozen=True)
class StandaloneProposalResult:
    """Adapter output for every row of a StandaloneProposalBatch."""

    token_ids: DataFrame
    """Integer frame of shape [len(keys), num_speculative_tokens]."""
    valid_lengths: DataFrame
    """Integer frame of shape [len(keys)]."""
    row_statuses: tuple[SpeculatorStatusCode, ...]


class StandaloneDraftAdapter(ABC):
    """Model-specific drafter behind the server.

    The server serializes all calls, so implementations need no locking.
    Capabilities must describe the adapter and checkpoint actually loaded,
    never be inferred from the method name.
    """

    @abstractmethod
    def capabilities(self) -> SpeculatorPlacementCapabilities: ...

    @abstractmethod
    def feature_schema(self) -> TargetFeatureSchema: ...

    @abstractmethod
    def supported_methods(self) -> Collection[str]: ...

    def draft_checkpoint_fingerprint(self) -> str:
        return ""

    @abstractmethod
    def open_sequence(self, key: SequenceKey) -> None: ...

    @abstractmethod
    def prefill(self, key: SequenceKey, inputs: PrefillInputs) -> None: ...

    @abstractmethod
    def advance_and_propose(
        self, batch: StandaloneProposalBatch
    ) -> StandaloneProposalResult: ...

    @abstractmethod
    def close_sequences(self, keys: tuple[SequenceKey, ...]) -> None: ...


class FakeDraftAdapter(StandaloneDraftAdapter):
    """Deterministic drafter for protocol tests.

    ``proposal[i, j] = (recovery_token[i] + j + 1) % vocab_size``, so a
    verifier can check end to end that the right rows came back in the
    right order. ``hold_batches`` makes a batch block until ``release`` is
    called, which is how slow-server and timeout paths are exercised.
    """

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        capabilities: SpeculatorPlacementCapabilities | None = None,
        methods: Collection[str] = REMOTE_DRAFT_SUPPORTED_METHODS,
        hold_timeout_s: float = 10.0,
        draft_checkpoint_fingerprint: str = "",
    ) -> None:
        self._vocab_size = vocab_size
        self._draft_checkpoint_fingerprint = draft_checkpoint_fingerprint
        self._capabilities = capabilities or SpeculatorPlacementCapabilities(
            state_dependency="own_kv",
            required_features=(TargetFeatureKind.TOKEN_IDS.value,),
            supports_parallel_drafting=True,
            standalone_weights="complete",
        )
        self._methods = frozenset(methods)
        self._schema = TargetFeatureSchema(
            schema_id=1,
            slots=(
                FeatureSlot(
                    kind=TargetFeatureKind.TOKEN_IDS.value,
                    dtype="int32",
                    trailing_shape=(),
                ),
            ),
        )
        self._prefixes: dict[SequenceKey, list[int]] = {}
        self._hold_timeout_s = hold_timeout_s
        self._release = threading.Event()
        self.hold_batches: set[int] = set()
        self.proposed_batches: list[int] = []

    def capabilities(self) -> SpeculatorPlacementCapabilities:
        return self._capabilities

    def feature_schema(self) -> TargetFeatureSchema:
        return self._schema

    def supported_methods(self) -> Collection[str]:
        return self._methods

    def draft_checkpoint_fingerprint(self) -> str:
        return self._draft_checkpoint_fingerprint

    def prefix(self, key: SequenceKey) -> list[int] | None:
        return self._prefixes.get(key)

    def release(self) -> None:
        self._release.set()

    def open_sequence(self, key: SequenceKey) -> None:
        self._prefixes[key] = []

    def prefill(self, key: SequenceKey, inputs: PrefillInputs) -> None:
        tokens = inputs.features[0]
        assert tokens is not None
        self._prefixes[key].extend(frame_to_ints(tokens))

    def advance_and_propose(
        self, batch: StandaloneProposalBatch
    ) -> StandaloneProposalResult:
        if batch.batch_id in self.hold_batches:
            self._release.wait(self._hold_timeout_s)
        self.proposed_batches.append(batch.batch_id)
        recovery = frame_to_ints(batch.recovery_tokens)
        num_rows = len(batch.keys)
        num_tokens = batch.num_speculative_tokens
        flat: list[int] = []
        for i in range(num_rows):
            last = recovery[i]
            flat.extend((last + j + 1) % self._vocab_size for j in range(num_tokens))
            if i in batch.active_rows:
                self._prefixes[batch.keys[i]].append(last)
        return StandaloneProposalResult(
            token_ids=ints_to_frame(
                0, flat, dtype="int64", shape=(num_rows, num_tokens)
            ),
            valid_lengths=ints_to_frame(0, [num_tokens] * num_rows),
            row_statuses=(SpeculatorStatusCode.OK,) * num_rows,
        )

    def close_sequences(self, keys: tuple[SequenceKey, ...]) -> None:
        for key in keys:
            self._prefixes.pop(key, None)


# ----------------------------------------------------------------------
# Sequence registry
# ----------------------------------------------------------------------


@dataclass
class SequenceEntry:
    """Server-side bookkeeping for one (verifier, sequence, generation)."""

    key: SequenceKey
    owner: str = ""
    """session_id of the connection that opened this generation."""
    state: SequenceState = SequenceState.OPENING
    applied_offset: int = 0
    last_chunk: tuple[int, int, int] | None = None
    """(offset, num_tokens, checksum) of the last applied prefill chunk."""
    prefix_length: int = 0
    last_batch_id: int = -1
    last_activity: float = 0.0


class OpenOutcome(NamedTuple):
    status: SpeculatorStatusCode
    detail: str
    created: bool
    replaced: SequenceKey | None


class SequenceRegistry:
    """Idempotent sequence lifecycle keyed by (verifier, sequence_id).

    A newer generation replaces the entry; commands carrying an older
    generation are rejected as STALE_GENERATION. Duplicate open, duplicate
    prefill chunk (same offset and checksum), and duplicate close are all
    acknowledged without side effects. Entries record the session that
    opened them, so the delayed disconnect of a superseded connection can
    only release its own leftovers, never a reconnected verifier's state.
    """

    def __init__(self, clock=time.monotonic) -> None:
        self._entries: dict[tuple[str, int], SequenceEntry] = {}
        self._clock = clock

    def __len__(self) -> int:
        return len(self._entries)

    def count(self, verifier_instance_id: str) -> int:
        return sum(1 for vid, _ in self._entries if vid == verifier_instance_id)

    def get(self, key: SequenceKey) -> SequenceEntry | None:
        entry = self._entries.get(_registry_key(key))
        return entry if entry is not None and entry.key == key else None

    def open(self, key: SequenceKey, owner: str = "") -> OpenOutcome:
        entry = self._entries.get(_registry_key(key))
        if entry is not None:
            if key.generation < entry.key.generation:
                return OpenOutcome(
                    SpeculatorStatusCode.STALE_GENERATION,
                    f"generation {key.generation} superseded by {entry.key.generation}",
                    False,
                    None,
                )
            if key.generation == entry.key.generation:
                entry.owner = owner
                entry.last_activity = self._clock()
                return OpenOutcome(SpeculatorStatusCode.OK, "already open", False, None)
        replaced = entry.key if entry is not None else None
        self._entries[_registry_key(key)] = SequenceEntry(
            key=key, owner=owner, last_activity=self._clock()
        )
        return OpenOutcome(SpeculatorStatusCode.OK, "", True, replaced)

    def prefill(
        self,
        key: SequenceKey,
        *,
        offset: int,
        num_tokens: int,
        checksum: int,
        is_final: bool,
    ) -> tuple[SpeculatorStatusCode, str, bool]:
        """Returns (status, detail, apply): apply is False for duplicates."""
        status, detail, entry = self._lookup(key)
        if entry is None:
            return status, detail, False
        entry.last_activity = self._clock()
        if entry.last_chunk is not None and offset == entry.last_chunk[0]:
            if (num_tokens, checksum) == entry.last_chunk[1:]:
                return SpeculatorStatusCode.OK, "duplicate chunk", False
            return (
                SpeculatorStatusCode.SEQUENCE_DESYNC,
                (
                    f"chunk at offset {offset} was already applied with a "
                    "different checksum"
                ),
                False,
            )
        if entry.state is SequenceState.READY:
            return (
                SpeculatorStatusCode.SEQUENCE_DESYNC,
                "prefill after final chunk",
                False,
            )
        if offset != entry.applied_offset:
            return (
                SpeculatorStatusCode.SEQUENCE_DESYNC,
                f"expected offset {entry.applied_offset}, got {offset}",
                False,
            )
        entry.applied_offset += num_tokens
        entry.prefix_length = entry.applied_offset
        entry.last_chunk = (offset, num_tokens, checksum)
        entry.state = SequenceState.READY if is_final else SequenceState.PREFILLING
        return SpeculatorStatusCode.OK, "", True

    def begin_round(
        self, key: SequenceKey, batch_id: int
    ) -> tuple[SpeculatorStatusCode, str]:
        status, detail, entry = self._lookup(key)
        if entry is None:
            return status, detail
        if entry.state is not SequenceState.READY:
            return SpeculatorStatusCode.SEQUENCE_DESYNC, "sequence is not prefilled"
        if batch_id <= entry.last_batch_id:
            return (
                SpeculatorStatusCode.ROUND_MISMATCH,
                f"batch {batch_id} is not newer than {entry.last_batch_id}",
            )
        entry.last_batch_id = batch_id
        entry.last_activity = self._clock()
        return SpeculatorStatusCode.OK, ""

    def advance(self, key: SequenceKey, accepted: int) -> None:
        entry = self.get(key)
        if entry is not None:
            # accepted drafts plus the bonus/recovery token
            entry.prefix_length += accepted + 1

    def close(self, key: SequenceKey) -> SequenceKey | None:
        """Remove the entry unless a newer generation owns it."""
        entry = self._entries.get(_registry_key(key))
        if entry is None or key.generation < entry.key.generation:
            return None
        del self._entries[_registry_key(key)]
        return entry.key

    def close_all(self, owner: str) -> tuple[SequenceKey, ...]:
        """Remove every entry still owned by one connection's session."""
        removed = tuple(
            entry.key for entry in self._entries.values() if entry.owner == owner
        )
        for key in removed:
            del self._entries[_registry_key(key)]
        return removed

    def _lookup(
        self, key: SequenceKey
    ) -> tuple[SpeculatorStatusCode, str, SequenceEntry | None]:
        entry = self._entries.get(_registry_key(key))
        if entry is None:
            return SpeculatorStatusCode.SEQUENCE_DESYNC, "unknown sequence", None
        if key.generation < entry.key.generation:
            return (
                SpeculatorStatusCode.STALE_GENERATION,
                f"generation {key.generation} superseded by {entry.key.generation}",
                None,
            )
        if key.generation > entry.key.generation:
            return (
                SpeculatorStatusCode.SEQUENCE_DESYNC,
                f"generation {key.generation} was never opened",
                None,
            )
        return SpeculatorStatusCode.OK, "", entry


def _registry_key(key: SequenceKey) -> tuple[str, int]:
    return key.verifier_instance_id, key.sequence_id


# ----------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------


class RemoteDraftServer:
    """Accepts verifier sessions and drives one adapter.

    ``start`` binds the endpoint and serves in background threads (one
    acceptor plus one per connection); ``stop`` tears everything down and
    joins. Adapter and registry access is serialized by one lock.
    """

    def __init__(
        self,
        adapter: StandaloneDraftAdapter,
        *,
        server_id: str | None = None,
        limits: RemoteServerLimits = DEFAULT_LIMITS,
        target_fingerprint: str = "",
        tokenizer_fingerprint: str = "",
        handshake_timeout_s: float = 30.0,
        clock=time.monotonic,
    ) -> None:
        self.adapter = adapter
        self.server_id = server_id or f"draft-{uuid.uuid4().hex[:12]}"
        self.limits = limits
        self.target_fingerprint = target_fingerprint
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.handshake_timeout_s = handshake_timeout_s
        self.registry = SequenceRegistry(clock)
        self.lock = threading.Lock()
        self._epoch = 0
        self._listener: Listener | None = None
        self._acceptor: threading.Thread | None = None
        self._workers: list[threading.Thread] = []
        self._connections: set[FramedConnection] = set()
        self._stopping = threading.Event()

    @property
    def endpoint(self) -> str | None:
        return self._listener.endpoint if self._listener is not None else None

    def start(self, endpoint: str) -> str:
        """Bind and serve in the background; returns the resolved endpoint."""
        if self._listener is not None:
            raise RuntimeError("server is already running")
        self._listener = listen(endpoint)
        self._stopping.clear()
        self._acceptor = threading.Thread(
            target=self._accept_loop, name="remote-draft-acceptor", daemon=True
        )
        self._acceptor.start()
        capabilities = self.adapter.capabilities()
        logger.info(
            "Remote draft server %s ready at %s: protocol %d.%d, methods %s, "
            "required features %s, transports %s, max_batch_size %d, "
            "max_model_len %d",
            self.server_id,
            self._listener.endpoint,
            PROTOCOL_MAJOR,
            PROTOCOL_MINOR,
            sorted(self.adapter.supported_methods()),
            list(capabilities.required_features),
            list(SUPPORTED_TRANSPORTS),
            self.limits.max_batch_size,
            self.limits.max_model_len,
        )
        return self._listener.endpoint

    def wait(self, timeout: float | None = None) -> bool:
        """Block until ``stop`` is called; returns False on timeout."""
        return self._stopping.wait(timeout)

    def stop(self) -> None:
        self._stopping.set()
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.close()
        for conn in list(self._connections):
            conn.close()
        if self._acceptor is not None:
            self._acceptor.join()
            self._acceptor = None
        for worker in self._workers:
            worker.join()
        self._workers.clear()

    def next_epoch(self) -> int:
        with self.lock:
            self._epoch += 1
            return self._epoch

    def _accept_loop(self) -> None:
        listener = self._listener
        assert listener is not None
        while not self._stopping.is_set():
            try:
                conn = listener.accept(timeout=_ACCEPT_POLL_S)
            except TransportTimeout:
                continue
            except TransportError:
                break
            self._connections.add(conn)
            worker = threading.Thread(
                target=self._serve_connection,
                args=(conn,),
                name="remote-draft-session",
                daemon=True,
            )
            self._workers.append(worker)
            worker.start()

    def _serve_connection(self, conn: FramedConnection) -> None:
        session = _ServerSession(self, conn)
        try:
            session.run()
        except Exception:
            logger.exception("Remote draft session %s crashed", session.session_id)
        finally:
            session.cleanup()
            conn.close()
            self._connections.discard(conn)


class _ServerSession:
    """One verifier connection: handshake, then a serial request loop."""

    def __init__(self, server: RemoteDraftServer, conn: FramedConnection) -> None:
        self.server = server
        self.conn = conn
        self.session_id = ""
        self.session_epoch = 0
        self.verifier_instance_id = ""
        self.num_speculative_tokens = 0
        self.inbound: dict[int, DataFrame] = {}
        self.last_batch_id = -1
        self.sequence_number = 0
        self.next_result_slot = 0
        self.cancelled_batches = 0

    def run(self) -> None:
        if not self._handshake():
            return
        while True:
            try:
                frame = self.conn.recv()
            except ConnectionClosed:
                return
            except TransportError as e:
                logger.warning(
                    "Session %s: unrecoverable frame error: %s", self.session_id, e
                )
                return
            if frame.kind == FRAME_CONTROL:
                self._handle_control(frame.body)
                continue
            try:
                data = decode_data_frame(frame.body)
            except TransportError as e:
                self._send(ErrorReply(SpeculatorStatusCode.PROTOCOL_ERROR, str(e)), 0)
                continue
            self.inbound[data.slot] = data

    def cleanup(self) -> None:
        if not self.session_id:
            return
        with self.server.lock:
            keys = self.server.registry.close_all(self.session_id)
            if keys:
                self.server.adapter.close_sequences(keys)

    # -- handshake -------------------------------------------------------

    def _handshake(self) -> bool:
        try:
            frame = self.conn.recv(timeout=self.server.handshake_timeout_s)
        except TransportError:
            return False
        if frame.kind != FRAME_CONTROL:
            self._send(
                ErrorReply(SpeculatorStatusCode.PROTOCOL_ERROR, "expected hello"), 0
            )
            return False
        try:
            envelope = decode_envelope(frame.body)
            message = decode_payload(envelope)
        except ProtocolError as e:
            self._send(ErrorReply(SpeculatorStatusCode.PROTOCOL_ERROR, str(e)), 0)
            return False
        if not isinstance(message, Hello):
            self._send(
                ErrorReply(
                    SpeculatorStatusCode.PROTOCOL_ERROR,
                    f"expected hello, got {envelope.message_type}",
                ),
                envelope.request_id,
            )
            return False
        reasons = self._hello_rejections(message)
        if reasons:
            self._send(
                ErrorReply(
                    SpeculatorStatusCode.UNSUPPORTED_FEATURE, "; ".join(reasons)
                ),
                envelope.request_id,
            )
            return False
        selected = next(
            t for t in message.supported_transports if t in SUPPORTED_TRANSPORTS
        )
        self.session_id = uuid.uuid4().hex
        self.session_epoch = self.server.next_epoch()
        self.verifier_instance_id = message.verifier_instance_id
        self.num_speculative_tokens = message.num_speculative_tokens
        adapter = self.server.adapter
        self._send(
            HelloAck(
                server_id=self.server.server_id,
                session_id=self.session_id,
                session_epoch=self.session_epoch,
                selected_transport=selected,
                capabilities=adapter.capabilities(),
                feature_schema=adapter.feature_schema(),
                limits=self.server.limits,
            ),
            envelope.request_id,
        )
        logger.info(
            "Session %s (epoch %d) opened for verifier %s over %s",
            self.session_id,
            self.session_epoch,
            self.verifier_instance_id,
            selected,
        )
        return True

    def _hello_rejections(self, hello: Hello) -> list[str]:
        server = self.server
        reasons: list[str] = []
        if hello.method not in server.adapter.supported_methods():
            reasons.append(
                f"method {hello.method!r} is not served by this adapter "
                f"(supported: {sorted(server.adapter.supported_methods())})"
            )
        if (
            server.target_fingerprint
            and hello.target_fingerprint != server.target_fingerprint
        ):
            reasons.append(
                f"target fingerprint {hello.target_fingerprint!r} does not match "
                f"the server's {server.target_fingerprint!r}"
            )
        if (
            server.tokenizer_fingerprint
            and hello.tokenizer_fingerprint != server.tokenizer_fingerprint
        ):
            reasons.append(
                f"tokenizer fingerprint {hello.tokenizer_fingerprint!r} does not "
                f"match the server's {server.tokenizer_fingerprint!r}"
            )
        expected_draft = server.adapter.draft_checkpoint_fingerprint()
        if expected_draft and hello.draft_checkpoint_fingerprint != expected_draft:
            reasons.append(
                f"draft checkpoint fingerprint "
                f"{hello.draft_checkpoint_fingerprint!r} does not match the "
                f"server's {expected_draft!r}"
            )
        if not any(t in SUPPORTED_TRANSPORTS for t in hello.supported_transports):
            reasons.append(
                f"no common transport: verifier offers "
                f"{list(hello.supported_transports)}, server supports "
                f"{list(SUPPORTED_TRANSPORTS)}"
            )
        return reasons

    # -- request loop ----------------------------------------------------

    def _send(self, message: Any, request_id: int) -> None:
        with contextlib.suppress(TransportError):
            self.conn.send_control(
                encode_message(
                    message, session_id=self.session_id, request_id=request_id
                )
            )

    def _send_frames(self, frames: list[DataFrame]) -> bool:
        try:
            for frame in frames:
                self.conn.send_data(frame)
        except TransportError:
            return False
        return True

    def _handle_control(self, body: bytes) -> None:
        try:
            envelope = decode_envelope(body)
            message = decode_payload(envelope)
        except ProtocolError as e:
            self._send(ErrorReply(SpeculatorStatusCode.PROTOCOL_ERROR, str(e)), 0)
            return
        request_id = envelope.request_id
        if envelope.session_id != self.session_id:
            self._send(
                ErrorReply(
                    SpeculatorStatusCode.PROTOCOL_ERROR,
                    f"session id {envelope.session_id!r} does not match this "
                    "connection",
                ),
                request_id,
            )
            return
        if isinstance(message, OpenSequence):
            self._open(message, request_id)
        elif isinstance(message, PrefillChunk):
            self._prefill(message, request_id)
        elif isinstance(message, AdvanceAndPropose):
            self._propose(message, request_id)
        elif isinstance(message, CloseSequence):
            self._close(message, request_id)
        elif isinstance(message, CancelBatch):
            self.cancelled_batches += 1
            logger.debug(
                "Session %s: cancel for batch %d (already served or never seen)",
                self.session_id,
                message.batch_id,
            )
        elif isinstance(message, Ping):
            with self.server.lock:
                active = self.server.registry.count(self.verifier_instance_id)
            self._send(
                Pong(nonce=message.nonce, queue_depth=0, active_sequences=active),
                request_id,
            )
        else:
            self._send(
                ErrorReply(
                    SpeculatorStatusCode.PROTOCOL_ERROR,
                    f"unexpected {envelope.message_type} from a verifier",
                ),
                request_id,
            )

    def _owns(self, key: SequenceKey) -> bool:
        return key.verifier_instance_id == self.verifier_instance_id

    def _ack(
        self,
        key: SequenceKey,
        request_id: int,
        status: SpeculatorStatusCode = SpeculatorStatusCode.OK,
        detail: str = "",
    ) -> None:
        self._send(SequenceAck(key=key, status=status, detail=detail), request_id)

    def _open(self, message: OpenSequence, request_id: int) -> None:
        key = message.key
        if not self._owns(key):
            self._ack(
                key,
                request_id,
                SpeculatorStatusCode.PROTOCOL_ERROR,
                "key belongs to another verifier",
            )
            return
        server = self.server
        with server.lock:
            registry = server.registry
            if (
                registry.get(key) is None
                and registry.count(self.verifier_instance_id)
                >= server.limits.max_sequences
            ):
                self._ack(
                    key,
                    request_id,
                    SpeculatorStatusCode.QUEUE_FULL,
                    f"max_sequences={server.limits.max_sequences} reached",
                )
                return
            outcome = registry.open(key, owner=self.session_id)
            if outcome.created:
                if outcome.replaced is not None:
                    server.adapter.close_sequences((outcome.replaced,))
                server.adapter.open_sequence(key)
        self._ack(key, request_id, outcome.status, outcome.detail)

    def _prefill(self, message: PrefillChunk, request_id: int) -> None:
        key = message.key
        schema = self.server.adapter.feature_schema()
        frames = self._take_frames(message.feature_slot, len(schema.slots))
        if not self._owns(key):
            self._ack(
                key,
                request_id,
                SpeculatorStatusCode.PROTOCOL_ERROR,
                "key belongs to another verifier",
            )
            return
        limits = self.server.limits
        if message.num_tokens > limits.max_feature_tokens:
            self._ack(
                key,
                request_id,
                SpeculatorStatusCode.PROTOCOL_ERROR,
                f"chunk of {message.num_tokens} tokens exceeds "
                f"max_feature_tokens={limits.max_feature_tokens}",
            )
            return
        if message.offset + message.num_tokens > limits.max_model_len:
            self._ack(
                key,
                request_id,
                SpeculatorStatusCode.OUT_OF_MEMORY,
                f"prefill to offset {message.offset + message.num_tokens} "
                f"exceeds max_model_len={limits.max_model_len}",
            )
            return
        problem = _frame_problems(frames, schema, message.num_tokens)
        if problem is not None:
            self._ack(key, request_id, SpeculatorStatusCode.PROTOCOL_ERROR, problem)
            return
        if _checksum(frames) != message.checksum:
            self._ack(
                key,
                request_id,
                SpeculatorStatusCode.SEQUENCE_DESYNC,
                "payload checksum does not match the chunk header",
            )
            return
        with self.server.lock:
            status, detail, apply = self.server.registry.prefill(
                key,
                offset=message.offset,
                num_tokens=message.num_tokens,
                checksum=message.checksum,
                is_final=message.is_final,
            )
            if apply:
                self.server.adapter.prefill(
                    key,
                    PrefillInputs(
                        offset=message.offset,
                        num_tokens=message.num_tokens,
                        is_final=message.is_final,
                        features=tuple(frames),
                    ),
                )
        self._ack(key, request_id, status, detail)

    def _propose(self, message: AdvanceAndPropose, request_id: int) -> None:
        schema = self.server.adapter.feature_schema()
        accepted = self.inbound.pop(message.accepted_counts_slot, None)
        recovery = self.inbound.pop(message.recovery_tokens_slot, None)
        frames = self._take_frames(message.feature_slot, len(schema.slots))
        if message.round_ids_slot is not None:
            self.inbound.pop(message.round_ids_slot, None)
        sequence_number = self.sequence_number
        self.sequence_number += 1

        def fail(status: SpeculatorStatusCode, detail: str) -> None:
            logger.warning(
                "Session %s: batch %d rejected: %s %s",
                self.session_id,
                message.batch_id,
                status.name,
                detail,
            )
            self._send(
                ProposalResponse(
                    batch_id=message.batch_id,
                    sequence_number=sequence_number,
                    status=status,
                ),
                request_id,
            )

        if message.batch_id <= self.last_batch_id:
            fail(
                SpeculatorStatusCode.ROUND_MISMATCH,
                f"batch id is not newer than {self.last_batch_id}",
            )
            return
        self.last_batch_id = message.batch_id
        num_rows = len(message.keys)
        if num_rows == 0 or num_rows > self.server.limits.max_batch_size:
            fail(
                SpeculatorStatusCode.PROTOCOL_ERROR,
                f"{num_rows} rows outside 1..{self.server.limits.max_batch_size}",
            )
            return
        if any(not self._owns(key) for key in message.keys):
            fail(SpeculatorStatusCode.PROTOCOL_ERROR, "key belongs to another verifier")
            return
        if accepted is None or recovery is None:
            fail(SpeculatorStatusCode.PROTOCOL_ERROR, "missing input frames")
            return
        try:
            accepted_counts = frame_to_ints(accepted)
        except TransportError as e:
            fail(SpeculatorStatusCode.PROTOCOL_ERROR, f"accepted_counts: {e}")
            return
        if len(accepted_counts) != num_rows or recovery.shape[:1] != (num_rows,):
            fail(SpeculatorStatusCode.PROTOCOL_ERROR, "input frames do not match rows")
            return
        num_tokens = self.num_speculative_tokens
        if any(not 0 <= count <= num_tokens for count in accepted_counts):
            fail(
                SpeculatorStatusCode.PROTOCOL_ERROR,
                f"accepted counts must be within 0..{num_tokens}",
            )
            return
        problem = _frame_problems(frames, schema, num_rows)
        if problem is not None:
            fail(SpeculatorStatusCode.PROTOCOL_ERROR, problem)
            return

        with self.server.lock:
            registry = self.server.registry
            row_statuses = [
                registry.begin_round(key, message.batch_id)[0] for key in message.keys
            ]
            max_model_len = self.server.limits.max_model_len
            for i, seq_key in enumerate(message.keys):
                if row_statuses[i] is not SpeculatorStatusCode.OK:
                    continue
                entry = registry.get(seq_key)
                if entry is not None and (
                    entry.prefix_length + accepted_counts[i] + 1 > max_model_len
                ):
                    row_statuses[i] = SpeculatorStatusCode.OUT_OF_MEMORY
            active = tuple(
                i
                for i, status in enumerate(row_statuses)
                if status is SpeculatorStatusCode.OK
            )
            if active:
                try:
                    result = self.server.adapter.advance_and_propose(
                        StandaloneProposalBatch(
                            batch_id=message.batch_id,
                            keys=message.keys,
                            active_rows=active,
                            num_speculative_tokens=num_tokens,
                            accepted_counts=accepted,
                            recovery_tokens=recovery,
                            features=tuple(frames),
                        )
                    )
                except Exception:
                    logger.exception(
                        "Session %s: adapter failed on batch %d",
                        self.session_id,
                        message.batch_id,
                    )
                    fail(SpeculatorStatusCode.INTERNAL_ERROR, "adapter raised")
                    return
                if not _result_well_formed(result, num_rows, num_tokens):
                    fail(
                        SpeculatorStatusCode.INTERNAL_ERROR,
                        "adapter returned a malformed result",
                    )
                    return
                for i in active:
                    row_statuses[i] = result.row_statuses[i]
                    if row_statuses[i] is SpeculatorStatusCode.OK:
                        registry.advance(message.keys[i], accepted_counts[i])
                tokens, valid = result.token_ids, result.valid_lengths
            else:
                tokens = ints_to_frame(
                    0,
                    [0] * (num_rows * num_tokens),
                    dtype="int64",
                    shape=(num_rows, num_tokens),
                )
                valid = ints_to_frame(0, [0] * num_rows)

        base = self.next_result_slot
        self.next_result_slot += _RESULT_FRAMES
        sent = self._send_frames(
            [
                msgspec.structs.replace(tokens, slot=base),
                msgspec.structs.replace(valid, slot=base + 1),
                ints_to_frame(base + 2, [int(s) for s in row_statuses]),
            ]
        )
        if sent:
            self._send(
                ProposalResponse(
                    batch_id=message.batch_id,
                    sequence_number=sequence_number,
                    result_slot=base,
                ),
                request_id,
            )

    def _close(self, message: CloseSequence, request_id: int) -> None:
        removed: list[SequenceKey] = []
        with self.server.lock:
            for key in message.keys:
                if not self._owns(key):
                    continue
                closed = self.server.registry.close(key)
                if closed is not None:
                    removed.append(closed)
            if removed:
                self.server.adapter.close_sequences(tuple(removed))
        for key in message.keys:
            if self._owns(key):
                self._ack(key, request_id)
            else:
                self._ack(
                    key,
                    request_id,
                    SpeculatorStatusCode.PROTOCOL_ERROR,
                    "key belongs to another verifier",
                )

    def _take_frames(self, base: int, count: int) -> list[DataFrame | None]:
        return [self.inbound.pop(base + i, None) for i in range(count)]


def _frame_problems(
    frames: list[DataFrame | None], schema: TargetFeatureSchema, num_rows: int
) -> str | None:
    for i, (spec, frame) in enumerate(zip(schema.slots, frames)):
        if frame is None:
            if spec.optional:
                continue
            return f"required feature slot {i} ({spec.kind}) is missing"
        if frame.shape[:1] != (num_rows,):
            return (
                f"feature slot {i} ({spec.kind}) has shape {frame.shape}, "
                f"expected {num_rows} rows"
            )
        if tuple(frame.shape[1:]) != tuple(spec.trailing_shape):
            return (
                f"feature slot {i} ({spec.kind}) has trailing shape "
                f"{tuple(frame.shape[1:])}, schema says {spec.trailing_shape}"
            )
        if frame.dtype != spec.dtype:
            return (
                f"feature slot {i} ({spec.kind}) has dtype {frame.dtype}, "
                f"schema says {spec.dtype}"
            )
    return None


def _result_well_formed(
    result: StandaloneProposalResult, num_rows: int, num_tokens: int
) -> bool:
    return (
        tuple(result.token_ids.shape) == (num_rows, num_tokens)
        and tuple(result.valid_lengths.shape) == (num_rows,)
        and len(result.row_statuses) == num_rows
        and result.token_ids.dtype in ("int32", "int64")
        and result.valid_lengths.dtype in ("int32", "int64")
    )


def _checksum(frames: list[DataFrame | None]) -> int:
    value = 0
    for frame in frames:
        if frame is not None:
            value = zlib.crc32(frame.payload, value)
    return value & 0x7FFFFFFF


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Serve a fake drafter; model adapters register here once they exist."""
    parser = argparse.ArgumentParser(
        description="Standalone speculator server for remote draft placement."
    )
    parser.add_argument(
        "--endpoint", required=True, help="tcp://host:port or unix:///path"
    )
    parser.add_argument(
        "--adapter", choices=("fake",), default="fake", help="drafter adapter to serve"
    )
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--target-fingerprint", default="")
    parser.add_argument("--tokenizer-fingerprint", default="")
    parser.add_argument(
        "--max-batch-size", type=int, default=DEFAULT_LIMITS.max_batch_size
    )
    parser.add_argument(
        "--max-model-len", type=int, default=DEFAULT_LIMITS.max_model_len
    )
    args = parser.parse_args(argv)
    limits = RemoteServerLimits(
        max_batch_size=args.max_batch_size,
        max_feature_tokens=DEFAULT_LIMITS.max_feature_tokens,
        max_sequences=DEFAULT_LIMITS.max_sequences,
        max_model_len=args.max_model_len,
    )
    server = RemoteDraftServer(
        FakeDraftAdapter(vocab_size=args.vocab_size),
        limits=limits,
        target_fingerprint=args.target_fingerprint,
        tokenizer_fingerprint=args.tokenizer_fingerprint,
    )
    endpoint = server.start(args.endpoint)
    # Parent processes read the resolved endpoint (tcp port 0 is common
    # in tests) from the first stdout line.
    print(endpoint, flush=True)
    try:
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
