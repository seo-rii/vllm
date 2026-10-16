# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline stream transport: endpoint parsing, framing, deadlines, and
integer frame helpers."""

import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, call

import msgspec
import pytest

from vllm.v1.spec_decode.remote import transport
from vllm.v1.spec_decode.remote.cuda_ipc import CudaIpcDescriptor
from vllm.v1.spec_decode.remote.transport import (
    FRAME_CONTROL,
    FRAME_DATA,
    MAX_FRAME_BYTES,
    ConnectionClosed,
    DataFrame,
    FramedConnection,
    TransportError,
    TransportTimeout,
    connect,
    decode_data_frame,
    frame_to_ints,
    ints_to_frame,
    is_local_endpoint,
    listen,
    parse_endpoint,
    validate_frame,
)


def ipc_descriptor(**overrides):
    fields = dict(
        buffer_id=1,
        generation=1,
        device_uuid="GPU-00000000-0000-0000-0000-000000000000",
        memory_handle=b"m" * 64,
        ready_event_handle=b"e" * 64,
        nbytes=8,
        capacity_bytes=8,
    )
    fields.update(overrides)
    return CudaIpcDescriptor(**fields)


def test_ipc_descriptor_round_trip_and_inline_backward_compatibility():
    old = {"slot": 0, "dtype": "int32", "shape": [1], "payload": b"\0" * 4}
    assert decode_data_frame(msgspec.msgpack.encode(old)).cuda_ipc is None
    frame = DataFrame(1, "float32", (1, 2), b"", ipc_descriptor())
    decoded = decode_data_frame(msgspec.msgpack.encode(frame))
    validate_frame(decoded)
    assert decoded == frame


@pytest.mark.parametrize(
    "fields",
    [
        {"payload": b"\0" * 8},
        {"dtype": "int32"},
        {"shape": (2, 2)},
        {"cuda_ipc": ipc_descriptor(memory_handle=b"short")},
        {"cuda_ipc": ipc_descriptor(capacity_bytes=4)},
        {"cuda_ipc": ipc_descriptor(generation=0)},
    ],
)
def test_invalid_ipc_frames_are_rejected_before_device_access(fields):
    frame = dict(
        slot=1, dtype="float32", shape=(1, 2), payload=b"", cuda_ipc=ipc_descriptor()
    )
    frame.update(fields)
    with pytest.raises(TransportError):
        validate_frame(DataFrame(**frame))


def test_ipc_descriptor_disallows_unrecognized_fields():
    frame = DataFrame(1, "float32", (1, 2), b"", ipc_descriptor())
    raw = msgspec.msgpack.decode(msgspec.msgpack.encode(frame))
    raw["cuda_ipc"]["python_object"] = "not allowed"
    with pytest.raises(TransportError, match="unknown field"):
        decode_data_frame(msgspec.msgpack.encode(raw))


@pytest.fixture
def pair():
    listener = listen("tcp://127.0.0.1:0")
    client = connect(listener.endpoint, timeout=5.0)
    server = listener.accept(timeout=5.0)
    yield client, server
    client.close()
    server.close()
    listener.close()


def test_parse_tcp_endpoint():
    assert parse_endpoint("tcp://127.0.0.1:5555") == (
        socket.AF_INET,
        ("127.0.0.1", 5555),
    )


def test_parse_ipv6_tcp_endpoint():
    assert parse_endpoint("tcp://[::1]:5555") == (socket.AF_INET6, ("::1", 5555))


def test_ipv6_listener_round_trip():
    try:
        listener = listen("tcp://[::1]:0")
    except TransportError:
        pytest.skip("IPv6 loopback unavailable")
    try:
        assert listener.endpoint.startswith("tcp://[::1]:")
        client = connect(listener.endpoint, timeout=5.0)
        server = listener.accept(timeout=5.0)
        client.send_control(b"v6")
        assert server.recv(timeout=5.0).body == b"v6"
        client.close()
        server.close()
    finally:
        listener.close()


@pytest.mark.parametrize(
    ("endpoint", "local"),
    [
        ("tcp://127.0.0.1:1", True),
        ("tcp://[::1]:1", True),
        ("tcp://localhost:1", True),
        ("tcp://0.0.0.0:1", False),
        ("tcp://10.0.0.5:1", False),
        ("tcp://draft-host:1", False),
        ("unix:///run/vllm/draft.sock", True),
    ],
)
def test_is_local_endpoint(endpoint, local):
    if endpoint.startswith("unix") and not hasattr(socket, "AF_UNIX"):
        pytest.skip("unix sockets unavailable")
    assert is_local_endpoint(endpoint) is local


def test_parse_unix_endpoint():
    if not hasattr(socket, "AF_UNIX"):
        pytest.skip("unix sockets unavailable")
    assert parse_endpoint("unix:///run/vllm/draft.sock") == (
        socket.AF_UNIX,
        "/run/vllm/draft.sock",
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1:1",
        "tcp://127.0.0.1",
        "tcp://127.0.0.1:99999",
        "tcp://:5555",
        "unix://",
        "draft",
    ],
)
def test_parse_rejects_bad_endpoints(endpoint):
    with pytest.raises(ValueError):
        parse_endpoint(endpoint)


def test_listen_resolves_ephemeral_port():
    listener = listen("tcp://127.0.0.1:0")
    try:
        assert listener.endpoint.startswith("tcp://127.0.0.1:")
        assert not listener.endpoint.endswith(":0")
    finally:
        listener.close()


def test_closed_listener_accept_reports_connection_closed():
    listener = listen("tcp://127.0.0.1:0")
    listener.close()
    with pytest.raises(ConnectionClosed, match="listener closed"):
        listener.accept(timeout=0.1)


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="unix sockets unavailable")
def test_unix_listener_refuses_live_socket(tmp_path):
    path = tmp_path / "draft.sock"
    listener = listen(f"unix://{path}")
    try:
        with pytest.raises(TransportError, match="cannot listen"):
            listen(listener.endpoint)
        client = connect(listener.endpoint, timeout=1.0)
        server = listener.accept(timeout=1.0)
        try:
            client.send_control(b"still listening")
            assert server.recv(timeout=1.0).body == b"still listening"
        finally:
            client.close()
            server.close()
    finally:
        listener.close()
    assert not path.exists()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="unix sockets unavailable")
def test_unix_listener_preserves_existing_file(tmp_path):
    path = tmp_path / "draft.sock"
    path.write_text("keep")
    with pytest.raises(TransportError, match="cannot listen"):
        listen(f"unix://{path}")
    assert path.read_text() == "keep"


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="unix sockets unavailable")
@pytest.mark.parametrize(
    "replace_live", [True, False], ids=["live-path", "closed-path"]
)
def test_unix_close_preserves_replacement_listener(tmp_path, replace_live):
    path = tmp_path / "draft.sock"
    listener = listen(f"unix://{path}")
    try:
        if replace_live:
            path.unlink()
        else:
            listener.close()
        replacement = listen(listener.endpoint)
        try:
            listener.close()
            client = connect(replacement.endpoint, timeout=1.0)
            server = replacement.accept(timeout=1.0)
            client.close()
            server.close()
        finally:
            replacement.close()
    finally:
        listener.close()
    assert not path.exists()


def test_int_frame_roundtrip():
    frame = ints_to_frame(3, [1, -2, 3, 4], dtype="int64", shape=(2, 2))
    assert frame.slot == 3
    assert frame.shape == (2, 2)
    assert frame_to_ints(frame) == [1, -2, 3, 4]


def test_int_frame_rejects_shape_and_dtype_mismatch():
    with pytest.raises(ValueError):
        ints_to_frame(0, [1, 2, 3], shape=(2,))
    with pytest.raises(TransportError, match="not an integer"):
        ints_to_frame(0, [1], dtype="bfloat16")
    truncated = DataFrame(slot=0, dtype="int32", shape=(2,), payload=b"\x00" * 7)
    with pytest.raises(TransportError, match="payload bytes"):
        frame_to_ints(truncated)


@pytest.mark.parametrize(
    ("dtype", "itemsize"),
    [
        ("int32", 4),
        ("int64", 8),
        ("float16", 2),
        ("bfloat16", 2),
        ("float32", 4),
        ("float64", 8),
    ],
)
def test_frame_validation_checks_payload_size(dtype, itemsize):
    validate_frame(DataFrame(0, dtype, (2, 3), bytes(6 * itemsize)))
    for length in (6 * itemsize - 1, 6 * itemsize + 1):
        with pytest.raises(TransportError, match="payload bytes"):
            validate_frame(DataFrame(0, dtype, (2, 3), bytes(length)))


@pytest.mark.parametrize(
    ("frame", "match"),
    [
        (DataFrame(0, "unknown", (1,), b"\x00"), "unknown dtype"),
        (DataFrame(0, "int32", (-1, -1), bytes(4)), "negative shape"),
        (DataFrame(0, "float32", (0, 3), bytes(4)), "payload bytes"),
    ],
)
def test_frame_validation_rejects_invalid_metadata(frame, match):
    with pytest.raises(TransportError, match=match):
        validate_frame(frame)


def test_control_and_data_frames_roundtrip(pair):
    client, server = pair
    client.send_control(b"hello")
    client.send_data(ints_to_frame(7, [5, 6], dtype="int32"))
    client.send_control(b"")

    frame = server.recv(timeout=5.0)
    assert (frame.kind, frame.body) == (FRAME_CONTROL, b"hello")
    frame = server.recv(timeout=5.0)
    assert frame.kind == FRAME_DATA
    data = decode_data_frame(frame.body)
    assert (data.slot, frame_to_ints(data)) == (7, [5, 6])
    frame = server.recv(timeout=5.0)
    assert (frame.kind, frame.body) == (FRAME_CONTROL, b"")


@pytest.mark.parametrize("drain_incoming", [False, True])
@pytest.mark.timeout(5)
def test_send_deadline_closes_connection_when_peer_does_not_read(drain_incoming):
    sock, peer = socket.socketpair()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    connection = FramedConnection(sock)
    try:
        started = time.monotonic()
        with pytest.raises(TransportTimeout, match="sending a frame"):
            connection.send_data(
                DataFrame(0, "int32", (1 << 18,), bytes(1 << 20)),
                deadline=started + 0.05,
                drain_incoming=drain_incoming,
            )
        assert time.monotonic() - started < 2.0
        assert connection.closed
        with pytest.raises(ConnectionClosed):
            connection.send_control(b"cannot reuse partial write")
    finally:
        connection.close()
        peer.close()


def test_send_frames_share_absolute_deadline(monkeypatch):
    now = [10.0]
    clock = Mock()
    clock.monotonic.side_effect = lambda: now[0]
    monkeypatch.setattr(transport, "time", clock)
    sock = Mock(spec=socket.socket)

    def sendall(data):
        now[0] += 2.0

    sock.sendall.side_effect = sendall
    connection = FramedConnection(sock)
    connection.send_control(b"first", deadline=15.0)
    with pytest.raises(TransportTimeout):
        connection.send_data(ints_to_frame(0, [1]), deadline=15.0)
    assert sock.settimeout.call_args_list == [
        call(None),
        call(5.0),
        call(3.0),
        call(1.0),
    ]
    assert sock.sendall.call_count == 3
    assert connection.closed


@pytest.mark.parametrize("drain_incoming", [False, True])
def test_expired_deadline_before_frame_keeps_stream_usable(pair, drain_incoming):
    # Nothing reached the wire, so frame boundaries are intact and the
    # connection must not be torn down.
    client, server = pair
    with pytest.raises(TransportTimeout, match="before sending"):
        client.send_control(
            b"too late", deadline=time.monotonic() - 1.0, drain_incoming=drain_incoming
        )
    assert not client.closed
    client.send_control(b"next")
    assert server.recv(timeout=1.0).body == b"next"


@pytest.mark.timeout(10)
def test_duplex_send_buffers_large_reply_before_peer_reads_request():
    sockets = socket.socketpair()
    for sock in sockets:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    client, server = (FramedConnection(sock) for sock in sockets)
    request, reply = b"q" * (1 << 20), b"a" * (1 << 20)

    def exchange():
        server.send_control(reply, deadline=time.monotonic() + 5)
        return server.recv(timeout=5).body

    with ThreadPoolExecutor(max_workers=1) as pool:
        received = pool.submit(exchange)
        try:
            client.send_control(
                request, deadline=time.monotonic() + 5, drain_incoming=True
            )
            assert client.recv(timeout=5).body == reply
            assert received.result(timeout=5) == request
        finally:
            client.close()
            server.close()


@pytest.mark.parametrize("close_at", ["register", "select"])
@pytest.mark.timeout(5)
def test_duplex_send_close_races_raise_connection_closed(monkeypatch, close_at):
    sock, peer = socket.socketpair()
    connection = FramedConnection(sock)
    selector = transport.selectors.DefaultSelector()
    original = getattr(selector, close_at)

    def close_then_call(*args, **kwargs):
        connection.close()
        return original(*args, **kwargs)

    monkeypatch.setattr(selector, close_at, close_then_call)
    monkeypatch.setattr(transport.selectors, "DefaultSelector", lambda: selector)
    try:
        with pytest.raises(ConnectionClosed):
            connection.send_control(b"request", drain_incoming=True)
    finally:
        connection.close()
        peer.close()


def test_send_without_deadline_resets_previous_receive_timeout(pair):
    client, server = pair
    with pytest.raises(TransportTimeout):
        client.recv(timeout=0.01)
    client.send_control(b"no send deadline")
    assert client._sock.gettimeout() is None
    assert server.recv(timeout=1.0).body == b"no send deadline"


def test_send_failure_closes_connection():
    sock = Mock(spec=socket.socket)
    sock.sendall.side_effect = BrokenPipeError("peer disconnected")
    connection = FramedConnection(sock)
    with pytest.raises(ConnectionClosed, match="send failed"):
        connection.send_control(b"cannot deliver")
    assert connection.closed
    sock.close.assert_called_once()


def test_partial_frame_survives_timeout():
    listener = listen("tcp://127.0.0.1:0")
    _, address = parse_endpoint(listener.endpoint)
    raw = socket.create_connection(address, timeout=5.0)
    server = listener.accept(timeout=5.0)
    try:
        raw.sendall(struct.pack("!BI", FRAME_CONTROL, 4) + b"ab")
        with pytest.raises(TransportTimeout):
            server.recv(timeout=0.05)
        raw.sendall(b"cd")
        frame = server.recv(timeout=5.0)
        assert frame.body == b"abcd"
    finally:
        raw.close()
        server.close()
        listener.close()


@pytest.mark.parametrize(
    ("header", "match"),
    [
        (struct.pack("!BI", 0x7F, 0), "unknown frame kind"),
        (struct.pack("!BI", FRAME_CONTROL, MAX_FRAME_BYTES + 1), "exceeds"),
    ],
    ids=["unknown-kind", "oversized"],
)
def test_bad_frame_headers_rejected(header, match):
    listener = listen("tcp://127.0.0.1:0")
    _, address = parse_endpoint(listener.endpoint)
    raw = socket.create_connection(address, timeout=5.0)
    server = listener.accept(timeout=5.0)
    try:
        raw.sendall(header)
        with pytest.raises(TransportError, match=match):
            server.recv(timeout=5.0)
        # The poisoned header stays buffered: such a stream cannot resync
        # and every further read reports the same error.
        with pytest.raises(TransportError, match=match):
            server.recv(timeout=5.0)
    finally:
        raw.close()
        server.close()
        listener.close()


def test_peer_close_raises_connection_closed(pair):
    client, server = pair
    client.close()
    with pytest.raises(ConnectionClosed):
        server.recv(timeout=5.0)
    with pytest.raises(ConnectionClosed):
        client.send_control(b"late")


def test_close_unblocks_pending_recv(pair):
    _, server = pair
    outcome: list[BaseException | None] = []

    def reader():
        try:
            server.recv()
        except BaseException as e:  # noqa: BLE001 - recorded for the assert
            outcome.append(e)
        else:
            outcome.append(None)

    thread = threading.Thread(target=reader)
    thread.start()
    server.close()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert isinstance(outcome[0], ConnectionClosed)
