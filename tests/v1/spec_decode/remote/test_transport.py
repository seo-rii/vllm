# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline stream transport: endpoint parsing, framing, deadlines, and
integer frame helpers."""

import socket
import struct
import threading

import pytest

from vllm.v1.spec_decode.remote.transport import (
    FRAME_CONTROL,
    FRAME_DATA,
    MAX_FRAME_BYTES,
    ConnectionClosed,
    DataFrame,
    TransportError,
    TransportTimeout,
    connect,
    decode_data_frame,
    frame_to_ints,
    ints_to_frame,
    listen,
    parse_endpoint,
)


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


def test_parse_unix_endpoint():
    if not hasattr(socket, "AF_UNIX"):
        pytest.skip("unix sockets unavailable")
    assert parse_endpoint("unix:///run/vllm/draft.sock") == (
        socket.AF_UNIX,
        "/run/vllm/draft.sock",
    )


@pytest.mark.parametrize(
    "endpoint", ["http://127.0.0.1:1", "tcp://127.0.0.1", "unix://", "draft"]
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
