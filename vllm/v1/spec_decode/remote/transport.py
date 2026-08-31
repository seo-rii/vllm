# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy-based stream transport shared by the control and data planes.

This is the ``inline`` transport: control frames carry encoded
MessageEnvelopes and data frames carry raw tensor bytes tagged with a slot
index, multiplexed on one stream connection. It is dependency-free and
correct rather than fast; zero-copy transports (CUDA IPC, pinned-host ring)
replace only the data-plane half and keep the control plane as is.

Frame layout: ``kind:u8 | length:u32 big-endian | body``. Data-plane
payloads are in native byte order, which is acceptable because remote
placement is same-host only.
"""

import array
import contextlib
import math
import os
import socket
import struct
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

import msgspec

INLINE_TRANSPORT = "inline"

FRAME_CONTROL = 0x01
FRAME_DATA = 0x02

MAX_FRAME_BYTES = 1 << 30

_HEADER = struct.Struct("!BI")
_RECV_CHUNK = 1 << 16
_RECV_MAX = 1 << 24
# torch dtype name -> (array typecode, itemsize) for the torch-free integer
# helpers used on the server side.
_INT_TYPECODES = {"int32": ("i", 4), "int64": ("q", 8)}


class TransportError(Exception):
    """Failure on the stream connection."""


class TransportTimeout(TransportError):
    """No complete frame arrived before the deadline."""


class ConnectionClosed(TransportError):
    """The peer closed the connection or the socket failed."""


class DataFrame(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    """One tensor on the data plane.

    ``dtype`` is the torch dtype name (``"int32"``, ``"bfloat16"``) and
    ``payload`` holds the contiguous row-major bytes in native byte order.
    """

    slot: int
    dtype: str
    shape: tuple[int, ...]
    payload: bytes


@dataclass(frozen=True)
class Frame:
    """One raw frame off the wire."""

    kind: int
    body: bytes


_frame_encoder = msgspec.msgpack.Encoder()
_frame_decoder = msgspec.msgpack.Decoder(DataFrame)


def decode_data_frame(body: bytes) -> DataFrame:
    """Decode the body of a FRAME_DATA frame."""
    try:
        return _frame_decoder.decode(body)
    except msgspec.DecodeError as e:
        raise TransportError(f"malformed data frame: {e}") from e


def ints_to_frame(
    slot: int,
    values: Sequence[int],
    *,
    dtype: str = "int32",
    shape: tuple[int, ...] | None = None,
) -> DataFrame:
    """Build an integer DataFrame without torch."""
    typecode, itemsize = _int_typecode(dtype)
    if shape is None:
        shape = (len(values),)
    elif math.prod(shape) != len(values):
        raise ValueError(f"shape {shape} does not hold {len(values)} values")
    data = array.array(typecode, values)
    if data.itemsize != itemsize:
        raise TransportError(f"array typecode {typecode!r} is not {dtype} here")
    return DataFrame(slot=slot, dtype=dtype, shape=tuple(shape), payload=data.tobytes())


def frame_to_ints(frame: DataFrame) -> list[int]:
    """Read an integer DataFrame without torch (row-major, flattened)."""
    typecode, _ = _int_typecode(frame.dtype)
    validate_integer_frame(frame)
    data = array.array(typecode)
    data.frombytes(frame.payload)
    return data.tolist()


def validate_integer_frame(frame: DataFrame) -> None:
    """Validate an integer frame's dtype and payload size without decoding it."""
    _, itemsize = _int_typecode(frame.dtype)
    expected = math.prod(frame.shape) * itemsize
    if len(frame.payload) != expected:
        raise TransportError(
            f"data frame {frame.slot} has {len(frame.payload)} payload bytes, "
            f"expected {expected} for {frame.dtype} shape {frame.shape}"
        )


def _int_typecode(dtype: str) -> tuple[str, int]:
    try:
        return _INT_TYPECODES[dtype]
    except KeyError:
        raise TransportError(
            f"dtype {dtype!r} is not an integer frame dtype ({sorted(_INT_TYPECODES)})"
        ) from None


def parse_endpoint(endpoint: str) -> tuple[int, tuple[str, int] | str]:
    """Parse ``tcp://host:port`` or ``unix:///path`` into a socket address."""
    parts = urlsplit(endpoint)
    if parts.scheme == "tcp":
        if parts.hostname is None or parts.port is None:
            raise ValueError(f"tcp endpoint needs host and port: {endpoint!r}")
        return socket.AF_INET, (parts.hostname, parts.port)
    if parts.scheme == "unix":
        path = parts.path or parts.netloc
        if not path:
            raise ValueError(f"unix endpoint needs a path: {endpoint!r}")
        family = getattr(socket, "AF_UNIX", None)
        if family is None:
            raise ValueError("unix sockets are not supported on this platform")
        return family, path
    raise ValueError(
        f"unsupported endpoint scheme {parts.scheme!r} in {endpoint!r}; "
        "use tcp://host:port or unix:///path"
    )


class FramedConnection:
    """Blocking, single-reader framed stream with per-call deadlines.

    A timeout leaves any partially received frame buffered, so the stream
    never desynchronizes across retries. ``close`` may be called from
    another thread to unblock a pending ``recv``.
    """

    def __init__(self, sock: socket.socket) -> None:
        sock.settimeout(None)
        self._sock = sock
        self._buffer = bytearray()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def send_control(self, data: bytes) -> None:
        self._send(FRAME_CONTROL, data)

    def send_data(self, frame: DataFrame) -> None:
        self._send(FRAME_DATA, _frame_encoder.encode(frame))

    def _send(self, kind: int, body: bytes) -> None:
        if self._closed:
            raise ConnectionClosed("connection is closed")
        if len(body) > MAX_FRAME_BYTES:
            raise TransportError(
                f"frame of {len(body)} bytes exceeds {MAX_FRAME_BYTES}"
            )
        try:
            self._sock.sendall(_HEADER.pack(kind, len(body)))
            if body:
                self._sock.sendall(body)
        except OSError as e:
            raise ConnectionClosed(f"send failed: {e}") from e

    def recv(self, timeout: float | None = None) -> Frame:
        """Receive one frame, waiting at most ``timeout`` seconds."""
        deadline = None if timeout is None else time.monotonic() + timeout
        self._fill(_HEADER.size, deadline)
        kind, length = _HEADER.unpack_from(self._buffer)
        if kind not in (FRAME_CONTROL, FRAME_DATA):
            raise TransportError(f"unknown frame kind {kind:#x}")
        if length > MAX_FRAME_BYTES:
            raise TransportError(f"frame length {length} exceeds {MAX_FRAME_BYTES}")
        end = _HEADER.size + length
        self._fill(end, deadline)
        body = bytes(self._buffer[_HEADER.size : end])
        del self._buffer[:end]
        return Frame(kind, body)

    def _fill(self, size: int, deadline: float | None) -> None:
        while len(self._buffer) < size:
            if self._closed:
                raise ConnectionClosed("connection is closed")
            remaining = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TransportTimeout("timed out waiting for a frame")
            want = min(max(_RECV_CHUNK, size - len(self._buffer)), _RECV_MAX)
            try:
                # settimeout can also fail if another thread closed the
                # socket to unblock this reader.
                self._sock.settimeout(remaining)
                chunk = self._sock.recv(want)
            except TimeoutError:
                raise TransportTimeout("timed out waiting for a frame") from None
            except OSError as e:
                raise ConnectionClosed(f"recv failed: {e}") from e
            if not chunk:
                raise ConnectionClosed("peer closed the connection")
            self._buffer += chunk

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()


class Listener:
    """Bound server socket handing out FramedConnections."""

    def __init__(
        self, sock: socket.socket, endpoint: str, unix_path: str | None
    ) -> None:
        self._sock = sock
        self._unix_path = unix_path
        self.endpoint = endpoint
        """Resolved endpoint, with the actual port for ``tcp://host:0``."""

    def accept(self, timeout: float | None = None) -> FramedConnection:
        self._sock.settimeout(timeout)
        try:
            conn, _ = self._sock.accept()
        except TimeoutError:
            raise TransportTimeout("no connection before the deadline") from None
        except OSError as e:
            raise ConnectionClosed(f"listener closed: {e}") from e
        _tune(conn)
        return FramedConnection(conn)

    def close(self) -> None:
        try:
            self._sock.close()
        finally:
            if self._unix_path is not None:
                _unlink_quietly(self._unix_path)


def _tune(sock: socket.socket) -> None:
    if sock.family == socket.AF_INET:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def _unlink_quietly(path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


def connect(endpoint: str, timeout: float | None = None) -> FramedConnection:
    """Open a client connection to ``endpoint``."""
    family, address = parse_endpoint(endpoint)
    try:
        if family == socket.AF_INET:
            assert isinstance(address, tuple)
            sock = socket.create_connection(address, timeout=timeout)
        else:
            assert isinstance(address, str)
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(address)
    except TimeoutError:
        raise TransportTimeout(f"connect to {endpoint} timed out") from None
    except OSError as e:
        raise ConnectionClosed(f"connect to {endpoint} failed: {e}") from e
    _tune(sock)
    return FramedConnection(sock)


def listen(endpoint: str, backlog: int = 16) -> Listener:
    """Bind ``endpoint`` and start listening."""
    family, address = parse_endpoint(endpoint)
    sock = socket.socket(family, socket.SOCK_STREAM)
    unix_path: str | None = None
    try:
        if family == socket.AF_INET:
            if sys.platform != "win32":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(address)
            host, port = sock.getsockname()[:2]
            resolved = f"tcp://{host}:{port}"
        else:
            assert isinstance(address, str)
            unix_path = address
            _unlink_quietly(unix_path)
            sock.bind(unix_path)
            resolved = f"unix://{unix_path}"
        sock.listen(backlog)
    except OSError as e:
        sock.close()
        raise TransportError(f"cannot listen on {endpoint}: {e}") from e
    return Listener(sock, resolved, unix_path)
