# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One CPU coordinator and a bounded, process-owned native draft TP group."""

import contextlib
import json
import math
import multiprocessing
import os
import signal
import socket
import tempfile
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import msgspec

from vllm.logger import init_logger
from vllm.v1.spec_decode.remote.errors import (
    CompletionUnknownError,
    CudaRuntimeError,
)
from vllm.v1.spec_decode.remote.protocol import SequenceKey
from vllm.v1.spec_decode.remote.runtime.token_state import (
    TokenForwardRow,
    TokenStepExecutor,
)
from vllm.v1.spec_decode.remote.transport import (
    FRAME_CONTROL,
    FramedConnection,
    TransportError,
    TransportTimeout,
)

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig

_MAX_COMMAND_BYTES = 16 << 20
_POLL_S = 0.1
logger = init_logger(__name__)
TPOperation = Literal["forward", "warmup", "release", "capacity", "close"]


class TPCommand(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    command_id: int
    operation: TPOperation
    rows: tuple[TokenForwardRow, ...] = ()
    keys: tuple[SequenceKey, ...] = ()
    num_blocks: int = 0


class TPGeometry(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    num_blocks: int
    block_size: int
    max_num_reqs: int
    max_num_tokens: int
    world_size: int
    device: str
    device_uuid: str = ""


class TPFailure(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    kind: str
    detail: str
    traceback: str
    cuda_error_code: int | None = None


class TPReply(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    command_id: int
    rank: int
    operation: Literal["ready", "forward", "warmup", "release", "capacity", "close"]
    geometry: TPGeometry | None = None
    tokens: tuple[int | None, ...] = ()
    error: TPFailure | None = None


class TPWorkerConnection:
    """Single-owner typed channel; the worker never receives arbitrary methods."""

    def __init__(self, sock: socket.socket, *, timeout_s: float):
        self.timeout_s = timeout_s
        self.connection = FramedConnection(sock)
        self.connection.set_receive_limits(
            max_control_bytes=_MAX_COMMAND_BYTES,
            max_data_bytes=0,
            max_buffered_bytes=_MAX_COMMAND_BYTES + 5,
            partial_frame_timeout=timeout_s,
        )
        self._encoder = msgspec.msgpack.Encoder()
        self._commands = msgspec.msgpack.Decoder(TPCommand)
        self._replies = msgspec.msgpack.Decoder(TPReply)

    def send(self, message: TPCommand | TPReply, deadline: float) -> None:
        body = self._encoder.encode(message)
        if len(body) > _MAX_COMMAND_BYTES:
            raise TransportError("draft TP command exceeds the byte budget")
        self.connection.send_control(body, deadline=deadline)

    def receive_command(self) -> TPCommand:
        frame = self.connection.recv()
        if frame.kind != FRAME_CONTROL:
            raise TransportError("draft TP requires control frames")
        return self._commands.decode(frame.body)

    def receive_reply(self, timeout: float) -> TPReply:
        frame = self.connection.recv(timeout)
        if frame.kind != FRAME_CONTROL:
            raise TransportError("draft TP requires control frames")
        return self._replies.decode(frame.body)

    def close(self) -> None:
        self.connection.close()


def _failure(error: BaseException) -> TPFailure:
    code = None
    seen: set[int] = set()
    cause: BaseException | None = error
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, CudaRuntimeError):
            code = cause.cuda_error_code
            break
        cause = cause.__cause__ or cause.__context__
    return TPFailure(
        type(error).__name__,
        str(error)[:4096],
        "".join(traceback.format_exception(error))[-16384:],
        code,
    )


def _run_process(
    sock: socket.socket,
    death: Connection,
    rank: int,
    timeout_s: float,
    target: Callable,
    args: tuple,
) -> None:
    # The owner handles group SIGINT and closes ranks through the typed channel.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    exiting = threading.Event()

    def parent_died() -> None:
        try:
            death.recv_bytes(1)
        except (EOFError, OSError):
            # A rank may be inside CUDA initialization or a collective. Exiting
            # the process is safe; issuing more CUDA cleanup is not.
            if not exiting.is_set():
                os._exit(1)

    threading.Thread(target=parent_died, name="draft-tp-parent", daemon=True).start()
    channel = TPWorkerConnection(sock, timeout_s=timeout_s)
    try:
        target(channel, rank, *args)
    except BaseException as error:
        with contextlib.suppress(Exception):
            channel.send(
                TPReply(0, rank, "ready", error=_failure(error)),
                time.monotonic() + timeout_s,
            )
        os._exit(1)
    finally:
        exiting.set()
        channel.close()
        death.close()
    # The owner completed explicit cleanup. Interpreter finalization can invoke
    # more native destructors and outlive the acknowledged close deadline.
    os._exit(0)


@dataclass
class _Rank:
    process: BaseProcess
    socket: socket.socket
    channel: TPWorkerConnection
    death: Connection
    pid: int | None = None
    exit_code: int | None = None
    closed: bool = False

    def alive(self) -> bool:
        return not self.closed and self.process.is_alive()


class TPProcessGroup:
    """Bounded all-rank RPC and ownership, also usable with real CPU groups."""

    def __init__(
        self,
        world_size: int,
        worker_target: Callable,
        *,
        worker_args: tuple = (),
        timeout_s: float = 30,
        startup_timeout_s: float = 600,
    ):
        if type(world_size) is not int or world_size < 2:
            raise ValueError("draft TP requires at least two ranks")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (timeout_s, startup_timeout_s)
        ):
            raise ValueError("draft TP timeouts must be finite and positive")
        self.timeout_s = timeout_s
        self._ranks: list[_Rank] = []
        self._rpc_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._reap_lock = threading.Lock()
        self._closing = threading.Event()
        self._closed = threading.Event()
        self._failed = threading.Event()
        self._command_id = 0
        context = multiprocessing.get_context("spawn")
        deadline = time.monotonic() + startup_timeout_s
        try:
            for rank in range(world_size):
                parent = child = death_reader = death_writer = process = None
                registered = False
                failure = None
                try:
                    parent, child = socket.socketpair()
                    death_reader, death_writer = context.Pipe(duplex=False)
                    channel = TPWorkerConnection(parent, timeout_s=timeout_s)
                    process = context.Process(
                        target=_run_process,
                        args=(
                            child,
                            death_reader,
                            rank,
                            timeout_s,
                            worker_target,
                            worker_args,
                        ),
                        name=f"draft-tp-{rank}",
                        daemon=True,
                    )
                    owner = _Rank(process, parent, channel, death_writer)
                    self._ranks.append(owner)
                    registered = True
                    try:
                        process.start()
                    finally:
                        owner.pid = process.pid
                except BaseException as error:
                    failure = error
                finally:
                    cleanup_errors = []
                    for resource in (
                        child,
                        death_reader,
                        None if registered else parent,
                        None if registered else death_writer,
                        None if registered else process,
                    ):
                        if resource is not None:
                            try:
                                resource.close()
                            except BaseException as error:
                                cleanup_errors.append(error)
                    if cleanup_errors:
                        if failure is not None:
                            cleanup_errors.insert(0, failure)
                        raise BaseExceptionGroup(
                            "draft TP startup resource cleanup failed", cleanup_errors
                        )
                if failure is not None:
                    raise failure
            self.ready = self._collect(0, "ready", deadline)
        except BaseException as error:
            try:
                self._abort()
            except BaseException as cleanup_error:
                error.add_note(f"draft TP startup cleanup failed: {cleanup_error}")
            raise CompletionUnknownError("draft TP startup failed") from error

    def worker_status(self) -> tuple[dict[str, int | bool | None], ...]:
        """Retain process outcomes after closing their native sentinel handles."""
        with self._reap_lock:
            return tuple(
                dict(
                    rank=index,
                    pid=rank.pid,
                    exitcode=rank.exit_code if rank.closed else rank.process.exitcode,
                    alive=rank.alive(),
                )
                for index, rank in enumerate(self._ranks)
            )

    def _collect(self, command_id: int, operation: str, deadline: float):
        pending = set(range(len(self._ranks)))
        replies: list[TPReply | None] = [None] * len(self._ranks)
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TransportTimeout("draft TP all-rank completion timed out")
            watched = pending if operation == "close" else range(len(self._ranks))
            sentinels = {
                self._ranks[index].process.sentinel: index for index in watched
            }
            sockets = {self._ranks[index].socket: index for index in pending}
            ready = wait([*sockets, *sentinels], min(remaining, _POLL_S))
            # A healthy close ACK can arrive immediately before process exit.
            for item in ready:
                if item not in sockets:
                    continue
                index = sockets[item]
                try:
                    reply = self._ranks[index].channel.receive_reply(
                        min(max(0.0, deadline - time.monotonic()), _POLL_S)
                    )
                except TransportTimeout:
                    continue
                if (
                    reply.command_id != command_id
                    or reply.rank != index
                    or reply.operation != operation
                ):
                    raise TransportError("draft TP returned an unexpected reply")
                if reply.error is not None:
                    failure = reply.error
                    detail = (
                        f"draft TP rank {index} {failure.kind}: {failure.detail}\n"
                        f"{failure.traceback}"
                    )
                    if failure.cuda_error_code is not None:
                        raise CudaRuntimeError(
                            detail, cuda_error_code=failure.cuda_error_code
                        )
                    raise RuntimeError(detail)
                replies[index] = reply
                pending.remove(index)
            for item in ready:
                if item in sentinels and (
                    operation != "close" or sentinels[item] in pending
                ):
                    raise TransportError(
                        f"draft TP rank {sentinels[item]} exited before completion"
                    )
        return tuple(reply for reply in replies if reply is not None)

    def _exchange(self, command: TPCommand, deadline: float):
        # Every rank must receive the command before waiting for a response: a
        # first rank may already be waiting for its peers in a model collective.
        for rank in self._ranks:
            rank.channel.send(command, deadline)
        return self._collect(command.command_id, command.operation, deadline)

    def execute(
        self,
        operation: TPOperation,
        *,
        rows: tuple[TokenForwardRow, ...] = (),
        keys: tuple[SequenceKey, ...] = (),
        num_blocks: int = 0,
    ) -> tuple[TPReply, ...]:
        with self._rpc_lock:
            if self._closing.is_set() or self._failed.is_set():
                raise CompletionUnknownError("draft TP group is unavailable")
            self._command_id += 1
            command = TPCommand(self._command_id, operation, rows, keys, num_blocks)
            try:
                return self._exchange(command, time.monotonic() + self.timeout_s)
            except BaseException as error:
                try:
                    self._abort()
                except BaseException as cleanup_error:
                    error.add_note(f"draft TP command cleanup failed: {cleanup_error}")
                raise CompletionUnknownError("draft TP command failed") from error

    def _reap(self, *, graceful: bool) -> None:
        with self._reap_lock:
            if self._closed.is_set():
                return
            failure: BaseException | None = None

            def attempt(action: Callable[[], None]) -> bool:
                nonlocal failure
                try:
                    action()
                except BaseException as error:
                    if failure is None:
                        failure = error
                    else:
                        failure.add_note(f"draft TP cleanup also failed: {error}")
                    return False
                return True

            for rank in self._ranks:
                attempt(rank.channel.close)
            if graceful:
                deadline = time.monotonic() + min(self.timeout_s, 5)
                for rank in self._ranks:
                    if rank.pid is not None and not rank.closed:
                        attempt(
                            partial(
                                rank.process.join,
                                max(0.0, deadline - time.monotonic()),
                            )
                        )
            for rank in self._ranks:
                attempt(rank.death.close)
            for rank in self._ranks:
                if rank.alive():
                    attempt(rank.process.terminate)
            deadline = time.monotonic() + min(self.timeout_s, 2)
            for rank in self._ranks:
                if rank.pid is not None and not rank.closed:
                    attempt(
                        partial(
                            rank.process.join, max(0.0, deadline - time.monotonic())
                        )
                    )
            for rank in self._ranks:
                if rank.alive():
                    attempt(rank.process.kill)
            deadline = time.monotonic() + min(self.timeout_s, 2)
            for rank in self._ranks:
                if rank.pid is not None and not rank.closed:
                    attempt(
                        partial(
                            rank.process.join, max(0.0, deadline - time.monotonic())
                        )
                    )
                    if not rank.alive():
                        rank.exit_code = rank.process.exitcode
                        if attempt(rank.process.close):
                            rank.closed = True
            if any(rank.alive() for rank in self._ranks):
                raise CompletionUnknownError(
                    "draft TP rank did not exit after termination; "
                    "its owner must be replaced"
                ) from failure
            self._closed.set()
            if failure is not None:
                raise failure

    def _abort(self) -> None:
        self._failed.set()
        self._closing.set()
        self._reap(graceful=False)

    def abort(self) -> None:
        """Retire all rank processes without issuing another device command."""
        self._abort()

    def close(self) -> None:
        with self._close_lock:
            if self._closed.is_set():
                return
            self._closing.set()
            acquired = self._rpc_lock.acquire(blocking=False)
            graceful = False
            failure: BaseException | None = None
            try:
                if acquired and not self._failed.is_set():
                    self._command_id += 1
                    self._exchange(
                        TPCommand(self._command_id, "close"),
                        time.monotonic() + self.timeout_s,
                    )
                    graceful = True
                else:
                    self._failed.set()
            except BaseException as error:
                self._failed.set()
                failure = error
            finally:
                if acquired:
                    self._rpc_lock.release()
            try:
                self._reap(graceful=graceful)
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    failure.add_note(f"draft TP shutdown cleanup failed: {error}")
            if failure is not None:
                raise CompletionUnknownError("draft TP shutdown failed") from failure


def _gpu_worker(
    channel: TPWorkerConnection,
    rank: int,
    config: "VllmConfig",
    devices: tuple["torch.device", ...],
    kv_cache_bytes: int,
    init_method: str,
    timeout_s: float,
) -> None:
    import torch

    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tensor_model_parallel_world_size,
    )
    from vllm.v1.spec_decode.remote.runtime.gpu import StandaloneParallelContext
    from vllm.v1.spec_decode.remote.runtime.token_gpu import TokenGPUStepExecutor

    executor = TokenGPUStepExecutor(
        config,
        device=devices[rank],
        kv_cache_bytes=kv_cache_bytes,
        parallel_context=StandaloneParallelContext(
            rank, len(devices), init_method, timeout_s
        ),
    )
    channel.send(
        TPReply(
            0,
            rank,
            "ready",
            geometry=TPGeometry(
                executor.num_blocks,
                executor.block_size,
                executor.max_num_reqs,
                executor.max_num_tokens,
                get_tensor_model_parallel_world_size(),
                str(executor.device),
                str(torch.cuda.get_device_properties(executor.device).uuid),
            ),
        ),
        time.monotonic() + timeout_s,
    )
    previous = 0
    while True:
        command = channel.receive_command()
        try:
            if command.command_id != previous + 1:
                raise ValueError("draft TP command is out of order")
            previous = command.command_id
            tokens: tuple[int | None, ...] = ()
            if command.operation == "forward":
                tokens = executor.forward_batch(command.rows)
            elif command.operation == "warmup":
                executor.warmup()
            elif command.operation == "release":
                executor.close_sequences(command.keys)
            elif command.operation == "capacity":
                if not 2 <= command.num_blocks <= executor.num_blocks:
                    raise ValueError("invalid common draft TP KV capacity")
                executor.num_blocks = command.num_blocks
            elif command.operation == "close":
                destroy_model_parallel()
                destroy_distributed_environment()
            channel.send(
                TPReply(command.command_id, rank, command.operation, tokens=tokens),
                time.monotonic() + timeout_s,
            )
            if command.operation == "close":
                return
        except BaseException as error:
            with contextlib.suppress(Exception):
                channel.send(
                    TPReply(
                        command.command_id,
                        rank,
                        command.operation,
                        error=_failure(error),
                    ),
                    time.monotonic() + timeout_s,
                )
            # Other ranks may still be in a collective. Let the CPU owner
            # terminate the group; do not issue distributed/GPU cleanup here.
            os._exit(1)


class TokenTPExecutor(TokenStepExecutor):
    def __init__(
        self,
        config: "VllmConfig",
        *,
        devices: tuple["torch.device", ...],
        kv_cache_bytes: int,
        timeout_s: float = 30,
        startup_timeout_s: float = 600,
    ):
        import torch

        from vllm.platforms import current_platform
        from vllm.v1.spec_decode.remote.runtime.token_schema import (
            create_token_runtime_contract,
        )

        spec = config.speculative_config
        assert spec is not None
        size = spec.draft_parallel_config.tensor_parallel_size
        if (
            size < 2
            or len(devices) != size
            or len(set(devices)) != size
            or any(device.type != "cuda" or device.index is None for device in devices)
        ):
            raise ValueError("draft TP requires one distinct indexed CUDA GPU per rank")
        if type(kv_cache_bytes) is not int or kv_cache_bytes <= 0:
            raise ValueError("draft TP KV cache budget must be positive")
        visible = torch.accelerator.device_count()
        physical = set()
        for device in devices:
            assert device.index is not None
            if device.index < 0 or device.index >= visible:
                raise ValueError("draft TP devices must be visible CUDA GPUs")
            physical.add(
                current_platform.visible_device_id_to_physical_device_id(device.index)
            )
        if len(physical) != size:
            raise ValueError("draft TP ranks must use distinct physical GPUs")
        self.contract = create_token_runtime_contract(config)
        self.device = torch.device("cpu")
        self._directory = tempfile.TemporaryDirectory(prefix="vllm-draft-tp-")
        self._group: TPProcessGroup | None = None
        try:
            init_method = Path(self._directory.name, "rendezvous").as_uri()
            self._group = TPProcessGroup(
                size,
                _gpu_worker,
                worker_args=(
                    config,
                    devices,
                    kv_cache_bytes,
                    init_method,
                    timeout_s,
                ),
                timeout_s=timeout_s,
                startup_timeout_s=startup_timeout_s,
            )
            geometries = tuple(reply.geometry for reply in self._group.ready)
            if any(geometry is None for geometry in geometries):
                raise ValueError("draft TP startup omitted KV geometry")
            if (
                len({other.device_uuid for other in geometries if other is not None})
                != size
            ):
                raise ValueError("draft TP ranks must bind distinct physical GPUs")
            geometry = geometries[0]
            assert geometry is not None
            for rank, other in enumerate(geometries):
                assert other is not None
                if (
                    other.world_size != size
                    or other.device != str(devices[rank])
                    or not other.device_uuid
                    or other.block_size != geometry.block_size
                    or other.max_num_reqs != geometry.max_num_reqs
                    or other.max_num_tokens != geometry.max_num_tokens
                    or other.num_blocks < 2
                    or other.block_size <= 0
                    or other.max_num_reqs <= 0
                    or other.max_num_tokens <= 0
                ):
                    raise ValueError(
                        "draft TP ranks disagree on their runtime geometry"
                    )
            self.num_blocks = min(
                other.num_blocks for other in geometries if other is not None
            )
            self.block_size = geometry.block_size
            self._max_num_reqs = geometry.max_num_reqs
            self._max_num_tokens = geometry.max_num_tokens
            self._group.execute("capacity", num_blocks=self.num_blocks)
            logger.info(
                "TOKEN_DRAFT_TP_READY %s",
                json.dumps(
                    dict(
                        world_size=size,
                        devices=[str(device) for device in devices],
                        num_blocks=self.num_blocks,
                        block_size=self.block_size,
                        max_num_reqs=self.max_num_reqs,
                        max_num_tokens=self.max_num_tokens,
                        ranks=[
                            dict(
                                status,
                                world_size=other.world_size,
                                device=other.device,
                                device_uuid=other.device_uuid,
                                num_blocks=other.num_blocks,
                            )
                            for status, other in zip(
                                self._group.worker_status(), geometries
                            )
                            if other is not None
                        ],
                    ),
                    sort_keys=True,
                ),
            )
        except BaseException as error:
            if self._group is not None:
                try:
                    self._group.abort()
                except BaseException as cleanup_error:
                    error.add_note(f"draft TP abort failed: {cleanup_error}")
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note(f"draft TP cleanup failed: {cleanup_error}")
            raise

    @property
    def max_num_reqs(self) -> int:
        return self._max_num_reqs

    @property
    def max_num_tokens(self) -> int:
        return self._max_num_tokens

    def forward(self, key, req_index, start, tokens, blocks, *, sample):
        return self.forward_batch(
            (TokenForwardRow(key, req_index, start, tokens, blocks, sample),)
        )[0]

    def forward_batch(self, rows: tuple[TokenForwardRow, ...]):
        if not rows:
            return ()
        assert self._group is not None
        replies = self._group.execute("forward", rows=rows)
        tokens = replies[0].tokens
        if len(tokens) != len(rows) or any(reply.tokens != tokens for reply in replies):
            self._group.abort()
            raise CompletionUnknownError("draft TP ranks disagree on sampled tokens")
        return tokens

    def warmup(self) -> None:
        assert self._group is not None
        self._group.execute("warmup")

    def close_sequences(self, keys: tuple[SequenceKey, ...]) -> None:
        if keys:
            assert self._group is not None
            self._group.execute("release", keys=keys)

    def close(self) -> None:
        try:
            if self._group is not None:
                self._group.close()
        finally:
            self._directory.cleanup()
