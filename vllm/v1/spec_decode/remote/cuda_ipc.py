# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded CUDA IPC buffers for trusted, same-host draft workers.

Only opaque CUDA handles and tensor metadata cross the control connection.
Export allocations are independent of PyTorch's caching allocator. Receivers
copy into their own GPU tensors before acknowledging a lease, so retained
prefill features never alias a subsequently reused export buffer.
"""

import ctypes
import math
import time
from dataclasses import dataclass
from typing import Any

import msgspec
import regex as re

_HANDLE_BYTES = 64
_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
_ITEMSIZE = {
    "int32": 4,
    "int64": 8,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}
_MAX_BYTES = 1 << 40
# A disconnected peer can still have an outstanding device read. Retain all
# uncertain allocations/mappings until CUDA destroys the process context.
_QUARANTINE: list[Any] = []


class CudaIpcError(RuntimeError):
    """Invalid IPC contract, unsupported topology, or failed CUDA operation."""


class CudaIpcDescriptor(  # type: ignore[call-arg]
    msgspec.Struct, frozen=True, forbid_unknown_fields=True
):
    buffer_id: int
    generation: int
    device_uuid: str
    memory_handle: bytes
    ready_event_handle: bytes
    nbytes: int
    capacity_bytes: int
    offset_bytes: int = 0


def validate_descriptor(
    descriptor: CudaIpcDescriptor, dtype: str, shape: tuple[int, ...]
) -> None:
    """Validate the wire contract without importing torch or initializing CUDA."""
    itemsize = _ITEMSIZE.get(dtype)
    if itemsize is None or len(shape) > 8 or any(size < 0 for size in shape):
        raise CudaIpcError("invalid CUDA IPC tensor dtype or shape")
    expected = math.prod(shape) * itemsize
    if (
        descriptor.buffer_id < 1
        or descriptor.generation < 1
        or not _UUID.fullmatch(descriptor.device_uuid)
        or len(descriptor.memory_handle) != _HANDLE_BYTES
        or len(descriptor.ready_event_handle) != _HANDLE_BYTES
        or not 0 < expected == descriptor.nbytes <= descriptor.capacity_bytes
        or descriptor.offset_bytes < 0
        or descriptor.offset_bytes + descriptor.nbytes > descriptor.capacity_bytes
        or descriptor.capacity_bytes > _MAX_BYTES
    ):
        raise CudaIpcError("invalid CUDA IPC descriptor")


class _Handle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * _HANDLE_BYTES)]


class _CudaRuntime:
    """CUDA-only ABI; in particular CUDA IPC handles are exactly 64 bytes."""

    def __init__(self) -> None:
        import torch

        from vllm import envs
        from vllm.utils.system_utils import find_loaded_library

        if torch.version.hip is not None:
            raise CudaIpcError("cuda_ipc requires NVIDIA CUDA")
        path = find_loaded_library("libcudart") or envs.VLLM_CUDART_SO_PATH
        if path is None:
            raise CudaIpcError("libcudart is not loaded")
        self.lib = ctypes.CDLL(path)
        p = ctypes.c_void_p
        pp = ctypes.POINTER(p)
        i = ctypes.c_int
        u = ctypes.c_uint
        signatures = {
            "cudaMalloc": [pp, ctypes.c_size_t],
            "cudaFree": [p],
            "cudaIpcGetMemHandle": [ctypes.POINTER(_Handle), p],
            "cudaIpcOpenMemHandle": [pp, _Handle, u],
            "cudaIpcCloseMemHandle": [p],
            "cudaEventCreateWithFlags": [pp, u],
            "cudaEventDestroy": [p],
            "cudaIpcGetEventHandle": [ctypes.POINTER(_Handle), p],
            "cudaIpcOpenEventHandle": [pp, _Handle],
            "cudaEventRecord": [p, p],
            "cudaStreamWaitEvent": [p, p, u],
            "cudaEventQuery": [p],
            "cudaMemcpyAsync": [p, p, ctypes.c_size_t, i, p],
        }
        for name, args in signatures.items():
            func = getattr(self.lib, name)
            func.argtypes, func.restype = args, i
        self.lib.cudaGetErrorString.argtypes = [i]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def call(self, name: str, *args: Any) -> None:
        result = getattr(self.lib, name)(*args)
        if result:
            message = self.lib.cudaGetErrorString(result).decode("utf-8")
            raise CudaIpcError(f"{name} failed: {message} ({result})")

    def malloc(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        self.call("cudaMalloc", ctypes.byref(ptr), nbytes)
        assert ptr.value is not None
        return ptr.value

    def free(self, ptr: int) -> None:
        self.call("cudaFree", ptr)

    def memory_handle(self, ptr: int) -> bytes:
        handle = _Handle()
        self.call("cudaIpcGetMemHandle", ctypes.byref(handle), ptr)
        return bytes(handle)

    def open_memory(self, handle: bytes) -> int:
        ptr = ctypes.c_void_p()
        self.call(
            "cudaIpcOpenMemHandle",
            ctypes.byref(ptr),
            _Handle.from_buffer_copy(handle),
            1,
        )
        assert ptr.value is not None
        return ptr.value

    def close_memory(self, ptr: int) -> None:
        self.call("cudaIpcCloseMemHandle", ptr)

    def event(self, *, interprocess: bool = False) -> int:
        event = ctypes.c_void_p()
        # cudaEventDisableTiming | cudaEventInterprocess
        self.call(
            "cudaEventCreateWithFlags",
            ctypes.byref(event),
            2 | (4 if interprocess else 0),
        )
        assert event.value is not None
        return event.value

    def event_handle(self, event: int) -> bytes:
        handle = _Handle()
        self.call("cudaIpcGetEventHandle", ctypes.byref(handle), event)
        return bytes(handle)

    def open_event(self, handle: bytes) -> int:
        event = ctypes.c_void_p()
        self.call(
            "cudaIpcOpenEventHandle",
            ctypes.byref(event),
            _Handle.from_buffer_copy(handle),
        )
        assert event.value is not None
        return event.value

    def destroy_event(self, event: int) -> None:
        self.call("cudaEventDestroy", event)

    def record(self, event: int, stream: int) -> None:
        self.call("cudaEventRecord", event, stream)

    def wait(self, event: int, stream: int) -> None:
        self.call("cudaStreamWaitEvent", stream, event, 0)

    def copy(self, dst: int, src: int, nbytes: int, stream: int) -> None:
        # cudaMemcpyDefault selects the peer route for imported device pointers.
        self.call("cudaMemcpyAsync", dst, src, nbytes, 4, stream)

    def synchronize(self, event: int, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            result = self.lib.cudaEventQuery(event)
            if result == 0:
                return
            if result != 600:  # cudaErrorNotReady
                message = self.lib.cudaGetErrorString(result).decode("utf-8")
                raise CudaIpcError(f"cudaEventQuery failed: {message} ({result})")
            if time.monotonic() >= deadline:
                raise CudaIpcError("CUDA IPC copy completion timed out")
            time.sleep(0.0001)


def _device_uuid(torch: Any, device: Any) -> str:
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    if not uuid.startswith("GPU-"):
        uuid = f"GPU-{uuid}"
    if not _UUID.fullmatch(uuid):
        raise CudaIpcError(f"CUDA IPC requires a full GPU UUID, got {uuid!r}")
    return uuid


def _require_peer_access(source_uuid: str, destination_uuid: str) -> None:
    """Check physical GPUs even when their CUDA_VISIBLE_DEVICES sets differ."""
    if source_uuid == destination_uuid:
        return
    from vllm.third_party import pynvml

    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        source = pynvml.nvmlDeviceGetHandleByUUID(source_uuid)
        destination = pynvml.nvmlDeviceGetHandleByUUID(destination_uuid)
        # READ/WRITE are 0/1 (the vendored READ constant is accidentally a tuple).
        statuses = [
            pynvml.nvmlDeviceGetP2PStatus(destination, source, capability)
            for capability in (0, 1)
        ]
        if any(status != pynvml.NVML_P2P_STATUS_OK for status in statuses):
            raise CudaIpcError(
                f"CUDA IPC peer access unavailable: {source_uuid} -> "
                f"{destination_uuid}, NVML read/write status={statuses}"
            )
    except pynvml.NVMLError as exc:
        raise CudaIpcError(f"cannot verify CUDA IPC peer access: {exc}") from exc
    finally:
        if initialized:
            pynvml.nvmlShutdown()


def _limits(max_buffers: int, max_bytes: int) -> None:
    if not 1 <= max_buffers <= 4096 or not 1 <= max_bytes <= _MAX_BYTES:
        raise ValueError("invalid CUDA IPC pool limits")


@dataclass
class _ExportBuffer:
    buffer_id: int
    event: int
    event_handle: bytes
    generation: int = 0
    lease: CudaIpcDescriptor | None = None


class CudaIpcExporter:
    """Single-threaded pool; release leases only after successful peer replies."""

    def __init__(
        self, device: Any, *, max_buffers: int = 32, max_bytes: int = 512 << 20
    ) -> None:
        import torch

        _limits(max_buffers, max_bytes)
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise CudaIpcError("CUDA IPC exporter requires a CUDA device")
        self.device_uuid = _device_uuid(torch, self.device)
        self.runtime = _CudaRuntime()
        self.max_buffers, self.max_bytes = max_buffers, max_bytes
        self.buffers: list[_ExportBuffer] = []
        self.closed = False
        self.exported = False
        # Allocate before target KV-cache profiling, so IPC storage cannot eat
        # into the target's profiled KV budget on the first generation step.
        with torch.accelerator.device_index(self.device.index):
            self.ptr = self.runtime.malloc(max_bytes)
            try:
                self.memory_handle = self.runtime.memory_handle(self.ptr)
            except BaseException:
                self.runtime.free(self.ptr)
                raise

    @property
    def reserved_bytes(self) -> int:
        return self.max_bytes

    def export(self, tensor: Any) -> CudaIpcDescriptor:
        if self.closed:
            raise CudaIpcError("CUDA IPC exporter is closed")
        if (
            not tensor.is_cuda
            or _device_uuid(self.torch, tensor.device) != self.device_uuid
        ):
            raise CudaIpcError("tensor does not belong to the exporting CUDA device")
        dtype = str(tensor.dtype).removeprefix("torch.")
        nbytes = tensor.numel() * tensor.element_size()
        if dtype not in _ITEMSIZE or nbytes <= 0:
            raise CudaIpcError("CUDA IPC requires a nonempty supported tensor")
        with self.torch.accelerator.device_index(self.device.index):
            tensor = tensor.contiguous()
            stream = self.torch.cuda.current_stream(self.device)
            # First-fit allocation within one persistent arena. Acknowledged
            # leases release their ranges immediately without closing mappings.
            offset = 0
            for active in sorted(
                (b.lease for b in self.buffers if b.lease is not None),
                key=lambda lease: lease.offset_bytes,
            ):
                if offset + nbytes <= active.offset_bytes:
                    break
                offset = (active.offset_bytes + active.nbytes + 255) & ~255
            if offset + nbytes > self.max_bytes:
                raise CudaIpcError("CUDA IPC export pool exhausted")
            available = [b for b in self.buffers if b.lease is None]
            if available:
                buffer = available[0]
            else:
                if len(self.buffers) >= self.max_buffers:
                    raise CudaIpcError("CUDA IPC export pool exhausted")
                event = self.runtime.event(interprocess=True)
                try:
                    buffer = _ExportBuffer(
                        len(self.buffers) + 1,
                        event,
                        self.runtime.event_handle(event),
                    )
                except BaseException:
                    self.runtime.destroy_event(event)
                    raise
                self.buffers.append(buffer)
            buffer.generation += 1
            descriptor = CudaIpcDescriptor(
                buffer.buffer_id,
                buffer.generation,
                self.device_uuid,
                self.memory_handle,
                buffer.event_handle,
                nbytes,
                self.max_bytes,
                offset,
            )
            buffer.lease = descriptor
            try:
                self.runtime.copy(
                    self.ptr + offset, tensor.data_ptr(), nbytes, stream.cuda_stream
                )
                tensor.record_stream(stream)
                self.runtime.record(buffer.event, stream.cuda_stream)
            except BaseException:
                self.quarantine()
                raise
            self.exported = True
            return descriptor

    def release(self, descriptor: CudaIpcDescriptor) -> None:
        if self.closed:
            raise CudaIpcError("CUDA IPC exporter is closed")
        for buffer in self.buffers:
            if buffer.buffer_id == descriptor.buffer_id and buffer.lease == descriptor:
                buffer.lease = None
                return
        raise CudaIpcError("stale or unknown CUDA IPC lease acknowledgement")

    def quarantine(self) -> None:
        if not self.closed:
            _QUARANTINE.append((self.runtime, self.device, self.ptr, self.buffers))
            self.buffers = []
            self.closed = True

    def close(self, *, peer_closed: bool = False) -> None:
        """Free only after the caller confirms all peer mappings are closed."""
        if self.closed:
            return
        if (self.exported and not peer_closed) or any(
            b.lease is not None for b in self.buffers
        ):
            self.quarantine()
            return
        with self.torch.accelerator.device_index(self.device.index):
            try:
                for buffer in self.buffers:
                    self.runtime.destroy_event(buffer.event)
                self.runtime.free(self.ptr)
            finally:
                self.closed = True
                self.buffers = []


@dataclass
class _ImportBuffer:
    descriptor: CudaIpcDescriptor
    event: int
    last_generation: int = 0


class CudaIpcReceiver:
    """Import on the draft device; ACK only after receive() successfully returns."""

    def __init__(
        self,
        device: Any,
        *,
        max_buffers: int = 32,
        max_bytes: int = 512 << 20,
        copy_timeout: float = 30.0,
    ) -> None:
        import torch

        _limits(max_buffers, max_bytes)
        if not math.isfinite(copy_timeout) or copy_timeout <= 0:
            raise ValueError("copy_timeout must be finite and positive")
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise CudaIpcError("CUDA IPC receiver requires a CUDA device")
        self.device_uuid = _device_uuid(torch, self.device)
        self.runtime = _CudaRuntime()
        self.max_buffers, self.max_bytes = max_buffers, max_bytes
        self.copy_timeout = copy_timeout
        self.buffers: dict[int, _ImportBuffer] = {}
        self.peers: set[str] = set()
        self.closed = False
        self.ptr: int | None = None
        self.arena: CudaIpcDescriptor | None = None
        with torch.accelerator.device_index(self.device.index):
            self.done_event = self.runtime.event()

    def receive(
        self, descriptor: CudaIpcDescriptor, dtype: str, shape: tuple[int, ...]
    ) -> Any:
        validate_descriptor(descriptor, dtype, shape)
        if self.closed:
            raise CudaIpcError("CUDA IPC receiver is closed")
        with self.torch.accelerator.device_index(self.device.index):
            if self.arena is not None and (
                self.arena.memory_handle != descriptor.memory_handle
                or self.arena.device_uuid != descriptor.device_uuid
                or self.arena.capacity_bytes != descriptor.capacity_bytes
            ):
                raise CudaIpcError("CUDA IPC arena identity changed")
            buffer = self.buffers.get(descriptor.buffer_id)
            if buffer is None:
                if (
                    len(self.buffers) >= self.max_buffers
                    or descriptor.capacity_bytes > self.max_bytes
                ):
                    raise CudaIpcError("CUDA IPC import pool exhausted")
                if descriptor.device_uuid not in self.peers:
                    _require_peer_access(descriptor.device_uuid, self.device_uuid)
                    self.peers.add(descriptor.device_uuid)
                if self.ptr is None:
                    self.ptr = self.runtime.open_memory(descriptor.memory_handle)
                    self.arena = descriptor
                event = self.runtime.open_event(descriptor.ready_event_handle)
                buffer = _ImportBuffer(descriptor, event)
                self.buffers[descriptor.buffer_id] = buffer
            original = buffer.descriptor
            if (
                original.memory_handle != descriptor.memory_handle
                or original.ready_event_handle != descriptor.ready_event_handle
                or original.device_uuid != descriptor.device_uuid
                or original.capacity_bytes != descriptor.capacity_bytes
                or descriptor.generation <= buffer.last_generation
            ):
                raise CudaIpcError("CUDA IPC buffer identity changed or lease replayed")
            tensor = self.torch.empty(
                shape, dtype=getattr(self.torch, dtype), device=self.device
            )
            stream = self.torch.cuda.current_stream(self.device)
            done = self.done_event
            try:
                self.runtime.wait(buffer.event, stream.cuda_stream)
                assert self.ptr is not None
                self.runtime.copy(
                    tensor.data_ptr(),
                    self.ptr + descriptor.offset_bytes,
                    descriptor.nbytes,
                    stream.cuda_stream,
                )
                tensor.record_stream(stream)
                self.runtime.record(done, stream.cuda_stream)
                self.runtime.synchronize(done, self.copy_timeout)
            except BaseException:
                # Even a timeout cannot cancel an already enqueued CUDA copy.
                _QUARANTINE.append(
                    (self.runtime, self.device, self.ptr, self.buffers, tensor, done)
                )
                self.buffers = {}
                self.closed = True
                raise
            buffer.last_generation = descriptor.generation
            return tensor

    def close(self) -> None:
        if self.closed:
            return
        with self.torch.accelerator.device_index(self.device.index):
            try:
                for buffer in self.buffers.values():
                    self.runtime.destroy_event(buffer.event)
                if self.ptr is not None:
                    self.runtime.close_memory(self.ptr)
                self.runtime.destroy_event(self.done_event)
            finally:
                self.buffers = {}
                self.closed = True
