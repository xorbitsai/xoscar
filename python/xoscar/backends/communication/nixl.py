# Copyright 2026 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Socket control messages and opt-in NIXL WRITE transfers for ``copy_to``."""

from __future__ import annotations

import asyncio
import sys
import uuid

import numpy as np

from ...core import BufferRef
from ...utils import classproperty
from ..message import ControlMessage, ControlMessageType, ErrorMessage, new_message_id
from .core import register_client, register_server
from .socket import SocketChannel, SocketClient, SocketServer

# Retain only the most recent batch per connection, limited by its byte size.
# Larger batches are registered for the duration of a single copy only.
_REGISTRATION_CACHE_BYTES = 256 * 1024**2


def _describe_buffer(buffer, writable=False):
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(buffer, torch.Tensor):
        if not buffer.is_contiguous():
            raise ValueError("NIXL requires contiguous buffers")
        if buffer.device.type not in ("cpu", "cuda"):
            raise ValueError("NIXL supports CPU and CUDA buffers")
        return (
            "VRAM" if buffer.is_cuda else "DRAM",
            (
                buffer.data_ptr(),
                buffer.numel() * buffer.element_size(),
                max(buffer.get_device(), 0),
            ),
        )
    if hasattr(buffer, "__cuda_array_interface__"):
        try:
            import cupy
        except ImportError as ex:
            raise ImportError(
                "Install a CUDA-compatible CuPy to transfer CUDA arrays via NIXL"
            ) from ex

        array = cupy.asarray(buffer)
        if not array.flags.c_contiguous:
            raise ValueError("NIXL requires contiguous buffers")
        if writable and buffer.__cuda_array_interface__["data"][1]:
            raise ValueError("NIXL target buffer is read-only")
        return "VRAM", (array.data.ptr, array.nbytes, array.device.id)
    view = memoryview(buffer)
    if not view.c_contiguous:
        raise ValueError("NIXL requires contiguous buffers")
    if writable and view.readonly:
        raise ValueError("NIXL target buffer is read-only")
    array = np.frombuffer(view, dtype=np.uint8)
    return "DRAM", (array.ctypes.data, array.nbytes, 0)


def _synchronize(buffers):
    # A device fence covers non-default producer/consumer streams too. NIXL's
    # host API does not establish CUDA stream dependencies on our behalf.
    torch = sys.modules.get("torch")
    torch_devices = set()
    cupy_devices = set()
    for buffer in buffers:
        if torch is not None and isinstance(buffer, torch.Tensor):
            if buffer.is_cuda:
                torch_devices.add(buffer.device.index)
        elif hasattr(buffer, "__cuda_array_interface__"):
            import cupy

            cupy_devices.add(cupy.asarray(buffer).device.id)
    for device in torch_devices:
        torch.cuda.synchronize(device)
    for device in cupy_devices:
        import cupy

        with cupy.cuda.Device(device):
            cupy.cuda.runtime.deviceSynchronize()


async def _wait_until_done(task):
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as ex:
            cancelled = ex
        except Exception:
            break
    return cancelled


class NixlChannel(SocketChannel):
    name = "nixl"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._agent = None
        self._agent_init = None
        self._buffers = []
        self._descriptions = []
        self._registrations = []
        self._generation = 0
        self._remote_generation = None
        self._remote_agent = None
        self._pending = None
        self._copy_lock = asyncio.Lock()
        self._inflight = None
        self._closing = False
        self._close_task = None

    @property
    def closed(self):
        return self._closing or super().closed

    @property
    def agent(self):
        if self._closing or self.closed:
            raise ConnectionError("NIXL channel is closed")
        if self._agent is None:
            raise RuntimeError("NIXL agent has not been initialized")
        return self._agent

    @staticmethod
    def _create_agent():
        if sys.platform != "linux":
            raise ImportError("NIXL transfers require Linux")
        try:
            from nixl._api import nixl_agent, nixl_agent_config
        except ImportError as ex:
            raise ImportError("Install xoscar[nixl] to use NIXL transfers") from ex
        return nixl_agent(
            f"xoscar-{uuid.uuid4().hex}",
            nixl_agent_config(
                enable_prog_thread=True,
                enable_listen_thread=False,
                backends=["UCX"],
            ),
        )

    async def _initialize_agent(self):
        self._agent = await asyncio.to_thread(self._create_agent)

    async def _ensure_agent(self):
        if self._closing or self.closed:
            raise ConnectionError("NIXL channel is closed")
        if self._agent is None:
            if self._agent_init is None:
                self._agent_init = asyncio.create_task(self._initialize_agent())
            # A cancelled caller must not abandon an agent still being created.
            await asyncio.shield(self._agent_init)
        return self.agent

    def _deregister(self):
        for registration in self._registrations:
            self._agent.deregister_memory(registration)
        self._registrations.clear()
        self._buffers.clear()
        self._descriptions.clear()

    def _register(self, buffers, descriptions=None):
        if descriptions is None:
            descriptions = [_describe_buffer(b) for b in buffers]
        _synchronize(buffers)
        if (
            len(buffers) == len(self._buffers)
            and all(a is b for a, b in zip(buffers, self._buffers))
            and descriptions == self._descriptions
        ):
            return descriptions
        agent = self.agent
        self._deregister()
        self._buffers = list(buffers)
        self._descriptions = descriptions
        self._generation += 1
        try:
            for mem_type in ("DRAM", "VRAM"):
                regions = [
                    (*desc, "")
                    for kind, desc in descriptions
                    if kind == mem_type and desc[1]
                ]
                if regions:
                    self._registrations.append(
                        agent.register_memory(regions, mem_type=mem_type)
                    )
        except BaseException:
            self._deregister()
            raise
        return descriptions

    def _trim_registration_cache(self):
        if sum(desc[1] for _, desc in self._descriptions) > _REGISTRATION_CACHE_BYTES:
            self._deregister()

    async def handle_buffers(self, content):
        operation, token, payload = content
        if operation == "prepare":
            if self._pending is not None:
                raise RuntimeError("A NIXL copy is already active on this channel")
            refs, sizes, known_generation = payload
            buffers = [BufferRef.get_buffer(BufferRef(addr, uid)) for addr, uid in refs]
            descriptions = [_describe_buffer(b, writable=True) for b in buffers]
            if len(sizes) != len(descriptions) or any(
                size != desc[1] for size, (_, desc) in zip(sizes, descriptions)
            ):
                raise ValueError("NIXL source and target buffer sizes must match")
            await self._ensure_agent()
            self._register(buffers, descriptions)
            metadata = (
                self.agent.get_agent_metadata()
                if known_generation != self._generation
                else None
            )
            self._pending = token
            return self._generation, metadata, descriptions
        if operation == "finish":
            if token != self._pending:
                raise ValueError("Unknown NIXL copy")
            _synchronize(self._buffers)
            self._pending = None
            self._trim_registration_cache()
            return True
        raise ValueError(f"Unknown NIXL buffer operation: {operation}")

    async def _write(self, descriptions, remote_descriptions):
        agent = self.agent
        # Each NIXL descriptor list has one memory type. Group mixed batches
        # without changing the correspondence between source and target buffers.
        groups = {}
        for (local_type, local), (remote_type, remote) in zip(
            descriptions, remote_descriptions
        ):
            if local[1]:
                group = groups.setdefault((local_type, remote_type), ([], []))
                group[0].append(local)
                group[1].append(remote)
        for (local_type, remote_type), (local, remote) in groups.items():
            handle = agent.initialize_xfer(
                "WRITE",
                agent.get_xfer_descs(local, mem_type=local_type),
                agent.get_xfer_descs(remote, mem_type=remote_type),
                self._remote_agent,
            )
            try:
                state = agent.transfer(handle)
                poll_delay = 0
                while state == "PROC":
                    # Even after control-channel EOF, wait for a terminal
                    # transport state before deregistering memory. The UCX
                    # backend's default peer error handling reports peer loss.
                    await asyncio.sleep(poll_delay)
                    poll_delay = min(poll_delay * 2 or 0.0001, 0.001)
                    state = agent.check_xfer_state(handle)
                if state != "DONE":
                    raise RuntimeError(f"NIXL transfer failed: {state}")
            finally:
                agent.release_xfer_handle(handle)

    async def copy_buffers(self, buffers, refs, call):
        # Queued copies can be cancelled before they acquire any registrations.
        async with self._copy_lock:
            task = asyncio.create_task(self._copy_buffers(buffers, refs, call))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # Repeated cancellation cannot abandon an active WRITE or ack.
                await _wait_until_done(task)
                try:
                    task.result()
                except BaseException:
                    pass
                raise

    async def _copy_buffers(self, buffers, refs, call):
        token = new_message_id()
        prepare_rejected = False

        async def command(operation, payload=None):
            nonlocal prepare_rejected
            message = ControlMessage(
                message_id=new_message_id(),
                control_message_type=ControlMessageType.switch_to_copy_to,
                content=(operation, token, payload),
            )
            result = await call(message)
            if isinstance(result, ErrorMessage):
                prepare_rejected = operation == "prepare"
                raise result.as_instanceof_cause()
            return result.result

        await self._ensure_agent()
        descriptions = self._register(buffers)
        try:
            generation, metadata, remote = await command(
                "prepare",
                (
                    [(ref.address, ref.uid) for ref in refs],
                    [desc[1] for _, desc in descriptions],
                    self._remote_generation,
                ),
            )
            if metadata is not None:
                if self._remote_agent is not None:
                    self.agent.remove_remote_agent(self._remote_agent)
                self._remote_agent = self.agent.add_remote_agent(metadata)
            self._remote_generation = generation
            self._inflight = asyncio.create_task(self._write(descriptions, remote))
            await asyncio.shield(self._inflight)
            await command("finish")
            self._trim_registration_cache()
        except BaseException:
            # A rejected prepare has not started a transfer. The connection
            # also carries actor RPCs, so leave it usable for those callers.
            if prepare_rejected:
                self._trim_registration_cache()
            else:
                await self.close()
            raise
        finally:
            self._inflight = None

    async def close(self):
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close())
        cancelled = await _wait_until_done(self._close_task)
        self._close_task.result()
        if cancelled is not None:
            raise cancelled

    async def _close(self):
        try:
            for task in (self._agent_init, self._inflight):
                if task is not None:
                    await _wait_until_done(task)
                    try:
                        task.result()
                    except BaseException:
                        pass
            # Deregister explicitly: an exception traceback can still retain
            # the Python agent wrapper, so dropping our reference is not enough.
            self._deregister()
        finally:
            self._agent = None
            self._pending = None
            await super().close()


@register_server
class NixlServer(SocketServer):
    scheme = "nixl"
    channel_class = NixlChannel

    @classproperty
    def client_type(self):
        return NixlClient


@register_client
class NixlClient(SocketClient):
    scheme = "nixl"
    channel_class = NixlChannel

    async def copy_buffers(self, buffers, refs, call):
        return await self.channel.copy_buffers(buffers, refs, call)
