# Copyright 2026 XProbe Inc.
# Licensed under the Apache License, Version 2.0.

import asyncio
import ctypes
from unittest import mock

import numpy as np
import pytest

from ....core import BufferRef
from ...message import ErrorMessage, ResultMessage
from .. import nixl


class FakeAgent:
    def __init__(self):
        self.registered = []
        self.deregistered = []
        self.released = []
        self.started = asyncio.Event()
        self.proceed = True
        self.fail = False
        self.metadata_loads = 0

    def register_memory(self, regions, mem_type):
        self.registered.append(regions)
        return regions

    def deregister_memory(self, regions):
        self.deregistered.append(regions)

    def get_agent_metadata(self):
        return b"metadata"

    def add_remote_agent(self, metadata):
        self.metadata_loads += 1
        return "peer"

    def remove_remote_agent(self, name):
        pass

    def get_xfer_descs(self, descs, mem_type):
        return descs

    def initialize_xfer(self, operation, local, remote, peer):
        assert operation == "WRITE"
        return local, remote

    def transfer(self, handle):
        self.started.set()
        return self.check_xfer_state(handle)

    def check_xfer_state(self, handle):
        if self.fail:
            return "ERR"
        if not self.proceed:
            return "PROC"
        for local, remote in zip(*handle):
            ctypes.memmove(remote[0], local[0], local[1])
        return "DONE"

    def release_xfer_handle(self, handle):
        self.released.append(handle)


def make_channel():
    writer = mock.Mock()
    writer.is_closing.return_value = False
    writer.wait_closed = mock.AsyncMock()
    writer.close.side_effect = lambda: setattr(writer.is_closing, "return_value", True)
    channel = nixl.NixlChannel(None, writer)
    channel._agent = FakeAgent()
    return channel


@pytest.fixture
async def peers():
    source, target = make_channel(), make_channel()

    async def call(message):
        try:
            result = await target.handle_buffers(message.content)
            return ResultMessage(message.message_id, result)
        except Exception as e:
            return ErrorMessage(message.message_id, error_type=type(e), error=e)

    yield source, target, call
    await source.close()
    await target.close()


def refs_for(buffers):
    return [
        BufferRef.create(b, "nixl://target:1234", str(i).encode())
        for i, b in enumerate(buffers)
    ]


async def test_batch_copy_reuses_registration_and_metadata(peers):
    source, target, call = peers
    buffers = [
        np.arange(64, dtype="i4"),
        np.arange(9, dtype="u1"),
        np.empty(0, dtype="u1"),
    ]
    destinations = [np.zeros_like(b) for b in buffers]
    refs = refs_for(destinations)
    for _ in range(3):
        buffers[0] += 1
        await source.copy_buffers(buffers, refs, call)
        for a, b in zip(buffers, destinations):
            np.testing.assert_array_equal(a, b)
    assert len(source.agent.registered) == len(target.agent.registered) == 1
    assert source.agent.metadata_loads == 1
    assert target._pending is None


async def test_replacement_and_cache_limit_release_memory(peers, monkeypatch):
    source, target, call = peers
    monkeypatch.setattr(nixl, "_REGISTRATION_CACHE_BYTES", 32)
    for size in (8, 64):
        a, b = np.arange(size, dtype="u1"), np.zeros(size, dtype="u1")
        await source.copy_buffers([a], refs_for([b]), call)
        np.testing.assert_array_equal(a, b)
    assert len(source.agent.deregistered) == 2
    assert len(target.agent.deregistered) == 2
    assert not source._buffers and not target._buffers


async def test_cancel_waits_for_write_and_remote_ack(peers):
    source, target, call = peers
    agent = source.agent
    agent.proceed = False
    a, b = np.arange(64, dtype="u1"), np.zeros(64, dtype="u1")
    task = asyncio.create_task(source.copy_buffers([a], refs_for([b]), call))
    await agent.started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert source._buffers and target._buffers
    agent.proceed = True
    with pytest.raises(asyncio.CancelledError):
        await task
    np.testing.assert_array_equal(a, b)
    assert target._pending is None
    assert len(agent.released) == 1


async def test_transfer_error_closes_channel(peers):
    source, target, call = peers
    agent = source.agent
    agent.fail = True
    a, b = np.arange(16, dtype="u1"), np.zeros(16, dtype="u1")
    with pytest.raises(RuntimeError, match="NIXL transfer failed"):
        await source.copy_buffers([a], refs_for([b]), call)
    assert source.closed
    assert source._agent is None
    assert not source._buffers
    assert len(agent.released) == 1


async def test_size_mismatch_is_rejected_before_write(peers):
    source, target, call = peers
    a, b = np.arange(16, dtype="u1"), np.zeros(15, dtype="u1")
    with pytest.raises(ValueError, match="sizes must match"):
        await source.copy_buffers([a], refs_for([b]), call)
    assert not target.agent.registered
    assert target._pending is None


def test_noncontiguous_and_readonly_targets():
    with pytest.raises(ValueError, match="contiguous"):
        nixl._describe_buffer(np.arange(10)[::2])
    with pytest.raises(ValueError, match="read-only"):
        nixl._describe_buffer(b"123", writable=True)


# Imported by spawned GPU workers; no CUDA initialization at module import.
from .... import Actor, buffer_ref, copy_to, no_lock


class NixlGpuActor(Actor):
    def __init__(self, device):
        import torch

        torch.cuda.set_device(device)
        self.device = device
        self.buffers = []
        self.started = False

    def allocate(self, sizes):
        import torch

        stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(stream):
            self.buffers = [
                torch.zeros(size, dtype=torch.uint8, device=f"cuda:{self.device}")
                for size in sizes
            ]
        return [buffer_ref(self.address, b) for b in self.buffers]

    def verify(self, value):
        return all(bool((b == value).all().item()) for b in self.buffers)

    def allocate_mixed(self):
        import torch

        self.buffers = [
            torch.zeros(512, dtype=torch.uint8, device=device)
            for device in ("cpu", f"cuda:{self.device}", "cpu", f"cuda:{self.device}")
        ]
        return [buffer_ref(self.address, b) for b in self.buffers]

    async def copy_mixed(self, target):
        import torch

        refs = await target.allocate_mixed()
        local = [np.full(512, 91, dtype="u1")] + [
            torch.full((512,), 91, dtype=torch.uint8, device=device)
            for device in ("cpu", f"cuda:{self.device}", f"cuda:{self.device}")
        ]
        await copy_to(local, refs)
        return await target.verify(91)

    async def reject_wrong_size(self, target):
        import torch

        refs = await target.allocate([(16,)])
        local = torch.ones(17, dtype=torch.uint8, device=f"cuda:{self.device}")
        try:
            await copy_to([local], refs)
        except ValueError:
            return True
        return False

    @no_lock
    def is_started(self):
        return self.started

    async def copy(self, target, sizes, concurrent=False, repeat=1):
        import torch

        refs = await target.allocate(sizes)
        stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(stream):
            local = [
                torch.full(size, 91, dtype=torch.uint8, device=f"cuda:{self.device}")
                for size in sizes
            ]
        self.started = True
        if concurrent:
            await asyncio.gather(*(copy_to(local, refs) for _ in range(4)))
        else:
            for _ in range(repeat):
                await copy_to(local, refs)
        return await target.verify(91)


@pytest.mark.cuda
@pytest.mark.parametrize("kill_peer", [False, True])
async def test_nixl_gpu_pool_transfer(kill_peer):
    import sys

    if sys.platform != "linux":
        pytest.skip("NIXL requires Linux")
    pytest.importorskip("nixl")
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA devices required")
    import xoscar as xo

    from ...allocate_strategy import ProcessIndex

    pool = await xo.create_actor_pool(
        "127.0.0.1",
        n_process=2,
        external_address_schemes=[None, "nixl", "nixl"],
        subprocess_start_method="spawn",
        use_uvloop=False,
    )
    try:
        actors = [
            await xo.create_actor(
                NixlGpuActor,
                i,
                address=pool.external_address,
                allocate_strategy=ProcessIndex(i + 1),
            )
            for i in range(2)
        ]
        if kill_peer:
            task = asyncio.create_task(
                actors[0].copy(actors[1], [(128 * 1024**2,)], repeat=1000)
            )
            while not await actors[0].is_started():
                await asyncio.sleep(0.01)
            await xo.kill_actor(actors[1])
            with pytest.raises((ConnectionError, xo.ServerClosed, RuntimeError)):
                await asyncio.wait_for(task, 30)
        else:
            sizes = [(0,), (16, 32), (4 * 1024**2,)]
            assert await actors[0].copy(actors[1], sizes, concurrent=True)
            assert await actors[1].copy(actors[0], sizes)
            assert await actors[0].copy_mixed(actors[1])
            assert await actors[0].reject_wrong_size(actors[1])
            assert await actors[0].copy(actors[1], sizes)
    finally:
        await pool.stop()


async def test_initialization_keeps_loop_responsive_and_close_waits(monkeypatch):
    import threading

    channel = make_channel()
    agent = channel._agent
    channel._agent = None
    started = threading.Event()
    release = threading.Event()
    calls = []

    def create():
        calls.append(threading.get_ident())
        started.set()
        assert release.wait(10)
        return agent

    monkeypatch.setattr(channel, "_create_agent", create)
    first = asyncio.create_task(channel._ensure_agent())
    second = asyncio.create_task(channel._ensure_agent())
    closing = None
    try:

        async def wait_started():
            while not started.is_set():
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_started(), 2)
        assert calls == [calls[0]]
        assert calls[0] != threading.get_ident()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        closing = asyncio.create_task(channel.close())
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        with pytest.raises(ConnectionError, match="closed"):
            await second
        await closing
        assert channel.closed
        assert channel._agent is None
    finally:
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)
        if closing is not None:
            await closing
        await channel.close()


def test_cuda_array_without_cupy(monkeypatch):
    import sys

    class CudaArray:
        __cuda_array_interface__ = {}

    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(ImportError, match="CUDA-compatible CuPy"):
        nixl._describe_buffer(CudaArray())


def test_cpu_tensor_description():
    torch = pytest.importorskip("torch")
    tensor = torch.zeros(4)
    assert nixl._describe_buffer(tensor) == (
        "DRAM",
        (tensor.data_ptr(), 4 * tensor.element_size(), 0),
    )
