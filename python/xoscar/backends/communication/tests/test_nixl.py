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

    agents = (source.agent, target.agent)
    yield source, target, call
    for agent in agents:
        agent.proceed = True
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
    assert source.agent.metadata_loads == 2
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
    assert not source._closing
    destination = np.zeros_like(a)
    await source.copy_buffers([a], refs_for([destination]), call)
    np.testing.assert_array_equal(a, destination)


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


async def test_cancelled_queued_copy_does_not_transfer(peers):
    source, target, call = peers
    agent = source.agent
    agent.proceed = False
    first_src, first_dst = np.arange(8, dtype="u1"), np.zeros(8, dtype="u1")
    queued_src, queued_dst = np.full(8, 91, dtype="u1"), np.zeros(8, dtype="u1")
    first = asyncio.create_task(
        source.copy_buffers([first_src], refs_for([first_dst]), call)
    )
    await agent.started.wait()
    queued = asyncio.create_task(
        source.copy_buffers([queued_src], refs_for([queued_dst]), call)
    )
    await asyncio.sleep(0)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(queued, 0.5)
    assert not first.done()
    agent.proceed = True
    await first
    np.testing.assert_array_equal(first_src, first_dst)
    assert not queued_dst.any()
    assert len(agent.released) == 1


@pytest.mark.parametrize("fail", [False, True])
async def test_repeated_cancellation_observes_transfer_result(peers, fail):
    source, target, call = peers
    agent = source.agent
    agent.proceed = False
    a, b = np.arange(8, dtype="u1"), np.zeros(8, dtype="u1")
    task = asyncio.create_task(source.copy_buffers([a], refs_for([b]), call))
    await agent.started.wait()
    for _ in range(3):
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    agent.fail = fail
    agent.proceed = True
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(agent.released) == 1
    if fail:
        assert source.closed and source._agent is None
    else:
        np.testing.assert_array_equal(a, b)
        assert target._pending is None


async def test_cancelled_close_waits_for_transfer_cleanup(peers):
    source, target, call = peers
    agent = source.agent
    agent.proceed = False
    a, b = np.arange(8, dtype="u1"), np.zeros(8, dtype="u1")
    copy = asyncio.create_task(source.copy_buffers([a], refs_for([b]), call))
    await agent.started.wait()
    closing = asyncio.create_task(source.close())
    await asyncio.sleep(0)
    for _ in range(3):
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
        assert not agent.deregistered
    agent.proceed = True
    await copy
    np.testing.assert_array_equal(a, b)
    assert target._pending is None
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert source.closed and source._agent is None
    assert not source._buffers and not source._registrations
    assert len(agent.deregistered) == 1
    await source.close()


@pytest.mark.parametrize("failure", ["readonly", "released", "active", "missing_nixl"])
async def test_prepare_rejection_keeps_channel_open(peers, monkeypatch, failure):
    source, target, call = peers
    a, b = np.arange(8, dtype="u1"), np.zeros(8, dtype="u1")
    refs = refs_for([b])
    if failure == "readonly":
        b.flags.writeable = False
        error = ValueError
    elif failure == "released":
        refs = [BufferRef(target.dest_address, b"missing-buffer")]
        error = KeyError
    elif failure == "active":
        target._pending = b"existing-copy"
        error = RuntimeError
    else:
        target._agent = None

        def missing():
            raise ImportError("Install xoscar[nixl]")

        monkeypatch.setattr(target, "_create_agent", missing)
        error = ImportError
    with pytest.raises(error):
        await source.copy_buffers([a], refs, call)
    assert not source.closed and not source._closing


async def test_prepare_transport_failure_closes_channel(peers):
    source, target, call = peers

    async def disconnected(message):
        raise ConnectionError("peer disconnected")

    a, b = np.arange(8, dtype="u1"), np.zeros(8, dtype="u1")
    with pytest.raises(ConnectionError):
        await source.copy_buffers([a], refs_for([b]), disconnected)
    assert source.closed and source._agent is None


async def test_finish_rejection_closes_channel(peers):
    source, target, call = peers

    async def reject_finish(message):
        if message.content[0] == "finish":
            error = ValueError("Unknown NIXL copy")
            return ErrorMessage(message.message_id, error_type=type(error), error=error)
        return await call(message)

    a, b = np.arange(8, dtype="u1"), np.zeros(8, dtype="u1")
    with pytest.raises(ValueError, match="Unknown NIXL copy"):
        await source.copy_buffers([a], refs_for([b]), reject_finish)
    np.testing.assert_array_equal(a, b)
    assert source.closed and source._agent is None


async def test_registration_failure_releases_partial_batch(peers):
    source, target, call = peers
    agent = source.agent
    a, b = np.arange(8, dtype="u1"), np.zeros(8, dtype="u1")
    descriptions = [nixl._describe_buffer(a), ("VRAM", nixl._describe_buffer(b)[1])]
    original = agent.register_memory

    def register(regions, mem_type):
        if mem_type == "VRAM":
            raise RuntimeError("registration failed")
        return original(regions, mem_type)

    with mock.patch.object(agent, "register_memory", side_effect=register):
        with pytest.raises(RuntimeError, match="registration failed"):
            source._register([a, b], descriptions)
    assert len(agent.deregistered) == 1
    assert not source._registrations and not source._buffers


class NixlCpuActor(Actor):
    def __init__(self, size=8):
        self.buffer = np.zeros(size, dtype="u1")
        self.started = False
        self.release = asyncio.Event()

    def allocate(self):
        return buffer_ref(self.address, self.buffer)

    def read(self):
        return self.buffer

    @no_lock
    async def wait(self):
        self.started = True
        await self.release.wait()
        return True

    @no_lock
    def waiting(self):
        return self.started

    @no_lock
    def finish_wait(self):
        self.release.set()


@pytest.mark.parametrize("scheme", [None, "nixl"])
async def test_client_only_first_copy_initializes_router(monkeypatch, scheme):
    import xoscar as xo

    from ...router import Router

    monkeypatch.setattr(nixl.NixlChannel, "_create_agent", staticmethod(FakeAgent))
    pool = await xo.create_actor_pool(
        "127.0.0.1:0", n_process=0, external_address_schemes=[scheme]
    )
    async with pool:
        actor = await xo.create_actor(NixlCpuActor, address=pool.external_address)
        ref = await actor.allocate()
        with monkeypatch.context() as patch:
            # A client receives a BufferRef without having made an actor RPC.
            patch.setattr(Router, "_instance", None)
            value = np.arange(8, dtype="u1")
            await copy_to([value], [ref])
            router = Router.get_instance()
            assert router is not None
            np.testing.assert_array_equal(await actor.read(), value)
            assert Router.get_instance() is router


async def test_cpu_pool_rejected_copy_preserves_inflight_rpc(monkeypatch):
    import xoscar as xo

    from ...router import Router

    monkeypatch.setattr(nixl.NixlChannel, "_create_agent", staticmethod(FakeAgent))
    pool = await xo.create_actor_pool(
        "127.0.0.1:0", n_process=0, external_address_schemes=["nixl"]
    )
    async with pool:
        actor = await xo.create_actor(NixlCpuActor, address=pool.external_address)
        with monkeypatch.context() as patch:
            patch.setattr(Router, "_instance", Router([], None))
            ref = await actor.allocate()
            pending = asyncio.create_task(actor.wait())
            try:

                async def wait_started():
                    while not await actor.waiting():
                        await asyncio.sleep(0)

                await asyncio.wait_for(wait_started(), 2)
                with pytest.raises(ValueError, match="sizes must match"):
                    await copy_to([np.ones(9, dtype="u1")], [ref])
                assert not pending.done()
                await actor.finish_wait()
                assert await asyncio.wait_for(pending, 2)
                value = np.arange(8, dtype="u1")
                await copy_to([value], [ref])
                np.testing.assert_array_equal(await actor.read(), value)
            finally:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)


def test_optional_agent_configuration(monkeypatch):
    import sys
    from types import ModuleType

    monkeypatch.setattr(sys, "platform", "linux")
    api = ModuleType("nixl._api")
    api.nixl_agent = mock.Mock(return_value=object())
    api.nixl_agent_config = mock.Mock(return_value="config")
    monkeypatch.setitem(sys.modules, "nixl._api", api)
    assert nixl.NixlChannel._create_agent() is api.nixl_agent.return_value
    api.nixl_agent_config.assert_called_once_with(
        enable_prog_thread=True, enable_listen_thread=False, backends=["UCX"]
    )


def test_optional_agent_requires_supported_platform_and_package(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(ImportError, match="require Linux"):
        nixl.NixlChannel._create_agent()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "nixl._api", None)
    with pytest.raises(ImportError, match="Install xoscar"):
        nixl.NixlChannel._create_agent()


def test_reject_strided_and_unsupported_torch_tensors():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="contiguous"):
        nixl._describe_buffer(torch.arange(8)[::2])
    with pytest.raises(ValueError, match="CPU and CUDA"):
        nixl._describe_buffer(torch.empty(8, device="meta"))


def test_cuda_array_description_and_validation(monkeypatch):
    import sys
    from types import SimpleNamespace

    array = SimpleNamespace(
        __cuda_array_interface__={"data": (1234, False)},
        flags=SimpleNamespace(c_contiguous=True),
        data=SimpleNamespace(ptr=1234),
        nbytes=16,
        device=SimpleNamespace(id=2),
    )
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(asarray=lambda a: a))
    assert nixl._describe_buffer(array, writable=True) == ("VRAM", (1234, 16, 2))
    array.__cuda_array_interface__["data"] = (1234, True)
    with pytest.raises(ValueError, match="read-only"):
        nixl._describe_buffer(array, writable=True)
    array.flags.c_contiguous = False
    with pytest.raises(ValueError, match="contiguous"):
        nixl._describe_buffer(array)


def test_device_fences_cover_each_device_once(monkeypatch):
    import sys
    from types import SimpleNamespace

    class Tensor:
        is_cuda = True
        device = SimpleNamespace(index=1)

    torch_sync = mock.Mock()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(Tensor=Tensor, cuda=SimpleNamespace(synchronize=torch_sync)),
    )
    cupy_device = mock.MagicMock()
    cupy_sync = mock.Mock()
    cupy = SimpleNamespace(
        asarray=lambda a: a,
        cuda=SimpleNamespace(
            Device=cupy_device, runtime=SimpleNamespace(deviceSynchronize=cupy_sync)
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    array = SimpleNamespace(__cuda_array_interface__={}, device=SimpleNamespace(id=2))
    nixl._synchronize([Tensor(), Tensor(), array, array])
    torch_sync.assert_called_once_with(1)
    cupy_device.assert_called_once_with(2)
    cupy_sync.assert_called_once_with()


async def test_unknown_protocol_operation_and_token(peers):
    source, target, call = peers
    with pytest.raises(ValueError, match="Unknown NIXL copy"):
        await target.handle_buffers(("finish", b"unknown-copy", None))
    with pytest.raises(ValueError, match="Unknown NIXL buffer operation"):
        await target.handle_buffers(("unsupported", b"unknown-copy", None))
    assert target._pending is None


async def test_agent_access_rejects_uninitialized_and_closed_channels():
    channel = make_channel()
    channel._agent = None
    with pytest.raises(RuntimeError, match="not been initialized"):
        channel.agent
    await channel.close()
    with pytest.raises(ConnectionError, match="closed"):
        await channel._ensure_agent()


async def test_rejected_prepare_trims_large_source_batch(peers, monkeypatch):
    source, target, call = peers
    agent = source.agent
    monkeypatch.setattr(nixl, "_REGISTRATION_CACHE_BYTES", 4)
    a, b = np.ones(8, dtype="u1"), np.zeros(7, dtype="u1")
    with pytest.raises(ValueError, match="sizes must match"):
        await source.copy_buffers([a], refs_for([b]), call)
    assert not source.closed
    assert not source._registrations and not source._buffers
    assert len(agent.deregistered) == 1


@pytest.mark.parametrize("cancel_first", [False, True])
async def test_concurrent_close_waits_for_shared_cleanup(peers, cancel_first):
    source, target, call = peers
    agent = source.agent
    agent.proceed = False
    a, b = np.arange(8, dtype="u1"), np.zeros(8, dtype="u1")
    copying = asyncio.create_task(source.copy_buffers([a], refs_for([b]), call))
    await agent.started.wait()
    first = asyncio.create_task(source.close())
    await asyncio.sleep(0)
    assert not source.writer.is_closing()
    second = asyncio.create_task(source.close())
    await asyncio.sleep(0)
    if cancel_first:
        first.cancel()
        await asyncio.sleep(0)
    assert not first.done() and not second.done()
    assert source.closed
    assert not agent.deregistered
    agent.proceed = True
    await copying
    if cancel_first:
        with pytest.raises(asyncio.CancelledError):
            await first
    else:
        await first
    await second
    assert source.writer.is_closing()
    assert source._agent is None and not source._registrations
    assert len(agent.deregistered) == 1


async def test_router_replaces_draining_nixl_client(peers, monkeypatch):
    from ...router import Router

    source, target, call = peers
    agent = source.agent
    agent.proceed = False
    a, b = np.arange(8, dtype="u1"), np.zeros(8, dtype="u1")
    copying = asyncio.create_task(source.copy_buffers([a], refs_for([b]), call))
    await agent.started.wait()
    router = Router([], None)
    address = "nixl://127.0.0.1:1234"
    old_client = nixl.NixlClient(
        local_address=None, dest_address=address, channel=source
    )
    replacement = nixl.NixlClient(
        local_address=None, dest_address=address, channel=make_channel()
    )
    factory = mock.AsyncMock(side_effect=[old_client, replacement])
    monkeypatch.setattr(Router, "_create_client", factory)
    assert await router.get_client(address) is old_client
    assert await router.get_client(address) is old_client
    factory.assert_awaited_once()
    closing = asyncio.create_task(source.close())
    await asyncio.sleep(0)
    try:
        assert await router.get_client(address) is replacement
        assert factory.await_count == 2
        assert not closing.done()
    finally:
        agent.proceed = True
        await copying
        await closing
        await replacement.close()


@pytest.mark.nixl
async def test_real_nixl_cpu_pool_transfer(monkeypatch):
    import sys

    if sys.platform != "linux":
        pytest.skip("NIXL requires Linux")
    pytest.importorskip("nixl")
    import xoscar as xo

    from ...allocate_strategy import ProcessIndex
    from ...router import Router

    # Transfer DRAM over TCP. NIXL also requires CUDA support in UCX_TLS
    # when it detects physical GPUs, even with CUDA_VISIBLE_DEVICES empty.
    monkeypatch.setenv("UCX_TLS", "tcp,cuda_copy")
    size = 1024**2
    pool = await xo.create_actor_pool(
        "127.0.0.1:0",
        n_process=1,
        external_address_schemes=[None, "nixl"],
        subprocess_start_method="spawn",
        use_uvloop=False,
    )
    async with pool:
        actor = await xo.create_actor(
            NixlCpuActor,
            size,
            address=pool.external_address,
            allocate_strategy=ProcessIndex(1),
        )
        ref = await actor.allocate()
        with monkeypatch.context() as patch:
            patch.setattr(Router, "_instance", None)
            for value in (17, 29, 43):
                source = np.full(size, value, dtype="u1")
                await copy_to([source], [ref])
                np.testing.assert_array_equal(await actor.read(), source)
