# Copyright 2026 XProbe Inc.
# Licensed under the Apache License, Version 2.0.

import gc
import multiprocessing as mp
import sys
import time
import weakref
from datetime import timedelta

import numpy as np
import pytest

from .. import xoscar_pygloo as xp


class MemoryStore:
    def __init__(self, values=None):
        self.values = {} if values is None else values

    def set_tcp(self, key, value):
        self.values[key] = value

    def get_tcp(self, key):
        return self.values[key]

    def wait(self, keys):
        deadline = time.monotonic() + 10
        while not all(key in self.values for key in keys):
            if time.monotonic() >= deadline:
                raise TimeoutError("Rendezvous keys were not published")
            time.sleep(0.001)


def test_prefix_store_rejects_none():
    with pytest.raises(TypeError):
        xp.rendezvous.PrefixStore("group", None)


@pytest.mark.parametrize("kind", ["hash", "file", "tcp", "custom"])
def test_prefix_store_owns_underlying_store(kind, tmp_path, unused_tcp_port):
    if kind == "hash":
        store = xp.rendezvous.HashStore()
    elif kind == "file":
        store = xp.rendezvous.FileStore(str(tmp_path))
    elif kind == "tcp":
        options = xp.rendezvous.TCPStoreOptions()
        options.port = unused_tcp_port
        options.isServer = True
        options.numWorkers = 1
        store = xp.rendezvous.TCPStore("127.0.0.1", options)
    else:
        memory = MemoryStore()
        memory_ref = weakref.ref(memory)
        store = xp.rendezvous.CustomStore(memory)
        del memory

    prefix = xp.rendezvous.PrefixStore("group", store)
    nested = xp.rendezvous.PrefixStore("nested", prefix)
    nested.set("key", list("value"))
    del store, prefix
    gc.collect()
    assert nested.get("key") == list("value")
    nested.set("other", list("updated"))
    assert nested.get("other") == list("updated")
    del nested
    gc.collect()
    if kind == "custom":
        assert memory_ref() is None


def test_tcp_store_destruction_releases_port(unused_tcp_port):
    options = xp.rendezvous.TCPStoreOptions()
    options.port = unused_tcp_port
    options.isServer = True
    options.numWorkers = 1
    options.timeout = timedelta(seconds=2)
    for _ in range(3):
        store = xp.rendezvous.TCPStore("127.0.0.1", options)
        prefix = xp.rendezvous.PrefixStore("group", store)
        del store
        prefix.set("key", list("value"))
        assert prefix.get("key") == list("value")
        del prefix
        gc.collect()


def test_rendezvous_context_casts_to_context():
    context = xp.rendezvous.Context(1, 3)
    assert context.rank == 1
    assert context.size == 3
    assert context.base == 2
    timeout = timedelta(seconds=2)
    context.setTimeout(timeout)
    assert context.getTimeout() == timeout


def _device():
    transport = xp.transport.tcp if sys.platform == "linux" else xp.transport.uv
    return transport.CreateDevice(transport.attr("127.0.0.1"))


@pytest.mark.parametrize("missing", ["store", "dev"])
def test_connect_full_mesh_rejects_none(missing):
    context = xp.rendezvous.Context(0, 1)
    store = None if missing == "store" else xp.rendezvous.HashStore()
    dev = None if missing == "dev" else _device()
    with pytest.raises(TypeError):
        context.connectFullMesh(store, dev)


def _collective_after_store_gc(rank, kind, directory, port, values):
    context = xp.rendezvous.Context(rank, 2)
    context.setTimeout(timedelta(seconds=10))
    memory_ref = None
    if kind == "file":
        store = xp.rendezvous.FileStore(directory)
    elif kind == "tcp":
        options = xp.rendezvous.TCPStoreOptions()
        options.port = port
        options.isServer = rank == 0
        options.numWorkers = 2
        options.timeout = timedelta(seconds=10)
        store = xp.rendezvous.TCPStore("127.0.0.1", options)
    else:
        memory = MemoryStore(values)
        memory_ref = weakref.ref(memory)
        store = xp.rendezvous.CustomStore(memory)
        del memory
    prefix = xp.rendezvous.PrefixStore("after-gc", store)
    context.connectFullMesh(prefix, _device())
    del store, prefix
    gc.collect()
    if memory_ref is not None and sys.platform == "linux":
        # TCP Context retains the shared store even after eager rendezvous.
        assert memory_ref() is not None

    sendbuf = np.full(4, rank + 1, dtype=np.float32)
    recvbuf = np.zeros_like(sendbuf)
    xp.allreduce(
        context,
        sendbuf.ctypes.data,
        recvbuf.ctypes.data,
        sendbuf.size,
        xp.GlooDataType_t.glooFloat32,
        xp.ReduceOp.SUM,
        xp.AllreduceAlgorithm.RING,
    )
    np.testing.assert_array_equal(recvbuf, np.full(4, 3, dtype=np.float32))
    xp.barrier(context)
    del context
    gc.collect()
    if memory_ref is not None:
        assert memory_ref() is None


@pytest.mark.parametrize("kind", ["file", "tcp", "custom"])
def test_collective_after_dropping_store_references(kind, tmp_path, unused_tcp_port):
    if kind == "file" and sys.platform == "win32":
        pytest.skip("FileStore is unsupported on Windows")
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        values = manager.dict() if kind == "custom" else None
        processes = [
            ctx.Process(
                target=_collective_after_store_gc,
                args=(rank, kind, str(tmp_path), unused_tcp_port, values),
            )
            for rank in range(2)
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(30)
            assert [process.exitcode for process in processes] == [0, 0]
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(5)
