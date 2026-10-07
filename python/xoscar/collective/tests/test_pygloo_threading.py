# Copyright 2026 XProbe Inc.
# Licensed under the Apache License, Version 2.0.

import multiprocessing as mp
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import numpy as np
import pytest

from .. import xoscar_pygloo as xp


def _transfer(operation, context, buffer, peer):
    getattr(xp, operation)(
        context,
        buffer.ctypes.data,
        buffer.size,
        xp.GlooDataType_t.glooUint8,
        peer,
        tag=17,
    )


def _threaded_transfer(rank, port, operation, bidirectional, progress):
    options = xp.rendezvous.TCPStoreOptions()
    options.port = port
    options.isServer = rank == 0
    options.numWorkers = 2
    options.timeout = timedelta(seconds=10)
    store = xp.rendezvous.TCPStore("127.0.0.1", options)
    prefix = xp.rendezvous.PrefixStore("threaded-transfer", store)
    transport = xp.transport.tcp if sys.platform == "linux" else xp.transport.uv
    device = transport.CreateDevice(transport.attr("127.0.0.1"))
    context = xp.rendezvous.Context(rank, 2)
    context.setTimeout(timedelta(seconds=10))
    context.connectFullMesh(prefix, device)
    xp.barrier(context)
    context.setTimeout(timedelta(seconds=3))

    sendbuf = np.full(192 * 1024, rank + 1, dtype=np.uint8)
    recvbuf = np.zeros_like(sendbuf)
    buffers = {"send": sendbuf, "recv": recvbuf}
    inverse = "recv" if operation == "send" else "send"
    peer = 1 - rank
    if bidirectional or rank == 0:
        entered = threading.Event()

        def transfer():
            entered.set()
            _transfer(operation, context, buffers[operation], peer)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(transfer)
            assert entered.wait(10)
            if bidirectional:
                _transfer(inverse, context, buffers[inverse], peer)
            else:
                # The peer waits for this Python thread before completing the
                # native operation. Let the transfer thread enter its wait.
                time.sleep(0.1)
                assert not future.done(), future.exception()
                progress.set()
            future.result(timeout=10)
    else:
        assert progress.wait(10)
        _transfer(inverse, context, buffers[inverse], peer)

    if bidirectional or (operation == "recv") == (rank == 0):
        np.testing.assert_array_equal(recvbuf, np.full_like(recvbuf, peer + 1))
    xp.barrier(context)


@pytest.mark.parametrize("operation", ["send", "recv"])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_send_recv_releases_gil(operation, bidirectional, unused_tcp_port):
    ctx = mp.get_context("spawn")
    progress = ctx.Event()
    processes = [
        ctx.Process(
            target=_threaded_transfer,
            args=(rank, unused_tcp_port, operation, bidirectional, progress),
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


@pytest.mark.parametrize("operation", ["send", "recv"])
def test_send_recv_propagates_native_error(operation):
    context = xp.rendezvous.Context(0, 1)
    buffer = np.zeros(1, dtype=np.uint8)
    with pytest.raises(RuntimeError, match="peer equals to current rank"):
        _transfer(operation, context, buffer, 0)


def _transfer_timeout(rank, port, operation, completed):
    options = xp.rendezvous.TCPStoreOptions()
    options.port = port
    options.isServer = rank == 0
    options.numWorkers = 2
    options.timeout = timedelta(seconds=10)
    store = xp.rendezvous.TCPStore("127.0.0.1", options)
    prefix = xp.rendezvous.PrefixStore("transfer-timeout", store)
    transport = xp.transport.tcp if sys.platform == "linux" else xp.transport.uv
    device = transport.CreateDevice(transport.attr("127.0.0.1"))
    context = xp.rendezvous.Context(rank, 2)
    context.setTimeout(timedelta(seconds=10))
    context.connectFullMesh(prefix, device)
    xp.barrier(context)
    if rank == 0:
        context.setTimeout(timedelta(milliseconds=200))
        buffer = np.zeros(192 * 1024, dtype=np.uint8)
        with pytest.raises(RuntimeError, match="[Tt]imed out"):
            _transfer(operation, context, buffer, 1)
        # Python remains usable after the native wait throws with the GIL released.
        completed.set()
    else:
        # Keep the connected peer alive without posting the matching operation.
        assert completed.wait(10)


@pytest.mark.parametrize("operation", ["send", "recv"])
def test_send_recv_propagates_connected_timeout(operation, unused_tcp_port):
    ctx = mp.get_context("spawn")
    completed = ctx.Event()
    processes = [
        ctx.Process(
            target=_transfer_timeout,
            args=(rank, unused_tcp_port, operation, completed),
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
