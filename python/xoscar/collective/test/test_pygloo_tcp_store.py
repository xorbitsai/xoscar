import multiprocessing as mp
import os
import shutil
import time

import numpy as np

from ...tests.core import require_unix
import platform

fileStore_path = "./collective"
system_name = platform.system()

def worker_allgather(rank):
    from ..gloo import xoscar_pygloo as xp
    from ..rendezvous import xoscar_store as xs

    if rank == 0:
        if os.path.exists(fileStore_path):
            shutil.rmtree(fileStore_path)
        os.makedirs(fileStore_path)
    else:
        time.sleep(0.5)

    context = xp.rendezvous.Context(rank, 2)

    if system_name == "Linux":
        attr = xp.transport.tcp.attr("localhost")
        dev = xp.transport.tcp.CreateDevice(attr)
    else:
        attr = xp.transport.uv.attr("localhost")
        dev = xp.transport.uv.CreateDevice(attr)
    opt = xs.TCPStoreOptions()
    opt.port = 25001
    opt.numWorkers = 2
    if rank == 0:
        opt.isServer = True
    else:
        opt.isServer = False

    store = xs.TCPStore("127.0.0.1", opt)
    custom_store = xp.rendezvous.CustomStore(store)
    store = xp.rendezvous.PrefixStore(str(2), custom_store)

    context.connectFullMesh(store, dev)

    sendbuf = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
    recvbuf = np.zeros([2] + list(sendbuf.shape), dtype=np.float32)
    sendptr = sendbuf.ctypes.data
    recvptr = recvbuf.ctypes.data

    assert sendbuf.size * 2 == recvbuf.size

    data_size = (
        sendbuf.size if isinstance(sendbuf, np.ndarray) else sendbuf.numpy().size
    )
    datatype = xp.glooDataType_t.glooFloat32

    xp.allgather(context, sendptr, recvptr, data_size, datatype)

    print(f"rank {rank} sends {sendbuf},\nreceives {recvbuf}")


@require_unix
def test_allgather():
    process1 = mp.Process(target=worker_allgather, args=(0,))
    process1.start()
    process2 = mp.Process(target=worker_allgather, args=(1,))
    process2.start()

    process1.join()
    process2.join()


def worker_allreduce(rank):
    from ..gloo import xoscar_pygloo as xp
    from ..rendezvous import xoscar_store as xs

    if rank == 0:
        if os.path.exists(fileStore_path):
            shutil.rmtree(fileStore_path)
        os.makedirs(fileStore_path)
    else:
        time.sleep(0.5)

    context = xp.rendezvous.Context(rank, 2)

    if system_name == "Linux":
        attr = xp.transport.tcp.attr("localhost")
        dev = xp.transport.tcp.CreateDevice(attr)
    else:
        attr = xp.transport.uv.attr("localhost")
        dev = xp.transport.uv.CreateDevice(attr)

    opt = xs.TCPStoreOptions()
    opt.port = 25001
    opt.numWorkers = 2
    if rank == 0:
        opt.isServer = True
    else:
        opt.isServer = False

    store = xs.TCPStore("127.0.0.1", opt)
    custom_store = xp.rendezvous.CustomStore(store)
    store = xp.rendezvous.PrefixStore(str(2), custom_store)

    context.connectFullMesh(store, dev)

    sendbuf = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
    recvbuf = np.zeros_like(sendbuf, dtype=np.float32)
    sendptr = sendbuf.ctypes.data
    recvptr = recvbuf.ctypes.data

    data_size = (
        sendbuf.size if isinstance(sendbuf, np.ndarray) else sendbuf.numpy().size
    )
    datatype = xp.glooDataType_t.glooFloat32
    op = xp.ReduceOp.SUM
    algorithm = xp.allreduceAlgorithm.RING

    xp.allreduce(context, sendptr, recvptr, data_size, datatype, op, algorithm)

    print(f"rank {rank} sends {sendbuf}, receives {recvbuf}")


@require_unix
def test_allreduce():
    process1 = mp.Process(target=worker_allreduce, args=(0,))
    process1.start()
    process2 = mp.Process(target=worker_allreduce, args=(1,))
    process2.start()

    process1.join()
    process2.join()


def worker_barrier(rank):
    from ..gloo import xoscar_pygloo as xp
    from ..rendezvous import xoscar_store as xs

    if rank == 0:
        if os.path.exists(fileStore_path):
            shutil.rmtree(fileStore_path)
        os.makedirs(fileStore_path)
    else:
        time.sleep(0.5)

    context = xp.rendezvous.Context(rank, 2)

    if system_name == "Linux":
        attr = xp.transport.tcp.attr("localhost")
        dev = xp.transport.tcp.CreateDevice(attr)
    else:
        attr = xp.transport.uv.attr("localhost")
        dev = xp.transport.uv.CreateDevice(attr)

    opt = xs.TCPStoreOptions()
    opt.port = 25001
    opt.numWorkers = 2
    if rank == 0:
        opt.isServer = True
    else:
        opt.isServer = False

    store = xs.TCPStore("127.0.0.1", opt)
    custom_store = xp.rendezvous.CustomStore(store)
    store = xp.rendezvous.PrefixStore(str(2), custom_store)

    context.connectFullMesh(store, dev)

    sendbuf = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
    recvbuf = np.zeros_like(sendbuf, dtype=np.float32)
    sendptr = sendbuf.ctypes.data
    recvptr = recvbuf.ctypes.data

    data_size = (
        sendbuf.size if isinstance(sendbuf, np.ndarray) else sendbuf.numpy().size
    )
    datatype = xp.glooDataType_t.glooFloat32
    op = xp.ReduceOp.SUM
    algorithm = xp.allreduceAlgorithm.RING

    xp.allreduce(context, sendptr, recvptr, data_size, datatype, op, algorithm)
    xp.barrier(context)

    print(f"rank {rank} sends {sendbuf}, receives {recvbuf}")


@require_unix
def test_barrier():
    process1 = mp.Process(target=worker_barrier, args=(0,))
    process1.start()
    process2 = mp.Process(target=worker_barrier, args=(1,))
    process2.start()

    process1.join()
    process2.join()


def worker_broadcast(rank):
    from ..gloo import xoscar_pygloo as xp
    from ..rendezvous import xoscar_store as xs

    if rank == 0:
        if os.path.exists(fileStore_path):
            shutil.rmtree(fileStore_path)
        os.makedirs(fileStore_path)
    else:
        time.sleep(0.5)

    context = xp.rendezvous.Context(rank, 2)

    if system_name == "Linux":
        attr = xp.transport.tcp.attr("localhost")
        dev = xp.transport.tcp.CreateDevice(attr)
    else:
        attr = xp.transport.uv.attr("localhost")
        dev = xp.transport.uv.CreateDevice(attr)

    opt = xs.TCPStoreOptions()
    opt.port = 25001
    opt.numWorkers = 2
    if rank == 0:
        opt.isServer = True
    else:
        opt.isServer = False

    store = xs.TCPStore("127.0.0.1", opt)
    custom_store = xp.rendezvous.CustomStore(store)
    store = xp.rendezvous.PrefixStore(str(2), custom_store)

    context.connectFullMesh(store, dev)

    if rank == 0:
        sendbuf = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
        sendptr = sendbuf.ctypes.data
    else:
        sendbuf = np.zeros((2, 3), dtype=np.float32)
        sendptr = -1
    recvbuf = np.zeros_like(sendbuf, dtype=np.float32)
    recvptr = recvbuf.ctypes.data

    data_size = (
        sendbuf.size if isinstance(sendbuf, np.ndarray) else sendbuf.numpy().size
    )
    datatype = xp.glooDataType_t.glooFloat32
    root = 0

    xp.broadcast(context, sendptr, recvptr, data_size, datatype, root)

    print(f"rank {rank} sends {sendbuf}, receives {recvbuf}")
    ## example output
    # (pid=36435) rank 1 sends [[0. 0. 0.]
    # (pid=36435)  [0. 0. 0.]], receives [[1. 2. 3.]
    # (pid=36435)  [1. 2. 3.]]
    # (pid=36432) rank 0 sends [[1. 2. 3.]
    # (pid=36432)  [1. 2. 3.]], receives [[1. 2. 3.]
    # (pid=36432)  [1. 2. 3.]]


@require_unix
def test_broadcast():
    process1 = mp.Process(target=worker_broadcast, args=(0,))
    process1.start()
    process2 = mp.Process(target=worker_broadcast, args=(1,))
    process2.start()

    process1.join()
    process2.join()


def worker_gather(rank):
    from ..gloo import xoscar_pygloo as xp
    from ..rendezvous import xoscar_store as xs

    if rank == 0:
        if os.path.exists(fileStore_path):
            shutil.rmtree(fileStore_path)
        os.makedirs(fileStore_path)
    else:
        time.sleep(0.5)

    context = xp.rendezvous.Context(rank, 3)

    if system_name == "Linux":
        attr = xp.transport.tcp.attr("localhost")
        dev = xp.transport.tcp.CreateDevice(attr)
    else:
        attr = xp.transport.uv.attr("localhost")
        dev = xp.transport.uv.CreateDevice(attr)

    opt = xs.TCPStoreOptions()
    opt.port = 25001
    opt.numWorkers = 3
    if rank == 0:
        opt.isServer = True
    else:
        opt.isServer = False

    store = xs.TCPStore("127.0.0.1", opt)
    custom_store = xp.rendezvous.CustomStore(store)
    store = xp.rendezvous.PrefixStore(str(3), custom_store)

    context.connectFullMesh(store, dev)

    sendbuf = np.array([rank, rank + 1], dtype=np.float32)
    sendptr = sendbuf.ctypes.data

    recvbuf = np.zeros((1, 3 * 2), dtype=np.float32)
    recvptr = recvbuf.ctypes.data

    data_size = (
        sendbuf.size if isinstance(sendbuf, np.ndarray) else sendbuf.numpy().size
    )
    datatype = xp.glooDataType_t.glooFloat32

    xp.gather(context, sendptr, recvptr, data_size, datatype, root=0)

    print(f"rank {rank} sends {sendbuf}, receives {recvbuf}")

    ## example output
    # (pid=23172) rank 2 sends [2. 3.], receives [[0. 0. 0. 0. 0. 0.]]
    # (pid=23171) rank 1 sends [1. 2.], receives [[0. 0. 0. 0. 0. 0.]]
    # (pid=23173) rank 0 sends [0. 1.], receives [[0. 1. 1. 2. 2. 3.]]


@require_unix
def test_gather():
    process1 = mp.Process(target=worker_gather, args=(0,))
    process1.start()
    process2 = mp.Process(target=worker_gather, args=(1,))
    process2.start()
    process3 = mp.Process(target=worker_gather, args=(2,))
    process3.start()

    process1.join()
    process2.join()
    process3.join()


def worker_reduce_scatter(rank):
    from ..gloo import xoscar_pygloo as xp
    from ..rendezvous import xoscar_store as xs

    if rank == 0:
        if os.path.exists(fileStore_path):
            shutil.rmtree(fileStore_path)
        os.makedirs(fileStore_path)
    else:
        time.sleep(0.5)

    context = xp.rendezvous.Context(rank, 3)

    if system_name == "Linux":
        attr = xp.transport.tcp.attr("localhost")
        dev = xp.transport.tcp.CreateDevice(attr)
    else:
        attr = xp.transport.uv.attr("localhost")
        dev = xp.transport.uv.CreateDevice(attr)

    opt = xs.TCPStoreOptions()
    opt.port = 25001
    opt.numWorkers = 3
    if rank == 0:
        opt.isServer = True
    else:
        opt.isServer = False

    store = xs.TCPStore("127.0.0.1", opt)
    custom_store = xp.rendezvous.CustomStore(store)
    store = xp.rendezvous.PrefixStore(str(3), custom_store)

    context.connectFullMesh(store, dev)

    sendbuf = np.array(
        [i + 1 for i in range(sum([j + 1 for j in range(3)]))], dtype=np.float32
    )
    sendptr = sendbuf.ctypes.data

    recvbuf = np.zeros((rank + 1,), dtype=np.float32)
    recvptr = recvbuf.ctypes.data
    recvElems = [i + 1 for i in range(3)]

    data_size = (
        sendbuf.size if isinstance(sendbuf, np.ndarray) else sendbuf.numpy().size
    )
    datatype = xp.glooDataType_t.glooFloat32
    op = xp.ReduceOp.SUM

    xp.reduce_scatter(context, sendptr, recvptr, data_size, recvElems, datatype, op)

    print(f"rank {rank} sends {sendbuf}, receives {recvbuf}")


@require_unix
def test_reduce_scatter():
    process1 = mp.Process(target=worker_reduce_scatter, args=(0,))
    process1.start()
    process2 = mp.Process(target=worker_reduce_scatter, args=(1,))
    process2.start()
    process3 = mp.Process(target=worker_reduce_scatter, args=(2,))
    process3.start()

    process1.join()
    process2.join()
    process3.join()


def worker_reduce(rank):
    from ..gloo import xoscar_pygloo as xp
    from ..rendezvous import xoscar_store as xs

    if rank == 0:
        if os.path.exists(fileStore_path):
            shutil.rmtree(fileStore_path)
        os.makedirs(fileStore_path)
    else:
        time.sleep(0.5)

    context = xp.rendezvous.Context(rank, 3)

    if system_name == "Linux":
        attr = xp.transport.tcp.attr("localhost")
        dev = xp.transport.tcp.CreateDevice(attr)
    else:
        attr = xp.transport.uv.attr("localhost")
        dev = xp.transport.uv.CreateDevice(attr)

    opt = xs.TCPStoreOptions()
    opt.port = 25001
    opt.numWorkers = 3
    if rank == 0:
        opt.isServer = True
    else:
        opt.isServer = False

    store = xs.TCPStore("127.0.0.1", opt)
    custom_store = xp.rendezvous.CustomStore(store)
    store = xp.rendezvous.PrefixStore(str(3), custom_store)

    context.connectFullMesh(store, dev)

    sendbuf = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
    recvbuf = np.zeros_like(sendbuf, dtype=np.float32)
    sendptr = sendbuf.ctypes.data
    recvptr = recvbuf.ctypes.data

    data_size = (
        sendbuf.size if isinstance(sendbuf, np.ndarray) else sendbuf.numpy().size
    )
    datatype = xp.glooDataType_t.glooFloat32
    op = xp.ReduceOp.SUM
    root = 0

    xp.reduce(context, sendptr, recvptr, data_size, datatype, op, root)

    print(f"rank {rank} sends {sendbuf}, receives {recvbuf}")


@require_unix
def test_reduce():
    process1 = mp.Process(target=worker_reduce, args=(0,))
    process1.start()
    process2 = mp.Process(target=worker_reduce, args=(1,))
    process2.start()
    process3 = mp.Process(target=worker_reduce, args=(2,))
    process3.start()

    process1.join()
    process2.join()
    process3.join()


def worker_scatter(rank):
    from ..gloo import xoscar_pygloo as xp
    from ..rendezvous import xoscar_store as xs

    if rank == 0:
        if os.path.exists(fileStore_path):
            shutil.rmtree(fileStore_path)
        os.makedirs(fileStore_path)
    else:
        time.sleep(0.5)

    context = xp.rendezvous.Context(rank, 2)

    if system_name == "Linux":
        attr = xp.transport.tcp.attr("localhost")
        dev = xp.transport.tcp.CreateDevice(attr)
    else:
        attr = xp.transport.uv.attr("localhost")
        dev = xp.transport.uv.CreateDevice(attr)

    opt = xs.TCPStoreOptions()
    opt.port = 25001
    opt.numWorkers = 2
    if rank == 0:
        opt.isServer = True
    else:
        opt.isServer = False

    store = xs.TCPStore("127.0.0.1", opt)
    custom_store = xp.rendezvous.CustomStore(store)
    store = xp.rendezvous.PrefixStore(str(2), custom_store)

    context.connectFullMesh(store, dev)

    sendbuf = [np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)] * 2
    recvbuf = np.zeros((2, 3), dtype=np.float32)
    sendptr = []
    for i in sendbuf:
        sendptr.append(i.ctypes.data)
    recvptr = recvbuf.ctypes.data

    data_size = (
        sendbuf[0].size
        if isinstance(sendbuf[0], np.ndarray)
        else sendbuf[0].numpy().size
    )
    datatype = xp.glooDataType_t.glooFloat32
    root = 0

    xp.scatter(context, sendptr, recvptr, data_size, datatype, root)

    print(f"rank {rank} sends {sendbuf}, receives {recvbuf}")
    ## example output, root is 0.
    # (pid=18951) rank 1 sends [array([[1., 2., 3.],
    # (pid=18951)        [1., 2., 3.]], dtype=float32), array([[1., 2., 3.],
    # (pid=18951)        [1., 2., 3.]], dtype=float32)], receives [[1. 2. 3.]
    # (pid=18951)  [1. 2. 3.]]
    # (pid=18952) rank 0 sends [array([[1., 2., 3.],
    # (pid=18952)        [1., 2., 3.]], dtype=float32), array([[1., 2., 3.],
    # (pid=18952)        [1., 2., 3.]], dtype=float32)], receives [[1. 2. 3.]
    # (pid=18952)  [1. 2. 3.]]


@require_unix
def test_scatter():
    process1 = mp.Process(target=worker_scatter, args=(0,))
    process1.start()
    process2 = mp.Process(target=worker_scatter, args=(1,))
    process2.start()

    process1.join()
    process2.join()


def worker_send_recv(rank):
    from ..gloo import xoscar_pygloo as xp
    from ..rendezvous import xoscar_store as xs

    if rank == 0:
        if os.path.exists(fileStore_path):
            shutil.rmtree(fileStore_path)
        os.makedirs(fileStore_path)
    else:
        time.sleep(0.5)

    context = xp.rendezvous.Context(rank, 2)

    if system_name == "Linux":
        attr = xp.transport.tcp.attr("localhost")
        dev = xp.transport.tcp.CreateDevice(attr)
    else:
        attr = xp.transport.uv.attr("localhost")
        dev = xp.transport.uv.CreateDevice(attr)

    opt = xs.TCPStoreOptions()
    opt.port = 25001
    opt.numWorkers = 2
    if rank == 0:
        opt.isServer = True
    else:
        opt.isServer = False

    store = xs.TCPStore("127.0.0.1", opt)
    custom_store = xp.rendezvous.CustomStore(store)
    store = xp.rendezvous.PrefixStore(str(2), custom_store)

    context.connectFullMesh(store, dev)

    if rank == 0:
        sendbuf = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
        sendptr = sendbuf.ctypes.data

        data_size = (
            sendbuf.size if isinstance(sendbuf, np.ndarray) else sendbuf.numpy().size
        )
        datatype = xp.glooDataType_t.glooFloat32
        peer = 1
        xp.send(context, sendptr, data_size, datatype, peer)
        print(f"rank {rank} sends {sendbuf}")

    elif rank == 1:
        recvbuf = np.zeros((2, 3), dtype=np.float32)
        recvptr = recvbuf.ctypes.data

        data_size = (
            recvbuf.size if isinstance(recvbuf, np.ndarray) else recvbuf.numpy().size
        )
        datatype = xp.glooDataType_t.glooFloat32
        peer = 0

        xp.recv(context, recvptr, data_size, datatype, peer)
        print(f"rank {rank} receives {recvbuf}")
    else:
        raise Exception(
            "Only support 2 process to test send function and recv function"
        )
    ## example output


@require_unix
def test_send_recv():
    process1 = mp.Process(target=worker_send_recv, args=(0,))
    process1.start()
    process2 = mp.Process(target=worker_send_recv, args=(1,))
    process2.start()

    process1.join()
    process2.join()
