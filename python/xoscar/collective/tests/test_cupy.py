# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing as mp

import cupy
import cupy.cuda.nccl as nccl
import numpy as np
import pytest

from .. import xoscar_cupy as xc

mp.set_start_method("spawn")


def worker_allreduce(rank, nccl_id, device_id):
    cupy.cuda.Device(device_id).use()

    context = xc.Context(n_devices=2, nccl_unique_id=nccl_id, rank=rank)
    sendbuf = cupy.array([[1, 2, 3], [1, 2, 3]], dtype=cupy.float32)
    recvbuf = cupy.zeros_like(sendbuf, dtype=cupy.float32)
    op = xc.ReduceOp.SUM

    xc.allreduce(context, sendbuf, recvbuf, op)
    cupy.testing.assert_array_equal(recvbuf, cupy.array(sendbuf * 2))


def worker_allgather(rank, nccl_id, device_id):
    cupy.cuda.Device(device_id).use()

    context = xc.Context(n_devices=2, nccl_unique_id=nccl_id, rank=rank)
    sendbuf = cupy.array([[1, 2, 3], [1, 2, 3]], dtype=cupy.float32)
    recvbuf = cupy.zeros([2] + list(sendbuf.shape), dtype=cupy.float32)

    xc.allgather(context, sendbuf, recvbuf)
    cupy.testing.assert_array_equal(recvbuf, cupy.array([sendbuf] * 2))


def worker_reduce(rank, nccl_id, device_id):
    cupy.cuda.Device(device_id).use()

    context = xc.Context(n_devices=2, nccl_unique_id=nccl_id, rank=rank)
    sendbuf = cupy.array([[1, 2, 3], [1, 2, 3]], dtype=cupy.float32)
    recvbuf = cupy.zeros_like(sendbuf, dtype=cupy.float32)
    op = xc.ReduceOp.SUM
    root = 0

    xc.reduce(context, sendbuf, recvbuf, root, op)
    if rank == root:
        cupy.testing.assert_array_equal(
            recvbuf,
            cupy.array(
                [
                    [
                        2.0,
                        4.0,
                        6.0,
                    ],
                    [2.0, 4.0, 6.0],
                ]
            ),
        )
    else:
        cupy.testing.assert_array_equal(
            recvbuf, cupy.zeros_like(sendbuf, dtype=cupy.float32)
        )


def worker_broadcast(rank, nccl_id, device_id):
    cupy.cuda.Device(device_id).use()

    context = xc.Context(n_devices=2, nccl_unique_id=nccl_id, rank=rank)

    if rank == 0:
        sendbuf = cupy.array([[1, 2, 3], [1, 2, 3]], dtype=cupy.float32)
    else:
        sendbuf = cupy.zeros((2, 3), dtype=cupy.float32)

    recvbuf = cupy.zeros_like(sendbuf, dtype=cupy.float32)
    root = 0

    xc.broadcast(context, sendbuf, recvbuf, root)

    cupy.testing.assert_array_equal(
        recvbuf, cupy.array([[1, 2, 3], [1, 2, 3]], dtype=cupy.float32)
    )


def worker_scatter(rank, nccl_id, device_id):
    cupy.cuda.Device(device_id).use()

    context = xc.Context(n_devices=2, nccl_unique_id=nccl_id, rank=rank)

    sendbuf = cupy.array([1, 2, 3, 4, 5, 6], dtype=cupy.float32)
    recvbuf = cupy.array([0, 0, 0], dtype=cupy.float32)
    root = 0

    xc.scatter(context, sendbuf, recvbuf, root)

    if rank == 0:
        cupy.testing.assert_array_equal(recvbuf, cupy.array([1.0, 2.0, 3.0]))
    elif rank == 1:
        cupy.testing.assert_array_equal(recvbuf, cupy.array([4.0, 5.0, 6.0]))


def worker_gather(rank, nccl_id, device_id):
    cupy.cuda.Device(device_id).use()

    context = xc.Context(n_devices=2, nccl_unique_id=nccl_id, rank=rank)

    if rank == 0:
        sendbuf = cupy.array([1, 2, 3], dtype=cupy.float32)
    else:
        sendbuf = cupy.array([4, 5, 6], dtype=cupy.float32)

    recvbuf = cupy.array([0, 0, 0, 0, 0, 0], dtype=cupy.float32)
    root = 0

    xc.gather(context, sendbuf, recvbuf, root)

    if rank == 0:
        cupy.testing.assert_array_equal(
            recvbuf, cupy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        )


def worker_reduce_scatter(rank, nccl_id, device_id):
    cupy.cuda.Device(device_id).use()

    context = xc.Context(n_devices=2, nccl_unique_id=nccl_id, rank=rank)

    sendbuf = cupy.array([1, 2, 3, 4], dtype=cupy.float32)
    recvbuf = cupy.array([0, 0], dtype=cupy.float32)
    op = xc.ReduceOp.SUM

    xc.reduce_scatter(context, sendbuf, recvbuf, op)

    if rank == 0:
        cupy.testing.assert_array_equal(recvbuf, cupy.array([2.0, 4.0]))
    elif rank == 1:
        cupy.testing.assert_array_equal(recvbuf, cupy.array([6.0, 8.0]))


def worker_all_to_all(rank, nccl_id, device_id):
    cupy.cuda.Device(device_id).use()

    context = xc.Context(n_devices=2, nccl_unique_id=nccl_id, rank=rank)

    if rank == 0:
        sendbuf = cupy.array([[1, 2], [3, 4]], dtype=cupy.float32)
    else:
        sendbuf = cupy.array([[2, 4], [6, 8]], dtype=cupy.float32)
    recvbuf = cupy.zeros((2, 2), dtype=cupy.float32)

    xc.all_to_all(context, sendbuf, recvbuf)

    if rank == 0:
        cupy.testing.assert_array_equal(recvbuf, cupy.array([[1.0, 2.0], [2.0, 4.0]]))
    elif rank == 1:
        cupy.testing.assert_array_equal(recvbuf, cupy.array([[3.0, 4.0], [6.0, 8.0]]))


def worker_barrier(rank, nccl_id, device_id):
    cupy.cuda.Device(device_id).use()

    context = xc.Context(n_devices=2, nccl_unique_id=nccl_id, rank=rank)
    sendbuf = cupy.array([[1, 2, 3], [1, 2, 3]], dtype=cupy.float32)
    recvbuf = cupy.zeros_like(sendbuf, dtype=cupy.float32)
    op = xc.ReduceOp.SUM

    xc.allreduce(context, sendbuf, recvbuf, op)
    xc.barrier(context)

    cupy.testing.assert_array_equal(recvbuf, cupy.array(sendbuf * 2))


def worker_scatter_buffer(rank, nccl_id, device_id):
    cupy.cuda.Device(device_id).use()

    context = xc.Context(n_devices=2, nccl_unique_id=nccl_id, rank=rank)

    with pytest.raises(
        ValueError,
        match="Buffer size is not exactly divisible by the number of devices",
    ):
        not_divisible_buffer = cupy.array([1, 2, 3], dtype=cupy.float32)
        xc.scatter_buffer(context, not_divisible_buffer)

    with pytest.raises(
        ValueError,
        match="Buffer size is not exactly divisible by the number of devices",
    ):
        not_divisible_buffer = cupy.array([[1, 2, 3, 4]], dtype=cupy.float32)
        xc.scatter_buffer(context, not_divisible_buffer)

    buffer = cupy.array([1, 2, 3, 4], dtype=cupy.float32)
    buffer_chunks = xc.scatter_buffer(context, buffer)
    assert len(buffer_chunks) == 2
    cupy.testing.assert_array_equal(buffer_chunks[0], cupy.array([1, 2]))
    cupy.testing.assert_array_equal(buffer_chunks[1], cupy.array([3, 4]))

    buffer = cupy.array([[1, 2], [3, 4]], dtype=cupy.float32)
    buffer_chunks = xc.scatter_buffer(context, buffer)
    assert len(buffer_chunks) == 2
    cupy.testing.assert_array_equal(buffer_chunks[0], cupy.array([[1, 2]]))
    cupy.testing.assert_array_equal(buffer_chunks[1], cupy.array([[3, 4]]))

    buffer = cupy.array([[1], [2], [3], [4]], dtype=cupy.float32)
    buffer_chunks = xc.scatter_buffer(context, buffer)
    assert len(buffer_chunks) == 2
    cupy.testing.assert_array_equal(buffer_chunks[0], cupy.array([[1], [2]]))
    cupy.testing.assert_array_equal(buffer_chunks[1], cupy.array([[3], [4]]))


@pytest.mark.parametrize(
    "worker_func",
    [
        worker_gather,
        worker_all_to_all,
        worker_allgather,
        worker_allreduce,
        worker_broadcast,
        worker_reduce,
        worker_reduce_scatter,
        worker_scatter,
        worker_barrier,
        worker_scatter_buffer,
    ],
)
def test_driver(worker_func):
    nccl_id = nccl.get_unique_id()
    device_id = [0, 1]

    process1 = mp.Process(target=worker_func, args=(0, nccl_id, device_id[0]))
    process1.start()
    process2 = mp.Process(target=worker_func, args=(1, nccl_id, device_id[1]))
    process2.start()

    process1.join()
    process2.join()

    assert process1.exitcode == 0, "Process 1 encountered an error."
    assert process2.exitcode == 0, "Process 2 encountered an error."


def test_get_buffer_ptr():
    buffer = cupy.array([1, 2, 3], dtype=cupy.float32)

    ptr = xc.get_buffer_ptr(buffer)
    assert isinstance(ptr, int)
    assert ptr == buffer.data.ptr

    with pytest.raises(ValueError, match="Buffer type not supported"):
        list_buffer = [1, 2, 3]
        xc.get_buffer_ptr(list_buffer)

    with pytest.raises(ValueError, match="Buffer type not supported"):
        np_buffer = np.array([1, 2, 3])
        xc.get_buffer_ptr(np_buffer)


def test_get_buffer_n_elements():
    buffer = cupy.array([1, 2, 3], dtype=cupy.float32)

    n = xc.get_buffer_n_elements(buffer)
    assert n == buffer.size

    with pytest.raises(ValueError, match="Buffer type not supported"):
        list_buffer = [1, 2, 3]
        xc.get_buffer_n_elements(list_buffer)

    with pytest.raises(ValueError, match="Buffer type not supported"):
        np_buffer = np.array([1, 2, 3])
        xc.get_buffer_n_elements(np_buffer)


def test_get_nccl_buffer_dtype():
    CUPY_NCCL_DTYPE_MAP = {
        cupy.uint8: nccl.NCCL_UINT8,
        cupy.uint32: nccl.NCCL_UINT32,
        cupy.uint64: nccl.NCCL_UINT64,
        cupy.int8: nccl.NCCL_INT8,
        cupy.int32: nccl.NCCL_INT32,
        cupy.int64: nccl.NCCL_INT64,
        cupy.half: nccl.NCCL_HALF,
        cupy.float16: nccl.NCCL_FLOAT16,
        cupy.float32: nccl.NCCL_FLOAT32,
        cupy.float64: nccl.NCCL_FLOAT64,
        cupy.double: nccl.NCCL_DOUBLE,
    }

    for cupy_dtype in CUPY_NCCL_DTYPE_MAP:
        buffer = cupy.array([1, 2, 3], dtype=cupy_dtype)
        assert xc.get_nccl_buffer_dtype(buffer) == CUPY_NCCL_DTYPE_MAP[cupy_dtype]

    with pytest.raises(ValueError, match="Buffer type not supported"):
        list_buffer = [1, 2, 3]
        xc.get_nccl_buffer_dtype(list_buffer)

    with pytest.raises(ValueError, match="Buffer type not supported"):
        np_buffer = np.array([1, 2, 3])
        xc.get_nccl_buffer_dtype(np_buffer)
