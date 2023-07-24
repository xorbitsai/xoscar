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

from enum import Enum
from typing import Tuple

import cupy
import cupy.cuda.nccl as nccl


class ReduceOp(Enum):
    SUM = nccl.NCCL_SUM
    PRODUCT = nccl.NCCL_PROD
    MIN = nccl.NCCL_MIN
    MAX = nccl.NCCL_MAX


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


class Context:
    def __init__(self, n_devices: int, nccl_unique_id: Tuple, rank: int):
        self.communicator = nccl.NcclCommunicator(n_devices, nccl_unique_id, rank)
        self.n_devices = n_devices
        self.rank = rank


def allreduce(
    context: Context,
    sendbuf: cupy.ndarray,
    recvbuf: cupy.ndarray,
    op: ReduceOp = ReduceOp.SUM,
):
    context.communicator.allReduce(
        get_buffer_ptr(sendbuf),
        get_buffer_ptr(recvbuf),
        get_buffer_n_elements(sendbuf),
        get_nccl_buffer_dtype(sendbuf),
        op.value,
        cupy.cuda.Stream.null.ptr,
    )


def allgather(
    context: Context,
    sendbuf: cupy.ndarray,
    recvbuf: cupy.ndarray,
):
    context.communicator.allGather(
        get_buffer_ptr(sendbuf),
        get_buffer_ptr(recvbuf),
        get_buffer_n_elements(sendbuf),
        get_nccl_buffer_dtype(sendbuf),
        cupy.cuda.Stream.null.ptr,
    )


def all_to_all(
    context: Context,
    sendbuf: cupy.ndarray,
    recvbuf: cupy.ndarray,
):
    assert context.n_devices == sendbuf.shape[0]

    for i in range(context.n_devices):
        if context.rank == i:
            buf_i = []
            for peer in range(context.n_devices):
                if peer == context.rank:
                    buf_i.append(sendbuf[i])
                else:
                    temp_recv = sendbuf[i].copy()
                    recv(context, temp_recv, peer)
                    buf_i.append(temp_recv)
            cupy_buf_i = cupy.concatenate(buf_i)
            recvbuf[:] = cupy_buf_i.reshape(recvbuf.shape)
        else:
            send(context, sendbuf[i], i)


def reduce(
    context: Context,
    sendbuf: cupy.ndarray,
    recvbuf: cupy.ndarray,
    root: int,
    op: ReduceOp = ReduceOp.SUM,
):
    context.communicator.reduce(
        get_buffer_ptr(sendbuf),
        get_buffer_ptr(recvbuf),
        get_buffer_n_elements(sendbuf),
        get_nccl_buffer_dtype(sendbuf),
        op.value,
        root,
        cupy.cuda.Stream.null.ptr,
    )


def scatter(
    context: Context,
    sendbuf: cupy.ndarray,
    recvbuf: cupy.ndarray,
    root: int,
):
    if context.rank == root:
        # make sendbuf equally scattered into context.n_devices chunks
        scattered_send_buf = scatter_buffer(context, sendbuf)
        for peer in range(context.n_devices):
            if peer == context.rank:
                continue
            send(context, scattered_send_buf[peer], peer)
        recvbuf[:] = scattered_send_buf[context.rank].reshape(recvbuf.shape)
    else:
        recv(context, recvbuf, root)


def gather(
    context: Context,
    sendbuf: cupy.ndarray,
    recvbuf: cupy.ndarray,
    root: int,
):
    if context.rank == root:
        buffs = []
        for peer in range(context.n_devices):
            if peer == context.rank:
                buffs.append(sendbuf)
            else:
                temp_recv = sendbuf.copy()
                recv(context, temp_recv, peer)
                buffs.append(temp_recv)

        cupy_buff = cupy.concatenate(buffs)
        recvbuf[:] = cupy_buff.reshape(recvbuf.shape)
    else:
        send(context, sendbuf, root)


def send(
    context: Context,
    sendbuf: cupy.ndarray,
    peer: int,
):
    context.communicator.send(
        get_buffer_ptr(sendbuf),
        get_buffer_n_elements(sendbuf),
        get_nccl_buffer_dtype(sendbuf),
        peer,
        cupy.cuda.Stream.null.ptr,
    )


def recv(
    context: Context,
    recvbuf: cupy.ndarray,
    peer: int,
):
    context.communicator.recv(
        get_buffer_ptr(recvbuf),
        get_buffer_n_elements(recvbuf),
        get_nccl_buffer_dtype(recvbuf),
        peer,
        cupy.cuda.Stream.null.ptr,
    )


def broadcast(
    context: Context,
    sendbuf: cupy.ndarray,
    recvbuf: cupy.ndarray,
    root: int,
):
    context.communicator.broadcast(
        get_buffer_ptr(sendbuf),
        get_buffer_ptr(recvbuf),
        get_buffer_n_elements(sendbuf),
        get_nccl_buffer_dtype(sendbuf),
        root,
        cupy.cuda.Stream.null.ptr,
    )


def reduce_scatter(
    context: Context,
    sendbuf: cupy.ndarray,
    recvbuf: cupy.ndarray,
    op: ReduceOp = ReduceOp.SUM,
):
    reduce_recv = sendbuf.copy()
    reduce(context, sendbuf, reduce_recv, 0, op)
    scatter(context, reduce_recv, recvbuf, 0)


def barrier(
    context: Context,
):
    barrier_tensors = [None] * context.n_devices
    for i in range(context.n_devices):
        with cupy.cuda.Device(i):
            barrier_tensors[i] = cupy.array([1])
    allreduce(context, barrier_tensors, barrier_tensors)


def scatter_buffer(context: Context, buffer):
    n_chunks = context.n_devices
    chunk_size = buffer.shape[0] // n_chunks
    if buffer.shape[0] % n_chunks != 0:
        raise ValueError(
            "Buffer size is not exactly divisible by the number of devices"
        )

    chunks = []
    ptr = buffer.data.ptr
    dtype_size = buffer.dtype.itemsize
    chunk_elements = chunk_size * buffer.size // buffer.shape[0]
    for i in range(n_chunks):
        cupy.cuda.Device(i).use()
        chunk_ptr = ptr + i * chunk_elements * dtype_size
        chunk_shape = (
            (chunk_size,) + buffer.shape[1:] if buffer.ndim > 1 else (chunk_size,)
        )
        chunk = cupy.ndarray(
            chunk_shape,
            dtype=buffer.dtype,
            memptr=cupy.cuda.MemoryPointer(
                cupy.cuda.memory.UnownedMemory(
                    chunk_ptr, chunk_elements * dtype_size, None
                ),
                0,
            ),
        )
        chunks.append(chunk)

    return chunks


def get_buffer_ptr(buffer):
    if isinstance(buffer, cupy.ndarray):
        return buffer.data.ptr

    raise ValueError("Buffer type not supported")


def get_buffer_n_elements(buffer):
    if isinstance(buffer, (cupy.ndarray)):
        return buffer.size

    raise ValueError("Buffer type not supported")


def get_nccl_buffer_dtype(buffer):
    if isinstance(buffer, cupy.ndarray):
        return CUPY_NCCL_DTYPE_MAP[buffer.dtype.type]

    raise ValueError("Buffer type not supported")
