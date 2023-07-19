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
from typing import Optional, Tuple

import cupy
import cupy.cuda.nccl as nccl


class ReduceOp(Enum):
    SUM = nccl.NCCL_SUM
    PRODUCT = nccl.NCCL_PROD
    MIN = nccl.NCCL_MIN
    MAX = nccl.NCCL_MAX


NUMPY_NCCL_DTYPE_MAP = {
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


def allreduce(
    context: Optional[Context] = None,
    sendbuf: Optional[cupy.ndarray] = None,
    recvbuf: Optional[cupy.ndarray] = None,
    reduceop: Optional[ReduceOp] = ReduceOp.SUM,
):
    context.communicator.allReduce(
        get_buffer_ptr(sendbuf),
        get_buffer_ptr(recvbuf),
        get_buffer_n_elements(sendbuf),
        get_nccl_buffer_dtype(sendbuf),
        reduceop.value,
        cupy.cuda.Stream.null.ptr,
    )


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
        return NUMPY_NCCL_DTYPE_MAP[buffer.dtype.type]

    raise ValueError("Unspported GPU buffer type")
