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

from ...tests.core import require_unix


def worker_allreduce(rank, nccl_id, device_id):
    from .. import xoscar_cupy as xc

    cupy.cuda.Device(device_id).use()

    context = xc.Context(n_devices=2, nccl_unique_id=nccl_id, rank=rank)
    sendbuf = cupy.array([[1, 2, 3], [1, 2, 3]], dtype=cupy.float32)
    recvbuf = cupy.zeros_like(sendbuf, dtype=cupy.float32)
    op = xc.ReduceOp.SUM

    xc.allreduce(context, sendbuf, recvbuf, op)
    cupy.testing.assert_array_equal(recvbuf, cupy.array(sendbuf * 2))


@require_unix
def test_allreduce():
    mp.set_start_method("spawn")

    nccl_id = nccl.get_unique_id()
    device_id = [0, 1]

    process1 = mp.Process(target=worker_allreduce, args=(0, nccl_id, device_id[0]))
    process1.start()
    process2 = mp.Process(target=worker_allreduce, args=(1, nccl_id, device_id[1]))
    process2.start()

    process1.join()
    process2.join()

    assert process1.exitcode == 0, "Process 1 encountered an error."
    assert process2.exitcode == 0, "Process 2 encountered an error."
