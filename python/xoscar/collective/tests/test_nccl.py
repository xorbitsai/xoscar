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

import asyncio
import multiprocessing as mp
import os

import numpy as np
import pytest

from ... import Actor, create_actor_pool, get_pool_config
from ...context import get_context
from ...tests.core import require_cupy
from ...utils import is_linux
from ..common import (
    COLLECTIVE_DEVICE_ID_ENV_KEY,
    RANK_ADDRESS_ENV_KEY,
    RENDEZVOUS_MASTER_IP_ENV_KEY,
    RENDEZVOUS_MASTER_PORT_ENV_KEY,
)
from ..core import (
    allgather,
    allreduce,
    alltoall,
    broadcast,
    gather,
    init_process_group,
    new_group,
    reduce,
    reduce_scatter,
    scatter,
)


class NcclWorkerActor(Actor):
    def __init__(self, rank, world, *args, **kwargs):
        self._rank = rank
        self._world = world

    async def init_process_group(self):
        os.environ[RANK_ADDRESS_ENV_KEY] = self.address
        return await init_process_group(self._rank, self._world, "nccl")

    async def test_reduce(self):
        import cupy as cp

        sendbuf = cp.array([1, 2, 3, 4], dtype=np.int32)
        recvbuf = cp.zeros((4,), dtype=np.int32)
        _group = [1, 0]
        group = await new_group(_group)
        root = 1
        if group is not None:
            await reduce(sendbuf, recvbuf, group_name=group, root=_group[root])
        if self._rank == _group[root]:
            cp.testing.assert_array_equal(recvbuf, sendbuf * 2)
        else:
            cp.testing.assert_array_equal(recvbuf, cp.zeros_like(sendbuf))

    async def test_allreduce(self):
        import cupy as cp

        sendbuf = cp.array([[1, 2, 3], [1, 2, 3]], dtype=np.int32)
        recvbuf = cp.zeros_like(sendbuf)
        _group = [0, 1]
        group = await new_group(_group)
        # This class handles the CUDA stream handle in RAII way, i.e.,
        # when an Stream instance is destroyed by the GC, its handle is also destroyed.
        # for more information about cupy.cuda.Stream, see: https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html
        stream = cp.cuda.Stream(null=False, non_blocking=True, ptds=False)
        if group is not None:
            await allreduce(sendbuf, recvbuf, group_name=group, stream=stream)
        if self._rank in _group:
            cp.testing.assert_array_equal(recvbuf, sendbuf * len(_group))
        else:
            cp.testing.assert_array_equal(recvbuf, cp.zeros_like(sendbuf))

    async def test_gather(self):
        import cupy as cp

        if self._rank == 0:
            sendbuf = cp.array(
                [[self._rank + 2, self._rank + 2], [self._rank + 2, self._rank + 2]],
                dtype=cp.int32,
            )
        else:
            sendbuf = cp.array(
                [[self._rank + 2, self._rank + 2], [self._rank + 2, self._rank + 2]],
                dtype=cp.int32,
            )
        recvbuf = cp.zeros((2, 4), dtype=np.int32)
        root = 0
        _group = [0, 1]
        group = await new_group(_group)
        if group is not None:
            await gather(sendbuf, recvbuf, group_name=group, root=root)

        if self._rank == _group[root]:
            cp.testing.assert_array_equal(
                recvbuf, np.array([[2, 2, 2, 2], [3, 3, 3, 3]], dtype=np.int32)
            )
        else:
            cp.testing.assert_array_equal(recvbuf, cp.zeros_like(recvbuf))

    async def test_allgather(self):
        import cupy as cp

        if self._rank == 0:
            sendbuf = cp.array([self._rank + 2, self._rank + 2], dtype=np.int32)
        else:
            sendbuf = cp.array([self._rank + 2, self._rank + 2], dtype=np.int32)
        recvbuf = cp.zeros((4,), dtype=np.int32)
        _group = [0, 1]
        group = await new_group(_group)
        if group is not None:
            await allgather(sendbuf, recvbuf, group_name=group)
        if self._rank in _group:
            cp.testing.assert_array_equal(
                recvbuf, cp.array([2, 2, 3, 3], dtype=np.int32)
            )
        else:
            cp.testing.assert_array_equal(recvbuf, cp.zeros_like(recvbuf))

    async def test_scatter(self):
        import cupy as cp

        _group = [0, 1]
        root = 0
        if self._rank == _group[root]:
            sendbuf1 = cp.array([10, 11], dtype=np.int32)
            sendbuf2 = cp.array([12, 13], dtype=np.int32)
        else:
            sendbuf1 = cp.zeros((2,), dtype=np.int32)
            sendbuf2 = cp.zeros((2,), dtype=np.int32)
        recvbuf = cp.zeros((2,), dtype=np.int32)
        send_list = [sendbuf1, sendbuf2]
        group = await new_group(_group)
        if group is not None:
            await scatter(send_list, recvbuf, group_name=group, root=root)

        if self._rank == _group[0]:
            cp.testing.assert_array_equal(recvbuf, cp.array([10, 11], dtype=np.int32))
        elif self._rank == _group[1]:
            cp.testing.assert_array_equal(recvbuf, cp.array([12, 13], dtype=np.int32))
        else:
            cp.testing.assert_array_equal(recvbuf, cp.zeros_like(recvbuf))

    async def test_reduce_scatter(self):
        import cupy as cp

        data = [self._rank + 1, self._rank + 2]
        sendbuf = cp.array(data, dtype=np.int32)
        recvbuf = cp.zeros((1,), dtype=np.int32)
        recv_elems = [1, 1]
        group = await new_group([0, 1])
        if group is not None:
            await reduce_scatter(sendbuf, recvbuf, recv_elems, group_name=group)
        cp.testing.assert_array_equal(recvbuf, cp.array([sum(data)], dtype=np.int32))

    async def test_alltoall(self):
        import cupy as cp

        sendbuf = cp.zeros((2,), dtype=np.float32) + self._rank + 2
        recvbuf = cp.zeros(sendbuf.shape, dtype=np.float32)
        group = await new_group([0, 1])
        if group is not None:
            await alltoall(sendbuf, recvbuf, group_name=group)
        cp.testing.assert_array_equal(recvbuf, cp.array([2, 3], dtype=np.float32))

    async def test_broadcast(self):
        import cupy as cp

        root = 1
        _group = [0, 1]
        sendbuf = cp.zeros((2, 3), dtype=np.int64)
        if self._rank == _group[root]:
            sendbuf = sendbuf + self._rank
        recvbuf = cp.zeros_like(sendbuf, dtype=np.int64)
        group = await new_group(_group)
        if group is not None:
            await broadcast(sendbuf, recvbuf, root=root, group_name=group)
        cp.testing.assert_array_equal(recvbuf, cp.zeros_like(recvbuf) + _group[root])

    async def test_collective_np(self):
        await self.test_broadcast()
        await self.test_reduce()
        await self.test_allreduce()
        await self.test_gather()
        await self.test_allgather()
        await self.test_scatter()
        if is_linux():
            await self.test_reduce_scatter()
        await self.test_alltoall()


@pytest.mark.asyncio
@require_cupy
@pytest.mark.skip(reason="There is only a GPU on CI, but this UT is required 2 GPU!")
async def test_collective():
    mp.set_start_method("spawn", force=True)
    pool = await create_actor_pool(
        "127.0.0.1",
        n_process=2,
        envs=[
            {
                RENDEZVOUS_MASTER_IP_ENV_KEY: "127.0.0.1",
                RENDEZVOUS_MASTER_PORT_ENV_KEY: "25001",
                COLLECTIVE_DEVICE_ID_ENV_KEY: "0",
            },
            {
                RENDEZVOUS_MASTER_IP_ENV_KEY: "127.0.0.1",
                RENDEZVOUS_MASTER_PORT_ENV_KEY: "25001",
                COLLECTIVE_DEVICE_ID_ENV_KEY: "1",
            },
        ],
    )
    config = (await get_pool_config(pool.external_address)).as_dict()
    all_addrs = list(config["mapping"].keys())
    async with pool:
        ctx = get_context()
        r0 = await ctx.create_actor(NcclWorkerActor, 0, 2, address=all_addrs[0])
        r1 = await ctx.create_actor(NcclWorkerActor, 1, 2, address=all_addrs[1])
        t0 = r0.init_process_group()
        t1 = r1.init_process_group()
        await asyncio.gather(t0, t1)
        t0 = r0.test_collective_np()
        t1 = r1.test_collective_np()
        await asyncio.gather(t0, t1)
