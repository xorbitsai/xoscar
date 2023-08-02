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
import os

import numpy as np
import pytest

from ... import Actor, create_actor_pool, get_pool_config
from ...context import get_context
from ...tests.core import require_cupy
from ...utils import is_linux
from ..common import (
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


class WorkerActor(Actor):
    def __init__(self, rank, device_id, world, commId, *args, **kwargs):
        self._rank = rank
        self.device_id = device_id
        self._world = world
        self._commId = commId

    async def init_process_group(self):
        os.environ[RANK_ADDRESS_ENV_KEY] = self.address
        return await init_process_group(
            self._rank, self._world, "nccl", self.device_id, self._commId
        )

    async def test_reduce(self, cid):
        sendbuf = np.array([1, 2, 3, 4], dtype=np.int32)
        recvbuf = np.zeros((4,), dtype=np.int32)
        _group = [0, 1]
        group = await new_group(_group, cid)
        root = 1
        if group is not None:
            await reduce(sendbuf, recvbuf, group_name=group, root=root)

        if self._rank == _group[root]:
            np.testing.assert_array_equal(recvbuf, sendbuf * 2)
        else:
            np.testing.assert_array_equal(recvbuf, np.zeros_like(sendbuf))

    async def test_allreduce(self, cid):
        sendbuf = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int32)
        recvbuf = np.zeros_like(sendbuf)
        _group = [0, 1]
        group = await new_group(_group, cid)
        if group is not None:
            await allreduce(sendbuf, recvbuf, group_name=group)
        if self._rank in _group:
            np.testing.assert_array_equal(recvbuf, sendbuf * len(_group))
        else:
            np.testing.assert_array_equal(recvbuf, np.zeros_like(sendbuf))

    async def test_gather(self, cid):
        if self._rank == 0:
            sendbuf = np.array(
                [[self._rank + 2, self._rank + 2], [self._rank + 2, self._rank + 2]],
                dtype=np.int32,
            )
        else:
            sendbuf = np.array(
                [[self._rank + 2, self._rank + 2], [self._rank + 2, self._rank + 2]],
                dtype=np.int32,
            )
        recvbuf = np.zeros((2, 4), dtype=np.int32)
        root = 0
        _group = [0, 1]
        group = await new_group(_group, cid)
        if group is not None:
            await gather(sendbuf, recvbuf, group_name=group, root=root)

        if self._rank == _group[root]:
            np.testing.assert_array_equal(
                recvbuf, np.array([[2, 2, 2, 2], [3, 3, 3, 3]], dtype=np.int32)
            )
        else:
            np.testing.assert_array_equal(recvbuf, np.zeros_like(recvbuf))

    async def test_allgather(self, cid):
        if self._rank == 0:
            sendbuf = np.array([self._rank + 2, self._rank + 2], dtype=np.int32)
        else:
            sendbuf = np.array([self._rank + 2, self._rank + 2], dtype=np.int32)
        recvbuf = np.zeros((4,), dtype=np.int32)
        _group = [0, 1]
        group = await new_group(_group, cid)
        if group is not None:
            await allgather(sendbuf, recvbuf, group_name=group)
        if self._rank in _group:
            np.testing.assert_array_equal(
                recvbuf, np.array([2, 2, 3, 3], dtype=np.int32)
            )
        else:
            np.testing.assert_array_equal(recvbuf, np.zeros_like(recvbuf))

    async def test_scatter(self, cid):
        _group = [0, 1]
        root = 0
        if self._rank == _group[root]:
            sendbuf1 = np.array([10, 11], dtype=np.int32)
            sendbuf2 = np.array([12, 13], dtype=np.int32)
        else:
            sendbuf1 = np.zeros((2,), dtype=np.int32)
            sendbuf2 = np.zeros((2,), dtype=np.int32)
        recvbuf = np.zeros((2,), dtype=np.int32)
        send_list = [sendbuf1, sendbuf2]
        group = await new_group(_group, cid)
        if group is not None:
            await scatter(send_list, recvbuf, group_name=group, root=root)

        if self._rank == _group[0]:
            np.testing.assert_array_equal(recvbuf, np.array([10, 11], dtype=np.int32))
        elif self._rank == _group[1]:
            np.testing.assert_array_equal(recvbuf, np.array([12, 13], dtype=np.int32))
        else:
            np.testing.assert_array_equal(recvbuf, np.zeros_like(recvbuf))

    async def test_reduce_scatter(self, cid):
        data = [self._rank + 1, self._rank + 2]
        sendbuf = np.array(data, dtype=np.int32)
        recvbuf = np.zeros((1,), dtype=np.int32)
        recv_elems = [1, 1]
        group = await new_group([0, 1], cid)
        if group is not None:
            await reduce_scatter(sendbuf, recvbuf, recv_elems, group_name=group)
        np.testing.assert_array_equal(recvbuf, np.array([sum(data)], dtype=np.int32))

    async def test_alltoall(self, cid):
        sendbuf = np.zeros((2,), dtype=np.float32) + self._rank
        recvbuf = np.zeros(sendbuf.shape, dtype=np.float32)
        group = await new_group([0, 1], cid)
        if group is not None:
            await alltoall(sendbuf, recvbuf, group_name=group)
        np.testing.assert_array_equal(recvbuf, np.array([0, 1], dtype=np.float32))

    async def test_broadcast(self, cid):
        root = 1
        _group = [0, 1]
        sendbuf = np.zeros((2, 3), dtype=np.int64)
        if self._rank == _group[root]:
            sendbuf = sendbuf + self._rank
        recvbuf = np.zeros_like(sendbuf, dtype=np.int64)
        group = await new_group(_group, cid)
        if group is not None:
            await broadcast(sendbuf, recvbuf, root=root, group_name=group)
        np.testing.assert_array_equal(recvbuf, np.zeros_like(recvbuf) + _group[root])

    async def test_collective_np(self, cid):
        await self.test_broadcast(cid)
        await self.test_reduce(cid)
        await self.test_allreduce(cid)
        await self.test_gather(cid)
        await self.test_allgather(cid)
        await self.test_scatter(cid)
        if is_linux():
            await self.test_reduce_scatter(cid)
        await self.test_alltoall(cid)


@pytest.mark.asyncio
@require_cupy
async def test_collective():
    from cupy.cuda import nccl

    pool = await create_actor_pool(
        "127.0.0.1",
        n_process=2,
        envs=[
            {
                RENDEZVOUS_MASTER_IP_ENV_KEY: "127.0.0.1",
                RENDEZVOUS_MASTER_PORT_ENV_KEY: "25001",
            }
        ]
        * 2,
    )
    config = (await get_pool_config(pool.external_address)).as_dict()
    all_addrs = list(config["mapping"].keys())
    async with pool:
        ctx = get_context()
        cid = nccl.get_unique_id()
        r0 = await ctx.create_actor(WorkerActor, 0, 0, 2, cid, address=all_addrs[0])
        r1 = await ctx.create_actor(WorkerActor, 1, 1, 2, cid, address=all_addrs[1])
        t0 = r0.init_process_group()
        t1 = r1.init_process_group()
        await asyncio.gather(*[t0, t1])
        cid = nccl.get_unique_id()
        t0 = r0.test_collective_np(cid)
        t1 = r1.test_collective_np(cid)
        await asyncio.gather(*[t0, t1])
