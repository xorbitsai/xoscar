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

from ... import Actor, ActorRefType, actor_ref, create_actor_pool, get_pool_config
from ...context import get_context
from ...tests.core import require_unix
from ...utils import is_linux
from ..common import (
    RANK_ADDRESS_ENV_KEY,
    RENDEZVOUS_MASTER_IP_ENV_KEY,
    RENDEZVOUS_MASTER_PORT_ENV_KEY,
)
from ..core import (
    RankActor,
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
from ..process_group import ProcessGroup


class WorkerActor(Actor):
    def __init__(self, rank, world, *args, **kwargs):
        self._rank = rank
        self._world = world

    async def init_process_group(self):
        os.environ[RANK_ADDRESS_ENV_KEY] = self.address
        return await init_process_group(self._rank, self._world)

    async def init_process_group_without_env(self):
        with pytest.raises(RuntimeError):
            await init_process_group(self._rank, self._world)

    async def test_params(self):
        rank_ref: ActorRefType[RankActor] = await actor_ref(
            address=self.address, uid="RankActor"
        )
        uid = rank_ref.uid
        assert uid == bytes(RankActor.default_uid(), "utf-8")

        rank = await rank_ref.rank()
        assert rank == self._rank

        world = await rank_ref.world()
        assert world == self._world

        backend = await rank_ref.backend()
        assert backend == "gloo"

        pg: ProcessGroup = await rank_ref.process_group("default")
        assert pg is not None

        assert pg.rank == self._rank
        assert pg.name == "default"
        assert pg.world_size == self._world
        assert pg.options is None

    async def test_reduce(self):
        sendbuf = np.array([1, 2, 3, 4], dtype=np.int32)
        recvbuf = np.zeros((4,), dtype=np.int32)
        _group = [0, 1, 2]
        group = await new_group(_group)
        root = 1
        if group is not None:
            await reduce(sendbuf, recvbuf, group_name=group, root=root)

        if self._rank == _group[root]:
            np.testing.assert_array_equal(recvbuf, sendbuf * 3)
        else:
            np.testing.assert_array_equal(recvbuf, np.zeros_like(sendbuf))

    async def test_allreduce(self):
        sendbuf = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int32)
        recvbuf = np.zeros_like(sendbuf)
        _group = [0, 1]
        group = await new_group(_group)
        if group is not None:
            await allreduce(sendbuf, recvbuf, group_name=group)
        if self._rank in _group:
            np.testing.assert_array_equal(recvbuf, sendbuf * len(_group))
        else:
            np.testing.assert_array_equal(recvbuf, np.zeros_like(sendbuf))

    async def test_gather(self):
        sendbuf = np.array([self._rank], dtype=np.int32)
        recvbuf = np.zeros((2,), dtype=np.int32)
        root = 0
        _group = [1, 2]
        group = await new_group(_group)
        if group is not None:
            await gather(sendbuf, recvbuf, group_name=group, root=root)

        if self._rank == _group[root]:
            np.testing.assert_array_equal(recvbuf, np.array(_group, dtype=np.int32))
        else:
            np.testing.assert_array_equal(recvbuf, np.zeros_like(recvbuf))

    async def test_allgather(self):
        sendbuf = np.array([self._rank], dtype=np.int32)
        recvbuf = np.zeros((3,), dtype=np.int32)
        _group = [0, 1, 2]
        group = await new_group(_group)
        if group is not None:
            await allgather(sendbuf, recvbuf, group_name=group)
        if self._rank in _group:
            np.testing.assert_array_equal(recvbuf, np.array(_group, dtype=np.int32))
        else:
            np.testing.assert_array_equal(recvbuf, np.zeros_like(recvbuf))

    async def test_scatter(self):
        _group = [1, 2]
        root = 0
        if self._rank == _group[root]:
            sendbuf1 = np.array([10, 11], dtype=np.int32)
            sendbuf2 = np.array([12, 13], dtype=np.int32)
        else:
            sendbuf1 = np.zeros((2,), dtype=np.int32)
            sendbuf2 = np.zeros((2,), dtype=np.int32)
        recvbuf = np.zeros((2,), dtype=np.int32)
        send_list = [sendbuf1, sendbuf2]
        group = await new_group(_group)
        if group is not None:
            await scatter(send_list, recvbuf, group_name=group, root=root)

        if self._rank == _group[0]:
            np.testing.assert_array_equal(recvbuf, np.array([10, 11], dtype=np.int32))
        elif self._rank == _group[1]:
            np.testing.assert_array_equal(recvbuf, np.array([12, 13], dtype=np.int32))
        else:
            np.testing.assert_array_equal(recvbuf, np.zeros_like(recvbuf))

    async def test_reduce_scatter(self):
        data = [self._rank, self._rank + 1, self._rank + 2]
        sendbuf = np.array(data, dtype=np.int32)
        recvbuf = np.zeros((1,), dtype=np.int32)
        recv_elems = [1, 1, 1]
        group = await new_group([0, 1, 2])
        if group is not None:
            await reduce_scatter(sendbuf, recvbuf, recv_elems, group_name=group)
        np.testing.assert_array_equal(recvbuf, np.array([sum(data)], dtype=np.int32))

    async def test_alltoall(self):
        sendbuf = np.zeros((3,), dtype=np.float32) + self._rank
        recvbuf = np.zeros(sendbuf.shape, dtype=np.float32)
        group = await new_group([0, 1, 2])
        if group is not None:
            await alltoall(sendbuf, recvbuf, group_name=group)
        np.testing.assert_array_equal(recvbuf, np.array([0, 1, 2], dtype=np.float32))

    async def test_broadcast(self):
        root = 1
        _group = [0, 1, 2]
        sendbuf = np.zeros((2, 3), dtype=np.int64)
        if self._rank == _group[root]:
            sendbuf = sendbuf + self._rank
        recvbuf = np.zeros_like(sendbuf, dtype=np.int64)
        group = await new_group(_group)
        if group is not None:
            await broadcast(sendbuf, recvbuf, root=root, group_name=group)
        np.testing.assert_array_equal(recvbuf, np.zeros_like(recvbuf) + _group[root])

    async def test_collective_np(self):
        await self.test_params()
        await self.test_reduce()
        await self.test_allreduce()
        await self.test_gather()
        await self.test_allgather()
        await self.test_scatter()
        # reduce_scatter has problem on non-linux os since uv has issue in gloo
        if is_linux():
            await self.test_reduce_scatter()
        await self.test_alltoall()
        await self.test_broadcast()


@pytest.mark.asyncio
@require_unix
async def test_collective():
    pool = await create_actor_pool(
        "127.0.0.1",
        n_process=3,
        envs=[
            {
                RENDEZVOUS_MASTER_IP_ENV_KEY: "127.0.0.1",
                RENDEZVOUS_MASTER_PORT_ENV_KEY: "25001",
            }
        ]
        * 3,
    )
    main_addr = pool.external_address
    config = (await get_pool_config(pool.external_address)).as_dict()
    all_addrs = list(config["mapping"].keys())
    all_addrs.remove(main_addr)

    async with pool:
        ctx = get_context()
        r0 = await ctx.create_actor(WorkerActor, 0, 3, address=all_addrs[0])
        r1 = await ctx.create_actor(WorkerActor, 1, 3, address=all_addrs[1])
        r2 = await ctx.create_actor(WorkerActor, 2, 3, address=all_addrs[2])
        t0 = r0.init_process_group()
        t1 = r1.init_process_group()
        t2 = r2.init_process_group()
        await asyncio.gather(*[t0, t1, t2])

        t0 = r0.test_collective_np()
        t1 = r1.test_collective_np()
        t2 = r2.test_collective_np()
        await asyncio.gather(*[t0, t1, t2])


@pytest.mark.asyncio
@require_unix
async def test_collective_without_env():
    pool = await create_actor_pool(
        "127.0.0.1",
        n_process=3,
    )
    main_addr = pool.external_address
    config = (await get_pool_config(pool.external_address)).as_dict()
    all_addrs = list(config["mapping"].keys())
    all_addrs.remove(main_addr)

    async with pool:
        ctx = get_context()
        r0 = await ctx.create_actor(WorkerActor, 0, 3, address=all_addrs[0])
        r1 = await ctx.create_actor(WorkerActor, 1, 3, address=all_addrs[1])
        r2 = await ctx.create_actor(WorkerActor, 2, 3, address=all_addrs[2])
        t0 = r0.init_process_group_without_env()
        t1 = r1.init_process_group_without_env()
        t2 = r2.init_process_group_without_env()
        await asyncio.gather(*[t0, t1, t2])

        t0 = r0.init_process_group()
        t1 = r1.init_process_group()
        t2 = r2.init_process_group()
        with pytest.raises(ValueError):
            await asyncio.gather(*[t0, t1, t2])
