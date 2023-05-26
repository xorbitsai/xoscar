# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2021 Alibaba Group Holding Ltd.
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
import sys
from typing import List, Optional

import numpy as np
import pytest

from .... import Actor, ActorRefType
from ....api import actor_ref, buffer_ref, copy_to
from ....backends.allocate_strategy import ProcessIndex
from ....backends.indigen.pool import MainActorPool
from ....context import get_context
from ....core import BufferRef
from ....tests.core import require_cupy
from ....utils import lazy_import
from ...pool import create_actor_pool

rmm = lazy_import("rmm")
cupy = lazy_import("cupy")
ucp = lazy_import("ucp")


class BufferTransferActor(Actor):
    def __init__(self):
        self._buffers = []

    def create_buffers(self, sizes: List[int], cpu: bool = True) -> List[BufferRef]:
        if cpu:
            buffers = [np.zeros(size, dtype="u1").data for size in sizes]
        else:
            assert cupy is not None
            buffers = [cupy.zeros(size, dtype="u1") for size in sizes]
        self._buffers.extend(buffers)
        res = [buffer_ref(self.address, buf) for buf in buffers]
        return res

    def create_arrays_from_buffer_refs(
        self, buf_refs: List[BufferRef], cpu: bool = True
    ):
        if cpu:
            return [
                np.frombuffer(BufferRef.get_buffer(ref), dtype="u1") for ref in buf_refs
            ]
        else:
            return [BufferRef.get_buffer(ref) for ref in buf_refs]

    async def copy_data(
        self, ref: ActorRefType["BufferTransferActor"], sizes, cpu: bool = True
    ):
        xp = np if cpu else cupy
        arrays1 = [np.random.randint(3, 12, dtype="u1", size=size) for size in sizes]
        arrays2 = [np.random.randint(6, 23, dtype="u1", size=size) for size in sizes]
        buffers1 = [a.data for a in arrays1]
        buffers2 = [a.data for a in arrays2]
        if not cpu:
            arrays1 = [cupy.asarray(a) for a in arrays1]
            arrays2 = [cupy.asarray(a) for a in arrays2]
            buffers1 = arrays1
            buffers2 = arrays2

        ref = await actor_ref(ref)
        buf_refs1 = await ref.create_buffers(sizes, cpu=cpu)
        buf_refs2 = await ref.create_buffers(sizes, cpu=cpu)
        tasks = [copy_to(buffers1, buf_refs1), copy_to(buffers2, buf_refs2)]
        await asyncio.gather(*tasks)

        new_arrays1 = await ref.create_arrays_from_buffer_refs(buf_refs1, cpu=cpu)
        assert len(arrays1) == len(new_arrays1)
        for a1, a2 in zip(arrays1, new_arrays1):
            xp.testing.assert_array_equal(a1, a2)
        new_arrays2 = await ref.create_arrays_from_buffer_refs(buf_refs2, cpu=cpu)
        assert len(arrays2) == len(new_arrays2)
        for a1, a2 in zip(arrays2, new_arrays2):
            xp.testing.assert_array_equal(a1, a2)


async def _copy_test(scheme: str, cpu: bool):
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )
    pool: MainActorPool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        subprocess_start_method=start_method,
        external_address_schemes=[None, scheme, scheme],
    )
    pool2: MainActorPool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        subprocess_start_method=start_method,
        external_address_schemes=[None, scheme, scheme],
    )

    async with pool:
        ctx = get_context()

        # actor on main pool
        actor_ref1 = await ctx.create_actor(
            BufferTransferActor,
            uid="test-1",
            address=pool.external_address,
            allocate_strategy=ProcessIndex(1),
        )
        actor_ref2 = await ctx.create_actor(
            BufferTransferActor,
            uid="test-2",
            address=pool.external_address,
            allocate_strategy=ProcessIndex(2),
        )
        actor_ref3 = await ctx.create_actor(
            BufferTransferActor,
            uid="test-3",
            address=pool.external_address,
            allocate_strategy=ProcessIndex(1),
        )

        actor_ref4 = await ctx.create_actor(
            BufferTransferActor,
            uid="test-4",
            address=pool2.external_address,
            allocate_strategy=ProcessIndex(1),
        )

        actor_ref5 = await ctx.create_actor(
            BufferTransferActor,
            uid="test-5",
            address=pool2.external_address,
            allocate_strategy=ProcessIndex(2),
        )

        sizes = [
            10 * 1024**2,
            3 * 1024**2,
            5 * 1024**2,
            8 * 1024**2,
            7 * 1024**2,
        ]
        await actor_ref1.copy_data(actor_ref2, sizes, cpu=cpu)
        await actor_ref1.copy_data(actor_ref3, sizes, cpu=cpu)
        await actor_ref1.copy_data(actor_ref4, sizes, cpu=cpu)
        await actor_ref1.copy_data(actor_ref5, sizes, cpu=cpu)

        # test small size of data
        sizes = [
            1 * 1024**2,
            2 * 1024**2,
            1 * 1024**2,
        ]
        tasks = [
            actor_ref2.copy_data(actor_ref1, sizes, cpu=cpu),
            actor_ref2.copy_data(actor_ref3, sizes, cpu=cpu),
            actor_ref2.copy_data(actor_ref4, sizes, cpu=cpu),
            actor_ref2.copy_data(actor_ref5, sizes, cpu=cpu),
        ]
        await asyncio.gather(*tasks)


schemes: List[Optional[str]] = [None]
if ucp is not None:
    schemes.append("ucx")


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme", schemes)
async def test_copy(scheme):
    await _copy_test(scheme, True)


@require_cupy
@pytest.mark.parametrize("scheme", schemes)
async def tests_gpu_copy(scheme):
    await _copy_test(scheme, False)
