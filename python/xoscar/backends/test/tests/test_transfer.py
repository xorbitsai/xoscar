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
import shutil
import sys
import tempfile
from typing import List, Optional

import numpy as np
import pytest

from .... import Actor, ActorRefType
from ....aio import AioFileObject
from ....api import actor_ref, buffer_ref, copy_to, file_object_ref
from ....backends.allocate_strategy import ProcessIndex
from ....backends.indigen.pool import MainActorPool
from ....context import get_context
from ....core import BufferRef, FileObjectRef
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

        with pytest.raises(ValueError):
            await copy_to([], [])

        ref = await actor_ref(ref)
        buf_refs1 = await ref.create_buffers(sizes, cpu=cpu)
        buf_refs2 = await ref.create_buffers(sizes, cpu=cpu)

        with pytest.raises(AssertionError):
            await copy_to(buffers1, buf_refs1, block_size=-1)

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


async def _copy_test(scheme1: Optional[str], scheme2: Optional[str], cpu: bool):
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
        external_address_schemes=[None, scheme1, scheme2],
    )
    pool2: MainActorPool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        subprocess_start_method=start_method,
        external_address_schemes=[None, scheme1, scheme2],
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
    await _copy_test(scheme, scheme, True)
    if "ucx" == scheme:
        await _copy_test(None, "ucx", True)
        await _copy_test("ucx", None, True)


@require_cupy
@pytest.mark.asyncio
@pytest.mark.parametrize("scheme", schemes)
async def tests_gpu_copy(scheme):
    await _copy_test(scheme, scheme, False)
    if "ucx" == scheme:
        await _copy_test(None, "ucx", False)
        await _copy_test("ucx", None, False)


class FileobjTransferActor(Actor):
    def __init__(self):
        self._fileobjs = []

    async def create_file_objects(self, names: List[str]) -> List[FileObjectRef]:
        refs = []
        for name in names:
            fobj = open(name, "w+b")
            afobj = AioFileObject(fobj)
            self._fileobjs.append(afobj)
            refs.append(file_object_ref(self.address, afobj))
        return refs

    async def close(self):
        for fobj in self._fileobjs:
            assert await fobj.tell() > 0
            await fobj.close()

    async def copy_data(
        self,
        ref: ActorRefType["FileobjTransferActor"],
        names1: List[str],
        names2: List[str],
        sizes: List[int],
    ):
        fobjs = []
        for name, size in zip(names1, sizes):
            fobj = open(name, "w+b")
            fobj.write(np.random.bytes(size))
            fobj.seek(0)
            fobjs.append(AioFileObject(fobj))

        ref = await actor_ref(ref)
        file_obj_refs = await ref.create_file_objects(names2)
        await copy_to(fobjs, file_obj_refs)
        _ = [await f.close() for f in fobjs]  # type: ignore
        await ref.close()

        for n1, n2 in zip(names1, names2):
            with open(n1, "rb") as f1, open(n2, "rb") as f2:
                b1 = f1.read()
                b2 = f2.read()
                assert b1 == b2


@pytest.mark.asyncio
async def test_copy_to_file_objects():
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )
    pool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        subprocess_start_method=start_method,
    )

    d = tempfile.mkdtemp()
    async with pool:
        ctx = get_context()

        # actor on main pool
        actor_ref1 = await ctx.create_actor(
            FileobjTransferActor,
            uid="test-1",
            address=pool.external_address,
            allocate_strategy=ProcessIndex(1),
        )
        actor_ref2 = await ctx.create_actor(
            FileobjTransferActor,
            uid="test-2",
            address=pool.external_address,
            allocate_strategy=ProcessIndex(2),
        )
        sizes = [
            10 * 1024**2,
            3 * 1024**2,
            int(0.5 * 1024**2),
            int(0.25 * 1024**2),
        ]
        names = []
        for _ in range(2 * len(sizes)):
            _, p = tempfile.mkstemp(dir=d)
            names.append(p)

        await actor_ref1.copy_data(actor_ref2, names[::2], names[1::2], sizes=sizes)
    try:
        shutil.rmtree(d)
    except PermissionError:
        pass
