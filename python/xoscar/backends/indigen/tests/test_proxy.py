# Copyright 2022-2025 XProbe Inc.
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
import multiprocessing
import os
import sys

import psutil
import pytest

import xoscar as xo

from ....utils import get_next_port
from ...router import Router


async def _run_actor_pool(started, address, proxy_config):
    pool = await xo.create_actor_pool(
        address,
        n_process=2,
        proxy_conf=proxy_config,
    )
    await pool.start()
    started.set()
    await pool.join()


def _run_in_process(started, address, proxy_config):
    asyncio.run(_run_actor_pool(started, address, proxy_config))


@pytest.fixture
async def actor_pools():
    addrs = addr1, addr2, addr3 = [f"127.0.0.1:{get_next_port()}" for _ in range(3)]
    processes = []
    try:
        for addr in addrs:
            proxy_conf = {
                "127.0.0.1": addr,
            }
            if addr == addr1:
                proxy_conf[addr3] = addr2
            elif addr == addr3:
                proxy_conf[addr1] = addr2
            s = multiprocessing.Event()
            p = multiprocessing.Process(
                target=_run_in_process, args=(s, addr, proxy_conf)
            )
            p.start()
            s.wait()

            ps = psutil.Process(p.pid).children()
            processes.append(psutil.Process(p.pid))
            processes.extend(ps)

        yield addr1, addr3
    finally:
        Router.set_instance(None)
        for p in processes:
            try:
                p.kill()
            except:
                continue


class TestActor(xo.Actor):
    def __init__(self):
        super().__init__()
        self.val = 0

    def acc(self):
        self.val += 1

    def get(self):
        return self.val

    def run(self, it):
        return it

    async def long_running(self):
        await asyncio.sleep(1000)


class CallerActor(xo.Actor):
    def __init__(self, actor_ref: xo.ActorRefType[TestActor]):
        super().__init__()
        self.ref = actor_ref

    async def call(self, method, *args, **kwargs):
        ref = await xo.actor_ref(self.ref)
        assert ref.proxy_addresses
        return getattr(ref, method)(*args, **kwargs)

    async def call2(self, address, uid, method, *args, **kwargs):
        ref = await xo.actor_ref(address, uid)
        assert ref.proxy_addresses
        return getattr(ref, method)(*args, **kwargs)


@pytest.mark.asyncio
async def test_client(actor_pools):
    addr1, addr2 = actor_pools

    actor_ref = await xo.create_actor(
        TestActor,
        address=addr1,
        uid="test",
        allocate_strategy=xo.allocate_strategy.RandomSubPool(),
    )
    assert actor_ref.proxy_addresses

    assert await xo.has_actor(actor_ref)

    assert await actor_ref.run(1) == 1

    await actor_ref.acc.tell()
    assert await actor_ref.get() == 1

    actor_ref2 = await xo.actor_ref(addr1, "test")
    assert actor_ref2 == actor_ref
    assert actor_ref2.proxy_addresses == actor_ref.proxy_addresses

    with pytest.raises(asyncio.CancelledError):
        task = asyncio.create_task(actor_ref.long_running())
        await asyncio.sleep(0)
        task.cancel()
        await task

    caller_ref = await xo.create_actor(
        CallerActor,
        actor_ref,
        address=addr2,
        uid="caller",
        allocate_strategy=xo.allocate_strategy.RandomSubPool(),
    )
    assert caller_ref.proxy_addresses

    assert await caller_ref.call("run", 1) == 1

    await caller_ref.call("acc")
    assert await caller_ref.call("get") == 2
    assert await caller_ref.call2(actor_ref.address, actor_ref.uid, "get") == 2


@pytest.mark.asyncio
async def test_actor_ref_with_parameters():
    # test `create_actor_ref` with parameters actor that use class object and parameters as arguments
    class ParameterActor(xo.Actor):
        def __init__(self, val1=0, val2=0, val3=0):
            super().__init__()
            self.val1 = val1
            self.val2 = val2

        def get_values(self):
            return self.val1, self.val2

        def update_values(self, val1=None, val2=None):
            if val1 is not None:
                self.val1 = val1
            if val2 is not None:
                self.val2 = val2
            return self.get_values()

        @classmethod
        def gen_uid(cls, band_name: str):
            return f"param_actor_{band_name}"

    pool = await xo.create_actor_pool(
        "127.0.0.1",
        n_process=2,
    )

    async with pool:
        io_addr = pool.external_address

        original_actor_ref = await xo.create_actor(
            ParameterActor,
            1,
            2,
            address=io_addr,
            uid=ParameterActor.gen_uid("numa-0"),
        )

        actor_ref_from_actor_ref_func = await xo.actor_ref(
            ParameterActor,
            1,
            2,
            address=io_addr,
            uid=ParameterActor.gen_uid("numa-0"),
        )

        assert await xo.has_actor(actor_ref_from_actor_ref_func)
        assert await original_actor_ref.get_values() == (1, 2)
        assert await actor_ref_from_actor_ref_func.get_values() == (1, 2)
