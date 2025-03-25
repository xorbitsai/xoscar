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
import os
import sys

import pytest

import xoscar as xo

from ....utils import get_next_port


@pytest.fixture
async def patched_actor_pools():
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )
    addr1 = f"127.0.0.1:{get_next_port()}"
    pool1 = await xo.create_actor_pool(
        addr1,
        n_process=2,
        subprocess_start_method=start_method,
        proxy_conf={"127.0.0.1": addr1},
    )
    await pool1.start()
    addr2 = f"127.0.0.1:{get_next_port()}"
    pool2 = await xo.create_actor_pool(
        addr2,
        n_process=1,
        subprocess_start_method=start_method,
        proxy_conf={"127.0.0.1": addr2},
    )
    await pool2.start()
    try:
        yield pool1, pool2
    finally:
        await pool1.stop()
        await pool2.stop()


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


@pytest.mark.asyncio
async def test_client(patched_actor_pools):
    pool1, pool2 = patched_actor_pools

    actor_ref = await xo.create_actor(
        TestActor,
        address=pool1.external_address,
        uid="test",
        allocate_strategy=xo.allocate_strategy.RandomSubPool(),
    )
    assert actor_ref.proxy_addresses

    assert await xo.has_actor(actor_ref)

    assert await actor_ref.run(1) == 1

    await actor_ref.acc.tell()
    assert await actor_ref.get() == 1

    actor_ref2 = await xo.actor_ref(pool1.external_address, "test")
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
        address=pool2.external_address,
        uid="caller",
        allocate_strategy=xo.allocate_strategy.RandomSubPool(),
    )
    assert caller_ref.proxy_addresses

    assert await caller_ref.call("run", 1) == 1

    await caller_ref.call("acc")
    assert await caller_ref.call("get") == 2
