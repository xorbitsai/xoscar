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
import gc
import os
import sys
import threading

import pytest

import xoscar as mo

from ...communication.dummy import DummyServer
from ...router import Router


class DummyActor(mo.Actor):
    def __init__(self, value):
        super().__init__()

        if value < 0:
            raise ValueError("value < 0")
        self.value = value

    async def add(self, value):
        if not isinstance(value, int):
            raise TypeError("add number must be int")
        self.value += value
        return self.value


@pytest.fixture
async def actor_pool_context():
    pool = await mo.create_actor_pool("test://127.0.0.1", n_process=2)
    async with pool:
        yield pool


@pytest.mark.asyncio
async def test_simple(actor_pool_context):
    pool = actor_pool_context
    actor_ref = await mo.create_actor(
        DummyActor,
        100,
        address=pool.external_address,
        allocate_strategy=mo.allocate_strategy.RandomSubPool(),
    )
    assert await actor_ref.add(1) == 101


def _cancel_all_tasks(loop):
    to_cancel = asyncio.all_tasks(loop)
    if not to_cancel:
        return

    for task in to_cancel:
        task.cancel()

    loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))

    for task in to_cancel:
        if task.cancelled():
            continue
        if task.exception() is not None:
            loop.call_exception_handler(
                {
                    "message": "unhandled exception during asyncio.run() shutdown",
                    "exception": task.exception(),
                    "task": task,
                }
            )


def _run_forever(loop):
    loop.run_forever()
    _cancel_all_tasks(loop)


@pytest.mark.asyncio
async def test_channel_cleanup(actor_pool_context):
    pool = actor_pool_context
    actor_ref = await mo.create_actor(
        DummyActor,
        0,
        address=pool.external_address,
        allocate_strategy=mo.allocate_strategy.RandomSubPool(),
    )

    curr_router = Router.get_instance()
    server_address = curr_router.get_internal_address(actor_ref.address)
    dummy_server = DummyServer.get_instance(server_address)

    async def inc():
        await asyncio.gather(*(actor_ref.add.tell(1) for _ in range(10)))

    loops = []
    threads = []
    futures = []
    for _ in range(10):
        loop = asyncio.new_event_loop()
        t = threading.Thread(target=_run_forever, args=(loop,))
        t.start()
        loops.append(loop)
        threads.append(t)
        fut = asyncio.run_coroutine_threadsafe(inc(), loop=loop)
        futures.append(fut)

    for fut in futures:
        fut.result()

    while True:
        if await actor_ref.add(0) == 100:
            break

    assert len(dummy_server._channels) == 12
    assert len(dummy_server._tasks) == 12

    for loop in loops:
        loop.call_soon_threadsafe(loop.stop)

    for t in threads:
        t.join()
    threads.clear()

    curr_router = Router.get_instance()
    server_address = curr_router.get_internal_address(actor_ref.address)
    dummy_server = DummyServer.get_instance(server_address)

    while True:
        gc.collect()
        # Two channels left:
        #   1. from the main pool to the actor
        #   2. from current main thread to the actor.
        if len(dummy_server._channels) == 2 and len(dummy_server._tasks) == 2:
            break
