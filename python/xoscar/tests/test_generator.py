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
import time

import pytest

import xoscar as xo

address = "127.0.0.1:12347"


class WorkerActor(xo.StatelessActor):
    @xo.generator
    def chat(self):
        for x in "hello oscar by sync":
            yield x
            time.sleep(0.1)

    @xo.generator
    async def achat(self):
        for x in "hello oscar by async":
            yield x
            await asyncio.sleep(0.1)

    @classmethod
    def uid(cls):
        return "worker"


class SupervisorActor(xo.StatelessActor):
    def get_all_generators(self):
        return list(self._generators.keys())

    @xo.generator
    async def chat(self):
        worker_actor: xo.ActorRef["WorkerActor"] = await xo.actor_ref(
            address=address, uid=WorkerActor.uid()
        )
        yield "sync"
        async for x in await worker_actor.chat():  # this is much confused. I will suggest use async generators only.
            yield x

        yield "async"
        async for x in await worker_actor.achat():
            yield x

    @xo.generator
    async def with_exception(self):
        yield 1
        raise Exception("intent raise")
        yield 2

    @xo.generator
    async def mix_gen(self, v):
        if v == 1:
            return self._gen()
        elif v == 2:
            return self._gen2()
        else:
            return 0

    @xo.generator
    def mix_gen2(self, v):
        if v == 1:
            return self._gen()
        elif v == 2:
            return self._gen2()
        else:
            return 0

    def _gen(self):
        for x in range(3):
            yield x

    async def _gen2(self):
        for x in range(3):
            yield x

    @classmethod
    def uid(cls):
        return "supervisor"


async def test_generator():
    await xo.create_actor_pool(address, 2)
    await xo.create_actor(WorkerActor, address=address, uid=WorkerActor.uid())
    superivsor_actor = await xo.create_actor(
        SupervisorActor, address=address, uid=SupervisorActor.uid()
    )

    all_gen = await superivsor_actor.get_all_generators()
    assert len(all_gen) == 0
    output = []
    async for x in await superivsor_actor.chat():
        all_gen = await superivsor_actor.get_all_generators()
        assert len(all_gen) == 1
        output.append(x)
    all_gen = await superivsor_actor.get_all_generators()
    assert len(all_gen) == 0
    assert output == [
        "sync",
        "h",
        "e",
        "l",
        "l",
        "o",
        " ",
        "o",
        "s",
        "c",
        "a",
        "r",
        " ",
        "b",
        "y",
        " ",
        "s",
        "y",
        "n",
        "c",
        "async",
        "h",
        "e",
        "l",
        "l",
        "o",
        " ",
        "o",
        "s",
        "c",
        "a",
        "r",
        " ",
        "b",
        "y",
        " ",
        "a",
        "s",
        "y",
        "n",
        "c",
    ]

    with pytest.raises(Exception, match="intent"):
        async for _ in await superivsor_actor.with_exception():
            pass
    all_gen = await superivsor_actor.get_all_generators()
    assert len(all_gen) == 0

    r = await superivsor_actor.with_exception()
    del r
    await asyncio.sleep(0)
    all_gen = await superivsor_actor.get_all_generators()
    assert len(all_gen) == 0

    for f in [superivsor_actor.mix_gen, superivsor_actor.mix_gen2]:
        out = []
        async for x in await f(1):
            out.append(x)
        assert out == [0, 1, 2]
        out = []
        async for x in await f(2):
            out.append(x)
        assert out == [0, 1, 2]
        assert 0 == await f(0)
