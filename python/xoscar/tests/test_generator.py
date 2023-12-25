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

    await asyncio.create_task(superivsor_actor.with_exception())
    await asyncio.sleep(0)
    all_gen = await superivsor_actor.get_all_generators()
    assert len(all_gen) == 0
