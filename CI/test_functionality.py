# Tests for wheel

import pytest

import xoscar as mo

import platform
import sys


class MyActor(mo.Actor):
    def __init__(self):
        self.i = 0

    def add(self, j: int) -> int:
        self.i += j
        return self.i

    def get(self) -> int:
        return self.i

    async def add_from(self, ref: mo.ActorRefType["MyActor"]) -> int:
        self.i += await ref.get()
        return self.i


@pytest.mark.asyncio
async def test_basic_cases():
    pool = await mo.create_actor_pool("127.0.0.1", n_process=2)
    async with pool:
        ref1 = await mo.create_actor(
            MyActor,
            address=pool.external_address,
            allocate_strategy=mo.allocate_strategy.ProcessIndex(1),
        )
        ref2 = await mo.create_actor(
            MyActor,
            address=pool.external_address,
            allocate_strategy=mo.allocate_strategy.ProcessIndex(2),
        )
        assert await ref1.add(1) == 1
        assert await ref2.add(2) == 2
        assert await ref1.add_from(ref2) == 3

def test_pygloo():
    is_windows = sys.platform.startswith('win')
    bit_number = platform.architecture()[0]
    if not (is_windows and bit_number=="32bit"):
        import xoscar.collective.xoscar_pygloo as xp
        print(type(xp.ReduceOp.SUM))
