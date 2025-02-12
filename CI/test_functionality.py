# Tests for wheel

import pytest
import logging
import asyncio

import xoscar as mo

import platform
import sys

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MyActor(mo.Actor):
    def __init__(self):
        self.i = 0
        logger.debug("MyActor initialized")

    def add(self, j: int) -> int:
        logger.debug(f"Adding {j} to {self.i}")
        self.i += j
        return self.i

    def get(self) -> int:
        logger.debug(f"Getting value: {self.i}")
        return self.i

    async def add_from(self, ref: mo.ActorRefType["MyActor"]) -> int:
        logger.debug("Starting add_from operation")
        self.i += await ref.get()
        logger.debug(f"After add_from: {self.i}")
        return self.i


@pytest.mark.asyncio
@pytest.mark.timeout(60)  # 设置更短的超时时间，以便更快发现问题
async def test_basic_cases():
    logger.debug("Starting test_basic_cases")
    pool = await mo.create_actor_pool(
        "127.0.0.1", 
        n_process=2,
        subprocess_start_method='spawn'  # 显式指定启动方法
    )
    logger.debug("Actor pool created")
    
    try:
        async with pool:
            logger.debug("Entering pool context")
            ref1 = await mo.create_actor(
                MyActor,
                address=pool.external_address,
                allocate_strategy=mo.allocate_strategy.ProcessIndex(1),
            )
            logger.debug("Created ref1")
            
            ref2 = await mo.create_actor(
                MyActor,
                address=pool.external_address,
                allocate_strategy=mo.allocate_strategy.ProcessIndex(2),
            )
            logger.debug("Created ref2")
            
            # 添加超时控制
            async with asyncio.timeout(30):
                assert await ref1.add(1) == 1
                logger.debug("ref1.add(1) completed")
                
                assert await ref2.add(2) == 2
                logger.debug("ref2.add(2) completed")
                
                assert await ref1.add_from(ref2) == 3
                logger.debug("ref1.add_from(ref2) completed")
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        raise
    finally:
        logger.debug("Test cleanup")

def test_pygloo():
    is_windows = sys.platform.startswith('win')
    bit_number = platform.architecture()[0]
    if not (is_windows and bit_number=="32bit"):
        import xoscar.collective.xoscar_pygloo as xp
        print(type(xp.ReduceOp.SUM))
