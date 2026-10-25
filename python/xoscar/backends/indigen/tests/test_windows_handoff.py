# Copyright 2026 Xorbits Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import subprocess
import sys

import psutil
import pytest

import xoscar as xo

from ...allocate_strategy import RandomSubPool
from ...router import Router
from ..handoff import SubPoolProcess
from .test_manager_handoff import (
    _TARGET,
    CounterActor,
    create_pool,
    read_json,
    wait_until,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object semantics"),
]


@pytest.fixture(autouse=True)
def clear_router():
    yield
    Router.set_instance(None)


_TWO_POOLS = """
import asyncio, json, os, sys
from pathlib import Path
import xoscar as xo
from xoscar.backends.allocate_strategy import RandomSubPool
class Actor(xo.StatelessActor):
    def pid(self):
        return os.getpid()
async def main():
    old = await xo.create_actor_pool("127.0.0.1", n_process=1)
    unrelated = await xo.create_actor_pool("127.0.0.1", n_process=1)
    ref = await xo.create_actor(Actor, uid="retained", address=old.external_address,
                               allocate_strategy=RandomSubPool())
    Path(sys.argv[1]).write_text(json.dumps({
        "address": ref.address, "pid": await ref.pid(),
        "unrelated": next(iter(unrelated.sub_processes.values())).identity,
    }))
    await old.handoff_to(sys.argv[2])
    os._exit(0)
asyncio.run(main())
"""


async def test_transferring_one_pool_does_not_retain_unrelated_jobs(tmp_path):
    async with await create_pool() as new:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            _TWO_POOLS,
            str(tmp_path / "ready"),
            new.external_address,
        )
        unrelated = None
        try:
            ready = await read_json(tmp_path / "ready")
            unrelated = SubPoolProcess(ready["unrelated"])
            assert await asyncio.wait_for(process.wait(), 15) == 0
            ref = await xo.actor_ref(address=ready["address"], uid="retained")
            assert await ref.pid() == ready["pid"]
            await wait_until(lambda: unrelated.returncode is not None, timeout=5)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
            if unrelated is not None and unrelated.returncode is None:
                unrelated.kill()


class ProcessActor(CounterActor):
    def spawn_child(self):
        self.child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(600)"]
        )
        return {
            "pid": self.child.pid,
            "create_time": psutil.Process(self.child.pid).create_time(),
        }


async def test_new_main_crash_also_cleans_actor_descendants(tmp_path):
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", _TARGET, str(tmp_path / "target")
    )
    descendant = None
    try:
        target = await read_json(tmp_path / "target")
        async with await create_pool(1) as old:
            ref = await xo.create_actor(
                ProcessActor,
                uid="family",
                address=old.external_address,
                allocate_strategy=RandomSubPool(),
            )
            child = old.sub_processes[ref.address]
            descendant = SubPoolProcess(await ref.spawn_child())
            await old.handoff_to(target["address"])
            assert (await ref.add(1))[0] == 1
            assert descendant.returncode is None
            process.kill()
            await process.wait()
            await wait_until(
                lambda: child.returncode is not None
                and descendant.returncode is not None,
                timeout=5,
            )
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        if descendant is not None and descendant.returncode is None:
            descendant.kill()
