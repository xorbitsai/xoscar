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
import os
import subprocess
import sys

import psutil
import pytest

import xoscar as xo

from ...allocate_strategy import RandomSubPool
from ...router import Router
from .. import fate_sharing, handoff
from ..fate_sharing import duplicate_win32_job
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
import psutil
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
    unrelated_process = next(iter(unrelated.sub_processes.values()))
    # Stop both watchdogs: only Job Object cleanup can end this process.
    psutil.Process(unrelated_process.pid).suspend()
    Path(sys.argv[1]).write_text(json.dumps({
        "address": ref.address, "pid": await ref.pid(),
        "unrelated": unrelated_process.identity,
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


@pytest.mark.parametrize("invalid_owner,error_code", [(True, 87), (False, 6)])
async def test_job_duplication_reports_windows_error(invalid_owner, error_code):
    # PID 0 is invalid for OpenProcess; handle 0 is invalid for DuplicateHandle.
    with pytest.raises(OSError) as exc:
        duplicate_win32_job(0 if invalid_owner else os.getpid(), 0)
    assert exc.value.winerror == error_code


async def test_owner_recheck_failure_closes_duplicated_job(monkeypatch):
    async with await create_pool(1) as old:
        source = next(iter(old.sub_processes.values()))
        owner = handoff._identity()
        duplicate = fate_sharing.duplicate_win32_job
        check = handoff._process
        prepared = []

        def record_duplicate(owner_pid, job_handle):
            job = duplicate(owner_pid, job_handle)
            prepared.append(job)
            return job

        def owner_changed(identity):
            if prepared:
                raise ProcessLookupError("owner identity changed")
            return check(identity)

        with monkeypatch.context() as patch:
            patch.setattr(fate_sharing, "duplicate_win32_job", record_duplicate)
            patch.setattr(handoff, "_process", owner_changed)
            with pytest.raises(ProcessLookupError, match="owner identity changed"):
                SubPoolProcess.adopted(source.identity, owner)
        assert len(prepared) == 1 and prepared[0].closed
        assert source._job is not None and not source._job.closed
        assert source.returncode is None


async def test_handoff_releases_job_handles_after_rollback_commit_and_exit(monkeypatch):
    async with await create_pool(1) as old, await create_pool() as new:
        ref = await xo.create_actor(
            CounterActor,
            uid="retained",
            address=old.external_address,
            allocate_strategy=RandomSubPool(),
        )
        source_process = old.sub_processes[ref.address]
        source_job = source_process._job
        assert source_job is not None and not source_job.closed
        prepared_jobs = []
        request = handoff._rpc

        async def lose_reserve_reply(pool, address, operation, **content):
            result = await request(pool, address, operation, **content)
            if pool is old and operation == "reserve_main":
                job = new._handoff["adopted"][ref.address]._job
                assert job is not None and not job.closed
                prepared_jobs.append(job)
                raise ConnectionResetError("reserve response lost")
            return result

        with monkeypatch.context() as patch:
            patch.setattr(handoff, "_rpc", lose_reserve_reply)
            with pytest.raises(ConnectionResetError, match="reserve response lost"):
                await old.handoff_to(new.external_address)
        assert old._handoff is None and new._handoff is None
        assert len(prepared_jobs) == 1 and prepared_jobs[0].closed
        assert source_process._job is source_job and not source_job.closed
        assert (await ref.add(1))[0] == 1

        await old.handoff_to(new.external_address)
        assert source_process._job is None and source_job.closed
        adopted = new.sub_processes[ref.address]
        adopted_job = adopted._job
        assert adopted_job is not None and not adopted_job.closed
        assert await ref.add(1) == (2, source_process.pid)
        adopted.kill()
        await asyncio.wait_for(adopted.wait(), 5)
        assert adopted._job is None and adopted_job.closed


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
