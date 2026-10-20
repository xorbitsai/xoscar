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
import contextlib
import json
import os
import sys
from pathlib import Path

import psutil
import pytest

import xoscar as xo

from ...allocate_strategy import AddressSpecified, RandomSubPool
from ...router import Router
from .. import handoff
from ..handoff import SubPoolProcess
from ..pool import MainActorPool

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        sys.platform == "win32", reason="Native handoff currently requires POSIX"
    ),
]


class CounterActor(xo.StatelessActor):
    def __init__(self, delay=0):
        super().__init__()
        self.value = 0
        self.delay = delay

    async def __post_create__(self):
        await asyncio.sleep(self.delay)

    def add(self, value):
        self.value += value
        return self.value, os.getpid()

    @xo.generator
    async def stream(self):
        for i in range(5):
            yield i
            await asyncio.sleep(0.02)

    async def wait_file(self, path):
        while not Path(path).exists():
            await asyncio.sleep(0.02)
        return self.value, os.getpid()

    async def create_other(self, uid, delay=0):
        return await xo.create_actor(CounterActor, delay, uid=uid, address=self.address)

    async def destroy_other(self, ref):
        await xo.destroy_actor(ref)

    async def lookup(self, uid):
        return await xo.actor_ref(address=self.address, uid=uid)


@pytest.fixture(autouse=True)
def clear_router():
    yield
    Router.set_instance(None)


async def create_pool(n_process=0, **kwargs):
    return await xo.create_actor_pool("127.0.0.1", n_process=n_process, **kwargs)


async def actor(pool, uid="counter", address=None):
    return await xo.create_actor(
        CounterActor,
        uid=uid,
        address=pool.external_address,
        allocate_strategy=AddressSpecified(address) if address else RandomSubPool(),
    )


async def wait_until(predicate, timeout=10):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.02)

    await asyncio.wait_for(wait(), timeout)


async def read_json(path):
    await wait_until(path.exists)
    return json.loads(path.read_text())


async def test_handoff_preserves_state_stream_and_main_actors():
    async with await create_pool(2) as old, await create_pool() as new:
        assert type(old) is MainActorPool and type(new) is MainActorPool
        assert not hasattr(xo, "take_over_actor_pool")
        main_old = await xo.create_actor(
            CounterActor, uid="control", address=old.external_address
        )
        main_new = await xo.create_actor(
            CounterActor, uid="control", address=new.external_address
        )
        refs = [
            await actor(old, f"counter{i}", addr)
            for i, addr in enumerate(old.sub_processes)
        ]
        ref = refs[0]
        assert (await ref.add(7))[0] == 7
        pid = (await ref.add(0))[1]
        stream = await ref.stream()
        assert await stream.__anext__() == 0
        try:
            addresses = await old.handoff_to(new.external_address)
            assert set(addresses) == {r.address for r in refs}
            assert not old.sub_processes
            assert new._monitor_task is not None
            assert main_old.uid in old._actors and main_new.uid in new._actors
            assert all(
                isinstance(p, SubPoolProcess) for p in new.sub_processes.values()
            )
            found = await xo.actor_ref(address=new.external_address, uid="counter0")
            assert await found.add(3) == (10, pid)
            assert (await refs[0].lookup("counter1")).address == refs[1].address
            assert [i async for i in stream] == [1, 2, 3, 4]
            assert await old.handoff_to(new.external_address) == addresses
            with pytest.raises(RuntimeError, match="handed off"):
                await old.append_sub_pool()
            await old.stop()
            assert await ref.add(1) == (11, pid)
        finally:
            await stream.destroy()
            stream._gc_destroy = False


async def test_adopted_pool_registration_scaling_and_recovery():
    async with await create_pool(1) as old, await create_pool() as new:
        ref = await actor(old)
        old_pid = (await ref.add(1))[1]
        await old.handoff_to(new.external_address)
        # Direct child lifecycle notifications must go to the new main.
        other = await ref.create_other("direct")
        assert (
            await xo.actor_ref(address=new.external_address, uid="direct")
        ).address == ref.address
        await ref.destroy_other(other)
        with pytest.raises(xo.ActorNotExist):
            await xo.actor_ref(address=new.external_address, uid="direct")
        addr = await new.append_sub_pool()
        another = await actor(new, "appended", addr)
        assert (await another.add(1))[0] == 1
        await new.remove_sub_pool(addr)
        new.sub_processes[ref.address].kill()
        await wait_until(lambda: new.sub_processes[ref.address].pid != old_pid)
        await xo.wait_actor_pool_recovered(ref.address, new.external_address)
        assert (await ref.add(1))[0] == 1


@pytest.mark.parametrize(
    "operation", ["reserve_main", "prepare_child", "stage_main", "switch_child"]
)
async def test_precommit_lost_response_rolls_back(monkeypatch, operation):
    async with await create_pool(2) as old, await create_pool() as new:
        refs = [
            await actor(old, f"counter{i}", addr)
            for i, addr in enumerate(old.sub_processes)
        ]
        await refs[0].add(5)
        request = handoff._rpc
        dropped = False

        async def drop_reply(pool, address, op, **content):
            nonlocal dropped
            result = await request(pool, address, op, **content)
            if pool is old and op == operation and not dropped:
                dropped = True
                raise ConnectionResetError("response lost")
            return result

        with monkeypatch.context() as m:
            m.setattr(handoff, "_rpc", drop_reply)
            with pytest.raises(ConnectionResetError, match="response lost"):
                await old.handoff_to(new.external_address)
        assert dropped and old._handoff is None and new._handoff is None
        assert old._monitor_task and not old._monitor_task.done()
        assert (await refs[0].add(2))[0] == 7
        # Both pools remain usable, and the complete transfer can be retried.
        await old.handoff_to(new.external_address)
        assert (await refs[0].add(1))[0] == 8


@pytest.mark.parametrize("execute", [False, True])
async def test_commit_response_loss_can_be_retried(monkeypatch, execute):
    async with await create_pool(1) as old, await create_pool() as new:
        ref = await actor(old)
        pid = (await ref.add(9))[1]
        request = handoff._rpc

        async def drop_reply(pool, address, op, **content):
            if pool is old and op == "commit_main":
                if execute:
                    await request(pool, address, op, **content)
                raise ConnectionResetError("commit response lost")
            return await request(pool, address, op, **content)

        with monkeypatch.context() as m:
            m.setattr(handoff, "_rpc", drop_reply)
            with pytest.raises(ConnectionResetError):
                await old.handoff_to(new.external_address)
        assert not old.sub_processes
        await old.handoff_to(new.external_address)
        assert await ref.add(1) == (10, pid)


async def test_handoff_drains_lifecycle_without_blocking_calls(monkeypatch):
    async with await create_pool(1) as old, await create_pool() as new:
        ref = await actor(old)
        # A direct child creation must finish registering before snapshotting.
        pending_create = asyncio.create_task(ref.create_other("slow", 0.3))
        await asyncio.sleep(0.05)
        request = handoff._rpc
        prepared, release = asyncio.Event(), asyncio.Event()

        async def pause(pool, address, op, **content):
            result = await request(pool, address, op, **content)
            if pool is old and op == "prepare_child":
                prepared.set()
                await release.wait()
            return result

        with monkeypatch.context() as m:
            m.setattr(handoff, "_rpc", pause)
            transfer = asyncio.create_task(old.handoff_to(new.external_address))
            await asyncio.wait_for(prepared.wait(), 5)
            assert (await ref.add(3))[0] == 3
            with pytest.raises(RuntimeError, match="handoff"):
                await old.append_sub_pool()
            with pytest.raises(RuntimeError, match="handoff"):
                await ref.create_other("frozen")
            release.set()
            await transfer
        slow = await pending_create
        assert (
            await xo.actor_ref(address=new.external_address, uid="slow")
        ).address == slow.address


async def test_cancellation_completes_the_transfer(monkeypatch):
    async with await create_pool(1) as old, await create_pool() as new:
        ref = await actor(old)
        request = handoff._rpc
        ready, release = asyncio.Event(), asyncio.Event()

        async def pause(pool, address, op, **content):
            result = await request(pool, address, op, **content)
            if pool is old and op == "stage_main":
                ready.set()
                await release.wait()
            return result

        with monkeypatch.context() as m:
            m.setattr(handoff, "_rpc", pause)
            transfer = asyncio.create_task(old.handoff_to(new.external_address))
            await ready.wait()
            transfer.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await transfer
        assert not old.sub_processes and ref.address in new.sub_processes
        assert (await ref.add(1))[0] == 1


async def test_preparation_timeout_rolls_back(monkeypatch):
    async with await create_pool(1) as old, await create_pool() as new:
        ref = await actor(old)
        request = handoff._rpc

        async def pause(pool, address, op, **content):
            result = await request(pool, address, op, **content)
            if pool is old and op == "stage_main":
                await asyncio.sleep(2)
            return result

        with monkeypatch.context() as m:
            m.setattr(handoff, "_rpc", pause)
            with pytest.raises(asyncio.TimeoutError):
                await old.handoff_to(new.external_address, timeout=0.2)
        assert (await ref.add(1))[0] == 1
        assert old._handoff is None and new._handoff is None


async def test_repeated_handoff():
    async with await create_pool(
        1
    ) as first, await create_pool() as second, await create_pool() as third:
        ref = await actor(first)
        pid = (await ref.add(4))[1]
        await first.handoff_to(second.external_address)
        await second.handoff_to(third.external_address)
        await first.stop()
        await second.stop()
        assert await ref.add(3) == (7, pid)
        assert (
            await xo.actor_ref(address=third.external_address, uid="counter")
        ).address == ref.address


async def test_reject_busy_target_and_actor_conflicts():
    async with await create_pool(1) as old, await create_pool(
        1
    ) as busy, await create_pool() as new:
        ref = await actor(old)
        with pytest.raises(ValueError, match="no sub pools"):
            await old.handoff_to(busy.external_address)
        await xo.create_actor(CounterActor, uid="counter", address=new.external_address)
        with pytest.raises(ValueError, match="conflicting actor"):
            await old.handoff_to(new.external_address)
        assert old._handoff is None and new._handoff is None
        assert (await ref.add(1))[0] == 1


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, "30"])
async def test_validate_timeout(timeout):
    async with await create_pool() as old:
        with pytest.raises(ValueError, match="timeout"):
            await old.handoff_to("127.0.0.1:1", timeout=timeout)


async def test_validate_target_and_process_identity():
    async with await create_pool() as old:
        with pytest.raises(ValueError, match="different"):
            await old.handoff_to(old.external_address)
        with pytest.raises(ValueError, match="no sub pools"):
            await old.handoff_to("127.0.0.1:1")
    identity = handoff._identity()
    with pytest.raises(ProcessLookupError, match="identity"):
        handoff._process({**identity, "create_time": identity["create_time"] - 1})
    for field in ("machine", "python", "protocol"):
        with pytest.raises(ValueError, match=field):
            handoff._check_identity({**identity, field: None})


_SOURCE = """
import asyncio, json, os, sys
from pathlib import Path
import xoscar as xo
from xoscar.backends.allocate_strategy import RandomSubPool
from xoscar.backends.indigen.tests.test_manager_handoff import CounterActor
root, target, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
async def main():
    pool = await xo.create_actor_pool("127.0.0.1", n_process=1)
    ref = await xo.create_actor(CounterActor, uid="counter", address=pool.external_address, allocate_strategy=RandomSubPool())
    value, pid = await ref.add(10)
    (root / "ready").write_text(json.dumps({"address": ref.address, "pid": pid}))
    while not (root / "go").exists():
        await asyncio.sleep(.02)
    if mode == "lost_commit":
        from xoscar.backends.indigen import handoff
        request = handoff._rpc
        async def drop(pool, address, op, **content):
            if op == "commit_main":
                raise ConnectionResetError("commit not delivered")
            return await request(pool, address, op, **content)
        handoff._rpc = drop
        try:
            await pool.handoff_to(target, timeout=1)
        except ConnectionResetError:
            pass
    elif mode == "partial_commit":
        try:
            await pool.handoff_to(target, timeout=1)
        except ConnectionResetError:
            pass
    else:
        await pool.handoff_to(target)
    if mode in ("stop", "lost_commit", "partial_commit"):
        await pool.stop()
    (root / "done").write_text("done")
asyncio.run(main())
if mode == "abrupt":
    os._exit(0)
"""


@pytest.mark.parametrize(
    "mode", ["stop", "loop_exit", "abrupt", "lost_commit", "partial_commit"]
)
async def test_old_main_process_really_exits(tmp_path, mode, monkeypatch):
    async with await create_pool() as new:
        if mode == "partial_commit":
            request = handoff._rpc
            dropped = False

            async def lose_ack(pool, address, op, **content):
                nonlocal dropped
                result = await request(pool, address, op, **content)
                if pool is new and op == "finish_child" and not dropped:
                    dropped = True
                    raise ConnectionResetError("child acknowledgement lost")
                return result

            monkeypatch.setattr(handoff, "_rpc", lose_ack)
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", _SOURCE, str(tmp_path), new.external_address, mode
        )
        ref, stream, pending = None, None, None
        try:
            ready = await read_json(tmp_path / "ready")
            ref = await xo.actor_ref(address=ready["address"], uid="counter")
            stream = await ref.stream()
            assert await stream.__anext__() == 0
            pending = asyncio.create_task(ref.wait_file(str(tmp_path / "release")))
            (tmp_path / "go").touch()
            assert await asyncio.wait_for(proc.wait(), 15) == 0
            assert (tmp_path / "done").exists()
            await wait_until(lambda: ready["address"] in new.sub_processes)
            assert await ref.add(2) == (12, ready["pid"])
            assert [i async for i in stream] == [1, 2, 3, 4]
            (tmp_path / "release").touch()
            assert await pending == (12, ready["pid"])
            found = await xo.actor_ref(address=new.external_address, uid="counter")
            assert await found.add(1) == (13, ready["pid"])
        finally:
            (tmp_path / "release").touch()
            if pending:
                with contextlib.suppress(Exception):
                    await pending
            if stream:
                await stream.destroy()
                stream._gc_destroy = False
            if proc.returncode is None:
                proc.kill()
                await proc.wait()


_TARGET = """
import asyncio, json, os, sys
from pathlib import Path
import xoscar as xo
async def main():
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    Path(sys.argv[1]).write_text(json.dumps({"address": pool.external_address}))
    await pool.join()
asyncio.run(main())
"""


async def test_new_main_crash_cleans_adopted_children(tmp_path):
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", _TARGET, str(tmp_path / "target")
    )
    try:
        target = await read_json(tmp_path / "target")
        async with await create_pool(1) as old:
            ref = await actor(old)
            process = old.sub_processes[ref.address]
            await old.handoff_to(target["address"])
            proc.kill()
            await proc.wait()
            await wait_until(lambda: process.returncode is not None, timeout=5)
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


async def test_normal_main_crash_still_cleans_children(tmp_path):
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", _SOURCE, str(tmp_path), "unused", "stop"
    )
    try:
        ready = await read_json(tmp_path / "ready")
        child = psutil.Process(ready["pid"])
        proc.kill()
        await proc.wait()
        await wait_until(
            lambda: not child.is_running() or child.status() == psutil.STATUS_ZOMBIE,
            timeout=5,
        )
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


async def test_partial_commit_response_loss(monkeypatch):
    async with await create_pool(2) as old, await create_pool() as new:
        refs = [
            await actor(old, f"counter{i}", addr)
            for i, addr in enumerate(old.sub_processes)
        ]
        request = handoff._rpc
        dropped = False

        async def drop_reply(pool, address, op, **content):
            nonlocal dropped
            result = await request(pool, address, op, **content)
            if pool is new and op == "finish_child" and not dropped:
                dropped = True
                raise ConnectionResetError("partial commit response lost")
            return result

        with monkeypatch.context() as m:
            m.setattr(handoff, "_rpc", drop_reply)
            with pytest.raises(ConnectionResetError):
                await old.handoff_to(new.external_address)
        # A thawed child may register actors before the finish retry.
        direct = await refs[0].create_other("new_during_commit")
        await old.handoff_to(new.external_address)
        assert (
            await xo.actor_ref(address=new.external_address, uid="new_during_commit")
        ).address == direct.address
        assert (await refs[1].add(1))[0] == 1


async def test_concurrent_handoffs_have_one_winner():
    async with await create_pool(
        1
    ) as old, await create_pool() as first, await create_pool() as second:
        ref = await actor(old)
        results = await asyncio.gather(
            old.handoff_to(first.external_address),
            old.handoff_to(second.external_address),
            return_exceptions=True,
        )
        assert sum(isinstance(result, list) for result in results) == 1
        assert (ref.address in first.sub_processes) != (
            ref.address in second.sub_processes
        )
        assert (await ref.add(1))[0] == 1


async def test_unconfirmed_rollback_does_not_resume_old_owner(monkeypatch):
    async with await create_pool(1) as old, await create_pool() as new:
        ref = await actor(old)
        request = handoff._rpc

        async def fail(pool, address, op, **content):
            if pool is old and op == "stage_main":
                raise ConnectionResetError("preparation failed")
            if op == "abort_child":
                raise ConnectionResetError("rollback not confirmed")
            return await request(pool, address, op, **content)

        with monkeypatch.context() as m:
            m.setattr(handoff, "_rpc", fail)
            with pytest.raises(RuntimeError, match="rollback was not confirmed"):
                await old.handoff_to(new.external_address)
        assert old._handoff["phase"] == "rollback_failed"
        assert new._handoff["phase"] == "aborting"
        with pytest.raises(RuntimeError, match="handoff"):
            await old.append_sub_pool()
        await old.handoff_to(new.external_address)
        assert (await ref.add(1))[0] == 1
