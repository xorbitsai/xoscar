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
import subprocess
import sys
from dataclasses import replace

import psutil
import pytest

import xoscar as xo

from ...allocate_strategy import AddressSpecified, RandomSubPool
from ...message import ControlMessage, ControlMessageType, ErrorMessage, new_message_id
from ...router import Router
from ..handoff import ActorPoolHandoff, HandoffActorPool


class CounterActor(xo.StatelessActor):
    def __init__(self):
        super().__init__()
        self.value = 0

    def add(self, value):
        self.value += value
        return self.value, os.getpid()

    @xo.generator
    async def stream(self):
        for i in range(5):
            yield i
            await asyncio.sleep(0.05)


class PoolFactoryActor(xo.StatelessActor):
    async def create_pool(self):
        pool = await create_pool(n_process=1)
        actor = await xo.create_actor(
            CounterActor, uid="counter", address=pool.external_address
        )
        await actor.add(17)
        return (await pool.handoff()).to_dict()


@pytest.fixture(autouse=True)
def clear_router():
    yield
    Router.set_instance(None)


async def create_pool(**kwargs):
    return await xo.create_actor_pool(
        "127.0.0.1",
        n_process=kwargs.pop("n_process", 0),
        enable_manager_handoff=True,
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("n_process", [0, 1])
async def test_handoff_preserves_actor_and_stream(n_process):
    old = await create_pool(n_process=n_process)
    new = None
    try:
        ref = await xo.create_actor(
            CounterActor,
            uid="counter",
            address=old.external_address,
            **({"allocate_strategy": RandomSubPool()} if n_process else {}),
        )
        value, pid = await ref.add(7)
        assert value == 7
        stream = await ref.stream()
        assert await stream.__anext__() == 0
        stale = old.handoff_handle
        handoff = await old.handoff()
        # Credentials survive JSON roundtrips, without exposing them in repr.
        assert handoff.token not in repr(handoff)
        handoff = json.loads(json.dumps(handoff.to_dict()))
        new = await xo.take_over_actor_pool(handoff)
        assert new.external_address == old.external_address
        assert new.pid == old.pid
        discovered = await new.list_actors()
        assert [(r.address, r.uid) for r in discovered] == [(ref.address, ref.uid)]
        found = await xo.actor_ref(address=new.external_address, uid="counter")
        assert await found.add(3) == (10, pid)
        assert [i async for i in stream] == [1, 2, 3, 4]
        await stream.destroy()
        stream._gc_destroy = False
        stale_owner = HandoffActorPool(stale)
        with pytest.raises(RuntimeError, match="stale or invalid"):
            await stale_owner.stop()
        with pytest.raises(RuntimeError, match="stale or invalid"):
            await xo.take_over_actor_pool(handoff)
        assert await ref.add(1) == (11, pid)
    finally:
        if new is not None:
            await new.stop()
        elif not old._closed:
            await old.stop()


@pytest.mark.asyncio
async def test_active_owner_cannot_be_stolen():
    async with await create_pool() as pool:
        with pytest.raises(RuntimeError, match="active owner"):
            await xo.take_over_actor_pool(pool.handoff_handle)
        with pytest.raises(RuntimeError, match="stale or invalid"):
            await xo.take_over_actor_pool(replace(pool.handoff_handle, token="wrong"))
        assert await pool.list_actors() == []


@pytest.mark.asyncio
async def test_only_one_new_owner_wins():
    pool = await create_pool()
    handoff = await pool.handoff()
    results = await asyncio.gather(
        xo.take_over_actor_pool(handoff),
        xo.take_over_actor_pool(handoff),
        return_exceptions=True,
    )
    winners = [r for r in results if isinstance(r, HandoffActorPool)]
    assert len(winners) == 1
    try:
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 1 and "stale or invalid" in str(errors[0])
    finally:
        await winners[0].stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["take_over", "handoff"])
async def test_handoff_retries_a_lost_response(monkeypatch, operation):
    old = await create_pool()
    request = HandoffActorPool._request
    dropped = False

    async def drop_response(self, name, **payload):
        nonlocal dropped
        response = await request(self, name, **payload)
        if name == operation and not dropped:
            dropped = True
            raise ConnectionResetError("management response lost")
        return response

    monkeypatch.setattr(HandoffActorPool, "_request", drop_response)
    credential = await old.handoff()
    async with await xo.take_over_actor_pool(credential) as new:
        assert dropped
        assert await new.list_actors() == []


@pytest.mark.asyncio
async def test_handoff_inside_context_does_not_stop_host():
    async with await create_pool() as old:
        credential = await old.handoff()
    async with await xo.take_over_actor_pool(credential) as new:
        assert new.pid == old.pid


@pytest.mark.asyncio
async def test_heartbeat_and_recovery_after_owner_lease_expires():
    old = await create_pool(lease_timeout=0.9, orphan_timeout=5)
    new = None
    try:
        ref = await xo.create_actor(
            CounterActor, uid="counter", address=old.external_address
        )
        before = await ref.add(4)
        # Healthy owners keep the host alive beyond its initial lease.
        await asyncio.sleep(1.2)
        assert await ref.add(0) == before
        await old._close()  # Simulate a manager that stopped renewing.
        await asyncio.sleep(1)
        expired_owner = HandoffActorPool(old.handoff_handle)
        try:
            with pytest.raises(RuntimeError, match="lease expired"):
                await expired_owner._request("renew")
        finally:
            await expired_owner._close()
        new = await xo.take_over_actor_pool(old.handoff_handle)
        assert await ref.add(0) == before
        with pytest.raises(RuntimeError, match="stale or invalid"):
            stale_owner = HandoffActorPool(old.handoff_handle)
            try:
                await stale_owner._request("renew")
            finally:
                await stale_owner._close()
    finally:
        if new is not None:
            await new.stop()
        elif not old._closed:
            await old.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("handed_off", [False, True])
async def test_unowned_pool_is_reclaimed(handed_off):
    pool = await create_pool(lease_timeout=0.6, orphan_timeout=0.6)
    if handed_off:
        await pool.handoff()
    else:
        await pool._close()
    await pool.join(timeout=10)
    assert not pool.handoff_handle._alive()


@pytest.mark.asyncio
async def test_reclaim_stops_host_and_sub_pools():
    pool = await create_pool(n_process=1, lease_timeout=0.9, orphan_timeout=0.6)
    actor = await xo.create_actor(
        CounterActor,
        address=pool.external_address,
        allocate_strategy=RandomSubPool(),
    )
    _, actor_pid = await actor.add(0)
    assert actor_pid != pool.pid
    await pool.handoff()
    await pool.join(timeout=10)
    try:
        assert psutil.Process(actor_pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass


@pytest.mark.asyncio
async def test_host_shutdown_requires_owner_handle():
    async with await create_pool() as pool:
        actor = await xo.create_actor(CounterActor, address=pool.external_address)
        result = await pool._caller.call(
            Router.get_instance_or_empty(),
            pool.external_address,
            ControlMessage(
                new_message_id(), pool.external_address, ControlMessageType.stop, None
            ),
        )
        assert isinstance(result, ErrorMessage) and "owner handle" in str(result.error)
        assert (await actor.add(1))[0] == 1


@pytest.mark.asyncio
async def test_pool_outlives_creating_actor_and_its_main_pool():
    manager = await xo.create_actor_pool("127.0.0.1", n_process=1)
    credential = None
    try:
        factory = await xo.create_actor(
            PoolFactoryActor,
            address=manager.external_address,
            allocate_strategy=RandomSubPool(),
        )
        credential = await factory.create_pool()
    finally:
        await manager.stop()
    async with await xo.take_over_actor_pool(credential) as pool:
        actor = await xo.actor_ref(address=pool.external_address, uid="counter")
        assert (await actor.add(0))[0] == 17


@pytest.mark.asyncio
async def test_sub_pool_management_after_handoff():
    old = await create_pool(auto_recover="actor")
    address = await old.append_sub_pool(label="model", use_uvloop=False)
    ref = await xo.create_actor(
        CounterActor,
        uid="counter",
        address=old.external_address,
        allocate_strategy=AddressSpecified(address),
    )
    assert (await ref.add(2))[0] == 2
    handoff = await old.handoff()
    async with await xo.take_over_actor_pool(handoff) as new:
        config = await new.get_config()
        assert address in config.get_external_addresses()
        assert len(await new.list_actors()) == 1
        # Recovery remains owned by the stable host, without another manager
        # replaying constructors merely because ownership changed.
        await xo.kill_actor(ref)
        await xo.wait_actor_pool_recovered(address, new.external_address)
        recovered = await xo.actor_ref(address=new.external_address, uid="counter")
        assert (await recovered.add(0))[0] == 0
        await new.remove_sub_pool(address)
        assert address not in (await new.get_config()).get_external_addresses()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["handoff", "crash"])
async def test_pool_survives_manager_process_and_event_loop_exit(tmp_path, mode):
    credential = tmp_path / "pool.json"
    script = """
import asyncio, json, os, sys
from pathlib import Path
import xoscar as xo
from xoscar.backends.indigen.tests.test_manager_handoff import CounterActor
async def main():
    pool = await xo.create_actor_pool('127.0.0.1', n_process=1,
        enable_manager_handoff=True, lease_timeout=1.5, orphan_timeout=15)
    ref = await xo.create_actor(CounterActor, uid='counter', address=pool.external_address)
    await ref.add(13)
    handle = await pool.handoff() if sys.argv[2] == 'handoff' else pool.handoff_handle
    Path(sys.argv[1]).write_text(json.dumps(handle.to_dict()))
    if sys.argv[2] == 'crash':
        os._exit(0)
asyncio.run(main())
"""
    manager = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        str(credential),
        mode,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(manager.communicate(), 30)
    assert manager.returncode == 0, (stdout, stderr)
    handle = ActorPoolHandoff.from_dict(json.loads(credential.read_text()))
    assert handle._alive()
    if mode == "crash":
        await asyncio.sleep(1.6)
    async with await xo.take_over_actor_pool(handle) as pool:
        ref = await xo.actor_ref(address=pool.external_address, uid="counter")
        assert (await ref.add(0))[0] == 13


@pytest.mark.asyncio
@pytest.mark.parametrize("option", ["lease_timeout", "orphan_timeout", "start_timeout"])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
async def test_invalid_timeouts(option, value):
    with pytest.raises(ValueError, match=option):
        await create_pool(**{option: value})


@pytest.mark.asyncio
async def test_invalid_manager_handoff_option_and_backend():
    with pytest.raises(ValueError, match="enable_manager_handoff"):
        await xo.create_actor_pool("127.0.0.1", enable_manager_handoff="yes")
    with pytest.raises(ValueError, match="TCP"):
        await xo.create_actor_pool("test://127.0.0.1", enable_manager_handoff=True)


@pytest.mark.asyncio
async def test_incompatible_or_reused_process_handle():
    async with await create_pool() as pool:
        with pytest.raises(ValueError, match="protocol"):
            await xo.take_over_actor_pool(replace(pool.handoff_handle, protocol=999))
        with pytest.raises(ValueError, match="Python"):
            await xo.take_over_actor_pool(
                replace(pool.handoff_handle, python_version=(2, 7))
            )
        with pytest.raises(ProcessLookupError, match="identity"):
            await xo.take_over_actor_pool(replace(pool.handoff_handle, create_time=0))


@pytest.mark.asyncio
async def test_startup_failure_is_reported():
    with pytest.raises(RuntimeError, match="failed to start"):
        await create_pool(auto_recover="invalid", start_timeout=10)


@pytest.mark.asyncio
async def test_startup_cancellation_does_not_leave_host(tmp_path, monkeypatch):
    from .. import handoff

    directories = []
    original = handoff.tempfile.mkdtemp

    def temporary(**kwargs):
        path = original(dir=tmp_path, **kwargs)
        directories.append(path)
        return path

    monkeypatch.setattr(handoff.tempfile, "mkdtemp", temporary)
    task = asyncio.create_task(create_pool())
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert directories
    assert list(tmp_path.iterdir()) == []
