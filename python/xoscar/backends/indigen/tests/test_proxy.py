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
import multiprocessing
import socket
from contextlib import ExitStack

import numpy as np
import psutil
import pytest

import xoscar as xo

from ....serialization.aio import AioSerializer
from ...context import IndigenActorContext
from ...message import CancelMessage, ForwardMessage, SendMessage
from ...router import Router


async def _run_actor_pool(started, address, proxy_config):
    pool = await xo.create_actor_pool(
        address,
        n_process=2,
        proxy_conf=proxy_config,
    )
    await pool.start()
    started.set()
    await pool.join()


def _run_in_process(started, address, proxy_config):
    asyncio.run(_run_actor_pool(started, address, proxy_config))


@pytest.fixture
async def actor_pools():
    # Binding port 0 avoids Windows reserved ranges that netstat cannot see.
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(3)]
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        addrs = addr1, addr2, addr3 = [
            f"127.0.0.1:{sock.getsockname()[1]}" for sock in sockets
        ]
    processes = []
    try:
        for addr in addrs:
            proxy_conf = {
                "127.0.0.1": addr,
            }
            if addr == addr1:
                proxy_conf[addr3] = addr2
            elif addr == addr3:
                proxy_conf[addr1] = addr2
            s = multiprocessing.Event()
            p = multiprocessing.Process(
                target=_run_in_process, args=(s, addr, proxy_conf)
            )
            p.start()
            process = psutil.Process(p.pid)
            processes.append(process)
            assert await asyncio.to_thread(
                s.wait, 60
            ), f"Actor pool at {addr} failed to start (exit code: {p.exitcode})"
            processes.extend(process.children())

        yield addr1, addr3
    finally:
        Router.set_instance(None)
        for p in processes:
            try:
                p.kill()
            except:
                continue


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

    async def call2(self, address, uid, method, *args, **kwargs):
        ref = await xo.actor_ref(address, uid)
        assert ref.proxy_addresses
        return getattr(ref, method)(*args, **kwargs)


@pytest.mark.asyncio
async def test_client(actor_pools):
    addr1, addr2 = actor_pools

    actor_ref = await xo.create_actor(
        TestActor,
        address=addr1,
        uid="test",
        allocate_strategy=xo.allocate_strategy.RandomSubPool(),
    )
    assert actor_ref.proxy_addresses

    assert await xo.has_actor(actor_ref)

    assert await actor_ref.run(1) == 1

    await actor_ref.acc.tell()
    assert await actor_ref.get() == 1

    actor_ref2 = await xo.actor_ref(addr1, "test")
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
        address=addr2,
        uid="caller",
        allocate_strategy=xo.allocate_strategy.RandomSubPool(),
    )
    assert caller_ref.proxy_addresses

    assert await caller_ref.call("run", 1) == 1

    await caller_ref.call("acc")
    assert await caller_ref.call("get") == 2
    assert await caller_ref.call2(actor_ref.address, actor_ref.uid, "get") == 2


@pytest.mark.asyncio
async def test_actor_ref_with_parameters():
    # test `create_actor_ref` with parameters actor that use class object and parameters as arguments
    class ParameterActor(xo.Actor):
        def __init__(self, val1=0, val2=0, val3=0):
            super().__init__()
            self.val1 = val1
            self.val2 = val2

        def get_values(self):
            return self.val1, self.val2

        def update_values(self, val1=None, val2=None):
            if val1 is not None:
                self.val1 = val1
            if val2 is not None:
                self.val2 = val2
            return self.get_values()

        @classmethod
        def gen_uid(cls, band_name: str):
            return f"param_actor_{band_name}"

    pool = await xo.create_actor_pool(
        "127.0.0.1",
        n_process=2,
    )

    async with pool:
        io_addr = pool.external_address

        original_actor_ref = await xo.create_actor(
            ParameterActor,
            1,
            2,
            address=io_addr,
            uid=ParameterActor.gen_uid("numa-0"),
        )

        actor_ref_from_actor_ref_func = await xo.actor_ref(
            ParameterActor,
            1,
            2,
            address=io_addr,
            uid=ParameterActor.gen_uid("numa-0"),
        )

        assert await xo.has_actor(actor_ref_from_actor_ref_func)
        assert await original_actor_ref.get_values() == (1, 2)
        assert await actor_ref_from_actor_ref_func.get_values() == (1, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("delay", ["connect", "serialize", "serialize_error"])
async def test_cancel_does_not_overtake_proxied_call(monkeypatch, delay):
    remote_started = asyncio.Event()
    remote_cancelled = asyncio.Event()
    forwarding_started = asyncio.Event()
    cancel_arrived = asyncio.Event()
    release_serialization = asyncio.Event()

    class BlockingActor(xo.Actor):
        async def long_running(self, payload):
            remote_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                remote_cancelled.set()

        def run(self):
            return True

    target = await xo.create_actor_pool("127.0.0.1:0", n_process=0)
    proxy = await xo.create_actor_pool("127.0.0.1:0", n_process=0)
    async with target, proxy:
        ref = await xo.create_actor(BlockingActor, address=target.external_address)
        ref.proxy_addresses = [proxy.external_address]
        with monkeypatch.context() as patch:
            # Exercise an external client, which has no local address mappings.
            patch.setattr(Router, "_instance", Router([], None))
            original_create_client = Router._create_client

            async def delayed_create_client(router, client_type, address, **kwargs):
                if (
                    delay == "connect"
                    and router is proxy.router
                    and address == target.external_address
                ):
                    forwarding_started.set()
                    await asyncio.sleep(0.2)
                return await original_create_client(
                    router, client_type, address, **kwargs
                )

            patch.setattr(Router, "_create_client", delayed_create_client)
            original_get_buffers = AioSerializer._get_buffers
            original_process = type(proxy).process_message

            async def delayed_get_buffers(serializer):
                message = serializer._obj
                if (
                    delay != "connect"
                    and isinstance(message, SendMessage)
                    and message.content[0] == "long_running"
                ):
                    forwarding_started.set()
                    await release_serialization.wait()
                    if delay == "serialize_error":
                        raise ValueError("Forward serialization failed")
                return await original_get_buffers(serializer)

            async def observe_cancel(pool, message, channel):
                if (
                    pool is proxy
                    and isinstance(message, ForwardMessage)
                    and isinstance(message.raw_message, CancelMessage)
                ):
                    cancel_arrived.set()
                return await original_process(pool, message, channel)

            patch.setattr(AioSerializer, "_get_buffers", delayed_get_buffers)
            patch.setattr(type(proxy), "process_message", observe_cancel)
            ctx = IndigenActorContext()
            # Distinct arrays cross serialize_with_spawn's 100-object threshold.
            payload = [np.arange(4, dtype="u1") for _ in range(150)]
            task = asyncio.create_task(
                ctx.send(ref, ("long_running", 0, (payload,), {}))
            )
            try:
                await asyncio.wait_for(forwarding_started.wait(), timeout=2)
                task.cancel()
                await asyncio.wait_for(cancel_arrived.wait(), timeout=2)
                await asyncio.sleep(0)
                release_serialization.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=2)
                if delay == "serialize_error":
                    assert not remote_started.is_set()
                else:
                    await asyncio.wait_for(remote_started.wait(), timeout=2)
                    assert remote_cancelled.is_set()
                # The cancelled method must release the actor lock for the next RPC.
                assert await asyncio.wait_for(
                    ctx.send(ref, ("run", 0, (), {})), timeout=2
                )
                if hasattr(proxy, "_forward_sends"):
                    assert not proxy._forward_sends
            finally:
                release_serialization.set()
                # Also clean up the remote call when running against the broken code.
                for process_task in list(target._process_messages.values()):
                    if process_task is not None:
                        process_task.cancel()
                await ctx._caller.stop()
