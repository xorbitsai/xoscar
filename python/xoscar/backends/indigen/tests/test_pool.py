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

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import re
import sys
import time
from unittest import mock

import psutil
import pytest

import xoscar

from .... import (
    Actor,
    create_actor,
    create_actor_ref,
    get_pool_config,
    kill_actor,
    wait_for,
)
from ....context import get_context
from ....errors import ActorNotExist, NoIdleSlot, SendMessageFailed, ServerClosed
from ....tests.core import require_ucx, require_unix
from ....utils import get_next_port
from ...allocate_strategy import (
    AddressSpecified,
    IdleLabel,
    MainPool,
    ProcessIndex,
    RandomSubPool,
)
from ...config import ActorPoolConfig
from ...message import (
    ActorRefMessage,
    CancelMessage,
    ControlMessage,
    ControlMessageType,
    CreateActorMessage,
    DestroyActorMessage,
    ErrorMessage,
    HasActorMessage,
    MessageType,
    SendMessage,
    TellMessage,
    new_message_id,
)
from ...pool import create_actor_pool
from ...router import Router
from ...test.pool import TestMainActorPool
from ..pool import MainActorPool, SubActorPool


class _CannotBePickled:
    def __getstate__(self):
        raise RuntimeError("cannot pickle")


class _CannotBeUnpickled:
    def __getstate__(self):
        return ()

    def __setstate__(self, state):
        raise RuntimeError("cannot unpickle")


class TestActor(Actor):
    __test__ = False

    def __init__(self):
        self.value = 0

    def add(self, val):
        self.value += val
        return self.value

    async def add_other(self, ref, val):
        self.value += await ref.add(val)
        return self.value

    async def sleep(self, second):
        try:
            await asyncio.sleep(second)
            return self.value
        except asyncio.CancelledError:
            return self.value + 1

    def return_cannot_unpickle(self):
        return _CannotBeUnpickled()

    def raise_cannot_pickle(self):
        raise ValueError(_CannotBePickled())


def _add_pool_conf(
    config: ActorPoolConfig,
    process_index: int,
    label: str,
    internal_address: str,
    external_address: str,
    env: dict | None = None,
):
    if sys.platform.startswith("win"):
        config.add_pool_conf(
            process_index, label, external_address, external_address, env=env
        )
    else:
        config.add_pool_conf(
            process_index, label, internal_address, external_address, env=env
        )


def _raise_if_error(message):
    if message.message_type == MessageType.error:
        raise message.error.with_traceback(message.traceback)


@pytest.fixture(autouse=True)
def clear_routers():
    yield
    Router.set_instance(None)


@pytest.mark.asyncio
@mock.patch("xoscar.backends.indigen.pool.SubActorPool.notify_main_pool_to_create")
@mock.patch("xoscar.backends.indigen.pool.SubActorPool.notify_main_pool_to_destroy")
async def test_sub_actor_pool(notify_main_pool_to_create, notify_main_pool_to_destroy):
    notify_main_pool_to_create.return_value = None
    notify_main_pool_to_destroy.return_value = None
    config = ActorPoolConfig()

    ext_address0 = f"127.0.0.1:{get_next_port()}"
    ext_address1 = f"127.0.0.1:{get_next_port()}"
    _add_pool_conf(config, 0, "main", "unixsocket:///0", ext_address0)
    _add_pool_conf(config, 1, "sub", "unixsocket:///1", ext_address1)

    pool = await SubActorPool.create({"actor_pool_config": config, "process_index": 1})
    await pool.start()

    try:
        create_actor_message = CreateActorMessage(
            new_message_id(),
            TestActor,
            b"test",
            tuple(),
            dict(),
            AddressSpecified(pool.external_address),
        )
        message = await pool.create_actor(create_actor_message)
        assert message.message_type == MessageType.result
        actor_ref = message.result
        assert actor_ref.address == pool.external_address
        assert actor_ref.uid == b"test"

        has_actor_message = HasActorMessage(new_message_id(), actor_ref)
        assert (await pool.has_actor(has_actor_message)).result is True

        actor_ref_message = ActorRefMessage(new_message_id(), actor_ref)
        assert (await pool.actor_ref(actor_ref_message)).result == actor_ref

        tell_message = TellMessage(
            new_message_id(), actor_ref, ("add", 0, (1,), dict())
        )
        message = await pool.tell(tell_message)
        assert message.result is None

        send_message = SendMessage(
            new_message_id(), actor_ref, ("add", 0, (3,), dict())
        )
        message = await pool.send(send_message)
        assert message.result == 4

        # test error message
        # type mismatch
        send_message = SendMessage(
            new_message_id(), actor_ref, ("add", 0, ("3",), dict())
        )
        result = await pool.send(send_message)
        assert result.message_type == MessageType.error
        assert isinstance(result.error, TypeError)

        send_message = SendMessage(
            new_message_id(),
            create_actor_ref(actor_ref.address, "non_exist"),
            ("add", 0, (3,), dict()),
        )
        result = await pool.send(send_message)
        assert isinstance(result.error, ActorNotExist)

        # test send message and cancel it
        send_message = SendMessage(
            new_message_id(), actor_ref, ("sleep", 0, (20,), dict())
        )
        result_task = asyncio.create_task(pool.send(send_message))
        await asyncio.sleep(0)
        start = time.time()
        cancel_message = CancelMessage(
            new_message_id(), actor_ref.address, send_message.message_id
        )
        cancel_task = asyncio.create_task(pool.cancel(cancel_message))
        result = await asyncio.wait_for(cancel_task, 3)
        assert result.message_type == MessageType.result
        assert result.result is True
        result = await result_task
        # test time
        assert time.time() - start < 3
        assert result.message_type == MessageType.result
        assert result.result == 5

        # test processing message on background
        async with await pool.router.get_client(pool.external_address) as client:
            send_message = SendMessage(
                new_message_id(), actor_ref, ("add", 0, (5,), dict())
            )
            await client.send(send_message)
            result = await client.recv()
            _raise_if_error(result)
            assert result.result == 9

            send_message = SendMessage(
                new_message_id(), actor_ref, ("add", 0, ("5",), dict())
            )
            await client.send(send_message)
            result = await client.recv()
            assert isinstance(result.error, TypeError)

        destroy_actor_message = DestroyActorMessage(new_message_id(), actor_ref)
        message = await pool.destroy_actor(destroy_actor_message)
        assert message.result == actor_ref.uid

        # send destroy failed
        message = await pool.destroy_actor(destroy_actor_message)
        assert isinstance(message.error, ActorNotExist)

        message = await pool.has_actor(has_actor_message)
        assert not message.result

        # test sync config
        _add_pool_conf(
            config, 1, "sub", "unixsocket:///1", f"127.0.0.1:{get_next_port()}"
        )
        sync_config_message = ControlMessage(
            new_message_id(), "", ControlMessageType.sync_config, config
        )
        message = await pool.handle_control_command(sync_config_message)
        assert message.result is True

        # test get config
        get_config_message = ControlMessage(
            new_message_id(), "", ControlMessageType.get_config, None
        )
        message = await pool.handle_control_command(get_config_message)
        config2 = message.result
        assert config.as_dict() == config2.as_dict()

        assert pool.router._mapping == Router.get_instance()._mapping
        assert (
            pool.router._curr_external_addresses
            == Router.get_instance()._curr_external_addresses
        )

        stop_message = ControlMessage(
            new_message_id(), "", ControlMessageType.stop, None
        )
        message = await pool.handle_control_command(stop_message)
        assert message.result is True

        await pool.join(0.05)
        assert pool.stopped
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_fail_when_create_subpool():
    config = ActorPoolConfig()
    my_label = "computation"
    main_address = f"127.0.0.1:{get_next_port()}"
    port = get_next_port()
    _add_pool_conf(config, 0, "main", "unixsocket:///0", main_address)

    # use the same port for sub pools, will raise `OSError` with "address already in use"
    _add_pool_conf(
        config, 1, my_label, "unixsocket:///1", f"127.0.0.1:{port}", env={"my_env": "1"}
    )
    _add_pool_conf(config, 2, my_label, "unixsocket:///2", f"127.0.0.1:{port}")

    with pytest.raises(OSError):
        await MainActorPool.create({"actor_pool_config": config})


@pytest.mark.asyncio
async def test_main_actor_pool():
    config = ActorPoolConfig()
    my_label = "computation"
    main_address = f"127.0.0.1:{get_next_port()}"
    _add_pool_conf(config, 0, "main", "unixsocket:///0", main_address)
    _add_pool_conf(
        config,
        1,
        my_label,
        "unixsocket:///1",
        f"127.0.0.1:{get_next_port()}",
        env={"my_env": "1"},
    )
    _add_pool_conf(
        config, 2, my_label, "unixsocket:///2", f"127.0.0.1:{get_next_port()}"
    )

    strategy = IdleLabel(my_label, "my_test")

    async with await MainActorPool.create({"actor_pool_config": config}) as pool:
        create_actor_message = CreateActorMessage(
            new_message_id(), TestActor, b"test", tuple(), dict(), MainPool()
        )
        message = await pool.create_actor(create_actor_message)
        actor_ref = message.result
        assert actor_ref.address == main_address

        create_actor_message1 = CreateActorMessage(
            new_message_id(), TestActor, b"test1", tuple(), dict(), strategy
        )
        message1 = await pool.create_actor(create_actor_message1)
        actor_ref1 = message1.result
        assert actor_ref1.address in config.get_external_addresses(my_label)

        create_actor_message2 = CreateActorMessage(
            new_message_id(), TestActor, b"test2", tuple(), dict(), strategy
        )
        message2 = await pool.create_actor(create_actor_message2)
        actor_ref2 = message2.result
        assert actor_ref2.address in config.get_external_addresses(my_label)
        assert actor_ref2.address != actor_ref1.address

        create_actor_message3 = CreateActorMessage(
            new_message_id(), TestActor, b"test3", tuple(), dict(), strategy
        )
        message3 = await pool.create_actor(create_actor_message3)
        # no slot to allocate the same label
        assert isinstance(message3.error, NoIdleSlot)

        has_actor_message = HasActorMessage(
            new_message_id(), create_actor_ref(main_address, b"test2")
        )
        assert (await pool.has_actor(has_actor_message)).result is True

        actor_ref_message = ActorRefMessage(
            new_message_id(), create_actor_ref(main_address, b"test2")
        )
        assert (await pool.actor_ref(actor_ref_message)).result == actor_ref2

        # tell
        tell_message = TellMessage(
            new_message_id(), actor_ref1, ("add", 0, (2,), dict())
        )
        message = await pool.tell(tell_message)
        assert message.result is None

        # send
        send_message = SendMessage(
            new_message_id(), actor_ref1, ("add", 0, (4,), dict())
        )
        assert (await pool.send(send_message)).result == 6

        # test error message
        # type mismatch
        send_message = SendMessage(
            new_message_id(), actor_ref1, ("add", 0, ("3",), dict())
        )
        result = await pool.send(send_message)
        assert isinstance(result.error, TypeError)

        # send and tell to main process
        tell_message = TellMessage(
            new_message_id(), actor_ref, ("add", 0, (2,), dict())
        )
        message = await pool.tell(tell_message)
        assert message.result is None
        send_message = SendMessage(
            new_message_id(), actor_ref, ("add", 0, (4,), dict())
        )
        assert (await pool.send(send_message)).result == 6

        # send and cancel
        send_message = SendMessage(
            new_message_id(), actor_ref1, ("sleep", 0, (20,), dict())
        )
        result_task = asyncio.create_task(pool.send(send_message))
        start = time.time()
        cancel_message = CancelMessage(
            new_message_id(), actor_ref1.address, send_message.message_id
        )
        cancel_task = asyncio.create_task(pool.cancel(cancel_message))
        result = await asyncio.wait_for(cancel_task, 3)
        assert result.message_type == MessageType.result
        assert result.result is True
        result = await result_task
        assert time.time() - start < 3
        assert result.message_type == MessageType.result
        assert result.result == 7

        # destroy
        destroy_actor_message = DestroyActorMessage(new_message_id(), actor_ref1)
        message = await pool.destroy_actor(destroy_actor_message)
        assert message.result == actor_ref1.uid

        tell_message = TellMessage(
            new_message_id(), actor_ref1, ("add", 0, (2,), dict())
        )
        message = await pool.tell(tell_message)
        assert isinstance(message, ErrorMessage)

        # destroy via connecting to sub pool directly
        async with await pool.router.get_client(
            config.get_external_addresses()[-1]
        ) as client:
            destroy_actor_message = DestroyActorMessage(new_message_id(), actor_ref2)
            await client.send(destroy_actor_message)
            result = await client.recv()
            _raise_if_error(result)
            assert result.result == actor_ref2.uid

        # test sync config
        config.add_pool_conf(
            3, "sub", "unixsocket:///3", f"127.0.0.1:{get_next_port()}"
        )
        sync_config_message = ControlMessage(
            new_message_id(),
            pool.external_address,
            ControlMessageType.sync_config,
            config,
        )
        message = await pool.handle_control_command(sync_config_message)
        assert message.result is True

        # test get config
        get_config_message = ControlMessage(
            new_message_id(),
            config.get_external_addresses()[1],
            ControlMessageType.get_config,
            None,
        )
        message = await pool.handle_control_command(get_config_message)
        config2 = message.result
        assert config.as_dict() == config2.as_dict()

    assert pool.stopped


@pytest.mark.asyncio
async def test_create_actor_pool():
    pool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
    )

    async with pool:
        # test global router
        global_router = Router.get_instance()
        # global router should not be the identical one with pool's router
        assert global_router is not pool.router
        assert pool.external_address in global_router._curr_external_addresses
        assert pool.external_address in global_router._mapping

        ctx = get_context()

        # actor on main pool
        actor_ref = await ctx.create_actor(
            TestActor, uid="test-1", address=pool.external_address
        )
        assert await actor_ref.add(3) == 3
        assert await actor_ref.add(1) == 4
        assert (await ctx.has_actor(actor_ref)) is True
        assert (await ctx.actor_ref(actor_ref)) == actor_ref
        # test cancel
        task = asyncio.create_task(actor_ref.sleep(20))
        await asyncio.sleep(0)
        task.cancel()
        assert await task == 5
        await ctx.destroy_actor(actor_ref)
        assert (await ctx.has_actor(actor_ref)) is False
        for f in actor_ref.add, ctx.actor_ref, ctx.destroy_actor:
            with pytest.raises(ActorNotExist):
                await f(actor_ref)

        # actor on sub pool
        actor_ref1 = await ctx.create_actor(
            TestActor, uid="test-main", address=pool.external_address
        )
        actor_ref2 = await ctx.create_actor(
            TestActor,
            uid="test-2",
            address=pool.external_address,
            allocate_strategy=RandomSubPool(),
        )
        assert (
            await ctx.actor_ref(uid="test-2", address=actor_ref2.address)
        ) == actor_ref2
        main_ref = await ctx.actor_ref(uid="test-main", address=actor_ref2.address)
        assert main_ref.address == pool.external_address
        main_ref = await ctx.actor_ref(actor_ref1)
        assert main_ref.address == pool.external_address
        assert actor_ref2.address != actor_ref.address
        assert await actor_ref2.add(3) == 3
        assert await actor_ref2.add(1) == 4
        with pytest.raises(RuntimeError):
            await actor_ref2.return_cannot_unpickle()
        with pytest.raises(SendMessageFailed):
            await actor_ref2.raise_cannot_pickle()
        assert (await ctx.has_actor(actor_ref2)) is True
        assert (await ctx.actor_ref(actor_ref2)) == actor_ref2
        # test cancel
        task = asyncio.create_task(actor_ref2.sleep(20))
        start = time.time()
        await asyncio.sleep(0)
        task.cancel()
        assert await task == 5
        assert time.time() - start < 3
        await ctx.destroy_actor(actor_ref2)
        assert (await ctx.has_actor(actor_ref2)) is False

    assert pool.stopped
    # after pool shutdown, global router must has been cleaned
    global_router = Router.get_instance()
    assert len(global_router._curr_external_addresses) == 0
    assert len(global_router._mapping) == 0


@pytest.mark.asyncio
async def test_create_actor_pool_extra_config():
    # create a actor pool based on socket rather than ucx
    # pass `extra_conf` to check if we can filter out ucx config
    pool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        extra_conf={
            "ucx": {
                "tcp": None,
                "nvlink": None,
                "infiniband": None,
                "rdmacm": None,
                "cuda-copy": None,
                "create-cuda-contex": None,
            }
        },
    )

    async with pool:
        # test global router
        global_router = Router.get_instance()
        # global router should not be the identical one with pool's router
        assert global_router is not pool.router
        assert pool.external_address in global_router._curr_external_addresses
        assert pool.external_address in global_router._mapping

        ctx = get_context()

        # actor on main pool
        actor_ref = await ctx.create_actor(
            TestActor, uid="test-1", address=pool.external_address
        )
        assert await actor_ref.add(3) == 3
        assert await actor_ref.add(1) == 4
        assert (await ctx.has_actor(actor_ref)) is True
        assert (await ctx.actor_ref(actor_ref)) == actor_ref
        # test cancel
        task = asyncio.create_task(actor_ref.sleep(20))
        await asyncio.sleep(0)
        task.cancel()
        assert await task == 5
        await ctx.destroy_actor(actor_ref)
        assert (await ctx.has_actor(actor_ref)) is False
        for f in actor_ref.add, ctx.actor_ref, ctx.destroy_actor:
            with pytest.raises(ActorNotExist):
                await f(actor_ref)

        # actor on sub pool
        actor_ref1 = await ctx.create_actor(
            TestActor, uid="test-main", address=pool.external_address
        )
        actor_ref2 = await ctx.create_actor(
            TestActor,
            uid="test-2",
            address=pool.external_address,
            allocate_strategy=RandomSubPool(),
        )
        assert (
            await ctx.actor_ref(uid="test-2", address=actor_ref2.address)
        ) == actor_ref2
        main_ref = await ctx.actor_ref(uid="test-main", address=actor_ref2.address)
        assert main_ref.address == pool.external_address
        main_ref = await ctx.actor_ref(actor_ref1)
        assert main_ref.address == pool.external_address
        assert actor_ref2.address != actor_ref.address
        assert await actor_ref2.add(3) == 3
        assert await actor_ref2.add(1) == 4
        with pytest.raises(RuntimeError):
            await actor_ref2.return_cannot_unpickle()
        with pytest.raises(SendMessageFailed):
            await actor_ref2.raise_cannot_pickle()
        assert (await ctx.has_actor(actor_ref2)) is True
        assert (await ctx.actor_ref(actor_ref2)) == actor_ref2
        # test cancel
        task = asyncio.create_task(actor_ref2.sleep(20))
        start = time.time()
        await asyncio.sleep(0)
        task.cancel()
        assert await task == 5
        assert time.time() - start < 3
        await ctx.destroy_actor(actor_ref2)
        assert (await ctx.has_actor(actor_ref2)) is False

    assert pool.stopped
    # after pool shutdown, global router must has been cleaned
    global_router = Router.get_instance()
    assert len(global_router._curr_external_addresses) == 0
    assert len(global_router._mapping) == 0


@pytest.mark.asyncio
@require_unix
async def test_create_actor_pool_elastic_ip():
    addr = f"111.111.111.111:{get_next_port()}"
    pool = await create_actor_pool(
        addr,
        pool_cls=MainActorPool,
        n_process=0,
        extra_conf={"listen_elastic_ip": True},
    )
    async with pool:
        # test global router
        global_router = Router.get_instance()
        # global router should not be the identical one with pool's router
        assert global_router is not pool.router
        assert pool.external_address in global_router._curr_external_addresses
        assert pool.external_address in global_router._mapping
        assert pool.external_address == addr

        ctx = get_context()

        # actor on main pool
        actor_ref = await ctx.create_actor(
            TestActor, uid="test-1", address=pool.external_address
        )
        assert await actor_ref.add(3) == 3
        await ctx.destroy_actor(actor_ref)
        assert (await ctx.has_actor(actor_ref)) is False

    assert pool.stopped
    # after pool shutdown, global router must has been cleaned
    global_router = Router.get_instance()
    assert len(global_router._curr_external_addresses) == 0
    assert len(global_router._mapping) == 0


@pytest.mark.asyncio
async def test_create_actor_pool_fix_all_zero_ip():
    port = get_next_port()
    addr = f"0.0.0.0:{port}"
    pool = await create_actor_pool(
        addr,
        pool_cls=MainActorPool,
        n_process=0,
    )
    async with pool:
        # test global router
        global_router = Router.get_instance()
        # global router should not be the identical one with pool's router
        assert global_router is not pool.router
        assert pool.external_address in global_router._curr_external_addresses
        assert pool.external_address in global_router._mapping
        assert pool.external_address == addr

        ctx = get_context()

        # actor on main pool
        actor_ref = await ctx.create_actor(
            TestActor, uid="test-1", address=pool.external_address
        )
        assert await actor_ref.add(3) == 3
        connect_addr = f"127.0.0.1:{port}"
        actor_ref2 = await ctx.actor_ref(address=connect_addr, uid="test-1")
        # test fix_all_zero_ip, the result is not 0.0.0.0
        assert actor_ref2.address == connect_addr
        assert await actor_ref.add(3) == 6

        await ctx.destroy_actor(actor_ref)
        assert (await ctx.has_actor(actor_ref)) is False

    assert pool.stopped
    # after pool shutdown, global router must has been cleaned
    global_router = Router.get_instance()
    assert len(global_router._curr_external_addresses) == 0
    assert len(global_router._mapping) == 0


@pytest.mark.asyncio
async def test_create_actor_pool_ipv6():
    port = get_next_port()
    addr = f":::{port}"
    pool = await create_actor_pool(
        addr,
        pool_cls=MainActorPool,
        n_process=0,
    )
    async with pool:
        # test global router
        global_router = Router.get_instance()
        # global router should not be the identical one with pool's router
        assert global_router is not pool.router
        assert pool.external_address in global_router._curr_external_addresses
        assert pool.external_address in global_router._mapping
        assert pool.external_address == addr

        ctx = get_context()

        # actor on main pool
        actor_ref = await ctx.create_actor(
            TestActor, uid="test-1", address=pool.external_address
        )
        assert await actor_ref.add(3) == 3
        await ctx.destroy_actor(actor_ref)
        assert (await ctx.has_actor(actor_ref)) is False

    assert pool.stopped
    # after pool shutdown, global router must has been cleaned
    global_router = Router.get_instance()
    assert len(global_router._curr_external_addresses) == 0
    assert len(global_router._mapping) == 0


@pytest.mark.asyncio
@require_unix
async def test_create_actor_pool_ipv6_elastic_ip():
    port = get_next_port()
    # ip not exists on local host
    addr = f"FFFF:34::55::1:{port}"
    pool = await create_actor_pool(
        addr,
        pool_cls=MainActorPool,
        n_process=0,
        extra_conf={"listen_elastic_ip": True},
    )
    async with pool:
        # test global router
        global_router = Router.get_instance()
        # global router should not be the identical one with pool's router
        assert global_router is not pool.router
        assert pool.external_address in global_router._curr_external_addresses
        assert pool.external_address in global_router._mapping
        assert pool.external_address == addr

        ctx = get_context()
        # actor on main pool
        actor_ref = await ctx.create_actor(
            TestActor, uid="test-1", address=pool.external_address
        )
        # test fix_all_zero_ip, the result is not :::port
        assert actor_ref.address == addr
        assert await actor_ref.add(3) == 3
        connect_addr = f"::1:{port}"
        actor_ref2 = await ctx.actor_ref(address=connect_addr, uid="test-1")
        assert await actor_ref2.add(4) == 7
        assert actor_ref2.address == addr

        await ctx.destroy_actor(actor_ref)
        assert (await ctx.has_actor(actor_ref)) is False

    assert pool.stopped
    # after pool shutdown, global router must has been cleaned
    global_router = Router.get_instance()
    assert len(global_router._curr_external_addresses) == 0
    assert len(global_router._mapping) == 0


@pytest.mark.asyncio
async def test_errors():
    with pytest.raises(ValueError):
        _ = await create_actor_pool(
            "127.0.0.1", pool_cls=MainActorPool, n_process=1, labels=["a"]
        )

    with pytest.raises(ValueError):
        _ = await create_actor_pool(
            f"127.0.0.1:{get_next_port()}",
            pool_cls=MainActorPool,
            n_process=1,
            ports=[get_next_port(), get_next_port()],
        )

    with pytest.raises(ValueError):
        _ = await create_actor_pool(
            "127.0.0.1", pool_cls=MainActorPool, n_process=1, ports=[get_next_port()]
        )

    with pytest.raises(ValueError):
        _ = await create_actor_pool(
            "127.0.0.1", pool_cls=MainActorPool, n_process=1, auto_recover="illegal"
        )

    with pytest.raises(ValueError, match="external_address_schemes"):
        _ = await create_actor_pool(
            "127.0.0.1",
            pool_cls=MainActorPool,
            n_process=1,
            external_address_schemes=["ucx"],
        )

    with pytest.raises(ValueError, match="enable_internal_addresses"):
        _ = await create_actor_pool(
            "127.0.0.1",
            pool_cls=MainActorPool,
            n_process=1,
            enable_internal_addresses=[True],
        )


@pytest.mark.asyncio
async def test_server_closed():
    pool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        auto_recover=False,
    )

    ctx = get_context()

    async with pool:
        actor_ref = await ctx.create_actor(
            TestActor, address=pool.external_address, allocate_strategy=ProcessIndex(1)
        )

        # check if error raised normally when subprocess killed
        task = asyncio.create_task(actor_ref.sleep(10))
        await asyncio.sleep(0.1)

        # kill subprocess 1
        process = list(pool._sub_processes.values())[0]
        process.kill()
        await process.wait()

        with pytest.raises(ServerClosed):
            # process already been killed,
            # ServerClosed will be raised
            await task

        assert process.returncode is not None

    with pytest.raises(RuntimeError):
        await pool.start()

    # test server unreachable
    with pytest.raises(ConnectionError):
        await ctx.has_actor(actor_ref)


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform.startswith("win"), reason="skip under Windows")
@pytest.mark.parametrize("auto_recover", [False, True, "actor", "process"])
async def test_auto_recover(auto_recover):
    recovered = asyncio.Event()

    def on_process_recover(*_):
        recovered.set()

    pool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        auto_recover=auto_recover,
        on_process_recover=on_process_recover,
    )

    async with pool:
        ctx = get_context()

        # wait for recover of main pool always returned immediately
        await ctx.wait_actor_pool_recovered(
            pool.external_address, pool.external_address
        )

        # create actor on main
        actor_ref = await ctx.create_actor(
            TestActor, address=pool.external_address, allocate_strategy=MainPool()
        )

        with pytest.raises(ValueError):
            # cannot kill actors on main pool
            await kill_actor(actor_ref)

        # create actor
        actor_ref = await ctx.create_actor(
            TestActor, address=pool.external_address, allocate_strategy=ProcessIndex(1)
        )
        # kill_actor will cause kill corresponding process
        await ctx.kill_actor(actor_ref)

        if auto_recover:
            # process must have been killed
            await ctx.wait_actor_pool_recovered(
                actor_ref.address, pool.external_address
            )
            assert recovered.is_set()

            expect_has_actor = True if auto_recover in ["actor", True] else False
            assert await ctx.has_actor(actor_ref) is expect_has_actor
        else:
            with pytest.raises((ServerClosed, ConnectionError)):
                await ctx.has_actor(actor_ref)


@pytest.mark.parametrize(
    "exception_config",
    [
        (Exception("recover exception"), False),
        (asyncio.CancelledError("cancel monitor"), True),
    ],
)
@pytest.mark.asyncio
async def test_monitor_sub_pool_exception(exception_config):
    recovered = asyncio.Event()
    exception, done = exception_config

    def on_process_recover(*_):
        recovered.set()
        raise exception

    pool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        on_process_recover=on_process_recover,
    )

    async with pool:
        ctx = get_context()
        task = await pool.start_monitor()

        # create actor
        actor_ref = await ctx.create_actor(
            TestActor, address=pool.external_address, allocate_strategy=ProcessIndex(1)
        )
        # kill_actor will cause kill corresponding process
        await ctx.kill_actor(actor_ref)

        await recovered.wait()
        assert task.done() is done


@pytest.mark.asyncio
async def test_two_pools():
    ctx = get_context()

    pool1 = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
    )
    pool2 = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
    )

    def is_interprocess_address(addr):
        if sys.platform.startswith("win"):
            return re.match(r"127\.0\.0\.1:\d+", addr)
        else:
            return addr.startswith("unixsocket://")

    try:
        actor_ref1 = await ctx.create_actor(
            TestActor, address=pool1.external_address, allocate_strategy=MainPool()
        )
        assert actor_ref1.address == pool1.external_address
        assert await actor_ref1.add(1) == 1
        assert (
            Router.get_instance()
            .get_internal_address(actor_ref1.address)
            .startswith("dummy://")
        )

        actor_ref2 = await ctx.create_actor(
            TestActor, address=pool1.external_address, allocate_strategy=RandomSubPool()
        )
        assert actor_ref2.address in pool1._config.get_external_addresses()[1:]
        assert await actor_ref2.add(3) == 3
        assert is_interprocess_address(
            Router.get_instance().get_internal_address(actor_ref2.address)
        )

        actor_ref3 = await ctx.create_actor(
            TestActor, address=pool2.external_address, allocate_strategy=MainPool()
        )
        assert actor_ref3.address == pool2.external_address
        assert await actor_ref3.add(5) == 5
        assert (
            Router.get_instance()
            .get_internal_address(actor_ref3.address)
            .startswith("dummy://")
        )

        actor_ref4 = await ctx.create_actor(
            TestActor, address=pool2.external_address, allocate_strategy=RandomSubPool()
        )
        assert actor_ref4.address in pool2._config.get_external_addresses()[1:]
        assert await actor_ref4.add(7) == 7
        assert is_interprocess_address(
            Router.get_instance().get_internal_address(actor_ref4.address)
        )

        assert await actor_ref2.add_other(actor_ref4, 3) == 13
    finally:
        await pool1.stop()
        await pool2.stop()


@pytest.mark.asyncio
async def test_parallel_allocate_idle_label():
    pool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        labels=[None, "my_label", "my_label"],
    )

    class _Actor(Actor):
        def get_pid(self):
            return os.getpid()

    async with pool:
        ctx = get_context()
        strategy = IdleLabel("my_label", "tests")
        tasks = [
            ctx.create_actor(
                _Actor, allocate_strategy=strategy, address=pool.external_address
            ),
            ctx.create_actor(
                _Actor, allocate_strategy=strategy, address=pool.external_address
            ),
        ]
        refs = await asyncio.gather(*tasks)
        # outputs identical process ids, while the result should be different
        assert len({await ref.get_pid() for ref in refs}) == 2


# equivalent to test-logging.conf
DICT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "formatter": {
            "format": "%(asctime)s %(name)-12s %(process)d %(levelname)-8s %(message)s",
        },
    },
    "handlers": {
        "stream_handler": {
            "class": "logging.StreamHandler",
            "formatter": "formatter",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "": {
            "level": "WARN",
            "handlers": ["stream_handler"],
        },
        "xoscar.backends.indigen.tests": {
            "level": "DEBUG",
            "handlers": ["stream_handler"],
            "propagate": False,
        },
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "logging_conf",
    [
        {
            "file": os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "test-logging.conf"
            )
        },
        {"level": logging.DEBUG},
        {"level": logging.DEBUG, "format": "%(asctime)s %(message)s"},
        {"dict": DICT_CONFIG},
    ],
)
async def test_logging_config(logging_conf):
    pool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=1,
        labels=[None, "my_label"],
        logging_conf=logging_conf,
    )

    class _Actor(Actor):
        def get_logger_level(self):
            logger = logging.getLogger(__name__)
            return logger.getEffectiveLevel()

    async with pool:
        ctx = get_context()
        strategy = IdleLabel("my_label", "tests")
        ref = await ctx.create_actor(
            _Actor, allocate_strategy=strategy, address=pool.external_address
        )
        assert await ref.get_logger_level() == logging.DEBUG


@pytest.mark.asyncio
async def test_ref_sub_pool_actor():
    pool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=1,
    )

    async with pool:
        ctx = get_context()
        ref1 = await ctx.create_actor(
            TestActor, address=pool.external_address, allocate_strategy=RandomSubPool()
        )
        sub_address = ref1.address
        ref2 = await ctx.create_actor(TestActor, address=sub_address)
        ref2_main = await ctx.actor_ref(ref2.uid, address=pool.external_address)
        assert ref2_main.address == sub_address

        await ctx.destroy_actor(create_actor_ref(pool.external_address, ref2.uid))
        assert not await ctx.has_actor(
            create_actor_ref(pool.external_address, ref2.uid)
        )
        assert not await ctx.has_actor(create_actor_ref(sub_address, ref2.uid))


class TestUCXActor(Actor):
    __test__ = False

    def __init__(self, init_val: int):
        self._init_val = init_val

    def verify(self, enabled_internal_addr: bool):
        router = Router.get_instance()
        assert router.external_address.startswith("ucx")  # type: ignore
        assert len(router._mapping) > 0  # type: ignore
        if not enabled_internal_addr:
            # no internal address
            assert all(v is None for v in router._mapping.values())  # type: ignore
        else:
            assert all(v is not None for v in router._mapping.values())  # type: ignore

    def add(self, n: int):
        return self._init_val + n

    async def foo(self, ref, n: int):
        assert self.address != ref.address
        return self._init_val + await ref.add(n)


@require_ucx
@pytest.mark.asyncio
@pytest.mark.parametrize("enable_internal_addr", [False, True])
async def test_ucx(enable_internal_addr: bool):
    pool = await create_actor_pool(  # type: ignore
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        external_address_schemes=["ucx"] * 3,
        enable_internal_addresses=[enable_internal_addr] * 3,
    )

    async with pool:
        ctx = get_context()
        ref1 = await ctx.create_actor(
            TestUCXActor,
            1,
            address=pool.external_address,
            allocate_strategy=ProcessIndex(0),
        )
        await ref1.verify(enable_internal_addr)
        ref2 = await ctx.create_actor(
            TestUCXActor,
            2,
            address=pool.external_address,
            allocate_strategy=ProcessIndex(1),
        )
        assert await ref1.foo(ref2, 3) == 6


@require_ucx
@pytest.mark.asyncio
async def test_ucx_elastic_ip():
    port = get_next_port()
    addr = f"111.111.111.111:{port}"
    pool = await create_actor_pool(  # type: ignore
        addr,
        pool_cls=MainActorPool,
        n_process=0,
        external_address_schemes=["ucx"],
        extra_conf={"listen_elastic_ip": True},
    )

    async with pool:
        ctx = get_context()
        ref1 = await ctx.create_actor(
            TestUCXActor, init_val=0, address=pool.external_address, uid="test-ucx"
        )
        assert ref1.address == "ucx://" + addr
        ref2 = await ctx.actor_ref(address=f"ucx://127.0.0.1:{port}", uid="test-ucx")
        assert await ref2.add(1) == 1
        assert await ref1.add(2) == 2
        assert ref2.address == "ucx://" + addr


@pytest.mark.asyncio
async def test_append_sub_pool_multiprocess():
    pool = await create_actor_pool(  # type: ignore
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
    )

    async with pool:
        config = await get_pool_config(pool.external_address)
        assert len(config.get_process_indexes()) == 3

        # test add a new sub pool
        sub_external_address = await pool.append_sub_pool(env={"foo": "bar"})
        assert sub_external_address is not None
        assert sub_external_address.startswith("127.0.0.1:")

        config = await get_pool_config(pool.external_address)
        assert len(config.get_process_indexes()) == 4
        process_index = config.get_process_indexes()[-1]
        sub_config = config.get_pool_config(process_index)
        assert sub_config["external_address"][0] == sub_external_address
        assert sub_config["env"] is not None
        assert sub_config["env"].get("foo", None) == "bar"

        class DummyActor(Actor):
            @staticmethod
            def test():
                return "this is dummy!"

        ref = await create_actor(DummyActor, address=sub_external_address)
        assert ref is not None
        assert ref.address == sub_external_address
        assert await ref.test() == "this is dummy!"

        # test remove
        await pool.remove_sub_pool(sub_external_address)
        config = await get_pool_config(pool.external_address)
        assert len(config.get_process_indexes()) == 3
        assert process_index not in config.get_process_indexes()
        with pytest.raises(KeyError):
            config.get_pool_config(process_index)
        with pytest.raises(Exception):
            await ref.test()


@pytest.mark.asyncio
@require_unix
async def test_append_sub_pool_multi_process_elastic_ip():
    pool = await create_actor_pool(  # type: ignore
        "111.111.111.111",
        pool_cls=MainActorPool,
        n_process=2,
        extra_conf={"listen_elastic_ip": True},
    )

    async with pool:
        config = await get_pool_config(pool.external_address)
        assert len(config.get_process_indexes()) == 3

        # test add a new sub pool
        sub_external_address = await pool.append_sub_pool(env={"foo": "bar"})
        assert sub_external_address is not None
        assert sub_external_address.startswith("111.111.111.111:")

        config = await get_pool_config(pool.external_address)
        assert len(config.get_process_indexes()) == 4
        process_index = config.get_process_indexes()[-1]
        sub_config = config.get_pool_config(process_index)
        assert sub_config["external_address"][0] == sub_external_address
        assert sub_config["env"] is not None
        assert sub_config["env"].get("foo", None) == "bar"

        class DummyActor(Actor):
            @staticmethod
            def test():
                return "this is dummy!"

        ref = await create_actor(DummyActor, address=sub_external_address)
        assert ref is not None
        assert ref.address == sub_external_address
        assert await ref.test() == "this is dummy!"

        # test remove
        await pool.remove_sub_pool(sub_external_address)
        config = await get_pool_config(pool.external_address)
        assert len(config.get_process_indexes()) == 3
        assert process_index not in config.get_process_indexes()
        with pytest.raises(KeyError):
            config.get_pool_config(process_index)
        with pytest.raises(Exception):
            await ref.test()


@pytest.mark.asyncio
@require_unix
async def test_append_sub_pool_single_process_elastic_ip():
    pool = await create_actor_pool(  # type: ignore
        f"111.111.111.111:{get_next_port()}",
        pool_cls=MainActorPool,
        n_process=0,
        extra_conf={"listen_elastic_ip": True},
    )

    async with pool:
        config = await get_pool_config(pool.external_address)
        assert len(config.get_process_indexes()) == 1

        # test add a new sub pool
        sub_external_address = await pool.append_sub_pool(env={"foo": "bar"})
        assert sub_external_address is not None
        assert sub_external_address.startswith("111.111.111.111:")

        config = await get_pool_config(pool.external_address)
        assert len(config.get_process_indexes()) == 2
        process_index = config.get_process_indexes()[-1]
        sub_config = config.get_pool_config(process_index)
        assert sub_config["external_address"][0] == sub_external_address
        assert sub_config["env"] is not None
        assert sub_config["env"].get("foo", None) == "bar"

        class DummyActor(Actor):
            @staticmethod
            def test():
                return "this is dummy!"

        ref = await create_actor(DummyActor, address=sub_external_address)
        assert ref is not None
        assert ref.address == sub_external_address
        assert await ref.test() == "this is dummy!"

        # test remove
        await pool.remove_sub_pool(sub_external_address)
        config = await get_pool_config(pool.external_address)
        assert len(config.get_process_indexes()) == 1
        assert process_index not in config.get_process_indexes()
        with pytest.raises(KeyError):
            config.get_pool_config(process_index)
        with pytest.raises(Exception):
            await ref.test()


@pytest.mark.asyncio
async def test_test_pool_append_sub_pool():
    pool = await create_actor_pool(  # type: ignore
        f"test://127.0.0.1:{get_next_port()}", pool_cls=TestMainActorPool, n_process=1
    )

    async with pool:
        config = await get_pool_config(pool.external_address)
        assert len(config.get_process_indexes()) == 2

        # test add a new sub pool
        sub_external_address = await pool.append_sub_pool(env={"foo": "bar"})
        assert sub_external_address is not None

        config = await get_pool_config(pool.external_address)
        assert len(config.get_process_indexes()) == 3
        process_index = config.get_process_indexes()[-1]
        sub_config = config.get_pool_config(process_index)
        assert sub_config["external_address"][0] == sub_external_address
        assert sub_config["env"] is not None
        assert sub_config["env"].get("foo", None) == "bar"

        class DummyActor(Actor):
            @staticmethod
            def test():
                return "this is dummy!"

        ref = await create_actor(DummyActor, address=sub_external_address)
        assert ref is not None

        assert ref.address == sub_external_address
        assert await ref.test() == "this is dummy!"

        # test remove
        await pool.remove_sub_pool(sub_external_address)
        config = await get_pool_config(pool.external_address)
        assert len(config.get_process_indexes()) == 2
        assert process_index not in config.get_process_indexes()
        with pytest.raises(KeyError):
            config.get_pool_config(process_index)


async def _run(started: multiprocessing.Event):  # type: ignore
    pool = await create_actor_pool(  # type: ignore
        "127.0.0.1", pool_cls=MainActorPool, n_process=1
    )

    class DummyActor(Actor):
        @staticmethod
        def test():
            return "this is dummy!"

    ref = await create_actor(
        DummyActor, address=pool.external_address, allocate_strategy=RandomSubPool()
    )
    assert ref is not None

    started.set()  # type: ignore
    await pool.join()


def _run_in_process(started: multiprocessing.Event):  # type: ignore
    asyncio.run(_run(started))


@pytest.mark.asyncio
async def test_sub_pool_quit_with_main_pool():
    s = multiprocessing.Event()
    p = multiprocessing.Process(target=_run_in_process, args=(s,))
    p.start()
    s.wait()

    processes = psutil.Process(p.pid).children()
    assert len(processes) == 1

    # kill main process
    p.kill()
    p.join()
    await asyncio.sleep(1)

    # subprocess should have died
    assert not psutil.pid_exists(processes[0].pid)


def _add(x: int) -> int:
    return x + 1


class _ProcessActor(Actor):
    def run(self, x: int):
        p = multiprocessing.Process(target=_add, args=(x,))
        p.start()
        p.join()
        return x + 1


@pytest.mark.asyncio
async def test_process_in_actor():
    pool = await create_actor_pool(  # type: ignore
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=1,
    )

    async with pool:
        ref = await create_actor(
            _ProcessActor,
            address=pool.external_address,
            allocate_strategy=RandomSubPool(),
        )
        assert 2 == await ref.run(1)


class AActor(Actor):
    async def call_destroy(self, ref):
        coro = xoscar.destroy_actor(ref)
        start = time.time()
        with pytest.raises(asyncio.TimeoutError):
            await wait_for(coro, timeout=1)
        return time.time() - start


class BActor(Actor):
    async def __pre_destroy__(self):
        time.sleep(20)


@pytest.mark.asyncio
async def test_pre_destroy_stuck():
    pool = await create_actor_pool(  # type: ignore
        "127.0.0.1", pool_cls=MainActorPool, n_process=2
    )

    async with pool:
        a = await xoscar.create_actor(
            AActor,
            address=pool.external_address,
            uid="a",
            allocate_strategy=xoscar.allocate_strategy.ProcessIndex(1),
        )
        b = await xoscar.create_actor(
            BActor,
            address=pool.external_address,
            uid="b",
            allocate_strategy=xoscar.allocate_strategy.ProcessIndex(2),
        )

        duration = await a.call_destroy(b)
        assert duration < 5

        await pool.kill_sub_pool(list(pool._sub_processes.values())[1])
