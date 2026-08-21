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
import threading
from unittest import mock

import pytest

from ..backends.core import ActorCallerThreadLocal
from ..backends.router import Router
from ..errors import ServerClosed


class FakeClient:
    def __init__(self, dest_address):
        self.dest_address = dest_address
        self.closed = False
        self.close_count = 0

    async def close(self):
        self.close_count += 1
        self.closed = True


def test_mapping_updates_preserve_other_thread_cache():
    router = Router(["main"], None)
    child_router = Router(["subpool"], None, {"subpool": "unixsocket"})
    ready = threading.Event()
    updated = threading.Event()
    result = {}

    def use_cache():
        result["before"] = router._cache
        ready.set()
        assert updated.wait(timeout=5)
        result["after"] = router._cache

    thread = threading.Thread(target=use_cache)
    thread.start()
    assert ready.wait(timeout=5)

    router.set_mapping({"worker": "127.0.0.1:1234"})
    router.add_router(child_router)
    router.remove_router(child_router)
    updated.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert result["after"] is result["before"]


@pytest.mark.asyncio
async def test_get_client_reuses_unchanged_route_and_closes_changed_route():
    router = Router([], None, {"worker": "127.0.0.1:1234"})
    clients = []

    async def create_client(_client_type, address, **_kwargs):
        client = FakeClient(address)
        clients.append(client)
        return client

    with mock.patch.object(Router, "_create_client", side_effect=create_client):
        first = await router.get_client("worker")
        router.set_mapping(
            {"worker": "127.0.0.1:1234", "other": "127.0.0.1:1235"}
        )
        assert await router.get_client("worker") is first
        assert first.close_count == 0

        router.set_mapping({"worker": "127.0.0.1:4321"})
        second = await router.get_client("worker")

    assert second is not first
    assert first.close_count == 1
    assert first.closed
    assert second.dest_address == "127.0.0.1:4321"
    assert len(clients) == 2


@pytest.mark.asyncio
async def test_closing_stale_client_does_not_hold_router_lock():
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    class BlockingCloseClient(FakeClient):
        async def close(self):
            close_started.set()
            await allow_close.wait()
            await super().close()

    router = Router([], None, {"worker": "127.0.0.1:1234"})
    first_client = True

    async def create_client(_client_type, address, **_kwargs):
        nonlocal first_client
        if first_client:
            first_client = False
            return BlockingCloseClient(address)
        return FakeClient(address)

    with mock.patch.object(Router, "_create_client", side_effect=create_client):
        await router.get_client("worker")
        router.set_mapping(
            {"worker": "127.0.0.1:4321", "other": "127.0.0.1:1235"}
        )
        changed_route = asyncio.create_task(router.get_client("worker"))
        await close_started.wait()

        other_client = await asyncio.wait_for(router.get_client("other"), timeout=1)
        assert other_client.dest_address == "127.0.0.1:1235"

        allow_close.set()
        await changed_route


@pytest.mark.asyncio
async def test_listener_removes_closed_client():
    class ClosingClient(FakeClient):
        async def recv(self):
            raise EOFError

    caller = ActorCallerThreadLocal()
    client = ClosingClient("127.0.0.1:1234")
    caller._listen_client(client)  # noqa: SLF001
    task = caller._clients[client]  # noqa: SLF001

    await task

    assert client not in caller._clients  # noqa: SLF001
    assert client not in caller._client_to_message_futures  # noqa: SLF001
    assert client.closed


@pytest.mark.asyncio
async def test_listener_cancellation_notifies_pending_futures():
    class BlockingClient(FakeClient):
        async def recv(self):
            await asyncio.Event().wait()

    caller = ActorCallerThreadLocal()
    client = BlockingClient("127.0.0.1:1234")
    caller._listen_client(client)  # noqa: SLF001
    task = caller._clients[client]  # noqa: SLF001
    pending = asyncio.get_running_loop().create_future()
    caller._client_to_message_futures[client][  # noqa: SLF001
        b"message-id"
    ] = pending

    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with pytest.raises(ServerClosed):
        await pending

    assert client not in caller._clients  # noqa: SLF001
    assert client not in caller._client_to_message_futures  # noqa: SLF001
    assert client.closed


@pytest.mark.asyncio
async def test_calls_reject_client_without_listener_state():
    class Message:
        message_id = b"message-id"

    caller = ActorCallerThreadLocal()
    client = FakeClient("127.0.0.1:1234")

    with pytest.raises(ServerClosed):
        await caller.call_with_client(client, Message())

    with pytest.raises(ServerClosed):
        await caller.call_send_buffers(client, [], Message())  # type: ignore[arg-type]
