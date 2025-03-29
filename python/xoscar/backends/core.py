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
import atexit
import copy
import logging
import threading
import weakref
from typing import Type, Union

from .._utils import Timer
from ..errors import ServerClosed
from ..profiling import get_profiling_data
from .communication import ChannelType, Client, UCXClient
from .message import (
    DeserializeMessageFailed,
    ErrorMessage,
    ForwardMessage,
    MessageType,
    ResultMessage,
    _MessageBase,
)
from .router import Router

ResultMessageType = Union[ResultMessage, ErrorMessage]
logger = logging.getLogger(__name__)


class ActorCallerThreadLocal:
    __slots__ = ("_client_to_message_futures", "_clients", "_profiling_data")

    _client_to_message_futures: dict[Client, dict[bytes, asyncio.Future]]
    _clients: dict[Client, asyncio.Task]

    def __init__(self):
        self._client_to_message_futures = dict()
        self._clients = dict()
        self._profiling_data = get_profiling_data()

    def _listen_client(self, client: Client):
        if client not in self._clients:
            self._clients[client] = asyncio.create_task(self._listen(client))
            self._client_to_message_futures[client] = dict()
            client_count = len(self._clients)
            if client_count >= 100:  # pragma: no cover
                if (client_count - 100) % 10 == 0:  # pragma: no cover
                    logger.warning(
                        "Actor caller has created too many clients (%s >= 100), "
                        "the global router may not be set.",
                        client_count,
                    )

    async def get_client_via_type(
        self, router: Router, dest_address: str, client_type: Type[Client]
    ) -> Client:
        client = await router.get_client_via_type(
            dest_address, client_type, from_who=self
        )
        self._listen_client(client)
        return client

    async def get_client(
        self,
        router: Router,
        dest_address: str,
        proxy_addresses: list[str] | None = None,
    ) -> Client:
        client = await router.get_client(
            dest_address, from_who=self, proxy_addresses=proxy_addresses
        )
        self._listen_client(client)
        return client

    async def _listen(self, client: Client):
        try:
            while not client.closed:
                try:
                    try:
                        message: _MessageBase = await client.recv()
                    except (EOFError, ConnectionError, BrokenPipeError) as e:
                        # AssertionError is from get_header
                        # remote server closed, close client and raise ServerClosed
                        logger.debug(f"{client.dest_address} close due to {e}")
                        try:
                            await client.close()
                        except (ConnectionError, BrokenPipeError):
                            # close failed, ignore it
                            pass
                        raise ServerClosed(
                            f"Remote server {client.dest_address} closed: {e}"
                        ) from None
                    future = self._client_to_message_futures[client].pop(
                        message.message_id
                    )
                    if not future.done():
                        future.set_result(message)
                except DeserializeMessageFailed as e:
                    message_id = e.message_id
                    future = self._client_to_message_futures[client].pop(message_id)
                    future.set_exception(e.__cause__)  # type: ignore
                except Exception as e:  # noqa: E722  # pylint: disable=bare-except
                    message_futures = self._client_to_message_futures[client]
                    self._client_to_message_futures[client] = dict()
                    for future in message_futures.values():
                        future.set_exception(copy.copy(e))
                finally:
                    # message may have Ray ObjectRef, delete it early in case next loop doesn't run
                    # as soon as expected.
                    try:
                        del message
                    except NameError:
                        pass
                    try:
                        del future
                    except NameError:
                        pass
                    await asyncio.sleep(0)

            message_futures = self._client_to_message_futures[client]
            self._client_to_message_futures[client] = dict()
            error = ServerClosed(f"Remote server {client.dest_address} closed")
            for future in message_futures.values():
                future.set_exception(copy.copy(error))
        finally:
            try:
                await client.close()
            except:  # noqa: E722  # nosec  # pylint: disable=bare-except
                # ignore all error if fail to close at last
                pass

    async def call_with_client(
        self, client: Client, message: _MessageBase, wait: bool = True
    ) -> ResultMessage | ErrorMessage | asyncio.Future:
        loop = asyncio.get_running_loop()
        wait_response = loop.create_future()
        self._client_to_message_futures[client][message.message_id] = wait_response

        with Timer() as timer:
            try:
                await client.send(message)
            except ConnectionError:
                try:
                    await client.close()
                except ConnectionError:
                    # close failed, ignore it
                    pass
                raise ServerClosed(f"Remote server {client.dest_address} closed")

            if not wait:
                r = wait_response
            else:
                r = await wait_response

        self._profiling_data.collect_actor_call(message, timer.duration)
        return r

    async def call_send_buffers(
        self,
        client: UCXClient,
        local_buffers: list,
        meta_message: _MessageBase,
        wait: bool = True,
    ) -> ResultMessage | ErrorMessage | asyncio.Future:
        loop = asyncio.get_running_loop()
        wait_response = loop.create_future()
        self._client_to_message_futures[client][meta_message.message_id] = wait_response

        with Timer() as timer:
            try:
                await client.send_buffers(local_buffers, meta_message)
            except ConnectionError:  # pragma: no cover
                try:
                    await client.close()
                except ConnectionError:
                    # close failed, ignore it
                    pass
                raise ServerClosed(f"Remote server {client.dest_address} closed")

            if not wait:  # pragma: no cover
                r = wait_response
            else:
                r = await wait_response

        self._profiling_data.collect_actor_call(meta_message, timer.duration)
        return r

    async def call(
        self,
        router: Router,
        dest_address: str,
        message: _MessageBase,
        wait: bool = True,
        proxy_addresses: list[str] | None = None,
    ) -> ResultMessage | ErrorMessage | asyncio.Future:
        client = await self.get_client(
            router, dest_address, proxy_addresses=proxy_addresses
        )
        if (
            client.channel_type == ChannelType.remote
            and client.dest_address != dest_address
            and message.message_type != MessageType.control
        ):
            # wrap message with forward message
            message = ForwardMessage(
                message_id=message.message_id, address=dest_address, raw_message=message
            )
        return await self.call_with_client(client, message, wait)

    async def stop(self):
        try:
            await asyncio.gather(*[client.close() for client in self._clients])
        except (ConnectionError, ServerClosed):
            pass
        try:
            self.cancel_tasks()
        except:
            pass

    def cancel_tasks(self):
        # cancel listening for all clients
        _ = [task.cancel() for task in self._clients.values()]


def _cancel_all_tasks(loop):
    to_cancel = asyncio.all_tasks(loop)
    if not to_cancel:
        return

    for task in to_cancel:
        task.cancel()

    loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))

    for task in to_cancel:
        if task.cancelled():
            continue
        if task.exception() is not None:
            loop.call_exception_handler(
                {
                    "message": "unhandled exception during asyncio.run() shutdown",
                    "exception": task.exception(),
                    "task": task,
                }
            )


def _safe_run_forever(loop):
    try:
        loop.run_forever()
    finally:
        _cancel_all_tasks(loop)
        loop.stop()


class ActorCaller:
    __slots__ = "_thread_local"

    class _RefHolder:
        pass

    _close_loop = asyncio.new_event_loop()
    _close_thread = threading.Thread(
        target=_safe_run_forever, args=(_close_loop,), daemon=True
    )
    _close_thread.start()
    atexit.register(_close_loop.call_soon_threadsafe, _close_loop.stop)

    def __init__(self):
        self._thread_local = threading.local()

    def __getattr__(self, item):
        try:
            actor_caller = self._thread_local.actor_caller
        except AttributeError:
            thread_info = str(threading.current_thread())
            logger.debug("Creating a new actor caller for thread: %s", thread_info)
            actor_caller = self._thread_local.actor_caller = ActorCallerThreadLocal()
            ref = self._thread_local.ref = ActorCaller._RefHolder()
            # If the thread exit, we clean the related actor callers and channels.

            def _cleanup():
                asyncio.run_coroutine_threadsafe(actor_caller.stop(), self._close_loop)
                logger.debug(
                    "Clean up the actor caller due to thread exit: %s", thread_info
                )

            weakref.finalize(ref, _cleanup)

        return getattr(actor_caller, item)
