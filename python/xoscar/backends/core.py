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
import copy
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple, Type, Union

from .._utils import Timer
from ..constants import XOSCAR_IDLE_TIMEOUT
from ..errors import ServerClosed
from ..profiling import get_profiling_data
from .communication import Client, DummyClient, UCXClient
from .message import DeserializeMessageFailed, ErrorMessage, ResultMessage, _MessageBase
from .router import Router

ResultMessageType = Union[ResultMessage, ErrorMessage]
logger = logging.getLogger(__name__)


class ActorCallerThreaded:
    """
    Just a proxy class to ActorCaller.
    Each thread as its own ActorCaller in case in multi threaded env.
    NOTE that it does not cleanup when thread exit
    """

    def __init__(self):
        self.local = threading.local()

    def _get_local(self) -> ActorCaller:
        try:
            return self.local.caller
        except AttributeError:
            caller = ActorCaller()
            self.local.caller = caller
            return caller

    async def get_copy_to_client(self, address: str) -> CallerClient:
        caller = self._get_local()
        return await caller.get_copy_to_client(address)

    async def call_with_client(
        self, client: CallerClient, message: _MessageBase, wait: bool = True
    ) -> ResultMessage | ErrorMessage | asyncio.Future:
        """
        Althoght we've already wrapped CallerClient in get_client(),
        might directly call from api (not recommended), Compatible with old usage.
        """
        caller = self._get_local()
        return await caller.call_with_client(client, message, wait)

    async def call_send_buffers(
        self,
        client: CallerClient,
        local_buffers: List,
        meta_message: _MessageBase,
        wait: bool = True,
    ) -> ResultMessage | ErrorMessage | asyncio.Future:
        """
        Althoght we've already wrapped CallerClient in get_client(),
        might directly call from api (not recommended), Compatible with old usage.
        """
        caller = self._get_local()
        return await caller.call_send_buffers(client, local_buffers, meta_message, wait)

    async def call(
        self,
        router: Router,
        dest_address: str,
        message: _MessageBase,
        wait: bool = True,
    ) -> ResultMessage | ErrorMessage | asyncio.Future:
        caller = self._get_local()
        return await caller.call(router, dest_address, message, wait)

    async def stop(self):
        local_caller = self._get_local()
        await local_caller.stop()

    def stop_nonblock(self):
        local_caller = self._get_local()
        local_caller.stop_nonblock()


class CallerClient:
    """
    A proxy class for under layer client, keep track for its ref_count.
    """

    _inner: Client
    _client_to_message_futures: dict[bytes, asyncio.Future]
    _ref_count: int
    _last_used: float
    _listen_task: Optional[asyncio.Task]
    _dest_address: str

    def __init__(self, client: Client, dest_address: str):
        self._inner = client
        self._ref_count = 0
        self._last_used = 0
        self._dest_address = dest_address
        self._listen_task = None
        self._client_to_message_futures = dict()

    def start_receiver(self):
        self._listen_task = asyncio.create_task(self._listen())

    def __repr__(self) -> str:
        return self._inner.__repr__()

    def abort(self):
        if self._listen_task is None:
            return
        try:
            self._listen_task.cancel()
        except:
            pass
        self._listen_task = None
        # Since listen task is aborted, need someone to cleanup
        self._cleanup(ServerClosed("Connection abort"))

    async def send(
        self,
        message: _MessageBase,
        wait_response: asyncio.Future,
        local_buffers: Optional[list] = None,
    ):
        self.add_ref()
        self._client_to_message_futures[message.message_id] = wait_response
        try:
            if local_buffers is None:
                await self._inner.send(message)
            else:
                assert isinstance(self._inner, UCXClient)
                await self._inner.send_buffers(local_buffers, message)
            self._last_used = time.time()
        except ConnectionError:
            try:
                # listen task will be notify by connection to exit and cleanup
                await self._inner.close()
            except ConnectionError:
                # close failed, ignore it
                pass
            raise ServerClosed(f"{self} closed")
        except:
            try:
                # listen task will be notify by connection to exit and cleanup
                await self._inner.close()
            except ConnectionError:
                # close failed, ignore it
                pass
            raise

    async def close(self):
        """
        Close connection.
        """
        self.abort()
        if not self.closed:
            await self._inner.close()

    @property
    def closed(self) -> bool:
        return self._inner.closed

    def _cleanup(self, e):
        message_futures = self._client_to_message_futures
        self._client_to_message_futures = dict()
        if e is None:
            e = ServerClosed(f"Remote server {self._inner.dest_address} closed")
        for future in message_futures.values():
            future.set_exception(copy.copy(e))

    async def _listen(self):
        client = self._inner
        while not client.closed:
            try:
                try:
                    message: _MessageBase = await client.recv()
                    self._last_used = time.time()
                except (EOFError, ConnectionError, BrokenPipeError) as e:
                    logger.debug(f"{client.dest_address} close due to {e}")
                    # remote server closed, close client and raise ServerClosed
                    try:
                        await client.close()
                    except (ConnectionError, BrokenPipeError):
                        # close failed, ignore it
                        pass
                    raise ServerClosed(
                        f"Remote server {client.dest_address} closed: {e}"
                    ) from None
                future = self._client_to_message_futures.pop(message.message_id)
                if not future.done():
                    future.set_result(message)
            except DeserializeMessageFailed as e:
                message_id = e.message_id
                future = self._client_to_message_futures.pop(message_id)
                future.set_exception(e.__cause__)  # type: ignore
            except Exception as e:  # noqa: E722  # pylint: disable=bare-except
                self._cleanup(e)
                logger.debug(f"{e}", exc_info=True)
            finally:
                # Counter part of self.add_ref() in send()
                self.de_ref()
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
        self._cleanup(None)

    def add_ref(self):
        self._ref_count += 1

    def de_ref(self):
        self._ref_count -= 1
        self._last_used = time.time()

    def get_ref(self) -> int:
        return self._ref_count

    def is_idle(self, idle_timeout: int) -> bool:
        return self.get_ref() == 0 and time.time() > idle_timeout + self._last_used


class ActorCaller:

    _clients: Dict[Client, CallerClient]
    # _addr_to_clients: A cache from dest_address to Caller, (regardless what mapping router did),
    #   if multiple ClientType only keep one
    _addr_to_clients: Dict[Tuple[str, Optional[Type[Client]]], CallerClient]
    _check_task: asyncio.Task

    def __init__(self):
        self._clients = dict()
        self._addr_to_clients = dict()
        self._profiling_data = get_profiling_data()
        self._check_task = None  #  Due to cython env If start task here will not get the shared copy of self.
        self._default_idle_timeout = int(
            os.environ.get("XOSCAR_IDLE_TIMEOUT", XOSCAR_IDLE_TIMEOUT)
        )
        self._loop = asyncio.get_running_loop()

    async def periodic_check(self):
        try:
            while True:
                router = Router.get_instance_or_empty()
                config = router.get_config()
                idle_timeout = config.get(
                    "idle_timeout",
                    self._default_idle_timeout,
                )
                await asyncio.sleep(idle_timeout)
                try_remove = []
                to_remove = []
                for client in self._clients.values():
                    if client.closed:
                        to_remove.append(client)
                    elif client.is_idle(idle_timeout):
                        try_remove.append(client)
                for client in to_remove:
                    self._force_remove_client(client)
                for client in try_remove:
                    await self._try_remove_client(client, idle_timeout)
                logger.debug("periodic_check: %d clients left", len(self._clients))

                addr_to_remove = []
                for key, client in self._addr_to_clients.items():
                    if client.closed:
                        addr_to_remove.append(key)
                for key in addr_to_remove:
                    self._addr_to_clients.pop(key, None)
        except Exception as e:
            logger.error(e, exc_info=True)

    async def _try_remove_client(self, client: CallerClient, idle_timeout):
        if client.closed:
            self._force_remove_client(client)
            logger.debug(f"Removed closed client {client}")
        elif client.is_idle(idle_timeout):
            self._force_remove_client(client)
            logger.debug(f"Removed idle client {client}")
            await client.close()

    def _force_remove_client(self, client: CallerClient):
        """
        Force removal client because is close
        """
        self._clients.pop(client._inner, None)
        client.abort()
        # althoght not necessarily dest_address is in _addr_to_clients, it's double ensurrence
        self._addr_to_clients.pop((client._dest_address, None), None)
        self._addr_to_clients.pop((client._dest_address, client._inner.__class__), None)

    def _add_client(
        self, dest_address: str, client_type: Optional[Type[Client]], client: Client
    ) -> CallerClient:
        caller_client = CallerClient(client, dest_address)
        caller_client.start_receiver()
        self._addr_to_clients[(dest_address, client_type)] = caller_client
        self._clients[caller_client._inner] = caller_client
        if self._check_task is None:
            # Delay the start of background task so that it get a ref of self
            self._check_task = asyncio.create_task(self.periodic_check())
        return caller_client

    async def get_copy_to_client(self, address: str) -> CallerClient:
        router = Router.get_instance()
        assert router is not None, "`copy_to` can only be used inside pools"
        client = await self.get_client(router, address)
        if isinstance(client._inner, DummyClient) or hasattr(
            client._inner, "send_buffers"
        ):
            return client
        client_types = router.get_all_client_types(address)
        # For inter-process communication, the ``self._caller.get_client`` interface would not look for UCX Client,
        # we still try to find UCXClient for this case.
        try:
            client_type = next(
                client_type
                for client_type in client_types
                if hasattr(client_type, "send_buffers")
            )
        except StopIteration:
            return client
        else:
            return await self.get_client_via_type(router, address, client_type)

    async def get_client_via_type(
        self, router: Router, dest_address: str, client_type: Type[Client]
    ) -> CallerClient:
        client = self._addr_to_clients.get((dest_address, client_type), None)
        if client is None or client.closed:
            _client = await router.get_client_via_type(dest_address, client_type)
            client = self._add_client(dest_address, client_type, _client)
        return client

    async def get_client(self, router: Router, dest_address: str) -> CallerClient:
        client = self._addr_to_clients.get((dest_address, None), None)
        if client is None or client.closed:
            _client = await router.get_client(dest_address)
            client = self._add_client(dest_address, None, _client)
        return client

    async def call_with_client(
        self, client: CallerClient, message: _MessageBase, wait: bool = True
    ) -> ResultMessage | ErrorMessage | asyncio.Future:
        """
        Althoght we've already wrapped CallerClient in get_client(),
        might directly call from api (not recommended), Compatible with old usage.
        """
        loop = asyncio.get_running_loop()
        wait_response = loop.create_future()
        with Timer() as timer:
            await client.send(message, wait_response)
            # NOTE: When send raise exception, we should not _force_remove_client,
            # let _listen_task exit normally on client close,
            # and set all futures in client to exception
            if not wait:
                r = wait_response
            else:
                r = await wait_response

        self._profiling_data.collect_actor_call(message, timer.duration)
        return r

    async def call_send_buffers(
        self,
        client: CallerClient,
        local_buffers: list,
        meta_message: _MessageBase,
        wait: bool = True,
    ) -> ResultMessage | ErrorMessage | asyncio.Future:
        """
        Althoght we've already wrapped CallerClient in get_client(),
        might directly call from api (not recommended), Compatible with old usage.
        """

        loop = asyncio.get_running_loop()
        wait_response = loop.create_future()
        with Timer() as timer:
            await client.send(meta_message, wait_response, local_buffers=local_buffers)
            # NOTE: When send raise exception, we should not _force_remove_client,
            # let _listen_task exit normally on client close,
            # and set all futures in client to exception
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
    ) -> ResultMessage | ErrorMessage | asyncio.Future:
        client = await self.get_client(router, dest_address)
        return await self.call_with_client(client, message, wait)

    async def stop(self):
        """Gracefully stop all client connections and background task"""
        try:
            await asyncio.gather(*[client.close() for client in self._clients])
        except (ConnectionError, ServerClosed):
            pass
        self.stop_nonblock()

    def stop_nonblock(self):
        """Clear all client without async closing, abort background task
        Use in non-async context"""
        if self._check_task is not None:
            try:
                self._check_task.cancel()
            except:
                pass
            self._check_task = None
        for client in self._clients.values():
            client.abort()
        self._clients = {}
        self._addr_to_clients = {}
