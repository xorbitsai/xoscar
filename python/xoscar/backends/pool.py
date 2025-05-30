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
import asyncio.subprocess
import concurrent.futures as futures
import contextlib
import itertools
import logging
import multiprocessing
import os
import threading
import traceback
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Callable, Coroutine, Optional, Type, TypeVar

import psutil

from .._utils import TypeDispatcher, create_actor_ref, to_binary
from ..api import Actor
from ..core import ActorRef, BufferRef, FileObjectRef, register_local_pool
from ..debug import debug_async_timeout, record_message_trace
from ..errors import (
    ActorAlreadyExist,
    ActorNotExist,
    CannotCancelTask,
    SendMessageFailed,
    ServerClosed,
)
from ..metrics import init_metrics
from ..utils import implements, is_zero_ip, register_asyncio_task_timeout_detector
from .allocate_strategy import AddressSpecified, allocated_type
from .communication import (
    Channel,
    Server,
    UCXChannel,
    gen_local_address,
    get_server_type,
)
from .communication.errors import ChannelClosed
from .config import ActorPoolConfig
from .core import ActorCaller, ResultMessageType
from .message import (
    DEFAULT_PROTOCOL,
    ActorRefMessage,
    CancelMessage,
    ControlMessage,
    ControlMessageType,
    CreateActorMessage,
    DestroyActorMessage,
    ErrorMessage,
    ForwardMessage,
    HasActorMessage,
    MessageType,
    ResultMessage,
    SendMessage,
    TellMessage,
    _MessageBase,
    new_message_id,
)
from .router import Router

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _disable_log_temporally():
    if os.getenv("CUDA_VISIBLE_DEVICES") == "-1":
        # disable logging when CUDA_VISIBLE_DEVICES == -1
        # many logging comes from ptxcompiler may distract users
        try:
            logging.disable(level=logging.ERROR)
            yield
        finally:
            logging.disable(level=logging.NOTSET)
    else:
        yield


class _ErrorProcessor:
    def __init__(self, address: str, message_id: bytes, protocol):
        self._address = address
        self._message_id = message_id
        self._protocol = protocol
        self.result = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.result is None:
            self.result = ErrorMessage(
                self._message_id,
                self._address,
                os.getpid(),
                exc_type,
                exc_val,
                exc_tb,
                protocol=self._protocol,
            )
            return True


def _register_message_handler(pool_type: Type["AbstractActorPool"]):
    pool_type._message_handler = dict()
    for message_type, handler in [
        (MessageType.create_actor, pool_type.create_actor),
        (MessageType.destroy_actor, pool_type.destroy_actor),
        (MessageType.has_actor, pool_type.has_actor),
        (MessageType.actor_ref, pool_type.actor_ref),
        (MessageType.send, pool_type.send),
        (MessageType.tell, pool_type.tell),
        (MessageType.cancel, pool_type.cancel),
        (MessageType.forward, pool_type.forward),
        (MessageType.control, pool_type.handle_control_command),
        (MessageType.copy_to_buffers, pool_type.handle_copy_to_buffers_message),
        (MessageType.copy_to_fileobjs, pool_type.handle_copy_to_fileobjs_message),
    ]:
        pool_type._message_handler[message_type] = handler  # type: ignore
    return pool_type


class AbstractActorPool(ABC):
    __slots__ = (
        "process_index",
        "label",
        "external_address",
        "internal_address",
        "env",
        "_servers",
        "_router",
        "_config",
        "_stopped",
        "_actors",
        "_caller",
        "_process_messages",
        "_asyncio_task_timeout_detector_task",
    )

    _message_handler: dict[MessageType, Callable]
    _process_messages: dict[bytes, asyncio.Future | asyncio.Task | None]

    def __init__(
        self,
        process_index: int,
        label: str,
        external_address: str,
        internal_address: str,
        env: dict,
        router: Router,
        config: ActorPoolConfig,
        servers: list[Server],
    ):
        # register local pool for local actor lookup.
        # The pool is weakrefed, so we don't need to unregister it.
        if not is_zero_ip(external_address):
            # Only register_local_pool when we listen on non-zero ip (because all-zero ip is wildcard address),
            # avoid mistaken with another remote service listen on non-zero ip with the same port.
            register_local_pool(external_address, self)
        self.process_index = process_index
        self.label = label
        self.external_address = external_address
        self.internal_address = internal_address
        self.env = env
        self._router = router
        self._config = config
        self._servers = servers

        self._stopped = asyncio.Event()

        # states
        # actor id -> actor
        self._actors: dict[bytes, Actor] = dict()
        # message id -> future
        self._process_messages = dict()

        # manage async actor callers
        self._caller = ActorCaller()
        self._asyncio_task_timeout_detector_task = (
            register_asyncio_task_timeout_detector()
        )
        # init metrics
        metric_configs = self._config.get_metric_configs()
        metric_backend = metric_configs.get("backend")
        init_metrics(metric_backend, config=metric_configs.get(metric_backend))

    @property
    def router(self):
        return self._router

    @abstractmethod
    async def create_actor(self, message: CreateActorMessage) -> ResultMessageType:
        """
        Create an actor.

        Parameters
        ----------
        message: CreateActorMessage
            message to create an actor.

        Returns
        -------
        result_message
            result or error message.
        """

    @abstractmethod
    async def has_actor(self, message: HasActorMessage) -> ResultMessage:
        """
        Check if an actor exists or not.

        Parameters
        ----------
        message: HasActorMessage
            message

        Returns
        -------
        result_message
            result message contains if an actor exists or not.
        """

    @abstractmethod
    async def destroy_actor(self, message: DestroyActorMessage) -> ResultMessageType:
        """
        Destroy an actor.

        Parameters
        ----------
        message: DestroyActorMessage
            message to destroy an actor.

        Returns
        -------
        result_message
            result or error message.
        """

    @abstractmethod
    async def actor_ref(self, message: ActorRefMessage) -> ResultMessageType:
        """
        Get an actor's ref.

        Parameters
        ----------
        message: ActorRefMessage
            message to get an actor's ref.

        Returns
        -------
        result_message
            result or error message.
        """

    @abstractmethod
    async def send(self, message: SendMessage) -> ResultMessageType:
        """
        Send a message to some actor.

        Parameters
        ----------
        message: SendMessage
            Message to send.

        Returns
        -------
        result_message
            result or error message.
        """

    @abstractmethod
    async def tell(self, message: TellMessage) -> ResultMessageType:
        """
        Tell message to some actor.

        Parameters
        ----------
        message: TellMessage
            Message to tell.

        Returns
        -------
        result_message
            result or error message.
        """

    @abstractmethod
    async def cancel(self, message: CancelMessage) -> ResultMessageType:
        """
        Cancel message that sent

        Parameters
        ----------
        message: CancelMessage
            Cancel message.

        Returns
        -------
        result_message
            result or error message
        """

    async def forward(self, message: ForwardMessage) -> ResultMessageType:
        """
        Forward message

        Parameters
        ----------
        message: ForwardMessage
            Forward message.

        Returns
        -------
        result_message
            result or error message
        """
        return await self.call(message.address, message.raw_message)

    def _sync_pool_config(self, actor_pool_config: ActorPoolConfig):
        self._config = actor_pool_config
        # remove router from global one
        global_router = Router.get_instance()
        global_router.remove_router(self._router)  # type: ignore
        # update router
        self._router.set_mapping(actor_pool_config.external_to_internal_address_map)
        # update global router
        global_router.add_router(self._router)  # type: ignore

    async def handle_control_command(
        self, message: ControlMessage
    ) -> ResultMessageType:
        """
        Handle control command.

        Parameters
        ----------
        message: ControlMessage
            Control message.

        Returns
        -------
        result_message
            result or error message.
        """
        with _ErrorProcessor(
            self.external_address, message.message_id, protocol=message.protocol
        ) as processor:
            content: bool | ActorPoolConfig = True
            if message.control_message_type == ControlMessageType.stop:
                await self.stop()
            elif message.control_message_type == ControlMessageType.sync_config:
                self._sync_pool_config(message.content)
            elif message.control_message_type == ControlMessageType.get_config:
                if message.content == "main_pool_address":
                    main_process_index = self._config.get_process_indexes()[0]
                    content = self._config.get_pool_config(main_process_index)[
                        "external_address"
                    ][0]
                else:
                    content = self._config
            else:  # pragma: no cover
                raise TypeError(
                    f"Unable to handle control message "
                    f"with type {message.control_message_type}"
                )
            processor.result = ResultMessage(
                message.message_id, content, protocol=message.protocol
            )

        return processor.result

    async def _run_coro(self, message_id: bytes, coro: Coroutine):
        self._process_messages[message_id] = asyncio.tasks.current_task()
        try:
            return await coro
        finally:
            self._process_messages.pop(message_id, None)

    async def _send_channel(
        self, result: _MessageBase, channel: Channel, resend_failure: bool = True
    ):
        try:
            await channel.send(result)
        except (ChannelClosed, ConnectionResetError):
            if not self._stopped.is_set() and not channel.closed:
                raise
        except Exception as ex:
            logger.exception(
                "Error when sending message %s from %s to %s",
                result.message_id.hex(),
                channel.local_address,
                channel.dest_address,
            )
            if not resend_failure:  # pragma: no cover
                raise

            with _ErrorProcessor(
                self.external_address, result.message_id, result.protocol
            ) as processor:
                error_msg = (
                    f"Error when sending message {result.message_id.hex()}. "
                    f"Caused by {ex!r}. "
                )
                if isinstance(result, ErrorMessage):
                    format_tb = "\n".join(traceback.format_tb(result.traceback))
                    error_msg += (
                        f"\nOriginal error: {result.error!r}"
                        f"Traceback: \n{format_tb}"
                    )
                else:
                    error_msg += "See server logs for more details"
                raise SendMessageFailed(error_msg) from None
            await self._send_channel(processor.result, channel, resend_failure=False)

    async def process_message(self, message: _MessageBase, channel: Channel):
        handler = self._message_handler[message.message_type]
        with _ErrorProcessor(
            self.external_address, message.message_id, message.protocol
        ) as processor:
            # use `%.500` to avoid print too long messages
            with debug_async_timeout(
                "process_message_timeout",
                "Process message %.500s of channel %s timeout.",
                message,
                channel,
            ):
                processor.result = await self._run_coro(
                    message.message_id, handler(self, message)
                )

        await self._send_channel(processor.result, channel)

    async def call(self, dest_address: str, message: _MessageBase) -> ResultMessageType:
        return await self._caller.call(self._router, dest_address, message)  # type: ignore

    @staticmethod
    def _parse_config(config: dict, kw: dict) -> dict:
        actor_pool_config: ActorPoolConfig = config.pop("actor_pool_config")
        kw["config"] = actor_pool_config
        kw["process_index"] = process_index = config.pop("process_index")
        curr_pool_config = actor_pool_config.get_pool_config(process_index)
        kw["label"] = curr_pool_config["label"]
        external_addresses = curr_pool_config["external_address"]
        kw["external_address"] = external_addresses[0]
        kw["internal_address"] = curr_pool_config["internal_address"]
        kw["router"] = Router(
            external_addresses,
            gen_local_address(process_index),
            actor_pool_config.external_to_internal_address_map,
            comm_config=actor_pool_config.get_comm_config(),
            proxy_config=actor_pool_config.get_proxy_config(),
        )
        kw["env"] = curr_pool_config["env"]

        if config:  # pragma: no cover
            raise TypeError(
                f"Creating pool got unexpected " f'arguments: {",".join(config)}'
            )

        return kw

    @classmethod
    @abstractmethod
    async def create(cls, config: dict) -> "AbstractActorPool":
        """
        Create an actor pool.

        Parameters
        ----------
        config: dict
            configurations.

        Returns
        -------
        actor_pool:
            Actor pool.
        """

    async def start(self):
        if self._stopped.is_set():
            raise RuntimeError("pool has been stopped, cannot start again")
        start_servers = [server.start() for server in self._servers]
        await asyncio.gather(*start_servers)

    async def join(self, timeout: float | None = None):
        wait_stopped = asyncio.create_task(self._stopped.wait())

        try:
            await asyncio.wait_for(wait_stopped, timeout=timeout)
        except (futures.TimeoutError, asyncio.TimeoutError):  # pragma: no cover
            wait_stopped.cancel()

    async def stop(self):
        try:
            # clean global router
            router = Router.get_instance()
            if router is not None:
                router.remove_router(self._router)
            stop_tasks = []
            # stop all servers
            stop_tasks.extend([server.stop() for server in self._servers])
            # stop all clients
            stop_tasks.append(self._caller.stop())
            await asyncio.gather(*stop_tasks)

            self._servers = []
            if self._asyncio_task_timeout_detector_task:  # pragma: no cover
                self._asyncio_task_timeout_detector_task.cancel()
        finally:
            self._stopped.set()

    async def handle_copy_to_buffers_message(self, message) -> ResultMessage:
        for addr, uid, start, _len, data in message.content:
            buffer = BufferRef.get_buffer(BufferRef(addr, uid))
            buffer[start : start + _len] = data
        return ResultMessage(message_id=message.message_id, result=True)

    async def handle_copy_to_fileobjs_message(self, message) -> ResultMessage:
        for addr, uid, data in message.content:
            file_obj = FileObjectRef.get_local_file_object(FileObjectRef(addr, uid))
            await file_obj.write(data)
        return ResultMessage(message_id=message.message_id, result=True)

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    async def _handle_ucx_meta_message(
        self, message: _MessageBase, channel: Channel
    ) -> bool:
        if (
            isinstance(message, ControlMessage)
            and message.message_type == MessageType.control
            and message.control_message_type == ControlMessageType.switch_to_copy_to
            and isinstance(channel, UCXChannel)
        ):
            with _ErrorProcessor(
                self.external_address, message.message_id, message.protocol
            ) as processor:
                # use `%.500` to avoid print too long messages
                with debug_async_timeout(
                    "process_message_timeout",
                    "Process message %.500s of channel %s timeout.",
                    message,
                    channel,
                ):
                    buffers = [
                        BufferRef.get_buffer(BufferRef(addr, uid))
                        for addr, uid in message.content
                    ]
                    await channel.recv_buffers(buffers)
                    processor.result = ResultMessage(
                        message_id=message.message_id, result=True
                    )
            asyncio.create_task(self._send_channel(processor.result, channel))
            return True
        return False

    async def on_new_channel(self, channel: Channel):
        try:
            while not self._stopped.is_set():
                try:
                    message = await channel.recv()
                except (EOFError, ConnectionError, BrokenPipeError) as e:
                    logger.debug(f"pool: close connection due to {e}")
                    # no data to read, check channel
                    try:
                        await channel.close()
                    except (ConnectionError, EOFError):
                        # close failed, ignore
                        pass
                    return
                if await self._handle_ucx_meta_message(message, channel):
                    continue
                asyncio.create_task(self.process_message(message, channel))
                # delete to release the reference of message
                del message
                await asyncio.sleep(0)
        finally:
            try:
                await channel.close()
            except:  # noqa: E722  # nosec  # pylint: disable=bare-except
                # ignore all error if fail to close at last
                pass

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


class ActorPoolBase(AbstractActorPool, metaclass=ABCMeta):
    __slots__ = ()

    @implements(AbstractActorPool.create_actor)
    async def create_actor(self, message: CreateActorMessage) -> ResultMessageType:
        with _ErrorProcessor(
            self.external_address, message.message_id, message.protocol
        ) as processor:
            actor_id = message.actor_id
            if actor_id in self._actors:
                raise ActorAlreadyExist(
                    f"Actor {actor_id!r} already exist, cannot create"
                )

            actor = message.actor_cls(*message.args, **message.kwargs)
            actor.uid = actor_id
            actor.address = address = self.external_address
            self._actors[actor_id] = actor
            await self._run_coro(message.message_id, actor.__post_create__())

            proxies = self._router.get_proxies(address)
            result = ActorRef(address, actor_id, proxy_addresses=proxies)
            # ensemble result message
            processor.result = ResultMessage(
                message.message_id, result, protocol=message.protocol
            )
        return processor.result

    @implements(AbstractActorPool.has_actor)
    async def has_actor(self, message: HasActorMessage) -> ResultMessage:
        result = ResultMessage(
            message.message_id,
            message.actor_ref.uid in self._actors,
            protocol=message.protocol,
        )
        return result

    @implements(AbstractActorPool.destroy_actor)
    async def destroy_actor(self, message: DestroyActorMessage) -> ResultMessageType:
        with _ErrorProcessor(
            self.external_address, message.message_id, message.protocol
        ) as processor:
            actor_id = message.actor_ref.uid
            try:
                actor = self._actors[actor_id]
            except KeyError:
                raise ActorNotExist(f"Actor {actor_id} does not exist")
            await self._run_coro(message.message_id, actor.__pre_destroy__())
            del self._actors[actor_id]

            processor.result = ResultMessage(
                message.message_id, actor_id, protocol=message.protocol
            )
        return processor.result

    @implements(AbstractActorPool.actor_ref)
    async def actor_ref(self, message: ActorRefMessage) -> ResultMessageType:
        with _ErrorProcessor(
            self.external_address, message.message_id, message.protocol
        ) as processor:
            actor_id = message.actor_ref.uid
            if actor_id not in self._actors:
                raise ActorNotExist(f"Actor {actor_id} does not exist")
            proxies = self._router.get_proxies(self.external_address)
            result = ResultMessage(
                message.message_id,
                ActorRef(self.external_address, actor_id, proxy_addresses=proxies),
                protocol=message.protocol,
            )
            processor.result = result
        return processor.result

    @implements(AbstractActorPool.send)
    async def send(self, message: SendMessage) -> ResultMessageType:
        with _ErrorProcessor(
            self.external_address, message.message_id, message.protocol
        ) as processor, record_message_trace(message):
            actor_id = message.actor_ref.uid
            if actor_id not in self._actors:
                raise ActorNotExist(f"Actor {actor_id} does not exist")
            coro = self._actors[actor_id].__on_receive__(message.content)
            result = await self._run_coro(message.message_id, coro)
            processor.result = ResultMessage(
                message.message_id,
                result,
                protocol=message.protocol,
                profiling_context=message.profiling_context,
            )
        return processor.result

    @implements(AbstractActorPool.tell)
    async def tell(self, message: TellMessage) -> ResultMessageType:
        with _ErrorProcessor(
            self.external_address, message.message_id, message.protocol
        ) as processor:
            actor_id = message.actor_ref.uid
            if actor_id not in self._actors:  # pragma: no cover
                raise ActorNotExist(f"Actor {actor_id} does not exist")
            call = self._actors[actor_id].__on_receive__(message.content)
            # asynchronously run, tell does not care about result
            asyncio.create_task(call)
            await asyncio.sleep(0)
            processor.result = ResultMessage(
                message.message_id,
                None,
                protocol=message.protocol,
                profiling_context=message.profiling_context,
            )
        return processor.result

    @implements(AbstractActorPool.cancel)
    async def cancel(self, message: CancelMessage) -> ResultMessageType:
        with _ErrorProcessor(
            self.external_address, message.message_id, message.protocol
        ) as processor:
            future = self._process_messages.get(message.cancel_message_id)
            if future is None or future.done():  # pragma: no cover
                raise CannotCancelTask(
                    "Task not exists, maybe it is done or cancelled already"
                )
            future.cancel()
            processor.result = ResultMessage(
                message.message_id, True, protocol=message.protocol
            )
        return processor.result

    @staticmethod
    def _set_global_router(router: Router):
        # be cautious about setting global router
        # for instance, multiple main pool may be created in the same process

        # get default router or create an empty one
        default_router = Router.get_instance_or_empty()
        Router.set_instance(default_router)
        # append this router to global
        default_router.add_router(router)

    @staticmethod
    def _update_stored_addresses(
        servers: list[Server],
        raw_addresses: list[str],
        actor_pool_config: ActorPoolConfig,
        kw: dict,
    ):
        process_index = kw["process_index"]
        curr_pool_config = actor_pool_config.get_pool_config(process_index)
        external_addresses = curr_pool_config["external_address"]
        external_address_set = set(external_addresses)

        kw["servers"] = servers

        new_external_addresses = [
            server.address
            for server, raw_address in zip(servers, raw_addresses)
            if raw_address in external_address_set
        ]

        if external_address_set != set(new_external_addresses):
            external_addresses = new_external_addresses
            actor_pool_config.reset_pool_external_address(
                process_index, external_addresses
            )
            external_addresses = curr_pool_config["external_address"]

            logger.debug(
                "External address of process index %s updated to %s",
                process_index,
                external_addresses[0],
            )
            if kw["internal_address"] == kw["external_address"]:
                # internal address may be the same as external address in Windows
                kw["internal_address"] = external_addresses[0]
            kw["external_address"] = external_addresses[0]

            kw["router"] = Router(
                external_addresses,
                gen_local_address(process_index),
                actor_pool_config.external_to_internal_address_map,
                comm_config=actor_pool_config.get_comm_config(),
                proxy_config=actor_pool_config.get_proxy_config(),
            )

    @classmethod
    async def _create_servers(
        cls, addresses: list[str], channel_handler: Callable, config: dict
    ):
        assert len(set(addresses)) == len(addresses)
        # create servers
        create_server_tasks = []
        for addr in addresses:
            server_type = get_server_type(addr)
            extra_config = server_type.parse_config(config)
            server_config = dict(address=addr, handle_channel=channel_handler)
            server_config.update(extra_config)
            task = asyncio.create_task(server_type.create(server_config))
            create_server_tasks.append(task)

        await asyncio.gather(*create_server_tasks)
        return [f.result() for f in create_server_tasks]

    @classmethod
    @implements(AbstractActorPool.create)
    async def create(cls, config: dict) -> "ActorPoolType":
        config = config.copy()
        kw: dict[str, Any] = dict()
        cls._parse_config(config, kw)
        process_index: int = kw["process_index"]
        actor_pool_config = kw["config"]  # type: ActorPoolConfig
        cur_pool_config = actor_pool_config.get_pool_config(process_index)
        external_addresses = cur_pool_config["external_address"]
        internal_address = kw["internal_address"]

        # import predefined modules
        modules = cur_pool_config["modules"] or []
        for mod in modules:
            __import__(mod, globals(), locals(), [])
        # make sure all lazy imports loaded
        with _disable_log_temporally():
            TypeDispatcher.reload_all_lazy_handlers()

        def handle_channel(channel):
            return pool.on_new_channel(channel)

        # create servers
        server_addresses = list(external_addresses)
        if internal_address:
            server_addresses.append(internal_address)
        server_addresses.append(gen_local_address(process_index))
        server_addresses = sorted(set(server_addresses))
        servers = await cls._create_servers(
            server_addresses, handle_channel, actor_pool_config.get_comm_config()
        )
        cls._update_stored_addresses(servers, server_addresses, actor_pool_config, kw)

        # set default router
        # actor context would be able to use exact client
        cls._set_global_router(kw["router"])

        # create pool
        pool = cls(**kw)
        return pool  # type: ignore


ActorPoolType = TypeVar("ActorPoolType", bound=AbstractActorPool)
MainActorPoolType = TypeVar("MainActorPoolType", bound="MainActorPoolBase")
SubProcessHandle = asyncio.subprocess.Process


class SubActorPoolBase(ActorPoolBase):
    __slots__ = ("_main_address", "_watch_main_pool_task")
    _watch_main_pool_task: Optional[asyncio.Task]

    def __init__(
        self,
        process_index: int,
        label: str,
        external_address: str,
        internal_address: str,
        env: dict,
        router: Router,
        config: ActorPoolConfig,
        servers: list[Server],
        main_address: str,
        main_pool_pid: Optional[int],
    ):
        super().__init__(
            process_index,
            label,
            external_address,
            internal_address,
            env,
            router,
            config,
            servers,
        )
        self._main_address = main_address
        if main_pool_pid:
            self._watch_main_pool_task = asyncio.create_task(
                self._watch_main_pool(main_pool_pid)
            )
        else:
            self._watch_main_pool_task = None

    async def _watch_main_pool(self, main_pool_pid: int):
        main_process = psutil.Process(main_pool_pid)
        while not self.stopped:
            try:
                await asyncio.to_thread(main_process.status)
                await asyncio.sleep(0.1)
                continue
            except (psutil.NoSuchProcess, ProcessLookupError, asyncio.CancelledError):
                # main pool died
                break

        if not self.stopped:
            await self.stop()

    async def notify_main_pool_to_destroy(
        self, message: DestroyActorMessage
    ):  # pragma: no cover
        await self.call(self._main_address, message)

    async def notify_main_pool_to_create(self, message: CreateActorMessage):
        reg_message = ControlMessage(
            new_message_id(),
            self.external_address,
            ControlMessageType.add_sub_pool_actor,
            (self.external_address, message.allocate_strategy, message),
        )
        await self.call(self._main_address, reg_message)

    @implements(AbstractActorPool.create_actor)
    async def create_actor(self, message: CreateActorMessage) -> ResultMessageType:
        result = await super().create_actor(message)
        if not message.from_main:
            await self.notify_main_pool_to_create(message)
        return result

    @implements(AbstractActorPool.actor_ref)
    async def actor_ref(self, message: ActorRefMessage) -> ResultMessageType:
        result = await super().actor_ref(message)
        if isinstance(result, ErrorMessage):
            # need a new message id to call main actor
            main_message = ActorRefMessage(
                new_message_id(),
                create_actor_ref(self._main_address, message.actor_ref.uid),
            )
            result = await self.call(self._main_address, main_message)
            # rewrite to message_id of the original request
            result.message_id = message.message_id
        return result

    @implements(AbstractActorPool.destroy_actor)
    async def destroy_actor(self, message: DestroyActorMessage) -> ResultMessageType:
        result = await super().destroy_actor(message)
        if isinstance(result, ResultMessage) and not message.from_main:
            # sync back to main actor pool
            await self.notify_main_pool_to_destroy(message)
        return result

    @implements(AbstractActorPool.handle_control_command)
    async def handle_control_command(
        self, message: ControlMessage
    ) -> ResultMessageType:
        if message.control_message_type == ControlMessageType.sync_config:
            self._main_address = message.address
        return await super().handle_control_command(message)

    @staticmethod
    def _parse_config(config: dict, kw: dict) -> dict:
        main_pool_pid = config.pop("main_pool_pid", None)
        kw = AbstractActorPool._parse_config(config, kw)
        pool_config: ActorPoolConfig = kw["config"]
        main_process_index = pool_config.get_process_indexes()[0]
        kw["main_address"] = pool_config.get_pool_config(main_process_index)[
            "external_address"
        ][0]
        kw["main_pool_pid"] = main_pool_pid
        return kw

    async def stop(self):
        await super().stop()
        if self._watch_main_pool_task:
            self._watch_main_pool_task.cancel()
            await self._watch_main_pool_task


class MainActorPoolBase(ActorPoolBase):
    __slots__ = (
        "_allocated_actors",
        "sub_actor_pool_manager",
        "_auto_recover",
        "_monitor_task",
        "_on_process_down",
        "_on_process_recover",
        "_recover_events",
        "_allocation_lock",
        "sub_processes",
    )

    def __init__(
        self,
        process_index: int,
        label: str,
        external_address: str,
        internal_address: str,
        env: dict,
        router: Router,
        config: ActorPoolConfig,
        servers: list[Server],
        auto_recover: str | bool = "actor",
        on_process_down: Callable[[MainActorPoolType, str], None] | None = None,
        on_process_recover: Callable[[MainActorPoolType, str], None] | None = None,
    ):
        super().__init__(
            process_index,
            label,
            external_address,
            internal_address,
            env,
            router,
            config,
            servers,
        )

        # auto recovering
        self._auto_recover = auto_recover
        self._monitor_task: Optional[asyncio.Task] = None
        self._on_process_down = on_process_down
        self._on_process_recover = on_process_recover
        self._recover_events: dict[str, asyncio.Event] = dict()

        # states
        self._allocated_actors: allocated_type = {
            addr: dict() for addr in self._config.get_external_addresses()
        }
        self._allocation_lock = threading.Lock()

        self.sub_processes: dict[str, SubProcessHandle] = dict()

    _process_index_gen = itertools.count()

    @classmethod
    def process_index_gen(cls, address):
        # make sure different processes does not share process indexes
        pid = os.getpid()
        for idx in cls._process_index_gen:
            yield pid << 16 + idx

    @property
    def _sub_processes(self):
        return self.sub_processes

    @implements(AbstractActorPool.create_actor)
    async def create_actor(self, message: CreateActorMessage) -> ResultMessageType:
        with _ErrorProcessor(
            address=self.external_address,
            message_id=message.message_id,
            protocol=message.protocol,
        ) as processor:
            allocate_strategy = message.allocate_strategy
            with self._allocation_lock:
                # get allocated address according to corresponding strategy
                address = allocate_strategy.get_allocated_address(
                    self._config, self._allocated_actors
                )
                # set placeholder to make sure this label is occupied
                self._allocated_actors[address][None] = (allocate_strategy, message)
            if address == self.external_address:
                # creating actor on main actor pool
                result = await super().create_actor(message)
                if isinstance(result, ResultMessage):
                    self._allocated_actors[self.external_address][result.result] = (
                        allocate_strategy,
                        message,
                    )
                processor.result = result
            else:
                # creating actor on sub actor pool
                # rewrite allocate strategy to AddressSpecified
                new_allocate_strategy = AddressSpecified(address)
                new_create_actor_message = CreateActorMessage(
                    message.message_id,
                    message.actor_cls,
                    message.actor_id,
                    message.args,
                    message.kwargs,
                    allocate_strategy=new_allocate_strategy,
                    from_main=True,
                    protocol=message.protocol,
                    message_trace=message.message_trace,
                )
                result = await self.call(address, new_create_actor_message)
                if isinstance(result, ResultMessage):
                    self._allocated_actors[address][result.result] = (
                        allocate_strategy,
                        new_create_actor_message,
                    )
                processor.result = result

            # revert placeholder
            self._allocated_actors[address].pop(None, None)

        return processor.result

    @implements(AbstractActorPool.has_actor)
    async def has_actor(self, message: HasActorMessage) -> ResultMessage:
        actor_ref = message.actor_ref
        # lookup allocated
        for address, item in self._allocated_actors.items():
            ref = create_actor_ref(address, to_binary(actor_ref.uid))
            if ref in item:
                return ResultMessage(
                    message.message_id, True, protocol=message.protocol
                )

        return ResultMessage(message.message_id, False, protocol=message.protocol)

    @implements(AbstractActorPool.destroy_actor)
    async def destroy_actor(self, message: DestroyActorMessage) -> ResultMessageType:
        actor_ref_message = ActorRefMessage(
            message.message_id, message.actor_ref, protocol=message.protocol
        )
        result = await self.actor_ref(actor_ref_message)
        if not isinstance(result, ResultMessage):
            return result
        real_actor_ref = result.result
        if real_actor_ref.address == self.external_address:
            result = await super().destroy_actor(message)
            if result.message_type == MessageType.error:
                return result
            del self._allocated_actors[self.external_address][real_actor_ref]
            return ResultMessage(
                message.message_id, real_actor_ref.uid, protocol=message.protocol
            )
        # remove allocated actor ref
        self._allocated_actors[real_actor_ref.address].pop(real_actor_ref, None)
        new_destroy_message = DestroyActorMessage(
            message.message_id,
            real_actor_ref,
            from_main=True,
            protocol=message.protocol,
        )
        return await self.call(real_actor_ref.address, new_destroy_message)

    @implements(AbstractActorPool.send)
    async def send(self, message: SendMessage) -> ResultMessageType:
        if message.actor_ref.uid in self._actors:
            return await super().send(message)
        actor_ref_message = ActorRefMessage(
            message.message_id, message.actor_ref, protocol=message.protocol
        )
        result = await self.actor_ref(actor_ref_message)
        if not isinstance(result, ResultMessage):
            return result
        actor_ref = result.result
        new_send_message = SendMessage(
            message.message_id,
            actor_ref,
            message.content,
            protocol=message.protocol,
            message_trace=message.message_trace,
        )
        return await self.call(actor_ref.address, new_send_message)

    @implements(AbstractActorPool.tell)
    async def tell(self, message: TellMessage) -> ResultMessageType:
        if message.actor_ref.uid in self._actors:
            return await super().tell(message)
        actor_ref_message = ActorRefMessage(
            message.message_id, message.actor_ref, protocol=message.protocol
        )
        result = await self.actor_ref(actor_ref_message)
        if not isinstance(result, ResultMessage):
            return result
        actor_ref = result.result
        new_tell_message = TellMessage(
            message.message_id,
            actor_ref,
            message.content,
            protocol=message.protocol,
            message_trace=message.message_trace,
        )
        return await self.call(actor_ref.address, new_tell_message)

    @implements(AbstractActorPool.actor_ref)
    async def actor_ref(self, message: ActorRefMessage) -> ResultMessageType:
        actor_ref = message.actor_ref
        actor_ref.uid = to_binary(actor_ref.uid)
        if actor_ref.address == self.external_address and actor_ref.uid in self._actors:
            actor_ref.proxy_addresses = self._router.get_proxies(actor_ref.address)
            return ResultMessage(
                message.message_id, actor_ref, protocol=message.protocol
            )

        # lookup allocated
        for address, item in self._allocated_actors.items():
            ref = create_actor_ref(address, actor_ref.uid)
            if ref in item:
                ref.proxy_addresses = self._router.get_proxies(ref.address)
                return ResultMessage(message.message_id, ref, protocol=message.protocol)

        with _ErrorProcessor(
            self.external_address, message.message_id, protocol=message.protocol
        ) as processor:
            raise ActorNotExist(
                f"Actor {actor_ref.uid} does not exist in {actor_ref.address}"
            )

        return processor.result

    @implements(AbstractActorPool.cancel)
    async def cancel(self, message: CancelMessage) -> ResultMessageType:
        if message.address == self.external_address:
            # local message
            return await super().cancel(message)
        # redirect to sub pool
        return await self.call(message.address, message)

    @implements(AbstractActorPool.handle_control_command)
    async def handle_control_command(
        self, message: ControlMessage
    ) -> ResultMessageType:
        with _ErrorProcessor(
            self.external_address, message.message_id, message.protocol
        ) as processor:
            if message.address == self.external_address:
                if message.control_message_type == ControlMessageType.sync_config:
                    # sync config, need to notify all sub pools
                    tasks = []
                    for addr in self.sub_processes:
                        control_message = ControlMessage(
                            new_message_id(),
                            message.address,
                            message.control_message_type,
                            message.content,
                            protocol=message.protocol,
                            message_trace=message.message_trace,
                        )
                        tasks.append(
                            asyncio.create_task(self.call(addr, control_message))
                        )
                    # call super
                    task = asyncio.create_task(super().handle_control_command(message))
                    tasks.append(task)
                    await asyncio.gather(*tasks)
                    processor.result = await task
                else:
                    processor.result = await super().handle_control_command(message)
            elif message.control_message_type == ControlMessageType.stop:
                timeout, force = (
                    message.content if message.content is not None else (None, False)
                )
                await self.stop_sub_pool(
                    message.address,
                    self.sub_processes[message.address],
                    timeout=timeout,
                    force=force,
                )
                processor.result = ResultMessage(
                    message.message_id, True, protocol=message.protocol
                )
            elif message.control_message_type == ControlMessageType.wait_pool_recovered:
                if self._auto_recover and message.address not in self._recover_events:
                    self._recover_events[message.address] = asyncio.Event()

                event = self._recover_events.get(message.address, None)
                if event is not None:
                    await event.wait()
                processor.result = ResultMessage(
                    message.message_id, True, protocol=message.protocol
                )
            elif message.control_message_type == ControlMessageType.add_sub_pool_actor:
                address, allocate_strategy, create_message = message.content
                create_message.from_main = True
                ref = create_actor_ref(address, to_binary(create_message.actor_id))
                self._allocated_actors[address][ref] = (
                    allocate_strategy,
                    create_message,
                )
                processor.result = ResultMessage(
                    message.message_id, True, protocol=message.protocol
                )
            else:
                processor.result = await self.call(message.address, message)
        return processor.result

    @staticmethod
    def _parse_config(config: dict, kw: dict) -> dict:
        kw["auto_recover"] = config.pop("auto_recover", "actor")
        kw["on_process_down"] = config.pop("on_process_down", None)
        kw["on_process_recover"] = config.pop("on_process_recover", None)
        kw = AbstractActorPool._parse_config(config, kw)
        return kw

    @classmethod
    @implements(AbstractActorPool.create)
    async def create(cls, config: dict) -> MainActorPoolType:
        config = config.copy()
        actor_pool_config: ActorPoolConfig = config.get("actor_pool_config")  # type: ignore
        if "process_index" not in config:
            config["process_index"] = actor_pool_config.get_process_indexes()[0]
        curr_process_index = config.get("process_index")
        old_config_addresses = set(actor_pool_config.get_external_addresses())

        tasks = []
        subpool_process_idxes = []
        # create sub actor pools
        n_sub_pool = actor_pool_config.n_pool - 1
        if n_sub_pool > 0:
            process_indexes = actor_pool_config.get_process_indexes()
            for process_index in process_indexes:
                if process_index == curr_process_index:
                    continue
                create_pool_task = asyncio.create_task(
                    cls.start_sub_pool(actor_pool_config, process_index)
                )
                await asyncio.sleep(0)
                # await create_pool_task
                tasks.append(create_pool_task)
                subpool_process_idxes.append(process_index)

        processes, ext_addresses = await cls.wait_sub_pools_ready(tasks)
        if ext_addresses:
            for process_index, ext_address in zip(subpool_process_idxes, ext_addresses):
                actor_pool_config.reset_pool_external_address(
                    process_index, ext_address
                )

        # create main actor pool
        pool: MainActorPoolType = await super().create(config)
        addresses = actor_pool_config.get_external_addresses()[1:]

        assert len(addresses) == len(
            processes
        ), f"addresses {addresses}, processes {processes}"
        for addr, proc in zip(addresses, processes):
            pool.attach_sub_process(addr, proc)

        new_config_addresses = set(actor_pool_config.get_external_addresses())
        if old_config_addresses != new_config_addresses:
            control_message = ControlMessage(
                message_id=new_message_id(),
                address=pool.external_address,
                control_message_type=ControlMessageType.sync_config,
                content=actor_pool_config,
            )
            await pool.handle_control_command(control_message)

        return pool

    async def start_monitor(self):
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self.monitor_sub_pools())
        return self._monitor_task

    @implements(AbstractActorPool.stop)
    async def stop(self):
        global_router = Router.get_instance()
        if global_router is not None:
            global_router.remove_router(self._router)

        # turn off auto recover to avoid errors
        self._auto_recover = False
        self._stopped.set()
        if self._monitor_task and not self._monitor_task.done():
            await self._monitor_task
            self._monitor_task = None
        await self.stop_sub_pools()
        await super().stop()

    @classmethod
    @abstractmethod
    async def start_sub_pool(
        cls,
        actor_pool_config: ActorPoolConfig,
        process_index: int,
        start_python: str | None = None,
    ):
        """Start a sub actor pool"""

    @classmethod
    @abstractmethod
    async def wait_sub_pools_ready(cls, create_pool_tasks: list[asyncio.Task]):
        """Wait all sub pools ready"""

    def attach_sub_process(self, external_address: str, process: SubProcessHandle):
        self.sub_processes[external_address] = process

    async def stop_sub_pools(self):
        to_stop_processes: dict[str, SubProcessHandle] = dict()  # type: ignore
        for address, process in self.sub_processes.items():
            if not await self.is_sub_pool_alive(process):
                continue
            to_stop_processes[address] = process

        tasks = []
        for address, process in to_stop_processes.items():
            tasks.append(self.stop_sub_pool(address, process))
        await asyncio.gather(*tasks)

    async def stop_sub_pool(
        self,
        address: str,
        process: SubProcessHandle,
        timeout: float | None = None,
        force: bool = False,
    ):
        if force:
            await self.kill_sub_pool(process, force=True)
            return

        stop_message = ControlMessage(
            new_message_id(),
            address,
            ControlMessageType.stop,
            None,
            protocol=DEFAULT_PROTOCOL,
        )
        try:
            if timeout is None:
                message = await self.call(address, stop_message)
                if isinstance(message, ErrorMessage):  # pragma: no cover
                    raise message.as_instanceof_cause()
            else:
                call = asyncio.create_task(self.call(address, stop_message))
                try:
                    await asyncio.wait_for(call, timeout)
                except (futures.TimeoutError, asyncio.TimeoutError):  # pragma: no cover
                    # timeout, just let kill to finish it
                    force = True
        except (ConnectionError, ServerClosed):  # pragma: no cover
            # process dead maybe, ignore it
            pass
        # kill process
        await self.kill_sub_pool(process, force=force)

    @abstractmethod
    async def kill_sub_pool(self, process: SubProcessHandle, force: bool = False):
        """Kill a sub actor pool"""

    @abstractmethod
    async def is_sub_pool_alive(self, process: SubProcessHandle):
        """
        Check whether sub pool process is alive
        Parameters
        ----------
        process : SubProcessHandle
            sub pool process handle
        Returns
        -------
        bool
        """

    @abstractmethod
    def recover_sub_pool(self, address):
        """Recover a sub actor pool"""

    def process_sub_pool_lost(self, address: str):
        if self._auto_recover in (False, "process"):
            # process down, when not auto_recover
            # or only recover process, remove all created actors
            self._allocated_actors[address] = dict()

    async def monitor_sub_pools(self):
        try:
            while not self._stopped.is_set():
                # Copy sub_processes to avoid changes during recover.
                for address, process in list(self.sub_processes.items()):
                    try:
                        recover_events_discovered = address in self._recover_events
                        if not await self.is_sub_pool_alive(
                            process
                        ):  # pragma: no cover
                            if self._on_process_down is not None:
                                self._on_process_down(self, address)
                            self.process_sub_pool_lost(address)
                            if self._auto_recover:
                                await self.recover_sub_pool(address)
                                if self._on_process_recover is not None:
                                    self._on_process_recover(self, address)
                        if recover_events_discovered:
                            event = self._recover_events.pop(address)
                            event.set()
                    except asyncio.CancelledError:
                        raise
                    except RuntimeError as ex:  # pragma: no cover
                        if "cannot schedule new futures" not in str(ex):
                            # to silence log when process exit, otherwise it
                            # will raise "RuntimeError: cannot schedule new futures
                            # after interpreter shutdown".
                            logger.exception("Monitor sub pool %s failed", address)
                    except Exception:
                        # log the exception instead of stop monitoring the
                        # sub pool silently.
                        logger.exception("Monitor sub pool %s failed", address)

                # check every half second
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:  # pragma: no cover
            # cancelled
            return

    @classmethod
    @abstractmethod
    def get_external_addresses(
        cls,
        address: str,
        n_process: int | None = None,
        ports: list[int] | None = None,
        schemes: list[Optional[str]] | None = None,
    ):
        """Returns external addresses for n pool processes"""

    @classmethod
    @abstractmethod
    def gen_internal_address(
        cls, process_index: int, external_address: str | None = None
    ) -> str | None:
        """Returns internal address for pool of specified process index"""


async def create_actor_pool(
    address: str,
    *,
    pool_cls: Type[MainActorPoolType] | None = None,
    n_process: int | None = None,
    labels: list[str] | None = None,
    ports: list[int] | None = None,
    envs: list[dict] | None = None,
    external_address_schemes: list[Optional[str]] | None = None,
    enable_internal_addresses: list[bool] | None = None,
    auto_recover: str | bool = "actor",
    modules: list[str] | None = None,
    suspend_sigint: bool | None = None,
    use_uvloop: str | bool = "auto",
    logging_conf: dict | None = None,
    proxy_conf: dict | None = None,
    on_process_down: Callable[[MainActorPoolType, str], None] | None = None,
    on_process_recover: Callable[[MainActorPoolType, str], None] | None = None,
    extra_conf: dict | None = None,
    **kwargs,
) -> MainActorPoolType:
    if n_process is None:
        n_process = multiprocessing.cpu_count()
    if labels and len(labels) != n_process + 1:
        raise ValueError(
            f"`labels` should be of size {n_process + 1}, got {len(labels)}"
        )
    if envs and len(envs) != n_process:
        raise ValueError(f"`envs` should be of size {n_process}, got {len(envs)}")
    if external_address_schemes and len(external_address_schemes) != n_process + 1:
        raise ValueError(
            f"`external_address_schemes` should be of size {n_process + 1}, "
            f"got {len(external_address_schemes)}"
        )
    if enable_internal_addresses and len(enable_internal_addresses) != n_process + 1:
        raise ValueError(
            f"`enable_internal_addresses` should be of size {n_process + 1}, "
            f"got {len(enable_internal_addresses)}"
        )
    elif not enable_internal_addresses:
        enable_internal_addresses = [True] * (n_process + 1)
    if auto_recover is True:
        auto_recover = "actor"
    if auto_recover not in ("actor", "process", False):
        raise ValueError(
            f'`auto_recover` should be one of "actor", "process", '
            f"True or False, got {auto_recover}"
        )
    if use_uvloop == "auto":
        try:
            import uvloop  # noqa: F401 # pylint: disable=unused-variable

            use_uvloop = True
        except ImportError:
            use_uvloop = False

    assert pool_cls is not None
    external_addresses = pool_cls.get_external_addresses(
        address, n_process=n_process, ports=ports, schemes=external_address_schemes
    )
    actor_pool_config = ActorPoolConfig()
    actor_pool_config.add_metric_configs(kwargs.get("metrics", {}))
    # add proxy config
    actor_pool_config.add_proxy_config(proxy_conf)
    # add main config
    process_index_gen = pool_cls.process_index_gen(address)
    main_process_index = next(process_index_gen)
    main_internal_address = (
        pool_cls.gen_internal_address(main_process_index, external_addresses[0])
        if enable_internal_addresses[0]
        else None
    )
    actor_pool_config.add_pool_conf(
        main_process_index,
        labels[0] if labels else None,
        main_internal_address,
        external_addresses[0],
        modules=modules,
        suspend_sigint=suspend_sigint,
        use_uvloop=use_uvloop,  # type: ignore
        logging_conf=logging_conf,
        kwargs=kwargs,
    )
    # add sub configs
    for i in range(n_process):
        sub_process_index = next(process_index_gen)
        internal_address = (
            pool_cls.gen_internal_address(sub_process_index, external_addresses[i + 1])
            if enable_internal_addresses[i + 1]
            else None
        )
        actor_pool_config.add_pool_conf(
            sub_process_index,
            labels[i + 1] if labels else None,
            internal_address,
            external_addresses[i + 1],
            env=envs[i] if envs else None,
            modules=modules,
            suspend_sigint=suspend_sigint,
            use_uvloop=use_uvloop,  # type: ignore
            logging_conf=logging_conf,
            kwargs=kwargs,
        )
    actor_pool_config.add_comm_config(extra_conf)

    pool: MainActorPoolType = await pool_cls.create(
        {
            "actor_pool_config": actor_pool_config,
            "process_index": main_process_index,
            "auto_recover": auto_recover,
            "on_process_down": on_process_down,
            "on_process_recover": on_process_recover,
        }
    )
    await pool.start()
    return pool
