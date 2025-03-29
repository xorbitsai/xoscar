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
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Type, Union

from .._utils import create_actor_ref, to_binary
from ..aio import AioFileObject
from ..api import Actor
from ..context import BaseActorContext
from ..core import ActorRef, BufferRef, FileObjectRef, create_local_actor_ref
from ..debug import debug_async_timeout, detect_cycle_send
from ..errors import CannotCancelTask
from ..utils import dataslots, fix_all_zero_ip
from .allocate_strategy import AddressSpecified, AllocateStrategy
from .communication import Client, DummyClient, UCXClient
from .core import ActorCaller
from .message import (
    DEFAULT_PROTOCOL,
    ActorRefMessage,
    CancelMessage,
    ControlMessage,
    ControlMessageType,
    CopyToBuffersMessage,
    CopyToFileObjectsMessage,
    CreateActorMessage,
    DestroyActorMessage,
    ErrorMessage,
    HasActorMessage,
    ResultMessage,
    SendMessage,
    _MessageBase,
    new_message_id,
)
from .router import Router

DEFAULT_TRANSFER_BLOCK_SIZE = 4 * 1024**2


@dataslots
@dataclass
class ProfilingContext:
    task_id: str


class IndigenActorContext(BaseActorContext):
    __slots__ = ("_caller", "_lock")

    support_allocate_strategy = True

    def __init__(self, address: str | None = None):
        BaseActorContext.__init__(self, address)
        self._caller = ActorCaller()
        self._lock = asyncio.Lock()

    def __del__(self):
        self._caller.cancel_tasks()

    async def _call(
        self,
        address: str,
        message: _MessageBase,
        wait: bool = True,
        proxy_addresses: list[str] | None = None,
    ) -> Union[ResultMessage, ErrorMessage, asyncio.Future]:
        return await self._caller.call(
            Router.get_instance_or_empty(),
            address,
            message,
            wait=wait,
            proxy_addresses=proxy_addresses,
        )

    async def _call_with_client(
        self, client: Client, message: _MessageBase, wait: bool = True
    ) -> Union[ResultMessage, ErrorMessage, asyncio.Future]:
        # NOTE: used by copyto, cannot support proxy
        return await self._caller.call_with_client(client, message, wait)

    async def _call_send_buffers(
        self,
        client: UCXClient,
        local_buffers: list,
        meta_message: _MessageBase,
        wait: bool = True,
    ) -> Union[ResultMessage, ErrorMessage, asyncio.Future]:
        return await self._caller.call_send_buffers(
            client, local_buffers, meta_message, wait
        )

    @staticmethod
    def _process_result_message(message: Union[ResultMessage, ErrorMessage]):
        if isinstance(message, ResultMessage):
            return message.result
        else:
            raise message.as_instanceof_cause()

    async def _wait(self, future: asyncio.Future, address: str, message: _MessageBase):
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            try:
                await self.cancel(address, message.message_id)
            except CannotCancelTask:
                # cancel failed, already finished
                raise asyncio.CancelledError
        except:  # noqa: E722  # nosec  # pylint: disable=bare-except
            pass
        return await future

    async def create_actor(
        self,
        actor_cls: Type[Actor],
        *args,
        uid=None,
        address: str | None = None,
        **kwargs,
    ) -> ActorRef:
        router = Router.get_instance_or_empty()
        address = address or self._address or router.external_address
        allocate_strategy = kwargs.get("allocate_strategy", None)
        if isinstance(allocate_strategy, AllocateStrategy):
            allocate_strategy = kwargs.pop("allocate_strategy")
        else:
            allocate_strategy = AddressSpecified(address)
        create_actor_message = CreateActorMessage(
            new_message_id(),
            actor_cls,
            to_binary(uid),
            args,
            kwargs,
            allocate_strategy,
            protocol=DEFAULT_PROTOCOL,
        )
        future = await self._call(address, create_actor_message, wait=False)
        result = await self._wait(future, address, create_actor_message)  # type: ignore
        return self._process_result_message(result)

    async def has_actor(self, actor_ref: ActorRef) -> bool:
        message = HasActorMessage(
            new_message_id(), actor_ref, protocol=DEFAULT_PROTOCOL
        )
        future = await self._call(
            actor_ref.address,
            message,
            wait=False,
            proxy_addresses=actor_ref.proxy_addresses,
        )
        result = await self._wait(future, actor_ref.address, message)  # type: ignore
        return self._process_result_message(result)

    async def destroy_actor(self, actor_ref: ActorRef):
        message = DestroyActorMessage(
            new_message_id(), actor_ref, protocol=DEFAULT_PROTOCOL
        )
        future = await self._call(
            actor_ref.address,
            message,
            wait=False,
            proxy_addresses=actor_ref.proxy_addresses,
        )
        result = await self._wait(future, actor_ref.address, message)  # type: ignore
        return self._process_result_message(result)

    async def kill_actor(self, actor_ref: ActorRef, force: bool = True):
        # get main_pool_address
        control_message = ControlMessage(
            new_message_id(),
            actor_ref.address,
            ControlMessageType.get_config,
            "main_pool_address",
            protocol=DEFAULT_PROTOCOL,
        )
        main_address = self._process_result_message(
            await self._call(actor_ref.address, control_message, proxy_addresses=actor_ref.proxy_addresses)  # type: ignore
        )
        real_actor_ref = await self.actor_ref(actor_ref)
        if real_actor_ref.address == main_address:
            raise ValueError("Cannot kill actor on main pool")
        stop_message = ControlMessage(
            new_message_id(),
            real_actor_ref.address,
            ControlMessageType.stop,
            # default timeout (3 secs) and force
            (3.0, force),
            protocol=DEFAULT_PROTOCOL,
        )
        # stop server
        result = await self._call(
            main_address, stop_message, proxy_addresses=actor_ref.proxy_addresses
        )
        return self._process_result_message(result)  # type: ignore

    async def actor_ref(self, *args, **kwargs):
        actor_ref = create_actor_ref(*args, **kwargs)
        connect_addr = actor_ref.address
        local_actor_ref = create_local_actor_ref(actor_ref.address, actor_ref.uid)
        if local_actor_ref is not None:
            return local_actor_ref
        message = ActorRefMessage(
            new_message_id(), actor_ref, protocol=DEFAULT_PROTOCOL
        )
        future = await self._call(
            actor_ref.address,
            message,
            wait=False,
            proxy_addresses=actor_ref.proxy_addresses,
        )
        result = await self._wait(future, actor_ref.address, message)
        res = self._process_result_message(result)
        if res.address != connect_addr:
            res.address = fix_all_zero_ip(res.address, connect_addr)
        return res

    async def send(
        self,
        actor_ref: ActorRef,
        message: Tuple,
        wait_response: bool = True,
        profiling_context: ProfilingContext | None = None,
    ):
        send_message = SendMessage(
            new_message_id(),
            actor_ref,
            message,
            protocol=DEFAULT_PROTOCOL,
            profiling_context=profiling_context,
        )

        # use `%.500` to avoid print too long messages
        with debug_async_timeout(
            "actor_call_timeout",
            "Calling %.500r on %s at %s timed out",
            send_message.content,
            actor_ref.uid,
            actor_ref.address,
        ):
            detect_cycle_send(send_message, wait_response)
            future = await self._call(
                actor_ref.address,
                send_message,
                wait=False,
                proxy_addresses=actor_ref.proxy_addresses,
            )
            if wait_response:
                result = await self._wait(future, actor_ref.address, send_message)  # type: ignore
                return self._process_result_message(result)
            else:
                return future

    async def cancel(self, address: str, cancel_message_id: bytes):
        message = CancelMessage(
            new_message_id(), address, cancel_message_id, protocol=DEFAULT_PROTOCOL
        )
        result = await self._call(address, message)
        return self._process_result_message(result)  # type: ignore

    async def wait_actor_pool_recovered(
        self, address: str, main_address: str | None = None
    ):
        if main_address is None:
            # get main_pool_address
            control_message = ControlMessage(
                new_message_id(),
                address,
                ControlMessageType.get_config,
                "main_pool_address",
                protocol=DEFAULT_PROTOCOL,
            )
            main_address = self._process_result_message(
                await self._call(address, control_message)  # type: ignore
            )

        # if address is main pool, it is never recovered
        if address == main_address:
            return

        control_message = ControlMessage(
            new_message_id(),
            address,
            ControlMessageType.wait_pool_recovered,
            None,
            protocol=DEFAULT_PROTOCOL,
        )
        self._process_result_message(await self._call(main_address, control_message))  # type: ignore

    async def get_pool_config(self, address: str):
        control_message = ControlMessage(
            new_message_id(),
            address,
            ControlMessageType.get_config,
            None,
            protocol=DEFAULT_PROTOCOL,
        )
        return self._process_result_message(await self._call(address, control_message))  # type: ignore

    @staticmethod
    def _gen_switch_to_copy_to_control_message(content: Any):
        return ControlMessage(
            message_id=new_message_id(),
            control_message_type=ControlMessageType.switch_to_copy_to,
            content=content,
        )

    @staticmethod
    def _gen_copy_to_buffers_message(content: Any):
        return CopyToBuffersMessage(message_id=new_message_id(), content=content)  # type: ignore

    @staticmethod
    def _gen_copy_to_fileobjs_message(content: Any):
        return CopyToFileObjectsMessage(message_id=new_message_id(), content=content)  # type: ignore

    async def _get_copy_to_client(self, router, address) -> Client:
        client = await self._caller.get_client(router, address)
        if isinstance(client, DummyClient) or hasattr(client, "send_buffers"):
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
            return await self._caller.get_client_via_type(router, address, client_type)

    async def _get_client(self, address: str) -> Client:
        router = Router.get_instance()
        assert router is not None, "`copy_to` can only be used inside pools"
        if router.get_proxy(address):
            raise RuntimeError("Cannot run `copy_to` when enabling proxy")
        return await self._get_copy_to_client(router, address)

    async def copy_to_buffers(
        self,
        local_buffers: list,
        remote_buffer_refs: List[BufferRef],
        block_size: Optional[int] = None,
    ):
        address = remote_buffer_refs[0].address
        client = await self._get_client(address)
        if isinstance(client, UCXClient):
            message = [(buf.address, buf.uid) for buf in remote_buffer_refs]
            await self._call_send_buffers(
                client,
                local_buffers,
                self._gen_switch_to_copy_to_control_message(message),
            )
        else:
            # ``local_buffers`` will be divided into buffers of the specified block size for transmission.
            # Smaller buffers will be accumulated and sent together,
            # while larger buffers will be divided and sent.
            current_buf_size = 0
            one_block_data = []
            block_size = block_size or DEFAULT_TRANSFER_BLOCK_SIZE
            for i, (l_buf, r_buf) in enumerate(zip(local_buffers, remote_buffer_refs)):
                if current_buf_size + len(l_buf) < block_size:
                    one_block_data.append(
                        (r_buf.address, r_buf.uid, 0, len(l_buf), l_buf)
                    )
                    current_buf_size += len(l_buf)
                    continue
                last_start = 0
                while current_buf_size + len(l_buf) > block_size:
                    remain = block_size - current_buf_size
                    one_block_data.append(
                        (r_buf.address, r_buf.uid, last_start, remain, l_buf[:remain])
                    )
                    await self._call_with_client(
                        client, self._gen_copy_to_buffers_message(one_block_data)
                    )
                    one_block_data = []
                    current_buf_size = 0
                    last_start += remain
                    l_buf = l_buf[remain:]

                if len(l_buf) > 0:
                    one_block_data.append(
                        (r_buf.address, r_buf.uid, last_start, len(l_buf), l_buf)
                    )
                    current_buf_size = len(l_buf)

            if one_block_data:
                await self._call_with_client(
                    client, self._gen_copy_to_buffers_message(one_block_data)
                )

    async def copy_to_fileobjs(
        self,
        local_fileobjs: List[AioFileObject],
        remote_fileobj_refs: List[FileObjectRef],
        block_size: Optional[int] = None,
    ):
        address = remote_fileobj_refs[0].address
        client = await self._get_client(address)
        block_size = block_size or DEFAULT_TRANSFER_BLOCK_SIZE
        one_block_data = []
        current_file_size = 0
        for file_obj, remote_ref in zip(local_fileobjs, remote_fileobj_refs):
            while True:
                file_data = await file_obj.read(block_size)  # type: ignore
                if file_data:
                    one_block_data.append(
                        (remote_ref.address, remote_ref.uid, file_data)
                    )
                    current_file_size += len(file_data)
                    if current_file_size >= block_size:
                        message = self._gen_copy_to_fileobjs_message(one_block_data)
                        await self._call_with_client(client, message)
                        one_block_data.clear()
                        current_file_size = 0
                else:
                    break

        if current_file_size > 0:
            message = self._gen_copy_to_fileobjs_message(one_block_data)
            await self._call_with_client(client, message)
            one_block_data.clear()
