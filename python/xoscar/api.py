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
from collections import defaultdict
from numbers import Number
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
from urllib.parse import urlparse

from .aio import AioFileObject
from .backend import get_backend
from .context import get_context
from .core import ActorRef, BufferRef, FileObjectRef, _Actor

if TYPE_CHECKING:
    from .backends.config import ActorPoolConfig
    from .backends.pool import MainActorPoolType


async def create_actor(
    actor_cls: Type, *args, uid=None, address=None, **kwargs
) -> ActorRef:
    # TODO: explain default values.
    """
    Create an actor.

    Parameters
    ----------
    actor_cls : Actor
        Actor class.
    args : tuple
        Positional arguments for ``actor_cls.__init__``.
    uid : identifier, default=None
        Actor identifier.
    address : str, default=None
        Address to locate the actor.
    kwargs : dict
        Keyword arguments for ``actor_cls.__init__``.

    Returns
    -------
    ActorRef
    """

    ctx = get_context()
    return await ctx.create_actor(actor_cls, *args, uid=uid, address=address, **kwargs)


async def has_actor(actor_ref: ActorRef) -> bool:
    """
    Check if the given actor exists.

    Parameters
    ----------
    actor_ref : ActorRef
        Reference to an actor.

    Returns
    -------
    bool
    """
    ctx = get_context()
    return await ctx.has_actor(actor_ref)


async def destroy_actor(actor_ref: ActorRef):
    """
    Destroy an actor by its reference.

    Parameters
    ----------
    actor_ref : ActorRef
        Reference to an actor.

    Returns
    -------
    bool
    """
    ctx = get_context()
    return await ctx.destroy_actor(actor_ref)


async def actor_ref(*args, **kwargs) -> ActorRef:
    """
    Create a reference to an actor.

    Returns
    -------
    ActorRef
    """
    # TODO: refine the argument list for better user experience.
    ctx = get_context()
    return await ctx.actor_ref(*args, **kwargs)


async def kill_actor(actor_ref: ActorRef):
    # TODO: explain the meaning of 'kill'
    """
    Forcefully kill an actor.

    It's important to note that this operation is potentially
    dangerous as it may result in the termination of other
    associated actors. Only proceed if you understand the
    potential impact on associated actors and can handle any
    resulting consequences.

    Parameters
    ----------
    actor_ref : ActorRef
        Reference to an actor.

    Returns
    -------
    bool
    """
    ctx = get_context()
    return await ctx.kill_actor(actor_ref)


async def create_actor_pool(
    address: str, n_process: int | None = None, **kwargs
) -> "MainActorPoolType":
    # TODO: explain default values.
    """
    Create an actor pool.

    Parameters
    ----------
    address: str
        Address of the actor pool.
    n_process: Optional[int], default=None
        Number of processes.
    kwargs : dict
        Other keyword arguments for the actor pool.

    Returns
    -------
    MainActorPoolType
    """
    if address is None:
        raise ValueError("address has to be provided")
    if "://" not in address:
        scheme = None
    else:
        scheme = urlparse(address).scheme or None

    return await get_backend(scheme).create_actor_pool(
        address, n_process=n_process, **kwargs
    )


def buffer_ref(address: str, buffer: Any) -> BufferRef:
    """
    Init buffer ref according address and buffer.

    Parameters
    ----------
    address
        The address of the buffer.
    buffer
        CPU / GPU buffer. Need to support for slicing and retrieving the length.

    Returns
    ----------
    BufferRef obj.
    """
    ctx = get_context()
    return ctx.buffer_ref(address, buffer)


def file_object_ref(address: str, fileobj: AioFileObject) -> FileObjectRef:
    """
    Init file object ref according to address and aio file obj.

    Parameters
    ----------
    address
        The address of the file obj.
    fileobj
        Aio file object.

    Returns
    ----------
    FileObjectRef obj.
    """
    ctx = get_context()
    return ctx.file_object_ref(address, fileobj)


async def copy_to(
    local_buffers_or_fileobjs: list,
    remote_refs: List[Union[BufferRef, FileObjectRef]],
    block_size: Optional[int] = None,
):
    """
    Copy data from local buffers to remote buffers or copy local file objects to remote file objects.

    Parameters
    ----------
    local_buffers_or_fileobjs
        Local buffers or file objects.
    remote_refs
        Remote buffer refs or file object refs.
    block_size
        Transfer block size when non-ucx
    """
    ctx = get_context()
    return await ctx.copy_to(local_buffers_or_fileobjs, remote_refs, block_size)


async def wait_actor_pool_recovered(address: str, main_pool_address: str | None = None):
    """
    Wait until the specified actor pool has recovered from failure.

    Parameters
    ----------
    address: str
        Address of the actor pool.
    main_pool_address: Optional[str], default=None
        Address of corresponding main actor pool.

    Returns
    -------
    """
    ctx = get_context()
    return await ctx.wait_actor_pool_recovered(address, main_pool_address)


async def get_pool_config(address: str) -> "ActorPoolConfig":
    """
    Get the configuration of specified actor pool.

    Parameters
    ----------
    address: str
        Address of the actor pool.

    Returns
    -------
    ActorPoolConfig
    """
    ctx = get_context()
    return await ctx.get_pool_config(address)


def setup_cluster(address_to_resources: Dict[str, Dict[str, Number]]):
    scheme_to_address_resources: defaultdict[str | None, dict] = defaultdict(dict)
    for address, resources in address_to_resources.items():
        if address is None:
            raise ValueError("address has to be provided")
        if "://" not in address:
            scheme = None
        else:
            scheme = urlparse(address).scheme or None

        scheme_to_address_resources[scheme][address] = resources
    for scheme, address_resources in scheme_to_address_resources.items():
        get_backend(scheme).get_driver_cls().setup_cluster(address_resources)


class AsyncActorMixin:
    @classmethod
    def default_uid(cls):
        return cls.__name__

    def __new__(cls, *args, **kwargs):
        try:
            return _actor_implementation[cls](*args, **kwargs)
        except KeyError:
            return super().__new__(cls, *args, **kwargs)

    async def __post_create__(self):
        """
        Method called after actor creation
        """
        return await super().__post_create__()

    async def __pre_destroy__(self):
        """
        Method called before actor destroy
        """
        return await super().__pre_destroy__()

    async def __on_receive__(self, message: Tuple[Any]):
        """
        Handle message from other actors and dispatch them to user methods

        Parameters
        ----------
        message : tuple
            Message shall be (method_name,) + args + (kwargs,)
        """
        return await super().__on_receive__(message)  # type: ignore


class Actor(AsyncActorMixin, _Actor):
    # Guard all the methods with an instance of __xoscar_lock_type__
    # Lock free if the __xoscar_lock_type__ is None.
    __xoscar_lock_type__: Optional[type[asyncio.locks.Lock]] = asyncio.locks.Lock


_actor_implementation: Dict[Type[Actor], Type[Actor]] = dict()


def register_actor_implementation(actor_cls: Type[Actor], impl_cls: Type[Actor]):
    _actor_implementation[actor_cls] = impl_cls


def unregister_actor_implementation(actor_cls: Type[Actor]):
    try:
        del _actor_implementation[actor_cls]
    except KeyError:
        pass
