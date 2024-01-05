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
import functools
import inspect
import logging
import threading
import uuid
from collections import defaultdict
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from urllib.parse import urlparse

from .aio import AioFileObject
from .backend import get_backend
from .context import get_context
from .core import ActorRef, BufferRef, FileObjectRef, _Actor, _StatelessActor

if TYPE_CHECKING:
    from .backends.config import ActorPoolConfig
    from .backends.pool import MainActorPoolType

logger = logging.getLogger(__name__)


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


T = TypeVar("T")


class IteratorWrapper(Generic[T]):
    def __init__(self, uid: str, actor_addr: str, actor_uid: str):
        self._uid = uid
        self._actor_addr = actor_addr
        self._actor_uid = actor_uid
        self._actor_ref = None
        self._gc_destroy = True

    async def destroy(self):
        if self._actor_ref is None:
            self._actor_ref = await actor_ref(
                address=self._actor_addr, uid=self._actor_uid
            )
        assert self._actor_ref is not None
        return await self._actor_ref.__xoscar_destroy_generator__(self._uid)

    def __del__(self):
        # It's not a good idea to spawn a new thread and join in __del__,
        # but currently it's the only way to GC the generator.
        # TODO(codingl2k1): This __del__ may hangs if the program is exiting.
        if self._gc_destroy:
            thread = threading.Thread(
                target=asyncio.run, args=(self.destroy(),), daemon=True
            )
            thread.start()
            thread.join()

    def __aiter__(self):
        return self

    def __getstate__(self):
        # Transfer gc destroy during serialization.
        state = self.__dict__.copy()
        state["_gc_destroy"] = True
        self._gc_destroy = False
        return state

    async def __anext__(self) -> T:
        if self._actor_ref is None:
            self._actor_ref = await actor_ref(
                address=self._actor_addr, uid=self._actor_uid
            )
        try:
            assert self._actor_ref is not None
            return await self._actor_ref.__xoscar_next__(self._uid)
        except Exception as e:
            if "StopIteration" in str(e):
                raise StopAsyncIteration
            else:
                raise


class AsyncActorMixin:
    @classmethod
    def default_uid(cls):
        return cls.__name__

    def __new__(cls, *args, **kwargs):
        try:
            return _actor_implementation[cls](*args, **kwargs)
        except KeyError:
            return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._generators: Dict[str, IteratorWrapper] = {}

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

    async def __xoscar_next__(self, generator_uid: str) -> Any:
        """
        Iter the next of generator.

        Parameters
        ----------
        generator_uid: str
            The uid of generator

        Returns
        -------
            The next value of generator
        """

        def _wrapper(_gen):
            try:
                return next(_gen)
            except StopIteration:
                return stop

        async def _async_wrapper(_gen):
            try:
                # anext is only available for Python >= 3.10
                return await _gen.__anext__()  # noqa: F821
            except StopAsyncIteration:
                return stop

        if gen := self._generators.get(generator_uid):
            stop = object()
            try:
                if inspect.isgenerator(gen):
                    r = await asyncio.to_thread(_wrapper, gen)
                elif inspect.isasyncgen(gen):
                    r = await asyncio.create_task(_async_wrapper(gen))
                else:
                    raise Exception(
                        f"The generator {generator_uid} should be a generator or an async generator, "
                        f"but a {type(gen)} is got."
                    )
            except Exception as e:
                logger.exception(
                    f"Destroy generator {generator_uid} due to an error encountered."
                )
                await self.__xoscar_destroy_generator__(generator_uid)
                del gen  # Avoid exception hold generator reference.
                raise e
            if r is stop:
                await self.__xoscar_destroy_generator__(generator_uid)
                del gen  # Avoid exception hold generator reference.
                raise Exception("StopIteration")
            else:
                return r
        else:
            raise RuntimeError(f"No iterator with id: {generator_uid}")

    async def __xoscar_destroy_generator__(self, generator_uid: str):
        """
        Destroy the generator.

        Parameters
        ----------
        generator_uid: str
            The uid of generator
        """
        logger.debug("Destroy generator: %s", generator_uid)
        self._generators.pop(generator_uid, None)


def generator(func):
    need_to_thread = not asyncio.iscoroutinefunction(func)

    @functools.wraps(func)
    async def _wrapper(self, *args, **kwargs):
        if need_to_thread:
            r = await asyncio.to_thread(func, self, *args, **kwargs)
        else:
            r = await func(self, *args, **kwargs)
        if inspect.isgenerator(r) or inspect.isasyncgen(r):
            gen_uid = uuid.uuid1().hex
            logger.debug("Create generator: %s", gen_uid)
            self._generators[gen_uid] = r
            return IteratorWrapper(gen_uid, self.address, self.uid)
        else:
            return r

    return _wrapper


class Actor(AsyncActorMixin, _Actor):
    pass


class StatelessActor(AsyncActorMixin, _StatelessActor):
    pass


_actor_implementation: Dict[Type[Actor], Type[Actor]] = dict()


def register_actor_implementation(actor_cls: Type[Actor], impl_cls: Type[Actor]):
    _actor_implementation[actor_cls] = impl_cls


def unregister_actor_implementation(actor_cls: Type[Actor]):
    try:
        del _actor_implementation[actor_cls]
    except KeyError:
        pass
