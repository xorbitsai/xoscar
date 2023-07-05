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
from typing import Any, List, Optional, Union
from urllib.parse import urlparse

from ._utils cimport new_actor_id, new_random_id
from .core cimport ActorRef, BufferRef, FileObjectRef


cdef dict _backend_context_cls = dict()

cdef object _context = None


cdef class BaseActorContext:
    # allocate strategy is for Indigen backend only
    support_allocate_strategy = False

    """
    Base class for actor context. Every backend need to implement
    actor context for their own.
    """

    def __init__(self, address: str = None):
        self._address = address

    async def create_actor(
        self,
        object actor_cls,
        *args,
        object uid=None,
        object address=None,
        **kwargs,
    ):
        """
        Stub method for creating an actor in current context.

        Parameters
        ----------
        actor_cls : Actor
            Actor class
        args : tuple
            args to be passed into actor_cls.__init__
        uid : identifier
            Actor identifier
        address : str
            Address to locate the actor
        kwargs : dict
            kwargs to be passed into actor_cls.__init__

        Returns
        -------
        ActorRef

        """
        raise NotImplementedError

    async def has_actor(self, ActorRef actor_ref):
        """
        Check if actor exists in current context

        Parameters
        ----------
        actor_ref : ActorRef
            Reference to an actor

        Returns
        -------
        bool
        """
        raise NotImplementedError

    async def destroy_actor(self, ActorRef actor_ref):
        """
        Destroy an actor by its reference

        Parameters
        ----------
        actor_ref : ActorRef
            Reference to an actor

        Returns
        -------
        bool
        """
        raise NotImplementedError

    async def kill_actor(self, ActorRef actor_ref):
        """
        Force to kill an actor, take care this is a dangerous operation,
        it may lead to the result that other actors are killed as well.
        Hence, unless you are knowing what you are doing and know how
        to recover possible effected actors, DO NOT USE this method!

        Parameters
        ----------
        actor_ref : ActorRef
            Reference to an actor

        Returns
        -------
        bool
        """

    async def send(
        self,
        ActorRef actor_ref,
        object message,
        bint wait_response=True,
        object profiling_context=None,
    ):
        """
        Send a message to given actor by its reference

        Parameters
        ----------
        actor_ref : ActorRef
            Reference to an actor
        message : object
            Message to send to an actor, need to comply to Actor.__on_receive__
        wait_response : bool
            Whether to wait for responses from the actor.
        profiling_context: ProfilingContext
            The profiling context.

        Returns
        -------
        object
        """
        raise NotImplementedError

    async def actor_ref(self, *args, **kwargs):
        """
        Create a reference to an actor

        Returns
        -------
        ActorRef
        """
        raise NotImplementedError

    async def wait_actor_pool_recovered(self, str address, str main_address = None):
        """
        Wait until an actor pool is recovered

        Parameters
        ----------
        address
            address of the actor pool
        main_address
            address of the main pool
        """
        raise NotImplementedError

    async def get_pool_config(self, str address):
        """
        Get config of actor pool with given address

        Parameters
        ----------
        address
            address of the actor pool

        Returns
        -------

        """
        raise NotImplementedError

    def buffer_ref(self, str address, object buf) -> BufferRef:
        """
        Create a reference to a buffer

        Parameters
        ----------
        address
            address of the actor pool
        buf
            buffer object

        Returns
        -------
        BufferRef
        """
        return BufferRef.create(buf, address, new_random_id(32))

    def file_object_ref(self, str address, object file_object) -> FileObjectRef:
        """
        Create a reference to an aio file object

        Parameters
        ----------
        address
            address of the actor pool
        file_object
            aio file object

        Returns
        -------
        FileObjectRef
        """
        return FileObjectRef.create(file_object, address, new_random_id(32))

    async def copy_to_buffers(self, local_buffers: List, remote_buffer_refs: List[BufferRef], block_size: Optional[int] = None):
        """
        Copy local buffers to remote buffers.
        Parameters
        ----------
        local_buffers
            Local buffers.
        remote_buffer_refs
            Remote buffer refs
        block_size
            Transfer block size when non-ucx
        """
        raise NotImplementedError

    async def copy_to_fileobjs(self, local_fileobjs: list, remote_fileobj_refs: List[FileObjectRef], block_size: Optional[int] = None):
        """
        Copy local file objs to remote file objs.
        Parameters
        ----------
        local_fileobjs
            Local file objs.
        remote_fileobj_refs
            Remote file object refs
        block_size
            Transfer block size when non-ucx
        """
        raise NotImplementedError


cdef class ClientActorContext(BaseActorContext):
    """
    Default actor context. This context will keep references to other contexts
    given their protocol scheme (i.e., `ray://xxx`).
    """
    cdef dict _backend_contexts

    def __init__(self, address: str = None):
        BaseActorContext.__init__(self, address)
        self._backend_contexts = dict()

    cdef inline object _get_backend_context(self, object address):
        if address is None:
            raise ValueError('address has to be provided')
        if '://' not in address:
            scheme = None
        else:
            scheme = urlparse(address).scheme or None
        try:
            return self._backend_contexts[scheme]
        except KeyError:
            context = self._backend_contexts[scheme] = \
                _backend_context_cls[scheme](address)
            return context

    def create_actor(
        self,
        object actor_cls,
        *args,
        object uid=None,
        object address=None,
        **kwargs,
    ):
        context = self._get_backend_context(address)
        uid = uid or new_actor_id()
        return context.create_actor(actor_cls, *args, uid=uid, address=address, **kwargs)

    def has_actor(self, ActorRef actor_ref):
        context = self._get_backend_context(actor_ref.address)
        return context.has_actor(actor_ref)

    def destroy_actor(self, ActorRef actor_ref):
        context = self._get_backend_context(actor_ref.address)
        return context.destroy_actor(actor_ref)

    def kill_actor(self, ActorRef actor_ref):
        context = self._get_backend_context(actor_ref.address)
        return context.kill_actor(actor_ref)

    def actor_ref(self, *args, **kwargs):
        from ._utils import create_actor_ref

        actor_ref = create_actor_ref(*args, **kwargs)
        context = self._get_backend_context(actor_ref.address)
        return context.actor_ref(actor_ref)

    def send(
        self,
        ActorRef actor_ref,
        object message,
        bint wait_response=True,
        object profiling_context=None
    ):
        context = self._get_backend_context(actor_ref.address)
        return context.send(
            actor_ref,
            message,
            wait_response=wait_response,
            profiling_context=profiling_context,
        )

    def wait_actor_pool_recovered(self, str address, str main_address = None):
        context = self._get_backend_context(address)
        return context.wait_actor_pool_recovered(address, main_address)

    def get_pool_config(self, str address):
        context = self._get_backend_context(address)
        return context.get_pool_config(address)

    def buffer_ref(self, str address, buf: Any) -> BufferRef:
        context = self._get_backend_context(address)
        return context.buffer_ref(address, buf)

    def file_object_ref(self, str address, object file_object) -> FileObjectRef:
        context = self._get_backend_context(address)
        return context.file_object_ref(address, file_object)

    def copy_to(self, local_buffers_or_fileobjs: list, remote_refs: List[Union[BufferRef, FileObjectRef]], block_size: Optional[int] = None):
        if len(local_buffers_or_fileobjs) == 0 or len(remote_refs) == 0:
            raise ValueError("Nothing to transfer since the length of `local_buffers_or_fileobjs` or `remote_refs` is 0.")
        assert (
            len({ref.address for ref in remote_refs}) == 1
        ), "remote_refs for `copy_to` can support only 1 destination"
        assert len(local_buffers_or_fileobjs) == len(remote_refs), (
            f"Buffers or fileobjs from local and remote must have same size, "
            f"local: {len(local_buffers_or_fileobjs)}, remote: {len(remote_refs)}"
        )
        if block_size is not None:
            assert (
                block_size > 0
            ), f"`block_size` option must be greater than 0, current value: {block_size}."
        remote_ref = remote_refs[0]
        address = remote_ref.address
        context = self._get_backend_context(address)
        if isinstance(remote_ref, BufferRef):
            return context.copy_to_buffers(local_buffers_or_fileobjs, remote_refs, block_size)
        else:
            return context.copy_to_fileobjs(local_buffers_or_fileobjs, remote_refs, block_size)


def register_backend_context(scheme, cls):
    assert issubclass(cls, BaseActorContext)
    _backend_context_cls[scheme] = cls


cpdef get_context():
    """
    Get an actor context. If not in an actor environment,
    ClientActorContext will be used
    """
    global _context
    if _context is None:
        _context = ClientActorContext()
    return _context
