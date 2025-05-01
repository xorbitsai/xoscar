# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2022 Alibaba Group Holding Ltd.
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
from enum import Enum
from types import TracebackType
from typing import Any, Type

from tblib import pickling_support

from ..core cimport ActorRef, BufferRef
from ..serialization.core cimport Serializer

from ..utils import wrap_exception

from .._utils cimport new_random_id

# make sure traceback can be pickled
pickling_support.install()

cdef int _DEFAULT_PROTOCOL = 0
DEFAULT_PROTOCOL = _DEFAULT_PROTOCOL


class MessageType(Enum):
    control = 0
    result = 1
    error = 2
    create_actor = 3
    destroy_actor = 4
    has_actor = 5
    actor_ref = 6
    send = 7
    tell = 8
    cancel = 9
    copy_to_buffers = 10
    copy_to_fileobjs = 11
    forward = 12


class ControlMessageType(Enum):
    stop = 0
    restart = 1
    sync_config = 2
    get_config = 3
    wait_pool_recovered = 4
    add_sub_pool_actor = 5
    # the new channel created is for data transfer only
    switch_to_copy_to = 6


cdef class _MessageSerialItem:
    cdef:
        tuple serialized
        list subs

    def __cinit__(self, tuple serialized, list subs):
        self.serialized = serialized
        self.subs = subs


cdef class _MessageBase:
    message_type: MessageType = None

    cdef:
        public int protocol
        public bytes message_id
        public list message_trace
        public object profiling_context

    def __init__(
        self,
        bytes message_id = None,
        int protocol = _DEFAULT_PROTOCOL,
        list message_trace = None,
        object profiling_context = None,
    ):
        self.message_id = message_id
        self.protocol = protocol
        # A message can be in the scope of other messages,
        # this is mainly used for detecting deadlocks,
        # e.g. Actor `A` sent a message(id: 1) to actor `B`,
        # in the processing of `B`, it sent back a message(id: 2) to `A`,
        # deadlock happens, because `A` is still waiting for reply from `B`.
        # In this case, the `scoped_message_ids` will be [1, 2],
        # `A` will find that id:1 already exists in inbox,
        # thus deadlock detected.
        self.message_trace = message_trace
        self.profiling_context = profiling_context

    cdef _MessageSerialItem serial(self):
        return _MessageSerialItem(
            (
                self.message_type.value,
                self.message_id,
                self.protocol,
                self.message_trace,
                self.profiling_context,
            ),
            [],
        )

    cdef deserial_members(self, tuple serialized, list subs):
        self.message_id = serialized[1]
        self.protocol = serialized[2]
        self.message_trace = serialized[3]
        self.profiling_context = serialized[4]

    def __repr__(self):
        cdef list attr_reprs = []
        for attr in dir(self):
            if attr.startswith("_") or attr == "message_type":
                continue
            val = getattr(self, attr)
            if callable(val):
                continue
            attr_reprs.append(f"{attr}={val!r}")
        values = ", ".join(attr_reprs)
        return f"{type(self).__name__}({values})"


cdef class ControlMessage(_MessageBase):
    message_type = MessageType.control

    cdef:
        public str address
        public object control_message_type
        public object content

    def __init__(
        self,
        bytes message_id = None,
        str address = None,
        object control_message_type: ControlMessageType = None,
        object content: Any = None,
        int protocol = _DEFAULT_PROTOCOL,
        list message_trace = None,
    ):
        _MessageBase.__init__(
            self, message_id, protocol=protocol, message_trace=message_trace
        )
        self.address = address
        self.control_message_type = control_message_type
        self.content = content

    cdef _MessageSerialItem serial(self):
        cdef _MessageSerialItem item = _MessageBase.serial(self)
        item.serialized += (
            self.address,
            self.control_message_type,
        )
        item.subs = [self.content]
        return item

    cdef deserial_members(self, tuple serialized, list subs):
        _MessageBase.deserial_members(self, serialized, subs)
        self.address = serialized[-2]
        self.control_message_type = serialized[-1]
        self.content = subs[0]


cdef class ResultMessage(_MessageBase):
    message_type = MessageType.result

    cdef:
        public object result

    def __init__(
        self,
        bytes message_id = None,
        object result: Any = None,
        int protocol = _DEFAULT_PROTOCOL,
        list message_trace = None,
        object profiling_context = None,
    ):
        _MessageBase.__init__(
            self,
            message_id,
            protocol=protocol,
            message_trace=message_trace,
            profiling_context=profiling_context,
        )
        self.result = result

    cdef _MessageSerialItem serial(self):
        cdef _MessageSerialItem item = _MessageBase.serial(self)
        item.subs = [self.result]
        return item

    cdef deserial_members(self, tuple serialized, list subs):
        _MessageBase.deserial_members(self, serialized, subs)
        self.result = subs[0]


class _AsCauseBase:
    def __str__(self):
        return f"[address={self.address}, pid={self.pid}] {str(self.__wrapped__)}"


cdef class ErrorMessage(_MessageBase):
    message_type = MessageType.error

    cdef:
        public str address
        public long pid
        public type error_type
        public object error
        public object traceback

    def __init__(
        self,
        bytes message_id = None,
        str address: str = None,
        long pid = -1,
        type error_type: Type[BaseException] = None,
        object error: BaseException = None,
        object traceback: TracebackType = None,
        int protocol = _DEFAULT_PROTOCOL,
        list message_trace = None,
    ):
        _MessageBase.__init__(
            self, message_id, protocol=protocol, message_trace=message_trace
        )
        self.address = address
        self.pid = pid
        self.error_type = error_type
        self.error = error
        self.traceback = traceback

    def as_instanceof_cause(self):
        # Check the as_instanceof_cause is not recursive.
        #
        # e.g. actor_a.method1 will reraise the exception raised
        # from actor_b.method2. But these two actors are in the same
        # process, so we don't want to append duplicated address and pid in the
        # error message.
        if issubclass(self.error_type, _AsCauseBase):
            return self.error.with_traceback(self.traceback)

        # for being compatible with Python 3.12 `asyncio.wait_for`
        # https://github.com/python/cpython/pull/113850
        if isinstance(self.error, asyncio.CancelledError):
            return asyncio.CancelledError(f"[address={self.address}, pid={self.pid}]").with_traceback(self.traceback)

        return wrap_exception(
            self.error,
            (_AsCauseBase,),
            traceback=self.traceback,
            attr_dict=dict(address=self.address, pid=self.pid),
        )

    cdef _MessageSerialItem serial(self):
        cdef _MessageSerialItem item = _MessageBase.serial(self)
        item.serialized += (self.address, self.pid)
        item.subs = [self.error_type, self.error, self.traceback]
        return item

    cdef deserial_members(self, tuple serialized, list subs):
        _MessageBase.deserial_members(self, serialized, subs)
        self.address = serialized[-2]
        self.pid = serialized[-1]
        self.error_type = subs[0]
        self.error = subs[1]
        self.traceback = subs[2]


cdef class CreateActorMessage(_MessageBase):
    message_type = MessageType.create_actor

    cdef:
        public type actor_cls
        public bytes actor_id
        public tuple args
        public dict kwargs
        public object allocate_strategy
        public object from_main

    def __init__(
        self,
        bytes message_id = None,
        type actor_cls = None,
        bytes actor_id = None,
        tuple args = None,
        dict kwargs = None,
        object allocate_strategy = None,
        object from_main: bool = False,
        int protocol = _DEFAULT_PROTOCOL,
        list message_trace = None,
    ):
        _MessageBase.__init__(
            self, message_id, protocol=protocol, message_trace=message_trace
        )
        self.actor_cls = actor_cls
        self.actor_id = actor_id
        self.args = args
        self.kwargs = kwargs
        self.allocate_strategy = allocate_strategy
        self.from_main = from_main

    cdef _MessageSerialItem serial(self):
        cdef _MessageSerialItem item = _MessageBase.serial(self)
        item.serialized += (
            self.actor_id, self.allocate_strategy, self.from_main
        )
        item.subs = [self.actor_cls, self.args, self.kwargs]
        return item

    cdef deserial_members(self, tuple serialized, list subs):
        _MessageBase.deserial_members(self, serialized, subs)
        self.actor_id = serialized[-3]
        self.allocate_strategy = serialized[-2]
        self.from_main = serialized[-1]
        self.actor_cls = subs[0]
        self.args = subs[1]
        self.kwargs = subs[2]


cdef class DestroyActorMessage(_MessageBase):
    message_type = MessageType.destroy_actor

    cdef:
        public ActorRef actor_ref
        public object from_main

    def __init__(
        self,
        bytes message_id = None,
        ActorRef actor_ref = None,
        object from_main: bool = False,
        int protocol = _DEFAULT_PROTOCOL,
        list message_trace = None,
    ):
        _MessageBase.__init__(
            self, message_id, protocol=protocol, message_trace=message_trace
        )
        self.actor_ref = actor_ref
        self.from_main = from_main

    cdef _MessageSerialItem serial(self):
        cdef _MessageSerialItem item = _MessageBase.serial(self)
        item.serialized += (
            self.actor_ref.address, self.actor_ref.uid,
            self.actor_ref.proxy_addresses, self.from_main
        )
        return item

    cdef deserial_members(self, tuple serialized, list subs):
        _MessageBase.deserial_members(self, serialized, subs)
        self.actor_ref = ActorRef(serialized[-4], serialized[-3], serialized[-2])
        self.from_main = serialized[-1]


cdef class HasActorMessage(_MessageBase):
    message_type = MessageType.has_actor

    cdef:
        public ActorRef actor_ref

    def __init__(
        self,
        bytes message_id = None,
        ActorRef actor_ref = None,
        int protocol = _DEFAULT_PROTOCOL,
        list message_trace = None,
    ):
        _MessageBase.__init__(
            self, message_id, protocol=protocol, message_trace=message_trace
        )
        self.actor_ref = actor_ref

    cdef _MessageSerialItem serial(self):
        cdef _MessageSerialItem item = _MessageBase.serial(self)
        item.serialized += (
            self.actor_ref.address, self.actor_ref.uid, self.actor_ref.proxy_addresses
        )
        return item

    cdef deserial_members(self, tuple serialized, list subs):
        _MessageBase.deserial_members(self, serialized, subs)
        self.actor_ref = ActorRef(serialized[-3], serialized[-2], serialized[-1])


cdef class ActorRefMessage(_MessageBase):
    message_type = MessageType.actor_ref

    cdef:
        public ActorRef actor_ref

    def __init__(
        self,
        bytes message_id = None,
        ActorRef actor_ref = None,
        int protocol = _DEFAULT_PROTOCOL,
        list message_trace = None,
    ):
        _MessageBase.__init__(
            self, message_id, protocol=protocol, message_trace=message_trace
        )
        self.actor_ref = actor_ref

    cdef _MessageSerialItem serial(self):
        cdef _MessageSerialItem item = _MessageBase.serial(self)
        item.serialized += (
            self.actor_ref.address, self.actor_ref.uid, self.actor_ref.proxy_addresses
        )
        return item

    cdef deserial_members(self, tuple serialized, list subs):
        _MessageBase.deserial_members(self, serialized, subs)
        self.actor_ref = ActorRef(serialized[-3], serialized[-2], serialized[-1])


cdef class SendMessage(_MessageBase):
    message_type = MessageType.send

    cdef:
        public ActorRef actor_ref
        public object content

    def __init__(
        self,
        bytes message_id = None,
        ActorRef actor_ref = None,
        object content = None,
        int protocol = _DEFAULT_PROTOCOL,
        list message_trace = None,
        object profiling_context = None,
    ):
        _MessageBase.__init__(
            self,
            message_id,
            protocol=protocol,
            message_trace=message_trace,
            profiling_context=profiling_context,
        )
        self.actor_ref = actor_ref
        self.content = content

    cdef _MessageSerialItem serial(self):
        cdef _MessageSerialItem item = _MessageBase.serial(self)
        item.serialized += (
            self.actor_ref.address, self.actor_ref.uid, self.actor_ref.proxy_addresses
        )
        item.subs = [self.content]
        return item

    cdef deserial_members(self, tuple serialized, list subs):
        _MessageBase.deserial_members(self, serialized, subs)
        self.actor_ref = ActorRef(serialized[-3], serialized[-2], serialized[-1])
        self.content = subs[0]


cdef class TellMessage(SendMessage):
    message_type = MessageType.tell


cdef class CancelMessage(_MessageBase):
    message_type = MessageType.cancel

    cdef:
        public str address
        public bytes cancel_message_id

    def __init__(
        self,
        bytes message_id = None,
        str address = None,
        bytes cancel_message_id = None,
        int protocol = _DEFAULT_PROTOCOL,
        list message_trace = None,
    ):
        _MessageBase.__init__(
            self, message_id, protocol=protocol, message_trace=message_trace
        )
        self.address = address
        self.cancel_message_id = cancel_message_id

    cdef _MessageSerialItem serial(self):
        cdef _MessageSerialItem item = _MessageBase.serial(self)
        item.serialized += (
            self.address, self.cancel_message_id
        )
        return item

    cdef deserial_members(self, tuple serialized, list subs):
        _MessageBase.deserial_members(self, serialized, subs)
        self.address = serialized[-2]
        self.cancel_message_id = serialized[-1]


cdef class CopyToBuffersMessage(_MessageBase):
    message_type = MessageType.copy_to_buffers

    cdef:
        public object content

    def __init__(
        self,
        bytes message_id = None,
        object content = None,
        int protocol = _DEFAULT_PROTOCOL,
        list message_trace = None,
    ):
        _MessageBase.__init__(
            self,
            message_id,
            protocol=protocol,
            message_trace=message_trace
        )
        self.content = content

    cdef _MessageSerialItem serial(self):
        cdef _MessageSerialItem item = _MessageBase.serial(self)
        item.subs = [self.content]
        return item

    cdef deserial_members(self, tuple serialized, list subs):
        _MessageBase.deserial_members(self, serialized, subs)
        self.content = subs[0]


cdef class CopyToFileObjectsMessage(CopyToBuffersMessage):
    message_type = MessageType.copy_to_fileobjs


cdef class ForwardMessage(_MessageBase):
    message_type =  MessageType.forward

    cdef:
        public str address
        public _MessageBase raw_message

    def __init__(
        self,
        bytes message_id = None,
        str address = None,
        _MessageBase raw_message = None,
        int protocol = DEFAULT_PROTOCOL,
        list message_trace = None,
    ):
        _MessageBase.__init__(
            self,
            message_id,
            protocol=protocol,
            message_trace=message_trace
        )
        self.address = address
        self.raw_message = raw_message

    cdef _MessageSerialItem serial(self):
        cdef _MessageSerialItem item = _MessageBase.serial(self)
        cdef _MessageSerialItem raw_message_serialized = self.raw_message.serial()
        item.serialized += (self.address,)
        item.serialized += raw_message_serialized.serialized
        item.subs += raw_message_serialized.subs
        return item

    cdef deserial_members(self, tuple serialized, list subs):
        # 5 is magic number that means serialized for _MessageBase
        base_serialized = serialized[:5]
        _MessageBase.deserial_members(self, base_serialized, [])
        self.address = serialized[5]
        # process raw message
        tp = _message_type_to_message_cls[serialized[6]]
        cdef _MessageBase raw_message = <_MessageBase>(tp())
        raw_message.deserial_members(serialized[6:], subs)
        self.raw_message = raw_message


cdef dict _message_type_to_message_cls = {
    MessageType.control.value: ControlMessage,
    MessageType.result.value: ResultMessage,
    MessageType.error.value: ErrorMessage,
    MessageType.create_actor.value: CreateActorMessage,
    MessageType.destroy_actor.value: DestroyActorMessage,
    MessageType.has_actor.value: HasActorMessage,
    MessageType.actor_ref.value: ActorRefMessage,
    MessageType.send.value: SendMessage,
    MessageType.tell.value: TellMessage,
    MessageType.cancel.value: CancelMessage,
    MessageType.copy_to_buffers.value: CopyToBuffersMessage,
    MessageType.copy_to_fileobjs.value: CopyToFileObjectsMessage,
    MessageType.forward.value: ForwardMessage,
}


class DeserializeMessageFailed(RuntimeError):
    def __init__(self, message_id):
        self.message_id = message_id

    def __str__(self):
        return f"Deserialize {self.message_id} failed"


cdef class MessageSerializer(Serializer):
    serializer_id = 32105

    cpdef serial(self, object obj, dict context):
        cdef _MessageBase msg = <_MessageBase>obj
        cdef _MessageSerialItem ser_item

        assert msg.protocol == _DEFAULT_PROTOCOL, "only support protocol 0 for now"
        ser_item = msg.serial()
        return ser_item.serialized, ser_item.subs, False

    cpdef deserial(self, tuple serialized, dict context, list subs):
        cdef _MessageBase msg

        msg_type = serialized[0]
        msg = _message_type_to_message_cls[msg_type]()
        msg.deserial_members(serialized, subs)
        return msg

    cpdef on_deserial_error(
        self,
        tuple serialized,
        dict context,
        list subs_serialized,
        int error_index,
        object exc,
    ):
        message_id = serialized[1]  # pos of message_id field
        try:
            raise DeserializeMessageFailed(message_id) from exc
        except BaseException as new_ex:
            return new_ex


# register message serializer
MessageSerializer.register(_MessageBase)


cpdef bytes new_message_id():
    return new_random_id(32)
