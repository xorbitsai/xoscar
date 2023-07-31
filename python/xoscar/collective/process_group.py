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
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from ..utils import is_linux
from . import xoscar_pygloo as xp
from .common import (
    RENDEZVOUS_MASTER_IP_ENV_KEY,
    RENDEZVOUS_MASTER_PORT_ENV_KEY,
    AllReduceAlgorithm,
    AllReduceAlgorithmMappingGloo,
    CollectiveReduceOp,
    ReduceOpMappingGloo,
    TypeMappingGloo,
)
from .utils import convert_data_to_np_array


class _World:
    def __init__(self):
        self._store = None
        self._device = None

    @property
    def store(self):
        return self._store

    @property
    def device(self):
        return self._device

    @store.setter  # type: ignore
    def store(self, store):
        self._store = store

    @device.setter  # type: ignore
    def device(self, device):
        self._device = device


_world = _World()


class ProcessGroup(ABC):
    class Options:
        master_ip: Optional[str] = None
        master_port: Optional[int] = None

    def __init__(
        self,
        rank: int,
        world_size: int,
        group_name: Optional[str] = None,
        pg_options: Optional[Options] = None,
    ):
        self._rank = rank
        self._world_size = world_size
        self._group_name = group_name
        self._option = pg_options

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def name(self):
        return self._group_name

    @property
    def options(self):
        return self._option

    @abstractmethod
    def allreduce(self, *args, **kwargs):
        ...

    @abstractmethod
    def reduce(self, *args, **kwargs):
        ...

    @abstractmethod
    def allgather(self, *args, **kwargs):
        ...

    @abstractmethod
    def gather(self, *args, **kwargs):
        ...

    @abstractmethod
    def scatter(self, *args, **kwargs):
        ...

    @abstractmethod
    def reduce_scatter(self, *args, **kwargs):
        ...

    @abstractmethod
    def alltoall(self, *args, **kwargs):
        ...

    @abstractmethod
    def broadcast(self, *args, **kwargs):
        ...


class ProcessGroupGloo(ProcessGroup):
    def __init__(
        self,
        ip: str,
        rank: int,
        world_size: int,
        group_name: Optional[str] = None,
        pg_options: Optional[ProcessGroup.Options] = None,
    ):
        super().__init__(rank, world_size, group_name, pg_options)
        if _world.store is None:
            master_ip = (
                pg_options.master_ip
                if pg_options is not None
                else os.environ.get(RENDEZVOUS_MASTER_IP_ENV_KEY, None)
            )
            master_port = (
                pg_options.master_port
                if pg_options is not None
                else os.environ.get(RENDEZVOUS_MASTER_PORT_ENV_KEY, None)
            )
            if master_ip is None or master_port is None:
                raise ValueError("Cannot find master ip or port for rendezvous")

            opt = xp.rendezvous.TCPStoreOptions()
            opt.port = int(master_port)
            opt.numWorkers = world_size
            opt.isServer = rank == 0

            store = xp.rendezvous.TCPStore(master_ip, opt)
            if not is_linux():
                attr = xp.transport.uv.attr(ip)  # type: ignore
                dev = xp.transport.uv.CreateDevice(attr)  # type: ignore
            else:
                attr = xp.transport.tcp.attr(ip)
                dev = xp.transport.tcp.CreateDevice(attr)  # type: ignore
            _world.store = store  # type: ignore
            _world.device = dev  # type: ignore
        else:
            store = _world.store
            dev = _world.device

        prefix_store = xp.rendezvous.PrefixStore(group_name or str(world_size), store)  # type: ignore
        context = xp.rendezvous.Context(rank, world_size)
        context.connectFullMesh(prefix_store, dev)
        self._context = context

    def reduce(
        self,
        send_data: Any,
        recv_data: Any,
        op: CollectiveReduceOp = CollectiveReduceOp.SUM,
        root: Optional[int] = 0,
        tag: Optional[int] = 0,
    ):
        send_buf = convert_data_to_np_array(send_data)
        recv_buf = convert_data_to_np_array(recv_data)
        size = send_buf.size
        dtype = send_buf.dtype
        sendptr = send_buf.ctypes.data
        recvptr = recv_buf.ctypes.data
        gloo_type = TypeMappingGloo[dtype.type]
        xp.reduce(
            self._context,
            sendptr,
            recvptr,
            size,
            gloo_type,
            ReduceOpMappingGloo[op],
            root,
            tag,
        )

    def allreduce(
        self,
        send_data: Any,
        recv_data: Any,
        op: CollectiveReduceOp = CollectiveReduceOp.SUM,
        algorithm: AllReduceAlgorithm = AllReduceAlgorithm.RING,
        tag: Optional[int] = 0,
    ):
        send_buf = convert_data_to_np_array(send_data)
        recv_buf = convert_data_to_np_array(recv_data)
        size = send_buf.size
        dtype = send_buf.dtype
        sendptr = send_buf.ctypes.data
        recvptr = recv_buf.ctypes.data
        gloo_type = TypeMappingGloo[dtype.type]
        xp.allreduce(
            self._context,
            sendptr,
            recvptr,
            size,
            gloo_type,
            ReduceOpMappingGloo[op],
            AllReduceAlgorithmMappingGloo[algorithm],
            tag,  # type: ignore
        )

    def gather(
        self,
        send_data: Any,
        recv_data: Any,
        root: Optional[int] = 0,
        tag: Optional[int] = 0,
    ):
        send_buf = convert_data_to_np_array(send_data)
        recv_buf = convert_data_to_np_array(recv_data)
        size = send_buf.size
        dtype = send_buf.dtype
        sendptr = send_buf.ctypes.data
        recvptr = recv_buf.ctypes.data
        gloo_type = TypeMappingGloo[dtype.type]
        xp.gather(self._context, sendptr, recvptr, size, gloo_type, root, tag)

    def allgather(self, send_data: Any, recv_data: Any, tag: Optional[int] = 0):
        send_buf = convert_data_to_np_array(send_data)
        recv_buf = convert_data_to_np_array(recv_data)
        size = send_buf.size
        dtype = send_buf.dtype
        sendptr = send_buf.ctypes.data
        recvptr = recv_buf.ctypes.data
        gloo_type = TypeMappingGloo[dtype.type]
        xp.allgather(self._context, sendptr, recvptr, size, gloo_type, tag)

    def scatter(
        self,
        send_data: List[Any],
        recv_data: Any,
        root: Optional[int] = 0,
        tag: Optional[int] = 0,
    ):
        send_bufs = [convert_data_to_np_array(d) for d in send_data]
        recv_buf = convert_data_to_np_array(recv_data)
        size = sum([d.size for d in send_bufs])
        dtype = recv_buf.dtype
        sendptrs = [d.ctypes.data for d in send_bufs]
        recvptr = recv_buf.ctypes.data
        gloo_type = TypeMappingGloo[dtype.type]
        xp.scatter(self._context, sendptrs, recvptr, size, gloo_type, root, tag)  # type: ignore

    def reduce_scatter(
        self,
        send_data: Any,
        recv_data: Any,
        recv_elems: List[int],
        op: CollectiveReduceOp = CollectiveReduceOp.SUM,
    ):
        send_buf = convert_data_to_np_array(send_data)
        recv_buf = convert_data_to_np_array(recv_data)
        sendptr = send_buf.ctypes.data
        recvptr = recv_buf.ctypes.data
        size = send_buf.size
        dtype = send_buf.dtype
        gloo_type = TypeMappingGloo[dtype.type]
        xp.reduce_scatter(
            self._context,
            sendptr,
            recvptr,
            size,
            recv_elems,
            gloo_type,
            ReduceOpMappingGloo[op],
        )

    def alltoall(self, send_data: Any, recv_data: Any, tag: Optional[int] = 0):
        send_buf = convert_data_to_np_array(send_data)
        recv_buf = convert_data_to_np_array(recv_data)
        size = send_buf.size
        dtype = send_buf.dtype
        sendptr = send_buf.ctypes.data
        recvptr = recv_buf.ctypes.data
        gloo_type = TypeMappingGloo[dtype.type]
        xp.all_to_all(self._context, sendptr, recvptr, size, gloo_type, tag)

    def broadcast(
        self,
        send_data: Any,
        recv_data: Any,
        root: Optional[int] = 0,
        tag: Optional[int] = 0,
    ):
        if send_data is not None:
            send_buf = convert_data_to_np_array(send_data)
            sendptr = send_buf.ctypes.data
        else:
            sendptr = None
        recv_buf = convert_data_to_np_array(recv_data)
        size = recv_buf.size
        dtype = recv_buf.dtype
        recvptr = recv_buf.ctypes.data
        gloo_type = TypeMappingGloo[dtype.type]
        xp.broadcast(
            self._context,
            recvptr if sendptr is None else sendptr,
            recvptr,
            size,
            gloo_type,
            root,
            tag,
        )
