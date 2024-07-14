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
import hashlib
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .. import Actor, actor_ref
from ..context import get_context
from ..utils import lazy_import
from .common import (
    COLLECTIVE_DEVICE_ID_ENV_KEY,
    INVOKE_ERROR_MESSAGE,
    RANK_ADDRESS_ENV_KEY,
    AllReduceAlgorithm,
    CollectiveReduceOp,
)
from .process_group import ProcessGroup, ProcessGroupGloo, ProcessGroupNCCL
from .utils import get_rank_address_via_env

cupy = lazy_import("cupy")


class RankActor(Actor):
    def __init__(
        self,
        rank: int,
        world: int,
        backend: str = "gloo",
        device_id: Optional[int] = None,
        pg_options: Optional[ProcessGroup.Options] = None,
    ):
        assert backend == "gloo" or (
            backend == "nccl" and cupy is not None
        ), "cupy is required when using nccl as backend."
        self._rank = rank
        self._device_id = device_id
        self._world = world
        self._backend = backend
        self.name_to_pg: Dict[str, Dict[str, "ProcessGroup"]] = defaultdict(dict)
        self._pg_options = pg_options

    @classmethod
    def default_uid(cls):
        return "RankActor"

    async def __post_create__(self):
        os.environ[RANK_ADDRESS_ENV_KEY] = self.address
        _ip = self._get_ip()
        if self._backend == "gloo":
            pg = ProcessGroupGloo(
                _ip,
                self._rank,
                self._world,
                group_name="default",
                pg_options=self._pg_options,
            )
            self.name_to_pg["gloo"]["default"] = pg
        elif self._backend == "nccl":
            pg = ProcessGroupNCCL(
                _ip,
                self._rank,
                self._device_id,
                self._world,
                pg_options=self._pg_options,
            )
            self.name_to_pg["nccl"]["default"] = pg
        else:
            raise NotImplementedError("Not impl other backends for now!")

    def process_group(self, pg_name: str) -> ProcessGroup:
        return self.name_to_pg[self._backend][pg_name]

    def rank(self) -> int:
        return self._rank

    def world(self) -> int:
        return self._world

    def device_id(self):
        return self._device_id

    def backend(self) -> str:
        return self._backend

    def _get_ip(self) -> str:
        return self.address.rsplit(":", 1)[0]

    def _process_group_name(self, ranks: List[int]) -> str:
        return hashlib.sha1(
            bytes(self._backend + "_".join(map(str, ranks)), "utf-8")
        ).hexdigest()

    def new_group(
        self,
        ranks: List[int],
        pg_options: Optional[ProcessGroup.Options] = None,
    ) -> Optional[str]:
        assert (
            len(ranks) <= self._world
        ), "``ranks`` in new_group cannot be larger than the world."
        assert all(
            [self._world > rank >= 0 for rank in ranks]
        ), "rank in ``ranks`` is illegal."
        assert len({rank for rank in ranks}) == len(
            ranks
        ), "there can be no duplicate ranks in the ``ranks``."
        if self._rank not in ranks:
            return None
        if len(ranks) == self._world:
            return "default"
        global_ranks = sorted(ranks)
        group_rank = global_ranks.index(self._rank)
        group_world = len(global_ranks)
        group_name = self._process_group_name(global_ranks)
        device_id = self._device_id
        if group_name in self.name_to_pg[self._backend]:
            return group_name
        _ip = self._get_ip()
        if self._backend == "gloo":
            pg_gloo = ProcessGroupGloo(
                _ip,
                group_rank,
                group_world,
                group_name=group_name,
                pg_options=pg_options,
            )
            self.name_to_pg[self._backend][group_name] = pg_gloo
        elif self._backend == "nccl":
            pg_nccl = ProcessGroupNCCL(
                _ip,
                group_rank,
                device_id,  # type: ignore
                group_world,
                group_name=group_name,
                pg_options=pg_options,
            )
            self.name_to_pg[self._backend][group_name] = pg_nccl
        else:
            raise NotImplementedError("Not impl other backends for now!")
        return group_name

    def reduce(
        self,
        send_data: Any,
        recv_data: Any,
        op: CollectiveReduceOp = CollectiveReduceOp.SUM,
        root: Optional[int] = 0,
        tag: Optional[int] = 0,
        pg_name: str = "default",
        stream: Optional[Any] = None,
    ):
        assert self.backend() == "nccl" or (
            self.backend() == "gloo" and stream is None
        ), "The parameter 'stream' can only be used when the backend of the group is 'nccl'"

        if self._backend == "gloo":
            self.name_to_pg[self._backend][pg_name].reduce(
                send_data, recv_data, op=op, root=root, tag=tag
            )
        else:
            self.name_to_pg[self._backend][pg_name].reduce(
                send_data,
                recv_data,
                op=op,
                root=root,
                stream=stream,
            )

    def allreduce(
        self,
        send_data: Any,
        recv_data: Any,
        op: CollectiveReduceOp = CollectiveReduceOp.SUM,
        algorithm: AllReduceAlgorithm = AllReduceAlgorithm.RING,
        tag: Optional[int] = 0,
        pg_name: str = "default",
        stream: Optional[Any] = None,
    ):
        if self._backend == "gloo":
            self.name_to_pg[self._backend][pg_name].allreduce(
                send_data, recv_data, op=op, algorithm=algorithm, tag=tag
            )
        else:
            self.name_to_pg[self._backend][pg_name].allreduce(
                send_data, recv_data, op=op, stream=stream
            )

    def gather(
        self,
        send_data: Any,
        recv_data: Any,
        root: Optional[int] = 0,
        tag: Optional[int] = 0,
        pg_name: str = "default",
        stream: Optional[Any] = None,
    ):
        assert self.backend() == "nccl" or (
            self.backend() == "gloo" and stream is None
        ), "The parameter 'stream' can only be used when the backend of the group is 'nccl'"

        if self._backend == "gloo":
            self.name_to_pg[self._backend][pg_name].gather(
                send_data, recv_data, root=root, tag=tag
            )
        else:
            self.name_to_pg[self._backend][pg_name].gather(
                send_data, recv_data, root=root, stream=stream
            )

    def allgather(
        self,
        send_data: Any,
        recv_data: Any,
        tag: Optional[int] = 0,
        pg_name: str = "default",
        stream: Optional[Any] = None,
    ):
        if self._backend == "gloo":
            self.name_to_pg[self._backend][pg_name].allgather(
                send_data, recv_data, tag=tag
            )
        else:
            self.name_to_pg[self._backend][pg_name].allgather(
                send_data, recv_data, stream=stream
            )

    def scatter(
        self,
        send_data: List[Any],
        recv_data: Any,
        root: Optional[int] = 0,
        tag: Optional[int] = 0,
        pg_name: str = "default",
        stream: Optional[Any] = None,
    ):
        assert self.backend() == "nccl" or (
            self.backend() == "gloo" and stream is None
        ), "The parameter 'stream' can only be used when the backend of the group is 'nccl'"

        if self._backend == "gloo":
            self.name_to_pg[self._backend][pg_name].scatter(
                send_data, recv_data, root=root, tag=tag
            )
        else:
            self.name_to_pg[self._backend][pg_name].scatter(
                send_data, recv_data, root=root, stream=stream
            )

    def reduce_scatter(
        self,
        send_data: Any,
        recv_data: Any,
        recv_elems: List[int],
        op: CollectiveReduceOp = CollectiveReduceOp.SUM,
        pg_name: str = "default",
        stream: Optional[Any] = None,
    ):
        assert self.backend() == "nccl" or (
            self.backend() == "gloo" and stream is None
        ), "The parameter 'stream' can only be used when the backend of the group is 'nccl'"

        if self._backend == "gloo":
            self.name_to_pg[self._backend][pg_name].reduce_scatter(
                send_data, recv_data, recv_elems, op
            )
        else:
            self.name_to_pg[self._backend][pg_name].reduce_scatter(
                send_data, recv_data, recv_elems, op, stream=stream
            )

    def alltoall(
        self,
        send_data: Any,
        recv_data: Any,
        tag: Optional[int] = 0,
        pg_name: str = "default",
        stream: Optional[Any] = None,
    ):
        assert self.backend() == "nccl" or (
            self.backend() == "gloo" and stream is None
        ), "The parameter 'stream' can only be used when the backend of the group is 'nccl'"

        if self._backend == "gloo":
            self.name_to_pg[self._backend][pg_name].alltoall(
                send_data, recv_data, tag=tag
            )
        else:
            self.name_to_pg[self._backend][pg_name].alltoall(
                send_data, recv_data, stream=stream
            )

    def broadcast(
        self,
        send_data: Any,
        recv_data: Any,
        root: Optional[int] = 0,
        tag: Optional[int] = 0,
        pg_name: str = "default",
        stream: Optional[Any] = None,
    ):
        assert self.backend() == "nccl" or (
            self.backend() == "gloo" and stream is None
        ), "The parameter 'stream' can only be used when the backend of the group is 'nccl'"

        if self._backend == "gloo":
            self.name_to_pg[self._backend][pg_name].broadcast(
                send_data, recv_data, root, tag=tag
            )
        else:
            self.name_to_pg[self._backend][pg_name].broadcast(
                send_data, recv_data, root, stream=stream
            )


async def init_process_group(
    rank: int,
    world_size: int,
    backend: str = "gloo",
    device_id: Optional[int] = None,
    address: Optional[str] = None,
):
    """
    Initializes the default distributed process group, and this will also
    initialize the distributed package.

    Args:
        rank (int): Rank of the current process (it should be a
                              number between 0 and ``world_size``-1).

        world_size (int): Number of processes participating in
                                    the job.

        backend (str optional): The backend to use. Depending on
                        build-time configurations, valid values include  ``gloo`` and
                        ``nccl``. If the backend is not provided, then  a ``gloo`` backend
                        will be created.

        device_id(int, optional): GPU ID the actor will bind, default ``None``
        If it is None and backend is gloo, it will try to get it from the environment variable COLLECTIVE_DEVICE_ID_ENV_KEY.
        If the environment variable is not set either, it will return an error.

        address(str, optional): actor address. default ``None``
    """
    env_device_id = os.environ.get(COLLECTIVE_DEVICE_ID_ENV_KEY, None)
    assert backend == "gloo" or (
        backend == "nccl"
        and (
            device_id is not None
            and device_id >= 0
            or env_device_id is not None
            and int(env_device_id) >= 0
        )
    ), "The device id should be set when using nccl as backend."
    assert backend == "gloo" or (
        backend == "nccl" and cupy is not None
    ), "cupy is required when using nccl as backend."
    address = address or os.environ.get(RANK_ADDRESS_ENV_KEY, None)
    if address is None:
        raise RuntimeError(
            "Cannot decide which process to involve in the collective communication."
        )
    ctx = get_context()
    if backend == "nccl" and device_id is None and env_device_id is not None:
        device_id = int(env_device_id)
    await ctx.create_actor(
        RankActor,
        rank,
        world_size,
        backend=backend,
        device_id=device_id,
        address=address,
        uid="RankActor",
    )


async def new_group(
    ranks: List[int],
    pg_options: Optional[ProcessGroup.Options] = None,
):
    """
    Creates a new distributed group.

    This function requires that all processes in the main group (i.e. all
    processes that are part of the distributed job) enter this function, even
    if they are not going to be members of the group. Additionally, groups
    should be created in the same order in all processes.

    Args:
        ranks (list[int]): List of ranks of group members. If ``None``, will be
            set to all ranks. Default is ``None``.

        pg_options (ProcessGroupOptions, optional): process group options
            specifying what additional options need to be passed in during
            the construction of specific process groups.

    Returns:
        A handle of distributed group that can be given to collective calls.
    """
    address = os.environ.get(RANK_ADDRESS_ENV_KEY, None)
    if address is None:
        raise RuntimeError(INVOKE_ERROR_MESSAGE)
    ref = await actor_ref(address=address, uid=f"RankActor")
    return await ref.new_group(ranks, pg_options)


async def reduce(
    send_data: Any,
    recv_data: Any,
    op: CollectiveReduceOp = CollectiveReduceOp.SUM,
    root: Optional[int] = 0,
    tag: Optional[int] = 0,
    group_name: str = "default",
    stream: Optional[Any] = None,
):
    """
    Reduces the numpy or cupy data across all machines.

    Only the process with rank ``root`` is going to receive the final result.

    Args:
        send_data (Any): Input of the collective. The function
            operates in-place.

        recv_data (Any): Output of the collective. The function
            operates in-place.

        root (int): Destination rank

        op (xoscar.collective.common.CollectiveReduceOp): One of the values from
            ``xoscar.collective.common.CollectiveReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
            Default is ``xoscar.collective.common.CollectiveReduceOp.SUM``.

        tag (int optional): Tag for this operation. Default is 0.

        group_name (str): The process group to work on. If None,
            the default process group will be used.

        stream (cupy.cuda.Stream, optional): stream handle for nccl, default is None.
    """
    address = get_rank_address_via_env(RANK_ADDRESS_ENV_KEY, INVOKE_ERROR_MESSAGE)
    ref = await actor_ref(address=address, uid=f"RankActor")
    await ref.reduce(
        send_data,
        recv_data,
        op=op,
        root=root,
        tag=tag,
        pg_name=group_name,
        stream=stream,
    )


async def allreduce(
    send_data: Any,
    recv_data: Any,
    op: CollectiveReduceOp = CollectiveReduceOp.SUM,
    tag: Optional[int] = 0,
    group_name: str = "default",
    stream: Optional[Any] = None,
):
    """
    Reduces the numpy or cupy data across all machines in such a way that all get
    the final result.

    Args:
        send_data (Any): Input of the collective. The function
            operates in-place.

        recv_data (Any): Output of the collective. The function
            operates in-place.

        op (xoscar.collective.common.CollectiveReduceOp): One of the values from
            ``xoscar.collective.common.CollectiveReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
            Default is ``xoscar.collective.common.CollectiveReduceOp.SUM``.

        algorithm (xoscar.collective.common.AllReduceAlgorithm): One of the values from
            ``xoscar.collective.common.AllReduceAlgorithm``
            enum.  Specifies an algorithm used for element-wise reductions.
            Default is ``xoscar.collective.common.AllReduceAlgorithm.RING``.

        tag (int optional): Tag for this operation. Default is 0.

        group_name (str): The process group to work on. If None,
            the default process group will be used.

        stream (cupy.cuda.Stream, optional): stream handle for nccl, default is None.
    """
    address = get_rank_address_via_env(RANK_ADDRESS_ENV_KEY, INVOKE_ERROR_MESSAGE)
    ref = await actor_ref(address=address, uid="RankActor")
    await ref.allreduce(
        send_data,
        recv_data,
        op=op,
        algorithm=AllReduceAlgorithm.RING,
        tag=tag,
        pg_name=group_name,
        stream=stream,
    )


async def gather(
    send_data: Any,
    recv_data: Any,
    root: Optional[int] = 0,
    tag: Optional[int] = 0,
    group_name: str = "default",
    stream: Optional[Any] = None,
):
    """
    Gathers a list of numpy or cupy data in a single process.

    Args:
        send_data (Any): Input data.

        recv_data (Any): Output data.

        root (int, optional): Destination rank. Default is 0.

        tag (int optional): Tag for this operation. Default is 0.

        group_name (str): The process group to work on. If None,
            the default process group will be used.

        stream (cupy.cuda.Stream, optional): stream handle for nccl, default is None.
    """
    address = get_rank_address_via_env(RANK_ADDRESS_ENV_KEY, INVOKE_ERROR_MESSAGE)
    ref = await actor_ref(address=address, uid=f"RankActor")
    await ref.gather(
        send_data,
        recv_data,
        root=root,
        tag=tag,
        pg_name=group_name,
        stream=stream,
    )


async def allgather(
    send_data: Any,
    recv_data: Any,
    tag: Optional[int] = 0,
    group_name: str = "default",
    stream: Optional[Any] = None,
):
    """
    Gathers a list of numpy or cupy data to all devices.

    Args:
        send_data (Any): Input data.

        recv_data (Any): Output data.

        tag (int optional): Tag for this operation. Default is 0.

        group_name (str): The process group to work on. If None,
            the default process group will be used.

        stream (cupy.cuda.Stream, optional): stream handle for nccl, default is None.
    """
    address = get_rank_address_via_env(RANK_ADDRESS_ENV_KEY, INVOKE_ERROR_MESSAGE)
    ref = await actor_ref(address=address, uid=f"RankActor")
    await ref.allgather(
        send_data,
        recv_data,
        tag=tag,
        pg_name=group_name,
        stream=stream,
    )


async def scatter(
    send_data: List[Any],
    recv_data: Any,
    root: Optional[int] = 0,
    tag: Optional[int] = 0,
    group_name: str = "default",
    stream: Optional[Any] = None,
):
    """
    Scatters a list of numpy or cupy data to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    recv_data.

    Args:
        send_data (List(Any)): Input data.

        recv_data (Any): Output data.

        root (int, optional): Source rank (default is 0).

        tag (int optional): Tag for this operation. Default is 0.

        group_name (str): The process group to work on. If None,
            the default process group will be used.

        stream (cupy.cuda.Stream, optional): stream handle for nccl, default is None.
    """
    address = get_rank_address_via_env(RANK_ADDRESS_ENV_KEY, INVOKE_ERROR_MESSAGE)
    ref = await actor_ref(address=address, uid=f"RankActor")
    await ref.scatter(
        send_data,
        recv_data,
        root=root,
        tag=tag,
        pg_name=group_name,
        stream=stream,
    )


async def reduce_scatter(
    send_data: Any,
    recv_data: Any,
    recv_elems: List[int],
    op: CollectiveReduceOp = CollectiveReduceOp.SUM,
    group_name: str = "default",
    stream: Optional[Any] = None,
):
    """
    Reduces, then scatters a list of numpy or cupy data to all processes in a group.

    Args:
        send_data (Any): Input data.

        recv_data (Any): Output data.

        recv_elems (List[int]): the size of recv data for each process

        op (xoscar.collective.common.CollectiveReduceOp): One of the values from
            ``xoscar.collective.common.CollectiveReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
            Default is ``xoscar.collective.common.CollectiveReduceOp.SUM``.

        group_name (str): The process group to work on. If None,
            the default process group will be used.

        stream (cupy.cuda.Stream, optional): stream handle for nccl, default is None.
    """
    address = get_rank_address_via_env(RANK_ADDRESS_ENV_KEY, INVOKE_ERROR_MESSAGE)
    ref = await actor_ref(address=address, uid=f"RankActor")
    await ref.reduce_scatter(
        send_data,
        recv_data,
        recv_elems,
        op,
        pg_name=group_name,
        stream=stream,
    )


async def alltoall(
    send_data: Any,
    recv_data: Any,
    tag: Optional[int] = 0,
    group_name: str = "default",
    stream: Optional[Any] = None,
):
    """
    Each process scatters list of numpy or cupy data to all processes in a group

    Complex tensors are supported.

    Args:
        send_data (Any): Input data.

        recv_data (Any): Output data.

        tag (int, optional): Tag for this operation. default is 0.

        group_name (str): The process group to work on. If None,
            the default process group will be used.

        stream (cupy.cuda.Stream, optional): stream handle for nccl, default is None.
    """
    address = get_rank_address_via_env(RANK_ADDRESS_ENV_KEY, INVOKE_ERROR_MESSAGE)
    ref = await actor_ref(address=address, uid=f"RankActor")
    await ref.alltoall(
        send_data,
        recv_data,
        tag=tag,
        pg_name=group_name,
        stream=stream,
    )


async def broadcast(
    send_data: Any,
    recv_data: Any,
    root: Optional[int] = 0,
    tag: Optional[int] = 0,
    group_name: str = "default",
    stream: Optional[Any] = None,
):
    """
    Broadcasts the tensor to the whole group.

    data must have the same number of elements in all processes
    participating in the collective.

    Args:
        send_data (Any): Input data.

        recv_data (Any): Output data.

        root (int, optional): Source rank. Default is 0.

        tag (int, optional): Tag for this operation. Default is 0.

        group_name (str): The process group to work on. If None,
            the default process group will be used.

        stream (cupy.cuda.Stream, optional): stream handle for nccl, default is None.
    """
    address = get_rank_address_via_env(RANK_ADDRESS_ENV_KEY, INVOKE_ERROR_MESSAGE)
    ref = await actor_ref(address=address, uid=f"RankActor")
    await ref.broadcast(
        send_data,
        recv_data,
        root,
        tag,
        pg_name=group_name,
        stream=stream,
    )
