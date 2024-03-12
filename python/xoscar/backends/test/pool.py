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
import multiprocessing
from typing import Any, Optional

from ..communication import DummyServer, gen_local_address
from ..config import ActorPoolConfig
from ..indigen.pool import MainActorPool, SubActorPool, SubpoolStatus
from ..message import ControlMessage, ControlMessageType, new_message_id
from ..pool import ActorPoolType


class TestMainActorPool(MainActorPool):
    @classmethod
    def get_external_addresses(
        cls,
        address: str,
        n_process: int | None = None,
        ports: list[int] | None = None,
        schemes: list[Optional[str]] | None = None,
    ):
        if "://" in address:
            address = address.split("://", 1)[1]
        return super().get_external_addresses(address, n_process=n_process, ports=ports)

    @classmethod
    def gen_internal_address(
        cls, process_index: int, external_address: str | None = None
    ) -> str:
        return f"dummy://{process_index}"

    @classmethod
    async def start_sub_pool(
        cls,
        actor_pool_config: ActorPoolConfig,
        process_index: int,
        start_method: str | None = None,
    ):
        status_queue: multiprocessing.Queue = multiprocessing.Queue()
        return (
            asyncio.create_task(
                cls._create_sub_pool(actor_pool_config, process_index, status_queue, 0)
            ),
            status_queue,
        )

    @classmethod
    async def wait_sub_pools_ready(cls, create_pool_tasks: list[asyncio.Task]):
        addresses = []
        tasks = []
        for t in create_pool_tasks:
            pool_task, queue = await t
            tasks.append(pool_task)
            status = await asyncio.to_thread(queue.get)
            addresses.append(status.external_addresses)
        return tasks, addresses

    @classmethod
    async def _create_sub_pool(
        cls,
        actor_config: ActorPoolConfig,
        process_index: int,
        status_queue: multiprocessing.Queue,
        main_pool_pid: int,
    ):
        pool: TestSubActorPool = await TestSubActorPool.create(
            {
                "actor_pool_config": actor_config,
                "process_index": process_index,
                "main_pool_pid": main_pool_pid,
            }
        )
        await pool.start()
        status_queue.put(
            SubpoolStatus(status=0, external_addresses=[pool.external_address])
        )
        actor_config.reset_pool_external_address(process_index, [pool.external_address])
        await pool.join()

    def _sync_pool_config(self, actor_pool_config: ActorPoolConfig):
        # test pool does not create routers, thus can skip this step
        pass

    async def append_sub_pool(
        self,
        label: str | None = None,
        internal_address: str | None = None,
        external_address: str | None = None,
        env: dict | None = None,
        modules: list[str] | None = None,
        suspend_sigint: bool | None = None,
        use_uvloop: bool | None = None,
        logging_conf: dict | None = None,
        start_method: str | None = None,
        kwargs: dict | None = None,
    ):
        external_address = (
            external_address
            or TestMainActorPool.get_external_addresses(
                self.external_address, n_process=1
            )[-1]
        )

        # use last process index's logging_conf and use_uv_loop config if not provide
        actor_pool_config = self._config.as_dict()
        last_process_index = self._config.get_process_indexes()[-1]
        last_logging_conf = actor_pool_config["pools"][last_process_index][
            "logging_conf"
        ]
        last_use_uv_loop = actor_pool_config["pools"][last_process_index]["use_uvloop"]
        _logging_conf = logging_conf or last_logging_conf
        _use_uv_loop = use_uvloop if use_uvloop is not None else last_use_uv_loop

        process_index = next(TestMainActorPool.process_index_gen(external_address))
        internal_address = internal_address or TestMainActorPool.gen_internal_address(
            process_index, external_address
        )

        self._config.add_pool_conf(
            process_index,
            label,
            internal_address,
            external_address,
            env,
            modules,
            suspend_sigint,
            _use_uv_loop,
            _logging_conf,
            kwargs,
        )
        pool_task = asyncio.create_task(
            TestMainActorPool.start_sub_pool(self._config, process_index)
        )
        tasks, addresses = await TestMainActorPool.wait_sub_pools_ready([pool_task])

        self.attach_sub_process(addresses[0][0], tasks[0])

        control_message = ControlMessage(
            message_id=new_message_id(),
            address=self.external_address,
            control_message_type=ControlMessageType.sync_config,
            content=self._config,
        )
        await self.handle_control_command(control_message)

        return addresses[0][0]

    async def kill_sub_pool(
        self, process: multiprocessing.Process, force: bool = False
    ):
        process.cancel()  # type: ignore

    async def is_sub_pool_alive(self, process: multiprocessing.Process):
        return not process.cancelled()  # type: ignore


class TestSubActorPool(SubActorPool):
    def _sync_pool_config(self, actor_pool_config: ActorPoolConfig):
        # test pool does not create routers, thus can skip this step
        pass

    @classmethod
    async def create(cls, config: dict) -> ActorPoolType:
        kw: dict[str, Any] = dict()
        cls._parse_config(config, kw)
        process_index: int = kw["process_index"]
        actor_pool_config = kw["config"]  # type: ActorPoolConfig
        external_addresses = actor_pool_config.get_pool_config(process_index)[
            "external_address"
        ]

        def handle_channel(channel):
            return pool.on_new_channel(channel)

        # create servers
        server_addresses = external_addresses + [gen_local_address(process_index)]
        server_addresses = sorted(set(server_addresses))
        servers = await cls._create_servers(
            server_addresses, handle_channel, actor_pool_config.get_comm_config()
        )
        cls._update_stored_addresses(servers, server_addresses, actor_pool_config, kw)

        # create pool
        pool = cls(**kw)
        return pool  # type: ignore

    async def stop(self):
        # do not close dummy server
        self._servers = [
            s for s in self._servers[:-1] if not isinstance(s, DummyServer)
        ]
        await super().stop()
