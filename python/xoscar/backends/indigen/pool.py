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
import configparser
import itertools
import logging.config
import os
import pickle
import random
import signal
import struct
import sys
import threading
import time
import uuid
from enum import IntEnum
from typing import List, Optional

import psutil

from ..._utils import reset_id_random_seed
from ...utils import ensure_coverage
from ..config import ActorPoolConfig
from ..message import (
    ControlMessage,
    ControlMessageType,
    CreateActorMessage,
    new_message_id,
)
from ..pool import MainActorPoolBase, SubActorPoolBase, _register_message_handler
from . import shared_memory
from .fate_sharing import create_subprocess_exec

_SUBPROCESS_SHM_SIZE = 10240
_is_windows: bool = sys.platform.startswith("win")

logger = logging.getLogger(__name__)


class _ShmSeq(IntEnum):
    INIT_PARAMS = 1
    INIT_RESULT = 2


def _shm_put_object(seq: _ShmSeq, shm: shared_memory.SharedMemory, o: object):
    serialized = pickle.dumps(o)
    assert (
        len(serialized) < _SUBPROCESS_SHM_SIZE - 8
    ), f"Serialized object {o} is too long."
    shm.buf[4:12] = struct.pack("<II", sys.hexversion, len(serialized))
    shm.buf[12 : 12 + len(serialized)] = serialized
    shm.buf[:4] = struct.pack("<I", seq)


def _shm_get_object(seq: _ShmSeq, shm: shared_memory.SharedMemory):
    recv_seq = struct.unpack("<I", shm.buf[:4])[0]
    if recv_seq != seq:
        return
    python_version_hex, size = struct.unpack("<II", shm.buf[4:12])
    if python_version_hex != sys.hexversion:
        raise RuntimeError(
            f"Python version mismatch, sender: {python_version_hex}, receiver: {sys.hexversion}"
        )
    return pickle.loads(shm.buf[12 : 12 + size])


@_register_message_handler
class MainActorPool(MainActorPoolBase):
    @classmethod
    def get_external_addresses(
        cls,
        address: str,
        n_process: int | None = None,
        ports: list[int] | None = None,
        schemes: list[Optional[str]] | None = None,
    ):
        """Get external address for every process"""
        assert n_process is not None
        if ":" in address:
            host, port_str = address.rsplit(":", 1)
            port = int(port_str)
            if ports:
                if len(ports) != n_process:
                    raise ValueError(
                        f"`ports` specified, but its count "
                        f"is not equal to `n_process`, "
                        f"number of ports: {len(ports)}, "
                        f"n_process: {n_process}"
                    )
                sub_ports = ports
            else:
                sub_ports = [0] * n_process
        else:
            host = address
            if ports and len(ports) != n_process + 1:
                # ports specified, the first of which should be main port
                raise ValueError(
                    f"`ports` specified, but its count "
                    f"is not equal to `n_process` + 1, "
                    f"number of ports: {len(ports)}, "
                    f"n_process + 1: {n_process + 1}"
                )
            elif not ports:
                ports = [0] * (n_process + 1)
            port = ports[0]
            sub_ports = ports[1:]
        if not schemes:
            prefix_iter = itertools.repeat("")
        else:
            prefix_iter = [f"{scheme}://" if scheme else "" for scheme in schemes]  # type: ignore
        return [
            f"{prefix}{host}:{port}"
            for port, prefix in zip([port] + sub_ports, prefix_iter)
        ]

    @classmethod
    def gen_internal_address(
        cls, process_index: int, external_address: str | None = None
    ) -> str | None:
        if hasattr(asyncio, "start_unix_server"):
            return f"unixsocket:///{process_index}"
        else:
            return external_address

    @classmethod
    async def start_sub_pool(
        cls,
        actor_pool_config: ActorPoolConfig,
        process_index: int,
        start_python: str | None = None,
    ):
        return await cls._create_sub_pool_from_parent(
            actor_pool_config, process_index, start_python
        )

    @classmethod
    async def wait_sub_pools_ready(cls, create_pool_tasks: List[asyncio.Task]):
        processes: list[asyncio.subprocess.Process] = []
        ext_addresses = []
        error = None
        for task in create_pool_tasks:
            process, address = await task
            processes.append(process)
            ext_addresses.append(address)
        if error:
            for p in processes:
                # error happens, kill all subprocesses
                p.kill()
            raise error
        return processes, ext_addresses

    @classmethod
    def _start_sub_pool_in_child(
        cls,
        shm_name: str,
    ):
        ensure_coverage()

        shm = shared_memory.SharedMemory(shm_name, track=False)
        try:
            config = _shm_get_object(_ShmSeq.INIT_PARAMS, shm)
            actor_config = config["actor_pool_config"]
            process_index = config["process_index"]
            main_pool_pid = config["main_pool_pid"]

            def _check_ppid():
                while True:
                    try:
                        # We can't simply check if the os.getppid() equals with main_pool_pid,
                        # as the double fork may result in a new process as the parent.
                        psutil.Process(main_pool_pid)
                    except psutil.NoSuchProcess:
                        logger.error("Exit due to main pool %s exit.", main_pool_pid)
                        os._exit(233)  # Special exit code for debugging.
                    except Exception as e:
                        logger.exception("Check ppid failed: %s", e)
                    time.sleep(10)

            t = threading.Thread(target=_check_ppid, daemon=True)
            t.start()

            # make sure enough randomness for every sub pool
            random.seed(uuid.uuid1().bytes)
            reset_id_random_seed()

            conf = actor_config.get_pool_config(process_index)
            suspend_sigint = conf["suspend_sigint"]
            if suspend_sigint:
                signal.signal(signal.SIGINT, lambda *_: None)

            logging_conf = conf["logging_conf"] or {}
            if isinstance(logging_conf, configparser.RawConfigParser):
                logging.config.fileConfig(logging_conf)
            elif logging_conf.get("dict"):
                logging.config.dictConfig(logging_conf["dict"])
            elif logging_conf.get("file"):
                logging.config.fileConfig(logging_conf["file"])
            elif logging_conf.get("level"):
                logging.getLogger("__main__").setLevel(logging_conf["level"])
                logging.getLogger("xoscar").setLevel(logging_conf["level"])
                if logging_conf.get("format"):
                    logging.basicConfig(format=logging_conf["format"])

            use_uvloop = conf["use_uvloop"]
            if use_uvloop:
                import uvloop

                asyncio.set_event_loop(uvloop.new_event_loop())
            else:
                asyncio.set_event_loop(asyncio.new_event_loop())

            coro = cls._create_sub_pool(actor_config, process_index, main_pool_pid, shm)
            asyncio.run(coro)
        finally:
            shm.close()

    @classmethod
    async def _create_sub_pool(
        cls,
        actor_config: ActorPoolConfig,
        process_index: int,
        main_pool_pid: int,
        shm: shared_memory.SharedMemory,
    ):
        cur_pool_config = actor_config.get_pool_config(process_index)
        env = cur_pool_config["env"]
        if env:
            os.environ.update(env)
        pool = await SubActorPool.create(
            {
                "actor_pool_config": actor_config,
                "process_index": process_index,
                "main_pool_pid": main_pool_pid,
            }
        )
        await pool.start()
        _shm_put_object(_ShmSeq.INIT_RESULT, shm, cur_pool_config["external_address"])
        await pool.join()

    @staticmethod
    async def _create_sub_pool_from_parent(
        actor_pool_config: ActorPoolConfig,
        process_index: int,
        start_python: str | None = None,
    ):
        # We check the Python version in _shm_get_object to make it faster,
        # as in most cases the Python versions are the same.
        if start_python is None:
            start_python = sys.executable

        external_addresses: List | None = None
        shm = shared_memory.SharedMemory(
            create=True, size=_SUBPROCESS_SHM_SIZE, track=False
        )
        try:
            _shm_put_object(
                _ShmSeq.INIT_PARAMS,
                shm,
                {
                    "actor_pool_config": actor_pool_config,
                    "process_index": process_index,
                    "main_pool_pid": os.getpid(),
                },
            )
            cmd = [
                start_python,
                "-m",
                "xoscar.backends.indigen",
                "start_sub_pool",
                "-sn",
                shm.name,
            ]
            # We need to inherit the parent environment to ensure the subprocess works correctly on Windows.
            new_env = dict(os.environ)
            env = actor_pool_config.get_pool_config(process_index).get("env") or {}
            new_env.update(env)
            logger.info("Creating sub pool via command: %s", cmd)
            process = await create_subprocess_exec(*cmd, env=new_env)

            def _get_external_addresses():
                try:
                    nonlocal external_addresses
                    while (
                        shm
                        and shm.buf is not None
                        and not (
                            external_addresses := _shm_get_object(
                                _ShmSeq.INIT_RESULT, shm
                            )
                        )
                    ):
                        time.sleep(0.1)
                except asyncio.CancelledError:
                    pass

            _, unfinished = await asyncio.wait(
                [
                    asyncio.create_task(process.wait()),
                    asyncio.create_task(asyncio.to_thread(_get_external_addresses)),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in unfinished:
                t.cancel()
        finally:
            shm.close()
            shm.unlink()
        if external_addresses is None:
            raise OSError(f"Start sub pool failed, returncode: {process.returncode}")
        return process, external_addresses

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
        start_python: str | None = None,
        kwargs: dict | None = None,
    ):
        # external_address has port 0, subprocess will bind random port.
        external_address = (
            external_address
            or MainActorPool.get_external_addresses(self.external_address, n_process=1)[
                -1
            ]
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

        process_index = next(MainActorPool.process_index_gen(external_address))
        internal_address = internal_address or MainActorPool.gen_internal_address(
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

        process, external_addresses = await self._create_sub_pool_from_parent(
            self._config, process_index, start_python
        )

        self._config.reset_pool_external_address(process_index, external_addresses[0])
        self.attach_sub_process(external_addresses[0], process)

        control_message = ControlMessage(
            message_id=new_message_id(),
            address=self.external_address,
            control_message_type=ControlMessageType.sync_config,
            content=self._config,
        )
        await self.handle_control_command(control_message)
        # The actual port will return in process_status.
        return external_addresses[0]

    async def remove_sub_pool(
        self, external_address: str, timeout: float | None = None, force: bool = False
    ):
        process = self.sub_processes[external_address]
        process_index = self._config.get_process_index(external_address)
        del self.sub_processes[external_address]
        self._config.remove_pool_config(process_index)
        await self.stop_sub_pool(external_address, process, timeout, force)

        control_message = ControlMessage(
            message_id=new_message_id(),
            address=self.external_address,
            control_message_type=ControlMessageType.sync_config,
            content=self._config,
        )
        await self.handle_control_command(control_message)

    async def kill_sub_pool(
        self, process: asyncio.subprocess.Process, force: bool = False
    ):
        try:
            p = psutil.Process(process.pid)
        except psutil.NoSuchProcess:
            return

        if not force:  # pragma: no cover
            p.terminate()
            try:
                p.wait(5)
            except psutil.TimeoutExpired:
                pass

        while p.is_running():
            p.kill()
            if not p.is_running():
                return
            logger.info("Sub pool can't be killed: %s", p)
            time.sleep(0.1)

    async def is_sub_pool_alive(self, process: asyncio.subprocess.Process):
        return process.returncode is None

    async def recover_sub_pool(self, address: str):
        process_index = self._config.get_process_index(address)
        # process dead, restart it
        # remember always use spawn to recover sub pool
        task = asyncio.create_task(self.start_sub_pool(self._config, process_index))
        self.sub_processes[address] = (await self.wait_sub_pools_ready([task]))[0][0]

        if self._auto_recover == "actor":
            # need to recover all created actors
            for _, message in self._allocated_actors[address].values():
                create_actor_message: CreateActorMessage = message  # type: ignore
                await self.call(address, create_actor_message)

    async def start(self):
        await super().start()
        await self.start_monitor()


@_register_message_handler
class SubActorPool(SubActorPoolBase):
    pass
