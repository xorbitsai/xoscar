# Copyright 2026 Xorbits Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Planned handoff between ordinary native main pools on the same machine."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import functools
import logging
import math
import os
import socket
import subprocess
import sys
import threading
import uuid

import psutil

from ..config import ActorPoolConfig
from ..message import (
    ControlMessage,
    ControlMessageType,
    ErrorMessage,
    ResultMessage,
    new_message_id,
)
from ..router import Router

logger = logging.getLogger(__name__)
_PROTOCOL = 1


def _identity() -> dict:
    return {
        "pid": os.getpid(),
        "create_time": psutil.Process().create_time(),
        "machine": (socket.gethostname(), psutil.boot_time()),
        "python": sys.version_info[:2],
        "protocol": _PROTOCOL,
    }


def _process(identity: dict) -> psutil.Process:
    process = psutil.Process(identity["pid"])
    if process.create_time() != identity["create_time"]:
        raise ProcessLookupError("Actor pool process identity changed")
    if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
        raise ProcessLookupError("Actor pool process has exited")
    return process


def _check_identity(identity: dict):
    local = _identity()
    for field in ("machine", "python", "protocol"):
        if identity[field] != local[field]:
            raise ValueError(f"Actor pool handoff requires matching {field}")
    _process(identity)


class SubPoolProcess:
    """An owned or adopted process, without an event-loop-owned transport.

    An adopted process is not necessarily an OS child. Polling its verified
    identity avoids waitpid() and PID reuse; the original parent alone reaps it.
    """

    def __init__(self, identity: dict, popen: subprocess.Popen | None = None, job=None):
        self.identity = dict(identity)
        self._popen = popen
        self._job = job
        if job is not None:
            self.identity["job_handle"] = int(job)

    @classmethod
    def spawned(cls, popen: subprocess.Popen, job=None):
        return cls(
            {"pid": popen.pid, "create_time": psutil.Process(popen.pid).create_time()},
            popen,
            job,
        )

    @classmethod
    def adopted(cls, identity: dict, owner: dict):
        job = None
        if sys.platform == "win32":
            from .fate_sharing import duplicate_win32_job

            if "job_handle" not in identity:
                raise NotImplementedError("Windows handoff requires per-sub-pool jobs")
            _process(owner)
            job = duplicate_win32_job(owner["pid"], identity["job_handle"])
            try:
                _process(owner)
            except BaseException:
                job.Close()
                raise
        return cls(identity, job=job)

    def _close_job(self):
        if self._job is not None:
            job, self._job = self._job, None
            job.Close()

    @property
    def pid(self):
        return self.identity["pid"]

    @property
    def returncode(self):
        if self._popen is not None:
            code = self._popen.poll()
        else:
            try:
                _process(self.identity)
                return None
            except (psutil.NoSuchProcess, ProcessLookupError):
                # Non-parents cannot collect the exit status.
                code = 0
        if code is not None:
            self._close_job()
        return code

    async def wait(self):
        while self.returncode is None:
            await asyncio.sleep(0.05)
        return self.returncode

    def terminate(self):
        try:
            _process(self.identity).terminate()
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass

    def kill(self):
        try:
            _process(self.identity).kill()
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass

    def detach(self):
        # Reaping must not keep asyncio.run() waiting for an executor worker.
        # Popen and a daemon thread do not kill the child when the loop closes.
        if self._popen is not None:
            threading.Thread(target=self._popen.wait, daemon=True).start()
            self._popen = None
        self._close_job()


class ParentWatcher:
    """The event-loop watcher and blocking-code watchdog share one owner."""

    def __init__(self, pid: int):
        process = psutil.Process(pid)
        self.identity = {"pid": pid, "create_time": process.create_time()}
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        threading.Thread(target=self._watch, daemon=True).start()

    def change_owner(self, identity: dict):
        _process(identity)
        with self._lock:
            self.identity = identity

    def alive(self):
        with self._lock:
            try:
                _process(self.identity)
                return True
            except (psutil.NoSuchProcess, ProcessLookupError):
                return False

    def _watch(self):
        while not self._stopped.wait(10):
            with self._lock:
                try:
                    _process(self.identity)
                except (psutil.NoSuchProcess, ProcessLookupError):
                    logger.error("Exit due to main pool %s exit.", self.identity["pid"])
                    os._exit(233)
                except Exception:
                    logger.exception("Checking main pool failed")

    def stop(self):
        self._stopped.set()


def pool_lifecycle(method):
    """Exclude actor/process mutation during handoff, while sends keep running."""

    @functools.wraps(method)
    async def locked(self, *args, **kwargs):
        self._check_mutable()
        task = asyncio.current_task()
        self._lifecycle_tasks[task] = self._lifecycle_tasks.get(task, 0) + 1
        self._lifecycle_active += 1
        self._lifecycle_idle.clear()
        try:
            return await method(self, *args, **kwargs)
        finally:
            self._lifecycle_tasks[task] -= 1
            if not self._lifecycle_tasks[task]:
                del self._lifecycle_tasks[task]
            self._lifecycle_active -= 1
            if not self._lifecycle_active:
                self._lifecycle_idle.set()

    return locked


async def _rpc(pool, address: str, operation: str, **content):
    result = await asyncio.wait_for(
        pool.call(
            address,
            ControlMessage(
                new_message_id(),
                address,
                ControlMessageType.pool_lifecycle,
                {"operation": operation, **content},
            ),
        ),
        timeout=5,
    )
    if isinstance(result, ErrorMessage):
        raise result.error.with_traceback(result.traceback)
    return result.result


class SubPoolHandoffMixin:
    def __init__(self, *args, **kwargs):
        owner = kwargs.get("main_pool_pid")
        self._parent_watcher = (
            owner
            if isinstance(owner, ParentWatcher)
            else (ParentWatcher(owner) if owner else None)
        )
        if self._parent_watcher:
            kwargs["main_pool_pid"] = self._parent_watcher.identity["pid"]
        self._lifecycle_lock = asyncio.Lock()
        self._lifecycle_active = 0
        self._lifecycle_tasks = {}
        self._lifecycle_idle = asyncio.Event()
        self._lifecycle_idle.set()
        self._handoff = None
        self._last_handoff = None
        super().__init__(*args, **kwargs)

    def _check_mutable(self):
        if self._handoff is not None:
            raise RuntimeError(
                "Actor pool handoff is in progress; retry actor lifecycle operations afterwards"
            )

    @pool_lifecycle
    async def create_actor(self, message):
        return await super().create_actor(message)

    @pool_lifecycle
    async def destroy_actor(self, message):
        return await super().destroy_actor(message)

    async def notify_main_pool_to_destroy(self, message):
        # A notification only updates the registry. It must not wait for the
        # main's mutation lock while the main waits for this child's prepare.
        await _rpc(
            self, self._main_address, "unregister_actor", actor_ref=message.actor_ref
        )

    async def _watch_main_pool(self, main_pool_pid):
        try:
            while not self.stopped:
                if not self._parent_watcher.alive():
                    asyncio.create_task(self.stop())
                    return
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            return

    async def stop(self):
        if self._parent_watcher:
            self._parent_watcher.stop()
        await super().stop()

    async def handle_control_command(self, message):
        if message.control_message_type != ControlMessageType.pool_lifecycle:
            if self._handoff and message.control_message_type in (
                ControlMessageType.stop,
                ControlMessageType.sync_config,
            ):
                raise RuntimeError("Actor pool handoff is in progress")
            return await super().handle_control_command(message)
        content = message.content
        operation, ticket = content["operation"], content["ticket"]
        result = True
        async with self._lifecycle_lock:
            if operation == "prepare_child":
                if self._handoff:
                    if self._handoff["ticket"] != ticket:
                        raise RuntimeError("Sub pool already has a pending handoff")
                else:
                    source = content["source"]
                    if (
                        self._main_address != source["address"]
                        or self._parent_watcher.identity["pid"] != source["pid"]
                        or self._parent_watcher.identity["create_time"]
                        != source["create_time"]
                    ):
                        raise RuntimeError("Sub pool owner changed")
                    _check_identity(source)
                    _check_identity(content["target"])
                    self._handoff = {
                        **content,
                        "old_config": self._config,
                        "old_owner": self._parent_watcher.identity,
                        "old_address": self._main_address,
                    }
                    await self._lifecycle_idle.wait()
            elif operation == "switch_child":
                state = self._handoff
                if not state or state["ticket"] != ticket:
                    raise RuntimeError("No matching prepared sub pool handoff")
                from .fate_sharing import clear_kill_on_parent_death

                clear_kill_on_parent_death()
                self._parent_watcher.change_owner(state["target"])
                self._main_address = state["target"]["address"]
                self._sync_pool_config(state["config"])
            elif operation == "abort_child":
                state = self._handoff
                if state and state["ticket"] == ticket:
                    self._parent_watcher.change_owner(state["old_owner"])
                    self._main_address = state["old_address"]
                    self._sync_pool_config(state["old_config"])
                    self._handoff = None
            elif operation == "child_ready":
                result = bool(
                    self._handoff
                    and self._handoff["ticket"] == ticket
                    and self._main_address == self._handoff["target"]["address"]
                )
            elif operation == "discard_child":
                if self._handoff and self._handoff["ticket"] == ticket:
                    asyncio.create_task(self.stop())
            elif operation == "finish_child":
                if self._last_handoff == ticket:
                    pass
                elif (
                    self._handoff
                    and self._handoff["ticket"] == ticket
                    and self._main_address == self._handoff["target"]["address"]
                ):
                    self._last_handoff = ticket
                    self._handoff = None
                else:
                    raise RuntimeError("Sub pool has not switched to the new main")
            else:
                raise ValueError(f"Unknown sub pool handoff operation {operation}")
        return ResultMessage(message.message_id, result, protocol=message.protocol)


class MainPoolHandoffMixin:
    external_address: str

    def __init__(self, *args, **kwargs):
        self._lifecycle_lock = asyncio.Lock()
        self._lifecycle_active = 0
        self._lifecycle_tasks = {}
        self._lifecycle_idle = asyncio.Event()
        self._lifecycle_idle.set()
        self._handoff = None
        self._completed_handoffs = {}
        self._handoff_expiry = None
        super().__init__(*args, **kwargs)

    def _check_mutable(self):
        if self.stopped:
            raise RuntimeError("Actor pool has stopped")
        if (
            self._handoff is not None
            and asyncio.current_task() not in self._lifecycle_tasks
        ):
            raise RuntimeError(
                "Actor pool handoff is in progress or its sub pools have been handed off"
            )

    def attach_sub_process(self, external_address, process):
        super().attach_sub_process(external_address, process)
        self._allocated_actors.setdefault(external_address, {})

    @pool_lifecycle
    async def create_actor(self, message):
        return await super().create_actor(message)

    @pool_lifecycle
    async def destroy_actor(self, message):
        return await super().destroy_actor(message)

    def _sync_handoff_config(self, config, addresses):
        # A client may already have calls in flight over TCP. Adding a Unix
        # route in the new main must not retire that live client connection.
        # Keep each process's existing data routes; the pool's own management
        # router still receives the full adopted configuration.
        router = Router.get_instance()
        routes = (
            {addr: router._mapping.get(addr) for addr in addresses} if router else {}
        )
        self._sync_pool_config(config)
        if router:
            for addr, previous in routes.items():
                if previous is None:
                    router._mapping.pop(addr, None)
                else:
                    router._mapping[addr] = previous

    async def stop(self):
        async with self._lifecycle_lock:
            if (
                self._handoff
                and self._handoff["role"] == "target"
                and self._handoff["phase"] != "committing"
            ):
                raise RuntimeError("Cannot stop a main pool during handoff")
            if self._handoff_expiry:
                self._handoff_expiry.cancel()
            await super().stop()

    async def handoff_to(self, address: str, *, timeout: float = 30) -> list[str]:
        """Transfer existing sub pools to another ordinary native main pool.

        The target must run on this machine, have no sub pools, and use a
        compatible Python/Xoscar environment. Main-pool actors stay here.
        On success this main may be stopped without stopping transferred
        actors. Existing references to sub-pool actors remain valid; callers
        using this main's address must reconnect to the target.

        Handoff supports native POSIX and Windows subprocess pools.
        Cancellation waits for commit or rollback.
        Before commit failures roll back; after commit retry this method with
        the same address to finish an interrupted acknowledgement.
        """
        if address == self.external_address:
            raise ValueError("The handoff target must be a different main pool")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a finite positive number")
        task = asyncio.create_task(self._handoff_to(address, timeout))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Shutdown may cancel the caller repeatedly, even during rollback.
            # Shield every wait so the ownership transfer reaches a final state.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()
            raise

    async def _rollback_handoff(self, state):
        results = await asyncio.gather(
            *[
                _rpc(self, addr, "abort_child", ticket=state["ticket"])
                for addr in state["prepared"]
            ],
            return_exceptions=True,
        )
        failed = []
        for addr, result in zip(state["prepared"], results):
            if isinstance(result, BaseException):
                try:
                    _process(state["processes"][addr])
                except (psutil.NoSuchProcess, ProcessLookupError):
                    continue
                failed.append(addr)
        with contextlib.suppress(Exception):
            await _rpc(
                self, state["target_address"], "abort_main", ticket=state["ticket"]
            )
        if failed:
            state["phase"] = "rollback_failed"
            raise RuntimeError(
                f"Handoff rollback was not confirmed for {failed}; retry handoff_to with the same target or stop this pool"
            )
        self._handoff = None
        await self.start_monitor()

    async def _handoff_to(self, address, timeout):
        async with self._lifecycle_lock:
            if (
                self._handoff
                and self._handoff["role"] == "source"
                and self._handoff["phase"] == "rollback_failed"
                and self._handoff["target_address"] == address
            ):
                await self._rollback_handoff(self._handoff)
            if (
                self._handoff
                and self._handoff["role"] == "source"
                and self._handoff["phase"] in ("committed", "done")
                and self._handoff["target_address"] == address
            ):
                state = self._handoff
                result = await _rpc(
                    self, address, "commit_main", ticket=state["ticket"]
                )
                state["phase"] = "done"
                return result
            self._check_mutable()
            if not self.sub_processes:
                raise ValueError("The source main pool has no sub pools to hand off")
            if not all(
                isinstance(p, SubPoolProcess) for p in self.sub_processes.values()
            ):
                raise NotImplementedError("Handoff requires native subprocess pools")
            state = self._handoff = {
                "role": "source",
                "phase": "preparing",
                "ticket": uuid.uuid4().hex,
                "target_address": address,
                "prepared": [],
                "processes": {},
            }
            source = {**_identity(), "address": self.external_address}
            processes = state["processes"]

            async def prepare():
                await self._lifecycle_idle.wait()
                if self._monitor_task:
                    monitor, self._monitor_task = self._monitor_task, None
                    monitor.cancel()
                    # A task cancelled before its coroutine first runs never
                    # reaches monitor_sub_pools()' cancellation handler. Consume
                    # that child cancellation while preserving cancellation of
                    # this preparation task (e.g. its timeout).
                    await asyncio.gather(monitor, return_exceptions=True)
                processes.update(
                    {addr: p.identity for addr, p in self.sub_processes.items()}
                )
                for identity in processes.values():
                    _process(identity)
                target, config = await _rpc(
                    self,
                    address,
                    "reserve_main",
                    ticket=state["ticket"],
                    source=source,
                    config=self._config,
                    processes=processes,
                    timeout=timeout,
                )
                _check_identity(target)
                for addr in processes:
                    state["prepared"].append(addr)
                    await _rpc(
                        self,
                        addr,
                        "prepare_child",
                        ticket=state["ticket"],
                        source=source,
                        target=target,
                        config=config,
                    )
                allocated = {
                    addr: dict(self._allocated_actors.get(addr, {}))
                    for addr in processes
                }
                await _rpc(
                    self,
                    address,
                    "stage_main",
                    ticket=state["ticket"],
                    allocated=allocated,
                )
                for addr in processes:
                    await _rpc(self, addr, "switch_child", ticket=state["ticket"])
                if state.get("cancelled"):
                    raise TimeoutError("Actor pool handoff expired before commit")

            try:
                await asyncio.wait_for(prepare(), timeout)
            except BaseException:
                # Include the child whose prepare reply was lost.
                await self._rollback_handoff(state)
                raise
            # The decision is irrevocable from here. Drop all old cleanup and
            # recovery authority before asking the new main to thaw actors.
            state["phase"] = "committed"
            for process in self.sub_processes.values():
                process.detach()
            self.sub_processes.clear()
            self._allocated_actors = {
                self.external_address: self._allocated_actors[self.external_address]
            }
            conf = copy.deepcopy(self._config.as_dict())
            main_conf = conf["pools"][self.process_index]
            conf["pools"] = {self.process_index: main_conf}
            conf["mapping"] = {
                addr: main_conf["internal_address"]
                for addr in main_conf["external_address"]
            }
            self._sync_handoff_config(ActorPoolConfig(conf), processes)
            result = await _rpc(self, address, "commit_main", ticket=state["ticket"])
            state["phase"] = "done"
            return result

    async def _expire_handoff(self, ticket, timeout):
        await asyncio.sleep(timeout)
        while self._handoff and self._handoff.get("ticket") == ticket:
            state = self._handoff
            try:
                if state["phase"] == "committing":
                    await self._target_command("commit_main", {"ticket": ticket})
                    return
                decision = await asyncio.wait_for(
                    _rpc(
                        self, state["source"]["address"], "expire_main", ticket=ticket
                    ),
                    2,
                )
            except Exception:
                try:
                    _process(state["source"])
                except (psutil.NoSuchProcess, ProcessLookupError):
                    # A source can exit after relinquishing its children but
                    # before the commit request/reply arrives. Finish only if
                    # the full registry is staged and every child switched.
                    ready = await asyncio.gather(
                        *[
                            _rpc(self, addr, "child_ready", ticket=ticket)
                            for addr in state["processes"]
                        ],
                        return_exceptions=True,
                    )
                    decision = (
                        "commit"
                        if state["phase"] == "staged"
                        and all(value is True for value in ready)
                        else "abort"
                    )
                else:
                    await asyncio.sleep(1)
                    continue
            try:
                await self._target_command(
                    "commit_main" if decision == "commit" else "abort_main",
                    {"ticket": ticket},
                )
                return
            except Exception:
                logger.exception("Finishing main pool handoff failed; retrying")
                await asyncio.sleep(1)

    async def _target_command(self, operation, content):
        ticket = content["ticket"]
        async with self._lifecycle_lock:
            if operation == "commit_main" and ticket in self._completed_handoffs:
                return self._completed_handoffs[ticket]
            if operation == "reserve_main":
                self._check_mutable()
                if self.sub_processes or self._config.n_pool != 1:
                    raise ValueError("The handoff target must have no sub pools")
                _check_identity(content["source"])
                source_config = content["config"]
                if (
                    source_config.get_comm_config() != self._config.get_comm_config()
                    or source_config.get_proxy_config()
                    != self._config.get_proxy_config()
                ):
                    raise ValueError(
                        "Handoff requires matching transport and proxy configuration"
                    )
                config = copy.deepcopy(self._config.as_dict())
                for index in source_config.get_process_indexes()[1:]:
                    if index in config["pools"]:
                        raise ValueError("Pool process indexes conflict")
                    conf = copy.deepcopy(source_config.get_pool_config(index))
                    config["pools"][index] = conf
                    for addr in conf["external_address"]:
                        config["mapping"][addr] = conf["internal_address"]
                for identity in content["processes"].values():
                    _process(identity)
                self._handoff = {
                    **content,
                    "role": "target",
                    "phase": "preparing",
                    "config": ActorPoolConfig(config),
                }
                await self._lifecycle_idle.wait()
                if self.sub_processes or self._config.n_pool != 1:
                    self._handoff = None
                    raise ValueError("The handoff target must have no sub pools")
                adopted = self._handoff["adopted"] = {}
                try:
                    for addr, identity in content["processes"].items():
                        adopted[addr] = SubPoolProcess.adopted(
                            identity, content["source"]
                        )
                except BaseException:
                    for process in adopted.values():
                        process.detach()
                    self._handoff = None
                    raise
                self._handoff_expiry = asyncio.create_task(
                    self._expire_handoff(ticket, content["timeout"])
                )
                return {**_identity(), "address": self.external_address}, self._handoff[
                    "config"
                ]
            state = self._handoff
            if not state or state["role"] != "target" or state["ticket"] != ticket:
                if operation == "abort_main":
                    return True
                raise RuntimeError("No matching prepared main pool handoff")
            if operation == "stage_main":
                allocated = content["allocated"]
                for actors in allocated.values():
                    if any(ref is None or ref.uid in self._actors for ref in actors):
                        raise ValueError("In-flight or conflicting actor registration")
                state["allocated"] = allocated
                state["phase"] = "staged"
            elif operation == "commit_main":
                if state["phase"] not in ("staged", "committing"):
                    raise RuntimeError("Actor registry has not been staged")
                state["phase"] = "committing"
                self._sync_handoff_config(state["config"], state["processes"])
                self.sub_processes = state["adopted"]
                self._allocated_actors.update(state["allocated"])
                for addr, process in self.sub_processes.items():
                    try:
                        await _rpc(self, addr, "finish_child", ticket=ticket)
                    except Exception:
                        if process.returncode is None:
                            raise
                        # The new owner recovers a process lost during commit.
                        logger.warning("Sub pool %s exited during handoff", addr)
                result = list(self.sub_processes)
                self._completed_handoffs[ticket] = result
                self._handoff = None
                if (
                    self._handoff_expiry
                    and self._handoff_expiry is not asyncio.current_task()
                ):
                    self._handoff_expiry.cancel()
                await self.start_monitor()
                return result
            elif operation == "abort_main":
                if state["phase"] == "committing":
                    raise RuntimeError("A committed handoff cannot be aborted")
                try:
                    _process(state["source"])
                    child_operation = "abort_child"
                except (psutil.NoSuchProcess, ProcessLookupError):
                    child_operation = "discard_child"
                state["phase"] = "aborting"
                results = await asyncio.gather(
                    *[
                        _rpc(self, addr, child_operation, ticket=ticket)
                        for addr in state["processes"]
                    ],
                    return_exceptions=True,
                )
                failed = []
                for addr, result in zip(state["processes"], results):
                    if not isinstance(result, BaseException):
                        continue
                    process = SubPoolProcess(state["processes"][addr])
                    if process.returncode is not None:
                        continue
                    if child_operation == "discard_child":
                        await self.kill_sub_pool(process, force=True)
                    else:
                        failed.append(addr)
                if failed:
                    raise RuntimeError(
                        f"Sub pool rollback is still pending for {failed}"
                    )
                for process in state["adopted"].values():
                    process.detach()
                self._handoff = None
                if (
                    self._handoff_expiry
                    and self._handoff_expiry is not asyncio.current_task()
                ):
                    self._handoff_expiry.cancel()
            else:
                raise ValueError(f"Unknown main pool handoff operation {operation}")
            return True

    async def handle_control_command(self, message):
        if message.control_message_type == ControlMessageType.pool_lifecycle:
            content = message.content
            operation = content["operation"]
            if operation == "unregister_actor":
                ref = content["actor_ref"]
                self._allocated_actors.get(ref.address, {}).pop(ref, None)
                result = True
            elif operation == "expire_main":
                state = self._handoff
                if (
                    state
                    and state["role"] == "source"
                    and state["ticket"] == content["ticket"]
                ):
                    result = (
                        "commit" if state["phase"] in ("committed", "done") else "abort"
                    )
                    if result == "abort":
                        state["cancelled"] = True
                else:
                    result = "abort"
            else:
                result = await self._target_command(operation, content)
            return ResultMessage(message.message_id, result, protocol=message.protocol)
        if (
            self._handoff
            and asyncio.current_task() not in self._lifecycle_tasks
            and message.control_message_type
            in (ControlMessageType.stop, ControlMessageType.sync_config)
            and message.address != self.external_address
        ):
            raise RuntimeError("Actor pool handoff is in progress")
        if (
            self._handoff
            and asyncio.current_task() not in self._lifecycle_tasks
            and message.control_message_type == ControlMessageType.sync_config
        ):
            raise RuntimeError("Actor pool handoff is in progress")
        return await super().handle_control_command(message)
