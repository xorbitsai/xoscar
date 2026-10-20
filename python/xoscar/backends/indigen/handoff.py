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

"""Switch actor pool managers without restarting actors.

The host remains the physical parent of its ordinary sub pools. A manager
replacement transfers a credential, never a Python object or an OS child
process handle. This preserves actor state and the existing recovery machinery.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import cloudpickle
import psutil

from ..._utils import create_actor_ref
from ...errors import ServerClosed
from ..core import ActorCaller
from ..message import (
    ControlMessage,
    ControlMessageType,
    ErrorMessage,
    ResultMessage,
    new_message_id,
)
from ..pool import _ErrorProcessor, _register_message_handler, create_actor_pool
from ..router import Router
from .pool import MainActorPool

logger = logging.getLogger(__name__)
_HANDOFF_PROTOCOL = 1


def _positive_timeout(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


@dataclass(frozen=True)
class ActorPoolHandoff:
    """Serializable local management credential. Do not publish its token."""

    address: str
    pool_id: str
    pid: int
    create_time: float
    token: str = field(repr=False)
    lease_timeout: float
    protocol: int = _HANDOFF_PROTOCOL
    python_version: tuple[int, int] = field(
        default_factory=lambda: sys.version_info[:2]
    )

    def to_dict(self) -> dict:
        """Return a JSON-compatible credential for storage or handoff."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> ActorPoolHandoff:
        value = dict(value)
        value["python_version"] = tuple(value["python_version"])
        return cls(**value)

    def _process(self) -> psutil.Process:
        process = psutil.Process(self.pid)
        if process.create_time() != self.create_time:
            raise ProcessLookupError("Handoff pool process identity changed")
        return process

    def _alive(self) -> bool:
        try:
            return self._process().status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, ProcessLookupError):
            return False


class HandoffActorPool:
    """A transferable manager handle for an independently hosted actor pool.

    Actor APIs use ``external_address`` normally. The handle only controls pool
    management; it does not emulate private attributes of a local MainActorPool.
    Its heartbeat must run on the event loop that created or took over the pool.
    """

    def __init__(self, handle: ActorPoolHandoff):
        self._handle = handle
        self._caller = ActorCaller()
        self._heartbeat_task: asyncio.Task | None = None
        self._closed = False

    @property
    def external_address(self) -> str:
        return self._handle.address

    @property
    def pid(self) -> int:
        return self._handle.pid

    @property
    def handoff_handle(self) -> ActorPoolHandoff:
        """Credential for crash recovery after this manager's lease expires."""
        return self._handle

    async def _request(self, operation: str, **payload):
        if self._closed:
            raise RuntimeError("Handoff pool handle is closed")
        request = ControlMessage(
            new_message_id(),
            self.external_address,
            ControlMessageType.pool_lifecycle,
            {
                "operation": operation,
                "pool_id": self._handle.pool_id,
                "token": self._handle.token,
                "protocol": self._handle.protocol,
                "python_version": self._handle.python_version,
                "payload": payload,
            },
        )
        result = await self._caller.call(
            Router.get_instance_or_empty(), self.external_address, request
        )
        if isinstance(result, ErrorMessage):
            raise result.error.with_traceback(result.traceback)
        return result.result

    def _start_heartbeat(self) -> None:
        self._heartbeat_task = asyncio.create_task(self._heartbeat())

    async def _heartbeat(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._handle.lease_timeout / 3)
                await asyncio.wait_for(
                    self._request("renew"), self._handle.lease_timeout / 3
                )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning(
                "Lost owner lease for handoff pool %s",
                self.external_address,
                exc_info=True,
            )

    async def _stop_heartbeat(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None

    async def _close(self) -> None:
        await self._stop_heartbeat()
        self._closed = True
        await self._caller.stop()

    @classmethod
    async def take_over(cls, handle: ActorPoolHandoff | dict) -> HandoffActorPool:
        if isinstance(handle, dict):
            handle = ActorPoolHandoff.from_dict(handle)
        if handle.protocol != _HANDOFF_PROTOCOL:
            raise ValueError("Incompatible actor pool manager handoff protocol")
        if handle.python_version != sys.version_info[:2]:
            raise ValueError("Handoff pool requires the same Python major/minor")
        handle._process()
        pool = cls(handle)
        next_token = secrets.token_urlsafe(32)
        try:
            # The host remembers the proposed credential, so a lost response
            # can be retried without issuing a second ownership transfer.
            for attempt in range(2):
                try:
                    response = await asyncio.wait_for(
                        pool._request("take_over", next_token=next_token),
                        max(1, handle.lease_timeout / 3),
                    )
                    break
                except (ConnectionError, ServerClosed, asyncio.TimeoutError):
                    if attempt:
                        raise
            pool._handle = ActorPoolHandoff.from_dict(response)
            pool._start_heartbeat()
            return pool
        except BaseException:
            await pool._close()
            raise

    async def handoff(self) -> ActorPoolHandoff:
        """Release ownership and obtain a one-use credential for the next owner.

        Actors keep serving. The host exits if no manager takes over within its
        configured orphan timeout. The returned handle permits immediate takeover.
        """
        for attempt in range(2):
            try:
                response = await self._request("handoff")
                break
            except (ConnectionError, ServerClosed):
                if attempt:
                    raise
        await self._close()
        return ActorPoolHandoff.from_dict(response)

    async def get_config(self):
        return await self._request("get_config")

    async def list_actors(self) -> list:
        """Discover current actor references without recreating any actor."""
        return await self._request("list_actors")

    async def append_sub_pool(self, **kwargs) -> str:
        return await self._request("append_sub_pool", **kwargs)

    async def remove_sub_pool(self, address: str, **kwargs) -> None:
        await self._request("remove_sub_pool", address=address, **kwargs)

    async def join(self, timeout: float | None = None) -> None:
        async def wait():
            while self._handle._alive():
                await asyncio.sleep(0.1)

        await asyncio.wait_for(wait(), timeout)

    async def stop(self, timeout: float = 30) -> None:
        """Stop the host and its sub pools, provided this lease still owns it."""
        try:
            await self._stop_heartbeat()
            await asyncio.wait_for(self._request("stop"), timeout)
            await self.join(timeout)
        finally:
            await self._close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        if not self._closed:
            await self.stop()


@_register_message_handler
class _HandoffMainActorPool(MainActorPool):
    """The stable resource owner, independent of any particular manager."""

    def configure_lifecycle(
        self,
        token: str,
        lease_timeout: float,
        orphan_timeout: float,
        start_timeout: float,
    ) -> None:
        self._owner_handle = ActorPoolHandoff(
            address=self.external_address,
            pool_id=uuid.uuid4().hex,
            pid=os.getpid(),
            create_time=psutil.Process().create_time(),
            token=token,
            lease_timeout=lease_timeout,
        )
        # Allow the creator's first RPC to initialize serialization/imports.
        # Its first renewal starts the normal, shorter owner lease.
        self._lease_deadline = time.monotonic() + start_timeout
        self._orphan_timeout = orphan_timeout
        self._awaiting_manager = False
        self._last_takeover: tuple[str, str] | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._exit_requested = asyncio.Event()

    def _owner_active(self) -> bool:
        return not self._awaiting_manager and time.monotonic() < self._lease_deadline

    def _check_envelope(self, request: dict) -> None:
        if request.get("protocol") != _HANDOFF_PROTOCOL:
            raise ValueError("Incompatible actor pool manager handoff protocol")
        if tuple(request.get("python_version", ())) != sys.version_info[:2]:
            raise ValueError("Handoff pool requires the same Python major/minor")
        if request.get("pool_id") != self._owner_handle.pool_id:
            raise RuntimeError("Handoff pool identity changed")
        if self._exit_requested.is_set():
            raise RuntimeError("Handoff pool is stopping")

    def _check_credential(self, request: dict) -> None:
        self._check_envelope(request)
        if not secrets.compare_digest(
            request.get("token", ""), self._owner_handle.token
        ):
            raise RuntimeError("Handoff pool ownership credential is stale or invalid")

    async def handle_control_command(self, message):
        if message.control_message_type != ControlMessageType.pool_lifecycle:
            if (
                message.control_message_type == ControlMessageType.stop
                and message.address == self.external_address
            ):
                with _ErrorProcessor(
                    self.external_address, message.message_id, message.protocol
                ) as processor:
                    raise RuntimeError(
                        "Stop a handoff pool host through its owner handle"
                    )
                return processor.result
            return await super().handle_control_command(message)
        with _ErrorProcessor(
            self.external_address, message.message_id, message.protocol
        ) as processor:
            request = message.content
            # Renewals must not wait behind a slow append/remove operation.
            if request.get("operation") == "renew":
                self._check_credential(request)
                if not self._owner_active():
                    raise RuntimeError("Handoff pool owner lease expired")
                self._lease_deadline = (
                    time.monotonic() + self._owner_handle.lease_timeout
                )
                result = None
            else:
                async with self._lifecycle_lock:
                    result = await self._manage(request)
            processor.result = ResultMessage(
                message.message_id, result, protocol=message.protocol
            )
        return processor.result

    async def _manage(self, request: dict):
        self._check_envelope(request)
        operation = request["operation"]
        payload = request.get("payload", {})
        if operation == "take_over" and self._last_takeover and self._owner_active():
            previous_token, next_token = self._last_takeover
            if secrets.compare_digest(
                request.get("token", ""), previous_token
            ) and secrets.compare_digest(payload.get("next_token", ""), next_token):
                return self._owner_handle.to_dict()
        self._check_credential(request)
        if operation == "take_over":
            if self._owner_active():
                raise RuntimeError("Handoff pool already has an active owner")
            if time.monotonic() >= self._lease_deadline + self._orphan_timeout:
                raise RuntimeError("Handoff pool handoff window expired")
            next_token = payload.get("next_token")
            if not isinstance(next_token, str) or len(next_token) < 32:
                raise ValueError("A takeover must propose a fresh ownership credential")
            self._last_takeover = self._owner_handle.token, next_token
            self._owner_handle = replace(self._owner_handle, token=next_token)
            self._awaiting_manager = False
            self._lease_deadline = time.monotonic() + self._owner_handle.lease_timeout
            return self._owner_handle.to_dict()
        if operation == "handoff" and self._awaiting_manager:
            return self._owner_handle.to_dict()
        if not self._owner_active():
            raise RuntimeError("Handoff pool owner lease expired or was handed off")
        if operation == "handoff":
            self._awaiting_manager = True
            self._lease_deadline = time.monotonic()
            return self._owner_handle.to_dict()
        if operation == "stop":
            self._exit_requested.set()
        elif operation == "get_config":
            return self._config
        elif operation == "list_actors":
            return [
                create_actor_ref(address, ref.uid)
                for address, actors in self._allocated_actors.items()
                for ref in actors
                if ref is not None
            ]
        elif operation == "append_sub_pool":
            address = await self.append_sub_pool(**payload)
            self._allocated_actors.setdefault(address, {})
            await self.start_monitor()
            return address
        elif operation == "remove_sub_pool":
            address = payload.pop("address")
            await self.remove_sub_pool(address, **payload)
            self._allocated_actors.pop(address, None)
        else:
            raise ValueError(f"Unknown handoff pool operation: {operation}")

    async def watch_owner(self) -> None:
        interval = min(1.0, self._owner_handle.lease_timeout / 3)
        while not self._exit_requested.is_set():
            if time.monotonic() >= self._lease_deadline + self._orphan_timeout:
                logger.info("Reclaiming unowned handoff pool %s", self.external_address)
                self._exit_requested.set()
                return
            await asyncio.sleep(interval)


async def create_handoff_pool(
    address: str,
    n_process: int | None = None,
    *,
    lease_timeout: float = 15,
    orphan_timeout: float = 300,
    start_timeout: float = 60,
    **kwargs,
) -> HandoffActorPool:
    """Start a native pool in an independent local host process."""
    lease_timeout = _positive_timeout("lease_timeout", lease_timeout)
    orphan_timeout = _positive_timeout("orphan_timeout", orphan_timeout)
    start_timeout = _positive_timeout("start_timeout", start_timeout)
    # Control-plane handoff is local. Actor traffic can still use the native
    # pool's optional transports; no Ray/dummy process handles are transferable.
    if "://" in address:
        raise ValueError("Handoff pool hosts require a native TCP address")
    directory = Path(tempfile.mkdtemp(prefix="xoscar-pool-"))
    config_path = directory / "config.pkl"
    ready_path = directory / "ready.json"
    log_path = directory / "host.log"
    token = secrets.token_urlsafe(32)
    config = {
        "address": address,
        "n_process": n_process,
        "kwargs": kwargs,
        "token": token,
        "lease_timeout": lease_timeout,
        "orphan_timeout": orphan_timeout,
        "start_timeout": start_timeout,
    }
    process = None
    pool = None
    try:
        with config_path.open("xb") as stream:
            os.chmod(config_path, 0o600)
            cloudpickle.dump(config, stream)
        # Do not use indigen.fate_sharing: this host must outlive its manager.
        options: dict[str, Any] = {}
        if sys.platform == "win32":
            options["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_BREAKAWAY_FROM_JOB
            )
        else:
            options["start_new_session"] = True
        with log_path.open("xb") as stream:
            # asyncio's subprocess transport kills a still-running process when
            # its creating event loop closes. Popen deliberately has no such
            # ownership, and the daemon reaper only waits for eventual exit.
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "xoscar.backends.indigen.handoff",
                    str(directory),
                ],
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                **options,
            )
        threading.Thread(target=process.wait, daemon=True).start()
        deadline = time.monotonic() + start_timeout
        while not ready_path.exists():
            if process.poll() is not None:
                raise RuntimeError(
                    "Handoff pool failed to start: " + log_path.read_text()[-8000:]
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out starting actor pool with manager handoff")
            await asyncio.sleep(0.05)
        handle = ActorPoolHandoff.from_dict(json.loads(ready_path.read_text()))
        pool = HandoffActorPool(handle)
        await asyncio.wait_for(pool._request("renew"), start_timeout)
        pool._start_heartbeat()
        return pool
    except BaseException:
        if pool is not None:
            await pool._close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, 10)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait)
        shutil.rmtree(directory, ignore_errors=True)
        raise


async def _serve(directory: Path) -> None:
    pool = None
    owner_watch = None
    try:
        config_path = directory / "config.pkl"
        with config_path.open("rb") as stream:
            config = cloudpickle.load(stream)
        config_path.unlink()
        pool = await create_actor_pool(
            config["address"],
            pool_cls=_HandoffMainActorPool,
            n_process=config["n_process"],
            **config["kwargs"],
        )
        pool.configure_lifecycle(
            config["token"],
            config["lease_timeout"],
            config["orphan_timeout"],
            config["start_timeout"],
        )
        # Publish readiness atomically, after all actor servers are listening.
        temporary = directory / "ready.tmp"
        temporary.write_text(json.dumps(pool._owner_handle.to_dict()))
        os.chmod(temporary, 0o600)
        temporary.replace(directory / "ready.json")
        owner_watch = asyncio.create_task(pool.watch_owner())
        await pool._exit_requested.wait()
        # Let the management response leave its channel before closing servers.
        await asyncio.sleep(0.1)
    finally:
        if owner_watch is not None:
            owner_watch.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await owner_watch
        if pool is not None:
            await pool.stop()
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(_serve(Path(sys.argv[1])))
