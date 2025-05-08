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
import dataclasses
import functools
import importlib
import importlib.util as importlib_utils
import inspect
import io
import logging
import os
import random
import socket
import sys
import time
import uuid
from abc import ABC
from functools import lru_cache
from types import TracebackType
from typing import Callable, Type, Union

from ._utils import (  # noqa: F401 # pylint: disable=unused-import
    NamedType,
    Timer,
    TypeDispatcher,
    to_binary,
    to_str,
)

# Please refer to https://bugs.python.org/issue41451
try:

    class _Dummy(ABC):
        __slots__ = ("__weakref__",)

    abc_type_require_weakref_slot = True
except TypeError:
    abc_type_require_weakref_slot = False


logger = logging.getLogger(__name__)


_memory_size_indices = {"": 0, "k": 1, "m": 2, "g": 3, "t": 4}


def parse_readable_size(value: str | int | float) -> tuple[float, bool]:
    if isinstance(value, (int, float)):
        return float(value), False

    value = value.strip().lower()
    num_pos = 0
    while num_pos < len(value) and value[num_pos] in "0123456789.-":
        num_pos += 1

    value, suffix = value[:num_pos], value[num_pos:]
    suffix = suffix.strip()
    if suffix.endswith("%"):
        return float(value) / 100, True

    try:
        return float(value) * (1024 ** _memory_size_indices[suffix[:1]]), False
    except (ValueError, KeyError):
        raise ValueError(f"Unknown limitation value: {value}")


def wrap_exception(
    exc: BaseException,
    bases: tuple[Type] | tuple | None = None,
    wrap_name: str | None = None,
    message: str | None = None,
    traceback: TracebackType | None = None,
    attr_dict: dict | None = None,
) -> BaseException:
    """Generate an exception wraps the cause exception."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        return getattr(exc, item)

    def __str__(self):
        return message or super(type(self), self).__str__()

    traceback = traceback or exc.__traceback__
    bases = bases or ()
    attr_dict = attr_dict or {}
    attr_dict.update(
        {
            "__init__": __init__,
            "__getattr__": __getattr__,
            "__str__": __str__,
            "__wrapname__": wrap_name,
            "__wrapped__": exc,
            "__module__": type(exc).__module__,
            "__cause__": exc.__cause__,
            "__context__": exc.__context__,
            "__suppress_context__": exc.__suppress_context__,
            "args": exc.args,
        }
    )
    new_exc_type = type(type(exc).__name__, bases + (type(exc),), attr_dict)
    return new_exc_type().with_traceback(traceback)


# from https://github.com/ericvsmith/dataclasses/blob/master/dataclass_tools.py
# released under Apache License 2.0
def dataslots(cls):
    # Need to create a new class, since we can't set __slots__
    #  after a class has been created.

    # Make sure __slots__ isn't already set.
    if "__slots__" in cls.__dict__:  # pragma: no cover
        raise TypeError(f"{cls.__name__} already specifies __slots__")

    # Create a new dict for our new class.
    cls_dict = dict(cls.__dict__)
    field_names = tuple(f.name for f in dataclasses.fields(cls))
    cls_dict["__slots__"] = field_names
    for field_name in field_names:
        # Remove our attributes, if present. They'll still be
        #  available in _MARKER.
        cls_dict.pop(field_name, None)
    # Remove __dict__ itself.
    cls_dict.pop("__dict__", None)
    # And finally create the class.
    qualname = getattr(cls, "__qualname__", None)
    cls = type(cls)(cls.__name__, cls.__bases__, cls_dict)
    if qualname is not None:
        cls.__qualname__ = qualname
    return cls


def implements(f: Callable):
    def decorator(g):
        g.__doc__ = f.__doc__
        return g

    return decorator


class classproperty:
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


LOW_PORT_BOUND = 10000
HIGH_PORT_BOUND = 65535
_local_occupied_ports: set = set()


def _get_ports_from_netstat() -> set[int]:
    import subprocess

    while True:
        p = subprocess.Popen("netstat -a -n -p tcp".split(), stdout=subprocess.PIPE)
        try:
            outs, _ = p.communicate(timeout=5)
            lines = outs.split(to_binary(os.linesep))
            occupied = set()
            for line in lines:
                if b"." not in line:
                    continue
                line_str: str = to_str(line)
                for part in line_str.split():
                    # in windows, netstat uses ':' to separate host and port
                    part = part.replace(":", ".")
                    if "." in part:
                        _, port_str = part.rsplit(".", 1)
                        if port_str == "*":
                            continue
                        port = int(port_str)
                        if LOW_PORT_BOUND <= port <= HIGH_PORT_BOUND:
                            occupied.add(int(port_str))
                        break
            return occupied
        except subprocess.TimeoutExpired:
            p.kill()
            continue


def get_next_port(typ: int | None = None, occupy: bool = True) -> int:
    import psutil

    if sys.platform.lower().startswith("win"):
        occupied = _get_ports_from_netstat()
    else:
        try:
            conns = psutil.net_connections()
            typ = typ or socket.SOCK_STREAM
            occupied = set(
                sc.laddr.port
                for sc in conns
                if sc.type == typ and LOW_PORT_BOUND <= sc.laddr.port <= HIGH_PORT_BOUND
            )
        except psutil.AccessDenied:
            occupied = _get_ports_from_netstat()

    occupied.update(_local_occupied_ports)
    random.seed(uuid.uuid1().bytes)
    randn = random.randint(0, 100000000)

    idx = int(randn % (1 + HIGH_PORT_BOUND - LOW_PORT_BOUND - len(occupied)))
    for i in range(LOW_PORT_BOUND, HIGH_PORT_BOUND + 1):
        if i in occupied:
            continue
        if idx == 0:
            if occupy:
                _local_occupied_ports.add(i)
            return i
        idx -= 1
    raise SystemError("No ports available.")


def lazy_import(
    name: str,
    package: str | None = None,
    globals: dict | None = None,  # pylint: disable=redefined-builtin
    locals: dict | None = None,  # pylint: disable=redefined-builtin
    rename: str | None = None,
    placeholder: bool = False,
):
    rename = rename or name
    prefix_name = name.split(".", 1)[0]
    globals = globals or inspect.currentframe().f_back.f_globals  # type: ignore

    class LazyModule:
        def __init__(self):
            self._on_loads = []

        def __getattr__(self, item):
            if item.startswith("_pytest") or item in ("__bases__", "__test__"):
                raise AttributeError(item)

            real_mod = importlib.import_module(name, package=package)
            if rename in globals:
                globals[rename] = real_mod
            elif locals is not None:
                locals[rename] = real_mod
            ret = getattr(real_mod, item)
            for on_load_func in self._on_loads:
                on_load_func()
            # make sure on_load hooks only executed once
            self._on_loads = []
            return ret

        def add_load_handler(self, func: Callable):
            self._on_loads.append(func)
            return func

    if importlib_utils.find_spec(prefix_name) is not None:
        return LazyModule()
    elif placeholder:
        return ModulePlaceholder(prefix_name)
    else:
        return None


def lazy_import_on_load(lazy_mod):
    def wrapper(fun):
        if lazy_mod is not None and hasattr(lazy_mod, "add_load_handler"):
            lazy_mod.add_load_handler(fun)
        return fun

    return wrapper


class ModulePlaceholder:
    def __init__(self, mod_name: str):
        self._mod_name = mod_name

    def _raises(self):
        raise AttributeError(f"{self._mod_name} is required but not installed.")

    def __getattr__(self, key):
        self._raises()

    def __call__(self, *_args, **_kwargs):
        self._raises()


def patch_asyncio_task_create_time():  # pragma: no cover
    new_loop = False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        new_loop = True
    loop_class = loop.__class__
    # Save raw loop_class.create_task and make multiple apply idempotent
    loop_create_task = getattr(
        patch_asyncio_task_create_time, "loop_create_task", loop_class.create_task
    )
    patch_asyncio_task_create_time.loop_create_task = loop_create_task

    def new_loop_create_task(*args, **kwargs):
        task = loop_create_task(*args, **kwargs)
        task.__xoscar_asyncio_task_create_time__ = time.time()
        return task

    if loop_create_task is not new_loop_create_task:
        loop_class.create_task = new_loop_create_task
    if not new_loop and loop.create_task is not new_loop_create_task:
        loop.create_task = functools.partial(new_loop_create_task, loop)


async def asyncio_task_timeout_detector(
    check_interval: int, task_timeout_seconds: int, task_exclude_filters: list[str]
):
    task_exclude_filters.append("asyncio_task_timeout_detector")
    while True:  # pragma: no cover
        await asyncio.sleep(check_interval)
        loop = asyncio.get_running_loop()
        current_time = (
            time.time()
        )  # avoid invoke `time.time()` frequently if we have plenty of unfinished tasks.
        for task in asyncio.all_tasks(loop=loop):
            # Some task may be create before `patch_asyncio_task_create_time` applied, take them as never timeout.
            create_time = getattr(
                task, "__xoscar_asyncio_task_create_time__", current_time
            )
            if current_time - create_time >= task_timeout_seconds:
                stack = io.StringIO()
                task.print_stack(file=stack)
                task_str = str(task)
                if any(
                    excluded_task in task_str for excluded_task in task_exclude_filters
                ):
                    continue
                logger.warning(
                    """Task %s in event loop %s doesn't finish in %s seconds. %s""",
                    task,
                    loop,
                    time.time() - create_time,
                    stack.getvalue(),
                )


def register_asyncio_task_timeout_detector(
    check_interval: int | None = None,
    task_timeout_seconds: int | None = None,
    task_exclude_filters: list[str] | None = None,
) -> asyncio.Task | None:  # pragma: no cover
    """Register a asyncio task which print timeout task periodically."""
    check_interval = check_interval or int(
        os.environ.get("XOSCAR_DEBUG_ASYNCIO_TASK_TIMEOUT_CHECK_INTERVAL", -1)
    )
    if check_interval > 0:
        patch_asyncio_task_create_time()
        task_timeout_seconds = task_timeout_seconds or int(
            os.environ.get("XOSCAR_DEBUG_ASYNCIO_TASK_TIMEOUT_SECONDS", check_interval)
        )
        if not task_exclude_filters:
            # Ignore Xoscar by default since it has some long-running coroutines.
            task_exclude_filter = os.environ.get(
                "XOSCAR_DEBUG_ASYNCIO_TASK_EXCLUDE_FILTERS", "xoscar"
            )
            task_exclude_filters = task_exclude_filter.split(";")
        if sys.version_info[:2] < (3, 7):
            logger.warning(
                "asyncio tasks timeout detector is not supported under python %s",
                sys.version,
            )
        else:
            loop = asyncio.get_running_loop()
            logger.info(
                "Create asyncio tasks timeout detector with check_interval %s task_timeout_seconds %s "
                "task_exclude_filters %s",
                check_interval,
                task_timeout_seconds,
                task_exclude_filters,
            )
            return loop.create_task(
                asyncio_task_timeout_detector(
                    check_interval, task_timeout_seconds, task_exclude_filters
                )
            )
    else:
        return None


def ensure_coverage():
    # make sure coverage is handled when starting with subprocess.Popen
    if (
        not sys.platform.startswith("win") and "COV_CORE_SOURCE" in os.environ
    ):  # pragma: no cover
        try:
            from pytest_cov.embed import cleanup_on_sigterm
        except ImportError:
            pass
        else:
            cleanup_on_sigterm()


def retry_callable(
    callable_,
    ex_type: type = Exception,
    wait_interval=1,
    max_retries=-1,
    sync: bool | None = None,
):
    if inspect.iscoroutinefunction(callable_) or sync is False:

        @functools.wraps(callable)
        async def retry_call(*args, **kwargs):
            num_retried = 0
            while max_retries < 0 or num_retried < max_retries:
                num_retried += 1
                try:
                    return await callable_(*args, **kwargs)
                except ex_type:
                    await asyncio.sleep(wait_interval)

    else:

        @functools.wraps(callable)
        def retry_call(*args, **kwargs):
            num_retried = 0
            ex = None
            while max_retries < 0 or num_retried < max_retries:
                num_retried += 1
                try:
                    return callable_(*args, **kwargs)
                except ex_type as e:
                    ex = e
                    time.sleep(wait_interval)
            assert ex is not None
            raise ex  # pylint: disable-msg=E0702

    return retry_call


_cupy = lazy_import("cupy")
_rmm = lazy_import("rmm")


def is_cuda_buffer(cuda_buffer: Union["_cupy.ndarray", "_rmm.DeviceBuffer"]) -> bool:  # type: ignore
    return hasattr(cuda_buffer, "__cuda_array_interface__")


def is_windows():
    return sys.platform.startswith("win")


def is_linux():
    return sys.platform.startswith("linux")


@lru_cache
def is_py_312():
    return sys.version_info[:2] == (3, 12)


def is_v4_zero_ip(ip_port_addr: str) -> bool:
    return ip_port_addr.split("://")[-1].startswith("0.0.0.0:")


def is_v6_zero_ip(ip_port_addr: str) -> bool:
    # tcp6 addr ":::123", ":: means all zero"
    arr = ip_port_addr.split("://")[-1].split(":")
    if len(arr) <= 2:  # Not tcp6 or udp6
        return False
    for part in arr[0:-1]:
        if part != "":
            if int(part, 16) != 0:
                return False
    return True


def is_zero_ip(ip_port_addr: str) -> bool:
    return is_v4_zero_ip(ip_port_addr) or is_v6_zero_ip(ip_port_addr)


def is_v6_ip(ip_port_addr: str) -> bool:
    arr = ip_port_addr.split("://", 1)[-1].split(":")
    return len(arr) > 1


def fix_all_zero_ip(remote_addr: str, connect_addr: str) -> str:
    """
    Use connect_addr to fix ActorRef.address return by remote server.
    When remote server listen on "0.0.0.0:port" or ":::port", it will return ActorRef.address set to listening addr,
    it cannot be use by client for the following interaction unless we fix it.
    (client will treat 0.0.0.0 as 127.0.0.1)

    NOTE: Server might return a different addr from a pool for load-balance purpose.
    """
    if remote_addr == connect_addr:
        return remote_addr
    if not is_v4_zero_ip(remote_addr) and not is_v6_zero_ip(remote_addr):
        # Remote server returns on non-zero ip
        return remote_addr
    if is_v4_zero_ip(connect_addr) or is_v6_zero_ip(connect_addr):
        # Client connect to local server
        return remote_addr
    remote_port = remote_addr.split(":")[-1]
    connect_ip = ":".join(connect_addr.split(":")[0:-1])  # Remote the port
    return f"{connect_ip}:{remote_port}"
