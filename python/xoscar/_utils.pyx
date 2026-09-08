# distutils: language = c++
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

import collections
import importlib
import itertools
import logging
import os
import threading
import time
import warnings
from functools import partial
from random import getrandbits
from typing import AsyncGenerator
from weakref import WeakSet

from libc.stdint cimport uint_fast64_t
from libc.stdlib cimport free, malloc

from .core cimport ActorRef, LocalActorRef
from .libcpp cimport mt19937_64


cdef mt19937_64 _rnd_gen
cdef bint _rnd_is_seed_set = False
_rnd_lock = threading.RLock()
NamedType = collections.namedtuple("NamedType", ["name", "type_"])
_type_dispatchers = WeakSet()
logger = logging.getLogger(__name__)


def _reset_random_lock_after_fork():
    global _rnd_lock, _rnd_is_seed_set
    _rnd_lock = threading.RLock()
    _rnd_is_seed_set = False
    for dispatcher in list(_type_dispatchers):
        (<TypeDispatcher>dispatcher)._lock = threading.RLock()
        (<TypeDispatcher>dispatcher)._loading = threading.local()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_random_lock_after_fork)

cdef class TypeDispatcher:
    def __init__(self):
        self._lock = threading.RLock()
        self._loading = threading.local()
        self._handlers = dict()
        self._lazy_handlers = dict()
        # store inherited handlers to facilitate unregistering
        self._inherit_handlers = dict()

        _type_dispatchers.add(self)

    cpdef void register(self, object type_, object handler):
        with self._lock:
            self._register(type_, handler)

    cdef _register(self, object type_, object handler):
        self._inherit_handlers.clear()
        if isinstance(type_, str):
            self._lazy_handlers[type_] = handler
        elif type(type_) is not NamedType and isinstance(type_, tuple):
            for t in type_:
                self.register(t, handler)
        else:
            self._handlers[type_] = handler

    cpdef void unregister(self, object type_):
        with self._lock:
            self._unregister(type_)

    cdef _unregister(self, object type_):
        if type(type_) is not NamedType and isinstance(type_, tuple):
            for t in type_:
                self.unregister(t)
        else:
            self._lazy_handlers.pop(type_, None)
            self._handlers.pop(type_, None)
            self._inherit_handlers.clear()

    cdef _reload_lazy_handlers(self):
        missing = object()
        active = getattr(self._loading, "keys", None)
        if active is None:
            active = self._loading.keys = set()
        with self._lock:
            pending = list(self._lazy_handlers.items())
        for k, v in pending:
            if k in active:
                # Recursive imports in this thread must not load the same key.
                continue
            active.add(k)
            mod_name, obj_name = k.rsplit('.', 1)
            try:
                with warnings.catch_warnings():
                    # the lazy imported cudf will warn no device found,
                    # when we set visible device to -1 for CPU processes,
                    # ignore the warning to not distract users
                    warnings.simplefilter("ignore")
                    mod = importlib.import_module(mod_name, __name__)
                obj_type = getattr(mod, obj_name)
            except Exception:
                # Lazy handlers are used for optional dependencies.  A module can
                # be discoverable in a child process yet fail to import there
                # because a transitive dependency or shared library is absent or
                # incompatible.  Skip only that handler so unrelated actor-pool
                # startup and serialization remain available.
                logger.debug("Failed to load lazy handler %s", k, exc_info=True)
                with self._lock:
                    if self._lazy_handlers.get(k, missing) is v:
                        self._lazy_handlers.pop(k)
                continue
            finally:
                active.remove(k)
            with self._lock:
                # Another thread may have consumed/replaced/unregistered it.
                # Keep pending keys visible during import so concurrent callers
                # can resolve them too, rather than spuriously missing a handler.
                if self._lazy_handlers.get(k, missing) is v:
                    self._lazy_handlers.pop(k)
                    self._handlers.setdefault(obj_type, v)
                    self._inherit_handlers.clear()

    cpdef get_handler(self, object type_):
        with self._lock:
            if type_ in self._handlers:
                return self._handlers[type_]
            if type_ in self._inherit_handlers:
                return self._inherit_handlers[type_]
        # Never hold the dispatcher lock while acquiring Python module locks.
        self._reload_lazy_handlers()
        with self._lock:
            return self._get_handler(type_)

    cdef _get_handler(self, object type_):
        try:
            return self._handlers[type_]
        except KeyError:
            pass

        try:
            return self._inherit_handlers[type_]
        except KeyError:
            if type(type_) is NamedType:
                named_type = partial(NamedType, type_.name)
                mro = itertools.chain(
                    *zip(map(named_type, type_.type_.__mro__),
                         type_.type_.__mro__)
                )
            else:
                mro = type_.__mro__
            for clz in mro:
                # only lookup self._handlers for mro clz
                handler = self._handlers.get(clz)
                if handler is not None:
                    if not self._lazy_handlers:
                        self._inherit_handlers[type_] = handler
                    return handler
            raise KeyError(f'Cannot dispatch type {type_}')

    def __call__(self, object obj, *args, **kwargs):
        return self.get_handler(type(obj))(obj, *args, **kwargs)

    @staticmethod
    def reload_all_lazy_handlers():
        cdef TypeDispatcher dispatcher
        for dispatcher in list(_type_dispatchers):
            dispatcher._reload_lazy_handlers()


cpdef str to_str(s, encoding='utf-8'):
    if type(s) is str:
        return <str>s
    elif isinstance(s, bytes):
        return (<bytes>s).decode(encoding)
    elif isinstance(s, str):
        return str(s)
    elif s is None:
        return s
    else:
        raise TypeError(f"Could not convert from {s} to str.")


cpdef bytes to_binary(s, encoding='utf-8'):
    if type(s) is bytes:
        return <bytes>s
    elif isinstance(s, unicode):
        return (<unicode>s).encode(encoding)
    elif isinstance(s, bytes):
        return bytes(s)
    elif s is None:
        return None
    else:
        raise TypeError(f"Could not convert from {s} to bytes.")


cpdef void reset_id_random_seed() except *:
    cdef bytes seed_bytes
    global _rnd_is_seed_set

    with _rnd_lock:
        seed_bytes = getrandbits(64).to_bytes(8, "little")
        _rnd_gen.seed((<uint_fast64_t *><char *>seed_bytes)[0])
        _rnd_is_seed_set = True


cpdef bytes new_random_id(int byte_len):
    cdef uint_fast64_t *res_ptr
    cdef uint_fast64_t res_data[4]
    cdef int i, qw_num = byte_len >> 3
    cdef bytes res

    if (qw_num << 3) < byte_len:
        qw_num += 1

    if qw_num <= 4:
        # use stack memory to accelerate
        res_ptr = res_data
    else:
        res_ptr = <uint_fast64_t *>malloc(qw_num << 3)

    try:
        with _rnd_lock:
            if not _rnd_is_seed_set:
                reset_id_random_seed()
            for i in range(qw_num):
                res_ptr[i] = _rnd_gen()
        return <bytes>((<char *>&(res_ptr[0]))[:byte_len])
    finally:
        # free memory if allocated by malloc
        if res_ptr != res_data:
            free(res_ptr)


cpdef bytes new_actor_id():
    return new_random_id(32)


cdef set _is_async_generator_typecache = set()


cdef bint is_async_generator(obj):
    cdef type tp = type(obj)
    if tp in _is_async_generator_typecache:
        return True

    if isinstance(obj, AsyncGenerator):
        if len(_is_async_generator_typecache) < 100:
            _is_async_generator_typecache.add(tp)
        return True
    else:
        return False


def create_actor_ref(*args, **kwargs):
    """
    Create an actor reference.

    Returns
    -------
    ActorRef
    """

    cdef str address
    cdef object uid
    cdef ActorRef existing_ref

    address = to_str(kwargs.pop('address', None))
    uid = kwargs.pop('uid', None)
    proxy_addresses = kwargs.pop("proxy_addresses", None)

    if kwargs:
        raise ValueError('Only `address` or `uid` keywords are supported')

    # args: [address, uid], or [address, uid, proxy_addresses]
    if 2 <= len(args) <= 3 and isinstance(args[0], (str, bytes)):
        if address:
            raise ValueError('address has been specified')
        address = to_str(args[0])
        uid = args[1]
        if len(args) == 3:
            proxy_addresses = args[2]
    elif len(args) == 1:
        tp0 = type(args[0])
        if tp0 is ActorRef or tp0 is LocalActorRef:
            existing_ref = <ActorRef>(args[0])
            uid = existing_ref.uid
            address = to_str(address or existing_ref.address)
            proxy_addresses = existing_ref.proxy_addresses
        else:
            uid = args[0]

    if uid is None:
        raise ValueError('Actor uid should be provided')

    return ActorRef(address, uid, proxy_addresses=proxy_addresses)


cdef class Timer:
    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *_):
        self.duration = time.time() - self._start
