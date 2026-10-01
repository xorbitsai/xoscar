# Copyright 2026 XProbe Inc.
# Licensed under the Apache License, Version 2.0.

import gc
import weakref
from datetime import timedelta

import pytest

from .. import xoscar_pygloo as xp


class MemoryStore:
    def __init__(self):
        self.values = {}

    def set_tcp(self, key, value):
        self.values[key] = value

    def get_tcp(self, key):
        return self.values[key]

    def wait(self, keys):
        assert all(key in self.values for key in keys)


def test_prefix_store_rejects_none():
    with pytest.raises(TypeError):
        xp.rendezvous.PrefixStore("group", None)


@pytest.mark.parametrize("kind", ["hash", "file", "tcp", "custom"])
def test_prefix_store_owns_underlying_store(kind, tmp_path, unused_tcp_port):
    if kind == "hash":
        store = xp.rendezvous.HashStore()
    elif kind == "file":
        store = xp.rendezvous.FileStore(str(tmp_path))
    elif kind == "tcp":
        options = xp.rendezvous.TCPStoreOptions()
        options.port = unused_tcp_port
        options.isServer = True
        options.numWorkers = 1
        store = xp.rendezvous.TCPStore("127.0.0.1", options)
    else:
        memory = MemoryStore()
        memory_ref = weakref.ref(memory)
        store = xp.rendezvous.CustomStore(memory)
        del memory

    prefix = xp.rendezvous.PrefixStore("group", store)
    nested = xp.rendezvous.PrefixStore("nested", prefix)
    nested.set("key", list("value"))
    del store, prefix
    gc.collect()
    assert nested.get("key") == list("value")
    nested.set("other", list("updated"))
    assert nested.get("other") == list("updated")
    del nested
    gc.collect()
    if kind == "custom":
        assert memory_ref() is None


def test_tcp_store_destruction_releases_port(unused_tcp_port):
    options = xp.rendezvous.TCPStoreOptions()
    options.port = unused_tcp_port
    options.isServer = True
    options.numWorkers = 1
    options.timeout = timedelta(seconds=2)
    for _ in range(3):
        store = xp.rendezvous.TCPStore("127.0.0.1", options)
        prefix = xp.rendezvous.PrefixStore("group", store)
        del store
        prefix.set("key", list("value"))
        assert prefix.get("key") == list("value")
        del prefix
        gc.collect()


def test_rendezvous_context_casts_to_context():
    context = xp.rendezvous.Context(1, 3)
    assert context.rank == 1
    assert context.size == 3
    assert context.base == 2
    timeout = timedelta(seconds=2)
    context.setTimeout(timeout)
    assert context.getTimeout() == timeout
