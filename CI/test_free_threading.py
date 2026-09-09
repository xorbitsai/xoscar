"""Run in a free-threaded interpreter without forcing PYTHON_GIL=0.

Importing an incompatible extension must fail the GIL assertions rather than
silently turning this into a test of a GIL-enabled interpreter.
"""

import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="requires a free-threaded interpreter",
)


def test_imports_keep_gil_disabled():
    import xoscar
    import xoscar.collective.xoscar_pygloo
    if sys.platform != "win32":
        import uvloop

    assert not sys._is_gil_enabled()


def test_concurrent_serialization_and_ids():
    from xoscar._utils import new_actor_id
    from xoscar.context import get_context
    from xoscar.serialization import deserialize, serialize

    def worker(index):
        ids = set()
        for i in range(500):
            value = {"index": index, "values": [i, b"payload", None]}
            header, buffers = serialize(value)
            assert deserialize(header, buffers) == value
            ids.add(new_actor_id())
        return ids, get_context()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker, range(8)))
    assert len(set.union(*(item[0] for item in results))) == 4000
    assert len({id(item[1]) for item in results}) == 1
    assert not sys._is_gil_enabled()


def test_concurrent_lazy_dispatch():
    import threading

    from xoscar._utils import TypeDispatcher

    dispatcher = TypeDispatcher()
    dispatcher.register("collections.OrderedDict", "handler")
    barrier = threading.Barrier(8)

    def worker(_):
        from collections import OrderedDict

        barrier.wait()
        return dispatcher.get_handler(OrderedDict)

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert list(executor.map(worker, range(8))) == ["handler"] * 8
