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
import multiprocessing

import pytest

from ....tests.core import require_unix


@require_unix
def test_tcp_store_options():
    from .. import xoscar_store as xs

    opt = xs.TCPStoreOptions()
    assert opt.numWorkers is None
    assert opt.isServer is False

    opt.numWorkers = 2
    assert opt.numWorkers == 2

    with pytest.raises(TypeError):
        opt.numWorkers = [5]


def server():
    from .. import xoscar_store as xs

    opt = xs.TCPStoreOptions()
    opt.port = 25001
    opt.numWorkers = 2
    opt.isServer = True

    store = xs.TCPStore("127.0.0.1", opt)
    val = store.get("test_key")
    assert val == b"test_12345"


def worker():
    from .. import xoscar_store as xs

    opt = xs.TCPStoreOptions()
    opt.port = 25001
    opt.numWorkers = 2
    opt.isServer = False

    store = xs.TCPStore("127.0.0.1", opt)
    store.set("test_key", b"test_12345")


@require_unix
def test_tcp_store():
    process1 = multiprocessing.Process(target=server)
    process1.start()
    process2 = multiprocessing.Process(target=worker)
    process2.start()

    process1.join()
    process2.join()
