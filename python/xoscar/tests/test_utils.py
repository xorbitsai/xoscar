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

import os
import shutil
import sys
import tempfile
import textwrap
import time

import pandas as pd
import pytest

from .. import utils


def test_string_conversion():
    s = None
    assert utils.to_binary(s) is None
    assert utils.to_str(s) is None

    s = "abcdefg"
    assert isinstance(utils.to_binary(s), bytes)
    assert utils.to_binary(s) == b"abcdefg"
    assert isinstance(utils.to_str(s), str)
    assert utils.to_str(s) == "abcdefg"

    ustr = type("ustr", (str,), {})
    assert isinstance(utils.to_str(ustr(s)), str)
    assert utils.to_str(ustr(s)) == "abcdefg"

    s = b"abcdefg"
    assert isinstance(utils.to_binary(s), bytes)
    assert utils.to_binary(s) == b"abcdefg"
    assert isinstance(utils.to_str(s), str)
    assert utils.to_str(s) == "abcdefg"

    ubytes = type("ubytes", (bytes,), {})
    assert isinstance(utils.to_binary(ubytes(s)), bytes)
    assert utils.to_binary(ubytes(s)) == b"abcdefg"

    s = "abcdefg"
    assert isinstance(utils.to_binary(s), bytes)
    assert utils.to_binary(s) == b"abcdefg"
    assert isinstance(utils.to_str(s), str)
    assert utils.to_str(s) == "abcdefg"

    with pytest.raises(TypeError):
        utils.to_binary(utils)
    with pytest.raises(TypeError):
        utils.to_str(utils)


def test_lazy_import():
    old_sys_path = sys.path
    mock_mod = textwrap.dedent(
        """
        __version__ = '0.1.0b1'
        """.strip()
    )
    mock_mod2 = textwrap.dedent(
        """
        from xoscar.utils import lazy_import
        mock_mod = lazy_import("mock_mod")

        def get_version():
            return mock_mod.__version__
        """
    )

    temp_dir = tempfile.mkdtemp(prefix="mars-utils-test-")
    sys.path += [temp_dir]
    try:
        with open(os.path.join(temp_dir, "mock_mod.py"), "w") as outf:
            outf.write(mock_mod)
        with open(os.path.join(temp_dir, "mock_mod2.py"), "w") as outf:
            outf.write(mock_mod2)

        non_exist_mod = utils.lazy_import("non_exist_mod", locals=locals())
        assert non_exist_mod is None

        non_exist_mod1 = utils.lazy_import("non_exist_mod1", placeholder=True)
        with pytest.raises(AttributeError) as ex_data:
            non_exist_mod1.meth()
        assert "required" in str(ex_data.value)

        mod = utils.lazy_import(
            "mock_mod", globals=globals(), locals=locals(), rename="mod"
        )
        assert mod is not None
        assert mod.__version__ == "0.1.0b1"

        glob = globals().copy()
        mod = utils.lazy_import("mock_mod", globals=glob, locals=locals(), rename="mod")
        glob["mod"] = mod
        assert mod is not None
        assert mod.__version__ == "0.1.0b1"
        assert type(glob["mod"]).__name__ == "module"

        import mock_mod2 as mod2

        assert type(mod2.mock_mod).__name__ != "module"
        assert mod2.get_version() == "0.1.0b1"
        assert type(mod2.mock_mod).__name__ == "module"
    finally:
        shutil.rmtree(temp_dir)
        sys.path = old_sys_path
        sys.modules.pop("mock_mod", None)
        sys.modules.pop("mock_mod2", None)


def test_type_dispatcher():
    dispatcher = utils.TypeDispatcher()

    type1 = type("Type1", (), {})
    type2 = type("Type2", (type1,), {})
    type3 = type("Type3", (), {})
    type4 = type("Type4", (type2,), {})
    type5 = type("Type5", (type4,), {})

    dispatcher.register(object, lambda x: "Object")
    dispatcher.register(type1, lambda x: "Type1")
    dispatcher.register(type4, lambda x: "Type4")
    dispatcher.register("pandas.DataFrame", lambda x: "DataFrame")
    dispatcher.register(utils.NamedType("ray", type1), lambda x: "RayType1")

    assert "Type1" == dispatcher(type2())
    assert "DataFrame" == dispatcher(pd.DataFrame())
    assert "Object" == dispatcher(type3())

    tp = utils.NamedType("ray", type1)
    assert dispatcher.get_handler(tp)(tp) == "RayType1"
    tp = utils.NamedType("ray", type2)
    assert dispatcher.get_handler(tp)(tp) == "RayType1"
    tp = utils.NamedType("xxx", type2)
    assert dispatcher.get_handler(tp)(tp) == "Type1"
    assert "Type1" == dispatcher(type2())
    tp = utils.NamedType("ray", type5)
    assert dispatcher.get_handler(tp)(tp) == "Type4"

    dispatcher.unregister(object)
    with pytest.raises(KeyError):
        dispatcher(type3())


def test_timer():
    with utils.Timer() as timer:
        time.sleep(0.1)

    assert timer.duration >= 0.1


def test_fix_all_zero_ip():
    assert utils.is_v4_zero_ip("0.0.0.0:1234") == True
    assert utils.is_v4_zero_ip("127.0.0.1:1234") == False
    assert utils.is_v6_zero_ip(":::1234") == True
    assert utils.is_v6_zero_ip("::FFFF:1234") == False
    return utils.is_v6_zero_ip("0000:0000:0000:0000:0000:0000:0000:0000:1234") == True
    return utils.is_v6_zero_ip("0:0:0:0:0:0:0:0:1234") == True
    return utils.is_v6_zero_ip("0:0:0:0:0:1234") == True
    assert utils.is_v6_zero_ip("2001:db8:3333:4444:5555:6666:7777:8888:1234") == False
    assert utils.is_v6_zero_ip("127.0.0.1:1234") == False
    assert utils.fix_all_zero_ip("127.0.0.1:1234", "127.0.0.1:5678") == "127.0.0.1:1234"
    assert utils.fix_all_zero_ip("0.0.0.0:1234", "0.0.0.0:5678") == "0.0.0.0:1234"
    assert (
        utils.fix_all_zero_ip("0.0.0.0:1234", "192.168.0.1:5678") == "192.168.0.1:1234"
    )
    assert utils.fix_all_zero_ip("127.0.0.1:1234", "0.0.0.0:5678") == "127.0.0.1:1234"
    assert (
        utils.fix_all_zero_ip(":::1234", "2001:0db8:0001:0000:0000:0ab9:C0A8:0102:5678")
        == "2001:0db8:0001:0000:0000:0ab9:C0A8:0102:1234"
    )
