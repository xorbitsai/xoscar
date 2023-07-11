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

import fnmatch
import itertools

import pytest

from ..utils import is_linux, is_windows, lazy_import

cupy = lazy_import("cupy")
cudf = lazy_import("cudf")
ucx = lazy_import("ucp")


def require_cupy(func):
    if pytest:
        func = pytest.mark.cuda(func)
    func = pytest.mark.skipif(cupy is None, reason="cupy not installed")(func)
    return func


def require_cudf(func):
    if pytest:
        func = pytest.mark.cuda(func)
    func = pytest.mark.skipif(cudf is None, reason="cudf not installed")(func)
    return func


def require_ucx(func):
    if pytest:
        func = pytest.mark.ucx(func)
    func = pytest.mark.skipif(ucx is None, reason="ucx not installed")(func)
    return func


def require_unix(func):
    if pytest:
        func = pytest.mark.unix(func)

    func = pytest.mark.skipif(is_windows(), reason="only unix is supported")(func)
    return func


def require_linux(func):
    if pytest:
        func = pytest.mark.linux(func)

    func = pytest.mark.skipif(not is_linux(), reason="only linux is supported")(func)
    return func


DICT_NOT_EMPTY = type("DICT_NOT_EMPTY", (object,), {})  # is check works for deepcopy


def check_dict_structure_same(a, b, prefix=None):
    def _p(k):
        if prefix is None:
            return k
        return ".".join(str(i) for i in prefix + [k])

    for ai, bi in itertools.zip_longest(
        a.items(), b.items(), fillvalue=("_KEY_NOT_EXISTS_", None)
    ):
        if ai[0] != bi[0]:
            if "*" in ai[0]:
                pattern, target = ai[0], bi[0]
            elif "*" in bi[0]:
                pattern, target = bi[0], ai[0]
            else:
                raise KeyError(f"Key {_p(ai[0])} != {_p(bi[0])}")
            if not fnmatch.fnmatch(target, pattern):
                raise KeyError(f"Key {_p(target)} not match {_p(pattern)}")

        if ai[1] is DICT_NOT_EMPTY:
            target = bi[1]
        elif bi[1] is DICT_NOT_EMPTY:
            target = ai[1]
        else:
            target = None
        if target is not None:
            if not isinstance(target, dict):
                raise TypeError(f"Value type of {_p(ai[0])} is not a dict.")
            if not target:
                raise TypeError(f"Value of {_p(ai[0])} empty.")
            continue

        if type(ai[1]) is not type(bi[1]):
            raise TypeError(f"Value type of {_p(ai[0])} mismatch {ai[1]} != {bi[1]}")
        if isinstance(ai[1], dict):
            check_dict_structure_same(
                ai[1], bi[1], [ai[0]] if prefix is None else prefix + [ai[0]]
            )
