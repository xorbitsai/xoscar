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

import pytest

from ..utils import lazy_import

cupy = lazy_import("cupy")
cudf = lazy_import("cudf")


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
