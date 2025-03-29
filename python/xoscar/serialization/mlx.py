# Copyright 2022-2025 XProbe Inc.
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

from typing import Any, List

import numpy as np

from ..utils import lazy_import
from .core import Serializer, buffered

mx = lazy_import("mlx.core")


dtype_map = {
    "b": np.int8,
    "B": np.uint8,
    "h": np.int16,
    "H": np.uint16,
    "i": np.int32,
    "I": np.uint32,
    "q": np.int64,
    "Q": np.uint64,
    "e": np.float16,
    "f": np.float32,
    "d": np.float64,
}


class MLXSerislizer(Serializer):
    @buffered
    def serial(self, obj: "mx.array", context: dict):  # type: ignore
        mv = memoryview(obj)
        header = dict(shape=mv.shape, format=mv.format)
        if not mv.c_contiguous:
            mv = memoryview(bytes(mv))
        return (header,), [mv], True

    def deserial(self, serialized: tuple, context: dict, subs: List[Any]):
        header = serialized[0]
        shape, format = header["shape"], header["format"]
        mv = memoryview(subs[0])
        if mv.format != format:
            dtype = dtype_map.get(format, np.uint8)
            np_arr = np.frombuffer(mv, dtype=dtype).reshape(shape)  # parse
            mv = memoryview(np_arr)  # recreate memoryview
        elif mv.shape != shape:
            mv = mv.cast(format, shape)  # cast directly
        return mx.array(mv)


if mx is not None:
    MLXSerislizer.register(mx.array)
