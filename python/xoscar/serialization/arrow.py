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

from typing import Any, Union

try:
    import pyarrow as pa

    pa_types = Union[pa.Table, pa.RecordBatch]
except ImportError:  # pragma: no cover
    pa = None
    pa_types = Any  # type: ignore

from .core import Serializer, buffered, pickle_buffers, unpickle_buffers


class ArrowBatchSerializer(Serializer):
    @buffered
    def serial(self, obj: pa_types, context: dict):
        header: dict = {}
        buffers = pickle_buffers(obj)
        return (header,), buffers, True

    def deserial(self, serialized: tuple, context: dict, subs: list):
        return unpickle_buffers(subs)


if pa is not None:  # pragma: no branch
    ArrowBatchSerializer.register(pa.Table)
    ArrowBatchSerializer.register(pa.RecordBatch)
