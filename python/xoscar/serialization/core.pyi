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

from concurrent.futures import Executor
from typing import Any, Callable

def buffered(func: Callable) -> Callable: ...
def fast_id(obj: Any) -> int: ...

class Serializer:
    serializer_id: int
    def serial(self, obj: Any, context: dict): ...
    def deserial(self, serialized: tuple, context: dict, subs: list[Any]): ...
    def on_deserial_error(
        self,
        serialized: tuple,
        context: dict,
        subs_serialized: list,
        error_index: int,
        exc: BaseException,
    ): ...
    @classmethod
    def register(cls, obj_type, name: str | None = None): ...
    @classmethod
    def unregister(cls, obj_type): ...

class Placeholder:
    id: int
    callbacks: list[Callable]
    def __init__(self, id_: int): ...
    def __hash__(self): ...
    def __eq__(self, other): ...

def serialize(obj: Any, context: dict | None = None): ...
async def serialize_with_spawn(
    obj: Any,
    context: dict | None = None,
    spawn_threshold: int = 100,
    executor: Executor | None = None,
): ...
def deserialize(headers: list, buffers: list, context: dict | None = None): ...
def pickle_buffers(obj: Any) -> list: ...
def unpickle_buffers(buffers: list) -> Any: ...
