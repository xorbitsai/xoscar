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

from ...backend import BaseActorBackend, register_backend
from ..context import IndigenActorContext
from .driver import IndigenActorDriver
from .pool import MainActorPool

__all__ = ["IndigenActorBackend"]


@register_backend
class IndigenActorBackend(BaseActorBackend):
    @staticmethod
    def name():
        # None means Indigen is default scheme
        # ucx can be recognized as Indigen backend as well
        return [None, "ucx"]

    @staticmethod
    def get_context_cls():
        return IndigenActorContext

    @staticmethod
    def get_driver_cls():
        return IndigenActorDriver

    @classmethod
    async def create_actor_pool(
        cls, address: str, n_process: int | None = None, **kwargs
    ):
        from ..pool import create_actor_pool

        assert n_process is not None
        return await create_actor_pool(
            address, pool_cls=MainActorPool, n_process=n_process, **kwargs
        )
