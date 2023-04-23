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

from ...backend import register_backend
from ..indigen.backend import IndigenActorBackend
from .pool import TestMainActorPool


@register_backend
class TestActorBackend(IndigenActorBackend):
    @staticmethod
    def name():
        return "test"

    @classmethod
    async def create_actor_pool(
        cls, address: str, n_process: int | None = None, **kwargs
    ):
        from ..pool import create_actor_pool

        assert n_process is not None
        return await create_actor_pool(
            address, pool_cls=TestMainActorPool, n_process=n_process, **kwargs
        )
