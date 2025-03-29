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

import os
import sys

import pytest

import xoscar as xo

from ....constants import XOSCAR_TEMP_DIR


class CheckTransformerActor(xo.Actor):
    def __init__(self, version):
        super().__init__()
        self.version = version

    def check(self):
        import transformers

        assert transformers.__version__ == self.version


@pytest.mark.asyncio
async def test_virtual_env():
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )
    virtual_env_confs = [
        {
            "clear": True,
            "env_type": "uv",
            "env_name": "env1",
            "packages": ["transformers==4.40.0"],
            "index_url": "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
        },
        {
            "clear": True,
            "env_type": "uv",
            "env_name": "env2",
            "packages": ["transformers==4.41.0"],
            "index_url": "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
        },
    ]
    pool = await xo.create_actor_pool(
        "127.0.0.1",
        n_process=2,
        subprocess_start_method=start_method,
        virtual_env_confs=virtual_env_confs,
    )

    async with pool:
        assert os.path.exists(os.path.join(XOSCAR_TEMP_DIR, "env1"))
        assert os.path.exists(os.path.join(XOSCAR_TEMP_DIR, "env2"))
        a1 = await xo.create_actor(
            CheckTransformerActor,
            "4.40.0",
            uid="check1",
            address=pool.external_address,
            allocate_strategy=xo.allocate_strategy.ProcessIndex(1),
        )
        await a1.check()
        a2 = await xo.create_actor(
            CheckTransformerActor,
            "4.41.0",
            uid="check2",
            address=pool.external_address,
            allocate_strategy=xo.allocate_strategy.ProcessIndex(2),
        )
        await a2.check()

    assert not os.path.exists(os.path.join(XOSCAR_TEMP_DIR, "env1"))
    assert not os.path.exists(os.path.join(XOSCAR_TEMP_DIR, "env2"))
