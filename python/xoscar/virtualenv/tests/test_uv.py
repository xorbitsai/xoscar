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

import os.path
import sys
import tempfile

import pytest

from .. import get_virtual_env_manager
from ..uv import UVVirtualEnvManager


@pytest.mark.skipif(not UVVirtualEnvManager.is_available(), reason="uv not installed")
@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="skip windows because some files cannot be deleted",
)
def test_uv_virtialenv_manager():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, ".env")
        manager = get_virtual_env_manager("uv", path)

        raw_sys_path = sys.path
        try:
            manager.create_env()
            assert os.path.exists(path)
            manager.install_packages(
                ["transformers==4.40.0"],
                index_url="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
            )

            sys.path.insert(0, manager.get_lib_path())

            import transformers

            assert transformers.__version__ == "4.40.0"

            manager.remove_env()
            assert not os.path.exists(path)
        finally:
            sys.path = raw_sys_path
