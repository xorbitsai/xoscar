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
import time

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


@pytest.mark.skipif(not UVVirtualEnvManager.is_available(), reason="uv not installed")
@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="skip windows because some files cannot be deleted",
)
def test_uv_virtualenv_manager_with_cancel():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, ".env")
        manager = get_virtual_env_manager("uv", path)

        raw_sys_path = sys.path
        try:
            # Create the virtual environment
            manager.create_env()
            assert os.path.exists(path)

            # Start the package installation in a separate thread
            import threading

            def install_task():
                manager.install_packages(
                    ["pygraphviz==1.8"],
                    index_url="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
                )

            # Start the installation thread
            install_thread = threading.Thread(target=install_task)
            install_thread.start()

            # Wait a bit to ensure installation has started
            time.sleep(1)

            # Call cancel_install to interrupt the installation process
            manager.cancel_install()

            # Wait for the installation thread to finish
            install_thread.join()

            # Ensure the installation was cancelled and the package wasn't installed
            with pytest.raises(ImportError):
                import pygraphviz  # noqa: F401 # pylint: disable=unused-import

            # Clean up the virtual environment
            manager.remove_env()
            assert not os.path.exists(path)

        finally:
            sys.path = raw_sys_path
