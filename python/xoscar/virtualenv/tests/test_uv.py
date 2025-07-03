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

import logging
import os.path
import sys
import tempfile
import time

import pytest

from ... import Actor, create_actor
from ...backends.indigen.pool import MainActorPool
from ...backends.pool import create_actor_pool
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
async def test_uv_virtialenv_pool():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, ".env")
        manager = get_virtual_env_manager("uv", path)

        raw_sys_path = sys.path
        try:
            manager.create_env()
            assert os.path.exists(path)
            manager.install_packages(
                ["xllamacpp==0.1.14"],
                index_url="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
            )

            pool = await create_actor_pool(
                "127.0.0.1",
                pool_cls=MainActorPool,
                n_process=0,
            )
            sub_external_address = await pool.append_sub_pool(
                start_python=manager.get_python_path()
            )

            class DummyActor(Actor):
                @staticmethod
                def test():
                    import xllamacpp

                    assert xllamacpp.__version__ == "0.1.14"
                    return sys.executable

            ref = await create_actor(DummyActor, address=sub_external_address)
            assert ref is not None
            assert ref.address == sub_external_address
            assert await ref.test() == manager.get_python_path()

            with pytest.raises((ImportError, AssertionError)):
                import xllamacpp

                assert xllamacpp.__version__ == "0.1.14"

            await pool.remove_sub_pool(sub_external_address)
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


@pytest.mark.skipif(not UVVirtualEnvManager.is_available(), reason="uv not installed")
@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="skip windows because some files cannot be deleted",
)
def test_uv_virtualenv_manager_with_log(caplog):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, ".env")
        manager = get_virtual_env_manager("uv", path)

        raw_sys_path = sys.path
        try:
            # Create the virtual environment
            manager.create_env()
            assert os.path.exists(path)

            # Start logging
            caplog.set_level(logging.INFO)

            # Install package with log enabled
            manager.install_packages(
                ["packaging==24.0"],
                index_url="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
                log=True,
            )

            if "packaging" in sys.modules:
                del sys.modules["packaging"]

            # Verify it's installed
            sys.path.insert(0, manager.get_lib_path())
            import packaging

            assert packaging.__version__ == "24.0"

            assert not manager._resolve_install_plan(["packaging"], {})
            assert manager._resolve_install_plan(["packaging==25.0"], {}) == [
                "packaging==25.0"
            ]

            # Check that logs are indeed captured
            assert any(
                "Installed 1 package in" in record.message for record in caplog.records
            )

            manager.remove_env()
            assert not os.path.exists(path)
        finally:
            sys.path = raw_sys_path


@pytest.mark.skipif(not UVVirtualEnvManager.is_available(), reason="uv not installed")
@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="skip windows because some files cannot be deleted",
)
def test_uv_virtualenv_manager_skip_system_package(caplog):
    import numpy

    system_numpy_version = numpy.__version__

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, ".env")
        manager = get_virtual_env_manager("uv", path)

        raw_sys_path = sys.path
        try:
            manager.create_env()
            assert os.path.exists(path)

            caplog.set_level(logging.INFO)

            # Install transformers and system numpy with skip_installed=True
            manager.install_packages(
                ["transformers==4.40.0", "#system_numpy#"],
                index_url="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
                skip_installed=True,
                log=True,
            )

            sys.path.insert(0, manager.get_lib_path())

            # Import and verify versions of transformers and numpy
            import numpy as numpy_in_env
            import transformers

            assert transformers.__version__ == "4.40.0"
            assert numpy_in_env.__version__ == system_numpy_version

            caplog.clear()

            # Confirm numpy is skipped (no installation needed)
            manager.install_packages(
                ["transformers>=4.40.0", "#system_numpy#"],
                index_url="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
                skip_installed=True,
                log=True,
            )

            caplog_lines = [r.message for r in caplog.records]
            # should be no logs since no packages to install
            assert "All required packages are already installed." in caplog_lines

        finally:
            sys.path = raw_sys_path
