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
from pathlib import Path
from unittest import mock

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
            manager.create_env(python_path=Path(sys.executable))
            assert os.path.exists(path)
            manager.install_packages(
                ["transformers==4.50.0"],
                index_url="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
            )

            sys.path.insert(0, manager.get_lib_path())

            import transformers

            assert transformers.__version__ == "4.50.0"

            manager.remove_env()
            assert not os.path.exists(path)
        finally:
            sys.path = raw_sys_path


@pytest.mark.skipif(not UVVirtualEnvManager.is_available(), reason="uv not installed")
@pytest.mark.skipif(
    sys.platform.startswith("win") or sys.version_info[:2] >= (3, 13),
    reason="skip windows because some files cannot be deleted, "
    "and skip python 3.13 since xllamacpp does not support",
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

            assert not manager._resolve_install_plan(
                ["packaging"], {}, index_url="https://pypi.org/simple"
            )
            assert manager._resolve_install_plan(
                ["packaging==25.0"], {}, index_url="https://pypi.org/simple"
            ) == ["packaging==25.0"]

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
            manager.create_env(python_path=Path(sys.executable))
            assert os.path.exists(path)

            caplog.set_level(logging.INFO)

            # Install transformers and system numpy with skip_installed=True
            manager.install_packages(
                ["transformers==4.50.0", "#system_numpy#"],
                index_url="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
                skip_installed=True,
                log=True,
            )

            sys.path.insert(0, manager.get_lib_path())

            # Import and verify versions of transformers and numpy
            import numpy as numpy_in_env
            import transformers

            assert transformers.__version__ == "4.50.0"
            assert numpy_in_env.__version__ == system_numpy_version

            caplog.clear()

            # Confirm numpy is skipped (no installation needed)
            manager.install_packages(
                ["transformers>=4.50.0", "#system_numpy#"],
                index_url="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
                skip_installed=True,
                log=True,
            )

            caplog_lines = [r.message for r in caplog.records]
            # should be no logs since no packages to install
            assert "All required packages are already installed." in caplog_lines

        finally:
            sys.path = raw_sys_path


@pytest.mark.skipif(not UVVirtualEnvManager.is_available(), reason="uv not installed")
@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="skip windows because some files cannot be deleted",
)
def test_uv_virtualenv_exists_env():
    """Test exists_env method and create_env with exists parameter."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, ".env")
        manager = get_virtual_env_manager("uv", path)

        # Initially environment should not exist
        assert not manager.exists_env()

        # Create environment
        manager.create_env()
        assert manager.exists_env()
        assert os.path.exists(path)

        # Test exists="error" - should raise FileExistsError
        with pytest.raises(FileExistsError, match="Virtual environment already exists"):
            manager.create_env(exists="error")

        # Test default behavior (exists="ignore") - should skip creation
        manager.create_env()
        assert manager.exists_env()

        # Test exists="clear" - should recreate environment
        manager.create_env(exists="clear")
        assert manager.exists_env()

        # Test invalid exists parameter
        with pytest.raises(ValueError, match="Invalid exists option"):
            manager.create_env(exists="invalid")

        manager.remove_env()
        assert not manager.exists_env()
        assert not os.path.exists(path)


def test_split_specs_with_extras():
    """Specs with extras must be resolved even if the base package is installed,
    since a bare install may lack the extra dependencies. The satisfied base
    distribution is pinned so the resolver only adds the missing extra deps."""
    installed = {"sglang": "0.5.7", "requests": "2.28.0", "oldpkg": "1.0"}

    keep, to_resolve, pinned = UVVirtualEnvManager._split_specs(
        [
            "sglang[diffusion]",
            "requests[socks]>=2.0",
            "oldpkg[x]>=2.0",
            "newpkg[y]",
        ],
        installed,
    )

    assert keep == []
    # all extras specs go to the resolver despite the base packages being installed
    assert to_resolve == [
        "sglang[diffusion]",
        "requests[socks]>=2.0",
        "oldpkg[x]>=2.0",
        "newpkg[y]",
    ]
    # installed base distributions satisfying the specifier are pinned so the
    # resolver won't upgrade them; unsatisfied or missing ones are left free
    assert pinned == {"sglang": "0.5.7", "requests": "2.28.0"}


@pytest.fixture
def uv_manager(tmp_path):
    env_path = tmp_path / "test_env"
    return UVVirtualEnvManager(env_path)


@pytest.mark.parametrize(
    "has_cuda, cuda_version, cuda_arch, has_npu, input_pkgs, expected_pkgs",
    [
        (
            True,
            "12.1",
            "sm_80",
            False,
            [
                "torch==2.0 ; has_cuda",
                "xformers ; cuda_version >= '12.0'",
                "cpu-lib ; not has_cuda",
                "npu-lib ; has_npu",
            ],
            ["torch==2.0", "xformers"],
        ),
        (
            False,
            None,
            None,
            False,
            ["torch==2.0 ; has_cuda", "cpu-lib ; not has_cuda", "npu-lib ; has_npu"],
            ["cpu-lib"],
        ),
        (
            False,
            None,
            None,
            True,
            ["torch==2.0 ; has_cuda", "cpu-lib ; not has_cuda", "npu-lib ; has_npu"],
            ["cpu-lib", "npu-lib"],
        ),
    ],
)
def test_install_packages_marker_filtering(
    uv_manager, has_cuda, cuda_version, cuda_arch, has_npu, input_pkgs, expected_pkgs
):
    # Mock environment detection functions and subprocess.Popen to avoid real installs
    with mock.patch(
        "xoscar.virtualenv.core.check_cuda_available", return_value=has_cuda
    ), mock.patch(
        "xoscar.virtualenv.core.get_cuda_version", return_value=cuda_version
    ), mock.patch(
        "xoscar.virtualenv.core.get_cuda_arch", return_value=cuda_arch
    ), mock.patch(
        "xoscar.virtualenv.core.check_npu_available", return_value=has_npu
    ), mock.patch(
        "subprocess.Popen"
    ) as mock_popen, mock.patch.object(
        UVVirtualEnvManager, "_get_uv_path", return_value="uv"
    ):

        # Mock the process and its return code
        process = mock.Mock()
        process.wait.return_value = 0
        mock_popen.return_value = process

        # Call install_packages with the input package list
        uv_manager.install_packages(input_pkgs)

        # Extract the command line passed to subprocess.Popen
        cmd = mock_popen.call_args[0][0]
        cmd_str = " ".join(cmd)

        # Assert that all expected packages appear in the command
        for pkg in expected_pkgs:
            assert pkg in cmd_str

        # Assert that packages that should be excluded are not present
        for pkg in input_pkgs:
            base_pkg = pkg.split(";")[0].strip()  # Strip marker if present
            if base_pkg not in expected_pkgs:
                assert base_pkg not in cmd_str


@pytest.mark.skipif(not UVVirtualEnvManager.is_available(), reason="uv not installed")
@pytest.mark.parametrize(
    "variables, input_pkgs, expected_pkgs",
    [
        # Test string variable substitution
        (
            {"engine": "vllm"},
            [
                'transformers==4.30.0; #engine# == "vllm"',
                'requests==2.28.0; #engine# == "sglang"',
            ],
            ["transformers==4.30.0"],
        ),
        # Test numeric variable substitution
        (
            {"count": 10},
            ["pkg1==1.0; #count# > 5", "pkg2==2.0; #count# < 5"],
            ["pkg1==1.0"],
        ),
        # Test boolean variable substitution
        (
            {"enabled": True},
            ["pkg1==1.0; #enabled# == True", "pkg2==2.0; #enabled# == False"],
            ["pkg1==1.0"],
        ),
        # Test multiple variables with AND
        (
            {"engine": "vllm", "mode": "local"},
            [
                'pkg1==1.0; #engine# == "vllm" and #mode# == "local"',
                'pkg2==2.0; #engine# == "sglang"',
            ],
            ["pkg1==1.0"],
        ),
        # Test mixed: variables with standard markers
        (
            {"engine": "vllm"},
            ['pkg1==1.0; #engine# == "vllm" and python_version >= "3.8"'],
            ["pkg1==1.0"],
        ),
        # Test no match - empty result
        (
            {"engine": "sglang"},
            ['pkg1==1.0; #engine# == "vllm"', 'pkg2==2.0; #engine# == "transformers"'],
            [],
        ),
    ],
)
def test_variable_substitution_in_install_packages(
    variables, input_pkgs, expected_pkgs
):
    """Test that #var# placeholders are correctly substituted during install_packages."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, ".env")
        uv_manager = UVVirtualEnvManager(Path(path))

        with mock.patch("subprocess.Popen") as mock_popen, mock.patch.object(
            UVVirtualEnvManager, "_get_uv_path", return_value="uv"
        ):
            # Mock the process and its return code
            process = mock.Mock()
            process.wait.return_value = 0
            mock_popen.return_value = process

            # Call install_packages with variables
            uv_manager.install_packages(input_pkgs, **variables)

            # If no packages expected, subprocess should not be called
            if not expected_pkgs:
                assert (
                    mock_popen.call_count == 0
                ), "Expected no subprocess call when no packages match"
                return

            # Extract the command line passed to subprocess.Popen
            cmd = mock_popen.call_args[0][0]
            cmd_str = " ".join(cmd)

            # Assert that all expected packages appear in the command
            for pkg in expected_pkgs:
                assert (
                    pkg in cmd_str
                ), f"Expected package '{pkg}' not found in command: {cmd_str}"

            # Assert that packages that should be excluded are not present
            for pkg in input_pkgs:
                base_pkg = pkg.split(";")[0].strip()  # Strip marker if present
                if base_pkg not in expected_pkgs:
                    assert (
                        base_pkg not in cmd_str
                    ), f"Unexpected package '{base_pkg}' found in command: {cmd_str}"


def test_install_packages_retry_without_system_pins(uv_manager, caplog):
    # first uv invocation fails (simulating a resolver conflict with the
    # host-aligned pin), the retry without pins succeeds
    calls = []

    def fake_popen(cmd, *args, **kwargs):
        calls.append(list(cmd))
        process = mock.Mock()
        process.wait.return_value = 1 if len(calls) == 1 else 0
        return process

    with mock.patch("importlib.metadata.version", return_value="1.26.4"), mock.patch(
        "subprocess.Popen", side_effect=fake_popen
    ), mock.patch.object(UVVirtualEnvManager, "_get_uv_path", return_value="uv"):
        with caplog.at_level(logging.WARNING, logger="xoscar.virtualenv.uv"):
            uv_manager.install_packages(["#system_numpy#", "vllm==0.21.0"])

    assert len(calls) == 2
    assert "numpy==1.26.4" in calls[0]
    assert "numpy==1.26.4" not in calls[1]
    assert "numpy" in calls[1]
    # explicit user/spec pins are never dropped
    assert "vllm==0.21.0" in calls[1]
    assert "retrying without them" in caplog.text


def test_install_packages_failure_without_system_pins_reraises(uv_manager):
    import subprocess

    with mock.patch("subprocess.Popen") as mock_popen, mock.patch.object(
        UVVirtualEnvManager, "_get_uv_path", return_value="uv"
    ):
        process = mock.Mock()
        process.wait.return_value = 1
        mock_popen.return_value = process

        with pytest.raises(subprocess.CalledProcessError):
            uv_manager.install_packages(["vllm==0.21.0"])
        # no system pins involved, so no retry
        assert mock_popen.call_count == 1


def test_install_packages_retry_also_fails(uv_manager):
    import subprocess

    with mock.patch("importlib.metadata.version", return_value="1.26.4"), mock.patch(
        "subprocess.Popen"
    ) as mock_popen, mock.patch.object(
        UVVirtualEnvManager, "_get_uv_path", return_value="uv"
    ):
        process = mock.Mock()
        process.wait.return_value = 1
        mock_popen.return_value = process

        with pytest.raises(subprocess.CalledProcessError):
            uv_manager.install_packages(["#system_numpy#", "vllm==0.21.0"])
        assert mock_popen.call_count == 2


def test_install_packages_retry_keeps_identical_explicit_pin(uv_manager):
    # an explicit user/spec pin spelling the same version as the resolved
    # #system_*# placeholder must survive the retry; only the
    # placeholder-derived entry is relaxed
    calls = []

    def fake_popen(cmd, *args, **kwargs):
        calls.append(list(cmd))
        process = mock.Mock()
        process.wait.return_value = 1 if len(calls) == 1 else 0
        return process

    with mock.patch("importlib.metadata.version", return_value="1.26.4"), mock.patch(
        "subprocess.Popen", side_effect=fake_popen
    ), mock.patch.object(UVVirtualEnvManager, "_get_uv_path", return_value="uv"):
        uv_manager.install_packages(["#system_numpy#", "numpy==1.26.4", "vllm==0.21.0"])

    assert len(calls) == 2
    # the placeholder-derived pin is relaxed to a bare name...
    assert "numpy" in calls[1]
    # ...while the explicit identical pin is preserved
    assert "numpy==1.26.4" in calls[1]
    assert "vllm==0.21.0" in calls[1]


def test_install_packages_retry_skip_installed_does_not_repin(uv_manager):
    # with skip_installed=True the retry must not let _split_specs pin the
    # relaxed placeholder back to the installed host version, otherwise the
    # second dry-run reuses the exact constraint that just failed
    import subprocess

    class FakeDist:
        metadata = {"Name": "numpy"}
        version = "1.26.4"

    plan_calls = []

    def fake_resolve(self, specs, pinned, *args, **kwargs):
        plan_calls.append((list(specs), dict(pinned)))
        if len(plan_calls) == 1:
            raise subprocess.CalledProcessError(1, "uv")
        return ["numpy==2.1.0", "vllm==0.21.0"]

    def fake_popen(cmd, *args, **kwargs):
        process = mock.Mock()
        process.wait.return_value = 0
        return process

    with mock.patch("importlib.metadata.version", return_value="1.26.4"), mock.patch(
        "xoscar.virtualenv.uv.distributions", return_value=[FakeDist()]
    ), mock.patch.object(
        UVVirtualEnvManager, "_resolve_install_plan", autospec=True
    ) as mock_plan, mock.patch(
        "subprocess.Popen", side_effect=fake_popen
    ), mock.patch.object(
        UVVirtualEnvManager, "_get_uv_path", return_value="uv"
    ):
        mock_plan.side_effect = fake_resolve
        uv_manager.install_packages(
            ["#system_numpy#", "vllm==0.21.0"], skip_installed=True
        )

    assert len(plan_calls) == 2
    # first attempt: the placeholder resolves to the host pin, so numpy is
    # satisfied and becomes a dry-run constraint
    assert plan_calls[0] == (["vllm==0.21.0"], {"numpy": "1.26.4"})
    # retry: numpy is sent to the resolver unconstrained instead of being
    # re-pinned to the host version
    assert "numpy" in plan_calls[1][0]
    assert "numpy" not in plan_calls[1][1]


def test_install_packages_no_retry_after_cancel(uv_manager):
    # cancel_install terminates uv, which also exits non-zero; that must not
    # be mistaken for a resolver conflict and trigger the pin-drop retry
    import subprocess

    calls = []

    class FakeProcess:
        def __init__(self):
            self._terminated = False

        def poll(self):
            return -15 if self._terminated else None

        def terminate(self):
            self._terminated = True

        def wait(self):
            if not self._terminated:
                # simulate the user cancelling while uv is running
                uv_manager.cancel_install()
            return -15

    def fake_popen(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return FakeProcess()

    with mock.patch("importlib.metadata.version", return_value="1.26.4"), mock.patch(
        "subprocess.Popen", side_effect=fake_popen
    ), mock.patch.object(UVVirtualEnvManager, "_get_uv_path", return_value="uv"):
        with pytest.raises(subprocess.CalledProcessError):
            uv_manager.install_packages(["#system_numpy#", "vllm==0.21.0"])

    assert len(calls) == 1
