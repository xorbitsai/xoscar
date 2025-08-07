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

from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest
from packaging.markers import default_environment

from ..core import VirtualEnvManager, filter_requirements


def test_system_package_with_plus_cpu():
    # simulate 'torch==2.1.2+cpu'
    with patch("importlib.metadata.version", return_value="2.1.2+cpu"):
        result = VirtualEnvManager.process_packages(["#system_torch#"])
        assert result == ["torch==2.1.2"]


def test_system_package_without_plus():
    with patch("importlib.metadata.version", return_value="1.26.4"):
        result = VirtualEnvManager.process_packages(["#system_numpy#"])
        assert result == ["numpy==1.26.4"]


def test_non_placeholder_package():
    result = VirtualEnvManager.process_packages(["requests>=2.0.0"])
    assert result == ["requests>=2.0.0"]


def test_package_not_found():
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
        with pytest.raises(RuntimeError, match="System package 'notexist' not found"):
            VirtualEnvManager.process_packages(["#system_notexist#"])


@pytest.mark.parametrize(
    "input_env, input_pkgs, expected_pkgs",
    [
        # Only built-in env vars (standard markers)
        (
            {"sys_platform": "linux", "python_version": "3.8"},
            ["foo ; sys_platform == 'linux'", "bar ; sys_platform == 'win32'"],
            ["foo"],
        ),
        # Only custom env vars, boolean check true
        (
            {"has_cuda": True},
            ["torch ; has_cuda", "cpu-lib ; not has_cuda"],
            ["torch"],
        ),
        # Only custom env vars, boolean check false
        (
            {"has_cuda": False},
            ["torch ; has_cuda", "cpu-lib ; not has_cuda"],
            ["cpu-lib"],
        ),
        # Mixed built-in and custom env vars, with comparison value
        (
            {"has_cuda": True, "sys_platform": "linux"},
            [
                "torch==2.0 ; has_cuda and sys_platform == 'linux'",
                "xformers ; sys_platform == 'win32' or not has_cuda",
            ],
            ["torch==2.0"],
        ),
        # Mixed with custom boolean false
        (
            {"has_cuda": False, "sys_platform": "win32"},
            [
                "torch==2.0 ; has_cuda and sys_platform == 'linux'",
                "xformers ; sys_platform == 'win32' or not has_cuda",
            ],
            ["xformers"],
        ),
        # Custom marker with simple comparison (custom marker with value)
        (
            {"cuda_version": "12.1"},
            ["torch ; cuda_version >= '12.0'", "torch_old ; cuda_version < '12.0'"],
            ["torch"],
        ),
        # Custom marker boolean AND with built-in OR
        (
            {"has_cuda": True, "sys_platform": "darwin"},
            [
                "torch ; has_cuda and (sys_platform == 'linux' or sys_platform == 'darwin')",
                "xformers ; sys_platform == 'win32' or not has_cuda",
            ],
            ["torch"],
        ),
        # Custom cuda_arch numeric comparison
        (
            {"cuda_arch": "sm_120"},
            ["fast-lib ; cuda_arch >= 'sm_80'", "old-lib ; cuda_arch < 'sm_80'"],
            ["fast-lib"],
        ),
        # Boolean false with negation and OR
        (
            {"has_cuda": False, "sys_platform": "linux"},
            [
                "lib1 ; not has_cuda and sys_platform == 'linux'",
                "lib2 ; has_cuda or sys_platform == 'win32'",
            ],
            ["lib1"],
        ),
    ],
)
def test_filter_requirements_with_combined_markers(
    input_env, input_pkgs, expected_pkgs
):
    # Convert all env values to string for marker.evaluate
    std_env = default_environment()
    std_env.update(
        {
            k: str(v).lower() if isinstance(v, bool) else str(v)
            for k, v in input_env.items()
        }
    )

    with patch("xoscar.virtualenv.core.get_env", return_value=std_env):
        filtered = filter_requirements(input_pkgs)
        assert set(filtered) == set(expected_pkgs)
