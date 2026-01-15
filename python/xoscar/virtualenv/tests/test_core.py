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

from ..core import VirtualEnvManager, filter_requirements, substitute_variables


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


class TestSubstituteVariables:
    """Test #var# substitution in markers."""

    def test_string_variable(self):
        """Test string variable substitution."""
        result = substitute_variables('#engine# == "vllm"', {"engine": "vllm"})
        assert result == '"vllm" == "vllm"'

    def test_string_variable_no_match(self):
        """Test when variable value doesn't match."""
        result = substitute_variables('#engine# == "vllm"', {"engine": "sglang"})
        assert result == '"sglang" == "vllm"'

    def test_numeric_variable(self):
        """Test numeric variable substitution."""
        result = substitute_variables("#count# > 5", {"count": 10})
        assert result == "10 > 5"

    def test_boolean_variable_true(self):
        """Test boolean variable substitution (True)."""
        result = substitute_variables("#enabled# == True", {"enabled": True})
        assert result == "True == True"

    def test_boolean_variable_false(self):
        """Test boolean variable substitution (False)."""
        result = substitute_variables("#enabled# == True", {"enabled": False})
        assert result == "False == True"

    def test_none_variable(self):
        """Test None variable substitution."""
        result = substitute_variables("#value# == None", {"value": None})
        assert result == "None == None"

    def test_string_with_quotes(self):
        """Test string variable with quotes gets escaped."""
        result = substitute_variables('#engine# == "test"', {"engine": 'my"engine'})
        assert result == '"my\\"engine" == "test"'

    def test_multiple_variables(self):
        """Test multiple variables in one marker."""
        result = substitute_variables(
            '#engine# == "vllm" and #mode# == "local"',
            {"engine": "vllm", "mode": "local"},
        )
        assert result == '"vllm" == "vllm" and "local" == "local"'

    def test_no_placeholder(self):
        """Test marker without placeholder remains unchanged."""
        result = substitute_variables("has_cuda", {"engine": "vllm"})
        assert result == "has_cuda"

    def test_variable_not_provided(self):
        """Test when placeholder exists but variable not provided."""
        result = substitute_variables('#engine# == "vllm"', {"mode": "local"})
        # Should not replace since 'engine' is not in variables
        assert result == '#engine# == "vllm"'


class TestFilterRequirementsWithVariables:
    """Test filter_requirements with dynamic variable substitution."""

    @pytest.mark.parametrize(
        "variables, input_pkgs, expected_pkgs",
        [
            # Simple string match
            (
                {"engine": "vllm"},
                ['pkg1; #engine# == "vllm"', 'pkg2; #engine# == "sglang"'],
                ["pkg1"],
            ),
            # Numeric comparison
            (
                {"count": 10},
                ["pkg1; #count# > 5", "pkg2; #count# < 5"],
                ["pkg1"],
            ),
            # Boolean check
            (
                {"enabled": True},
                ["pkg1; #enabled# == True", "pkg2; #enabled# == False"],
                ["pkg1"],
            ),
            # Multiple variables with AND
            (
                {"engine": "vllm", "mode": "local"},
                [
                    'pkg1; #engine# == "vllm" and #mode# == "local"',
                    'pkg2; #engine# == "sglang" or #mode# == "remote"',
                ],
                ["pkg1"],
            ),
            # No match - empty result
            (
                {"engine": "sglang"},
                ['pkg1; #engine# == "vllm"', 'pkg2; #engine# == "sglang"'],
                ["pkg2"],
            ),
        ],
    )
    def test_variable_substitution(self, variables, input_pkgs, expected_pkgs):
        """Test that #var# placeholders are correctly substituted and filtered."""
        filtered = filter_requirements(input_pkgs, **variables)
        assert set(filtered) == set(expected_pkgs)

    def test_variables_with_standard_markers(self):
        """Test dynamic variables work alongside standard markers."""
        std_env = default_environment()
        with patch("xoscar.virtualenv.core.get_env", return_value=std_env):
            filtered = filter_requirements(
                ['pkg1; #engine# == "vllm" and python_version >= "3.8"'], engine="vllm"
            )
            # Should match if python_version >= 3.8 (which is likely true in test env)
            assert filtered == ["pkg1"]
