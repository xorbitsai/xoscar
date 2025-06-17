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

from ..core import VirtualEnvManager


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
