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

from __future__ import annotations

from pathlib import Path

from .core import VirtualEnvManager
from .uv import UVVirtualEnvManager

_name_to_managers = {"uv": UVVirtualEnvManager}


def get_virtual_env_manager(env_name: str, env_path: str | Path) -> VirtualEnvManager:
    try:
        manager_cls = _name_to_managers[env_name]
    except KeyError:
        raise ValueError(
            f"Unknown virtualenv manager {env_name}, available: {list(_name_to_managers)}"
        )

    path = Path(env_path)
    return manager_cls(path)
