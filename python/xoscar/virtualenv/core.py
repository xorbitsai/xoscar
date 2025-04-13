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

from abc import ABC, abstractmethod
from pathlib import Path


class VirtualEnvManager(ABC):
    @classmethod
    @abstractmethod
    def is_available(cls):
        pass

    def __init__(self, env_path: Path):
        self.env_path = env_path.resolve()

    @abstractmethod
    def create_env(self, python_path: Path | None = None) -> None:
        pass

    @abstractmethod
    def install_packages(self, packages: list[str], **kwargs):
        pass

    @abstractmethod
    def cancel_install(self):
        pass

    @abstractmethod
    def get_lib_path(self) -> str:
        pass

    @abstractmethod
    def remove_env(self):
        pass
