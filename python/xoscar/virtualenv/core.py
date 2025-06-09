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

import importlib
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

    @staticmethod
    def process_packages(packages: list[str]) -> list[str]:
        """
        Process a list of package names, replacing placeholders like #system_<package>#
        with the installed version of the corresponding package from the system environment.

        Example:
            "#system_torch#" -> "torch==2.1.0" (if torch 2.1.0 is installed)

        Args:
            packages (list[str]): A list of package names, which may include placeholders.

        Returns:
            list[str]: A new list with resolved package names and versions.

        Raises:
            RuntimeError: If a specified system package is not found in the environment.
        """
        processed = []

        for pkg in packages:
            if pkg.startswith("#system_") and pkg.endswith("#"):
                real_pkg = pkg[
                    len("#system_") : -1
                ]  # Extract actual package name, e.g., "torch"
                try:
                    version = importlib.metadata.version(real_pkg)
                except importlib.metadata.PackageNotFoundError:
                    raise RuntimeError(
                        f"System package '{real_pkg}' not found. Cannot resolve '{pkg}'."
                    )
                processed.append(f"{real_pkg}=={version}")
            else:
                processed.append(pkg)

        return processed

    @abstractmethod
    def cancel_install(self):
        pass

    @abstractmethod
    def get_python_path(self) -> str | None:
        pass

    @abstractmethod
    def get_lib_path(self) -> str:
        pass

    @abstractmethod
    def remove_env(self):
        pass
