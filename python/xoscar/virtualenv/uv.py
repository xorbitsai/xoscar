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

import shutil
import subprocess
import sysconfig
from pathlib import Path

from .core import VirtualEnvManager


class UVVirtualEnvManager(VirtualEnvManager):
    def create_env(self, python_path: Path | None = None) -> None:
        cmd = ["uv", "venv", str(self.env_path)]
        if python_path:
            cmd += ["--python", str(python_path)]
        subprocess.run(cmd, check=True)

    def install_packages(self, packages: list[str], index_url: str | None = None):
        if not packages:
            return
        cmd = ["uv", "pip", "install", "-p", str(self.env_path)] + packages
        if index_url:
            cmd += ["-i", index_url]  # specify index url

        subprocess.run(cmd, check=True)

    def get_lib_path(self) -> str:
        return sysconfig.get_path("purelib", vars={"base": str(self.env_path)})

    def remove_env(self):
        if self.env_path.exists():
            shutil.rmtree(self.env_path, ignore_errors=True)
