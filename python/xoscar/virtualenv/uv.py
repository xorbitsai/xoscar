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
from typing import Optional

from .core import VirtualEnvManager


class UVVirtualEnvManager(VirtualEnvManager):
    def __init__(self, env_path: Path):
        super().__init__(env_path)
        self._install_process: Optional[subprocess.Popen] = None

    @classmethod
    def is_available(cls):
        return shutil.which("uv") is not None

    def create_env(self, python_path: Path | None = None) -> None:
        cmd = ["uv", "venv", str(self.env_path)]
        if python_path:
            cmd += ["--python", str(python_path)]
        subprocess.run(cmd, check=True)

    def install_packages(self, packages: list[str], **kwargs):
        """
        Install packages into the virtual environment using uv.
        Supports pip-compatible kwargs: index_url, extra_index_url, find_links.
        """
        if not packages:
            return

        cmd = ["uv", "pip", "install", "-p", str(self.env_path)] + packages

        # Handle known pip-related kwargs
        if "index_url" in kwargs and kwargs["index_url"]:
            cmd += ["-i", kwargs["index_url"]]
        if "extra_index_url" in kwargs and kwargs["extra_index_url"]:
            cmd += ["--extra-index-url", kwargs["extra_index_url"]]
        if "find_links" in kwargs and kwargs["find_links"]:
            cmd += ["-f", kwargs["find_links"]]
        if "trusted_host" in kwargs and kwargs["trusted_host"]:
            cmd += ["--trusted-host", kwargs["trusted_host"]]

        self._install_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        stdout, stderr = self._install_process.communicate()
        returncode = self._install_process.returncode

        self._install_process = None  # install finished, clear reference

        if returncode != 0:
            raise subprocess.CalledProcessError(
                returncode, cmd, output=stdout, stderr=stderr
            )

    def cancel_install(self):
        if self._install_process and self._install_process.poll() is None:
            self._install_process.terminate()
            self._install_process.wait()

    def get_lib_path(self) -> str:
        return sysconfig.get_path("purelib", vars={"base": str(self.env_path)})

    def remove_env(self):
        if self.env_path.exists():
            shutil.rmtree(self.env_path, ignore_errors=True)
