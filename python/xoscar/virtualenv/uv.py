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

import logging
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from importlib.metadata import distributions
from pathlib import Path
from typing import Optional

from .core import VirtualEnvManager
from .utils import run_subprocess_with_logger

UV_PATH = os.getenv("XOSCAR_UV_PATH")
SKIP_INSTALLED = bool(int(os.getenv("XOSCAR_VIRTUAL_ENV_SKIP_INSTALLED", "0")))
logger = logging.getLogger(__name__)


def _is_in_pyinstaller():
    return hasattr(sys, "_MEIPASS")


class UVVirtualEnvManager(VirtualEnvManager):
    def __init__(self, env_path: Path):
        super().__init__(env_path)
        self._install_process: Optional[subprocess.Popen] = None

    @classmethod
    def is_available(cls):
        if UV_PATH is not None:
            # user specified uv, just treat it as existed
            return True
        return shutil.which("uv") is not None

    @staticmethod
    def _get_uv_path() -> str:
        if (uv_path := UV_PATH) is None:
            try:
                from uv import find_uv_bin

                uv_path = find_uv_bin()
            except (ImportError, FileNotFoundError):
                logger.warning("Fail to find uv bin, use system one")
                uv_path = "uv"
        return uv_path

    def create_env(self, python_path: Path | None = None) -> None:
        uv_path = self._get_uv_path()
        cmd = [uv_path, "venv", str(self.env_path), "--system-site-packages"]
        if python_path:
            cmd += ["--python", str(python_path)]
        elif _is_in_pyinstaller():
            # in pyinstaller, uv would find the system python
            # in this case we'd better specify the same python version
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            cmd += ["--python", python_version]

        logger.info("Creating virtualenv via command: %s", cmd)
        subprocess.run(cmd, check=True)

    def _get_all_install_packages(self, packages: list[str]) -> list[str]:
        """
        Get the full list of packages (with versions) that would be installed for the given packages.

        This uses `uv pip install --dry-run` to simulate installation and parse the output.

        Args:
            packages: List of top-level packages to install (e.g., ['transformers']).

        Returns:
            A list of package specifications in the form ['transformers==4.53.0', 'numpy==2.3.1', ...].
        """
        cmd = ["uv", "pip", "install", *packages, "--dry-run", "-p", str(self.env_path)]
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        deps = []
        for line in result.stderr.splitlines():
            line = line.strip()
            match = re.match(r"^\+ (\S+)==(\S+)$", line)
            if match:
                name, version = match.groups()
                deps.append(f"{name}=={version}")
        return deps

    def _filter_packages_not_installed(self, packages: list[str]) -> list[str]:
        """
        Filter out packages that are already installed with the same version.
        """
        to_install = self._get_all_install_packages(packages)
        logger.debug(f"Resolved install list: {to_install}")

        if not to_install:
            # no packages to install
            return []

        installed = {
            dist.metadata["Name"].lower(): dist.version
            for dist in distributions()
            if dist.metadata and "Name" in dist.metadata
        }
        logger.debug(f"Currently installed packages: {installed}")

        final = []
        for item in to_install:
            name, version = item.split("==")
            key = name.lower()
            if key not in installed or installed[key] != version:
                final.append(item)
        logger.debug(f"Filtered install list: {final}")
        return final

    def install_packages(self, packages: list[str], **kwargs):
        """
        Install packages into the virtual environment using uv.
        Supports pip-compatible kwargs: index_url, extra_index_url, find_links.
        """
        if not packages:
            return

        packages = self.process_packages(packages)
        log = kwargs.pop("log", False)
        skip_installed = kwargs.pop("skip_installed", SKIP_INSTALLED)
        uv_path = self._get_uv_path()

        if skip_installed:
            packages = self._filter_packages_not_installed(packages)
            if not packages:
                logger.info("All required packages are already installed.")
                return

            cmd = [
                uv_path,
                "pip",
                "install",
                "-p",
                str(self.env_path),
                "--color=always",
                "--no-deps",
            ] + packages
        else:
            cmd = [
                uv_path,
                "pip",
                "install",
                "-p",
                str(self.env_path),
                "--color=always",
            ] + packages

        if "index_url" in kwargs and kwargs["index_url"]:
            cmd += ["-i", kwargs["index_url"]]
        param_and_option = [
            ("extra_index_url", "--extra-index-url"),
            ("find_links", "-f"),
            ("trusted_host", "--trusted-host"),
        ]
        for param, option in param_and_option:
            if param in kwargs and kwargs[param]:
                val = kwargs[param]
                cmd += (
                    [option, val]
                    if isinstance(val, str)
                    else [opt for v in val for opt in (option, v)]
                )

        if kwargs.get("no_build_isolation", False):
            cmd += ["--no-build-isolation"]

        logger.info("Installing packages via command: %s", cmd)
        if not log:
            self._install_process = process = subprocess.Popen(cmd)
            returncode = process.wait()
        else:
            with run_subprocess_with_logger(cmd) as process:
                self._install_process = process
            returncode = process.returncode

        self._install_process = None
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)

    def cancel_install(self):
        if self._install_process and self._install_process.poll() is None:
            self._install_process.terminate()
            self._install_process.wait()

    def get_python_path(self) -> str | None:
        if self.env_path.exists():
            return str(self.env_path.joinpath("bin/python"))
        return None

    def get_lib_path(self) -> str:
        return sysconfig.get_path("purelib", vars={"base": str(self.env_path)})

    def remove_env(self):
        if self.env_path.exists():
            shutil.rmtree(self.env_path, ignore_errors=True)
