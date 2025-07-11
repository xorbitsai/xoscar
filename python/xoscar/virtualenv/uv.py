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
import tempfile
from importlib.metadata import distributions
from pathlib import Path
from typing import Optional

from packaging.requirements import Requirement
from packaging.version import Version

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

    def _resolve_install_plan(
        self, specs: list[str], pinned: dict[str, str]
    ) -> list[str]:
        """
        Run uv --dry-run with pinned constraints and return
        a list like ['package==version', ...].
        """
        with tempfile.NamedTemporaryFile("w+", delete=True) as f:
            for name, ver in pinned.items():
                f.write(f"{name}=={ver}\n")
            f.flush()  # make sure content is on disk

            cmd = [
                self._get_uv_path(),
                "pip",
                "install",
                "-p",
                str(self.env_path),
                "--dry-run",
                "--constraint",
                f.name,
                *specs,
            ]
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)

        # the temp file is automatically deleted here
        deps = [
            f"{m.group(1)}=={m.group(2)}"
            for line in result.stderr.splitlines()
            if (m := re.match(r"^\+ (\S+)==(\S+)$", line.strip()))
        ]
        return deps

    @staticmethod
    def _split_specs(
        specs: list[str], installed: dict[str, str]
    ) -> tuple[list[str], dict[str, str]]:
        """
        Split the given requirement specs into:
        - to_resolve: specs that need to be passed to the resolver (unsatisfied ones)
        - pinned: already satisfied specs, used for constraint to lock their versions
        """
        to_resolve: list[str] = []
        pinned: dict[str, str] = {}

        for spec_str in specs:
            req = Requirement(spec_str)
            name = req.name.lower()
            cur_ver = installed.get(name)

            if cur_ver is None:
                # Package not installed, needs resolution
                to_resolve.append(spec_str)
                continue

            if not req.specifier:
                # No version constraint, already satisfied
                pinned[name] = cur_ver
                continue

            try:
                if Version(cur_ver) in req.specifier:
                    # Version satisfies the specifier, pin it
                    pinned[name] = cur_ver
                else:
                    # Version does not satisfy, needs resolution
                    to_resolve.append(spec_str)
            except Exception:
                # Parsing error, be conservative and resolve it
                to_resolve.append(spec_str)

        return to_resolve, pinned

    def _filter_packages_not_installed(self, packages: list[str]) -> list[str]:
        """
        Filter out packages that are already installed with the same version.
        """

        # all the installed packages in system site packages
        installed = {
            dist.metadata["Name"].lower(): dist.version
            for dist in distributions()
            if dist.metadata and "Name" in dist.metadata
        }

        # exclude those packages that satisfied in system site packages
        to_resolve, pinned = self._split_specs(packages, installed)
        if not to_resolve:
            logger.debug("All requirement specifiers satisfied by system packages.")
            return []

        resolved = self._resolve_install_plan(to_resolve, pinned)
        logger.debug(f"Resolved install list: {resolved}")
        if not resolved:
            # no packages to install
            return []

        final = []
        for item in resolved:
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
