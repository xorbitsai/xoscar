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

from .core import VirtualEnvManager, collect_system_pins, relax_system_requirement
from .utils import is_vcs_url, run_subprocess_with_logger

UV_PATH = os.getenv("XOSCAR_UV_PATH")
SKIP_INSTALLED = bool(int(os.getenv("XOSCAR_VIRTUAL_ENV_SKIP_INSTALLED", "0")))
logger = logging.getLogger(__name__)


def _is_in_pyinstaller():
    return hasattr(sys, "_MEIPASS")


class UVVirtualEnvManager(VirtualEnvManager):
    def __init__(self, env_path: Path):
        super().__init__(env_path)
        self._install_process: Optional[subprocess.Popen] = None
        self._install_cancelled = False

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

    def exists_env(self) -> bool:
        """Check if virtual environment already exists."""
        return self.env_path.exists() and (self.env_path / "pyvenv.cfg").exists()

    def create_env(
        self, python_path: Path | None = None, exists: str = "ignore"
    ) -> None:
        """
        Create virtual environment.

        Args:
            python_path: Path to Python interpreter to use
            exists: How to handle existing environment:
                - "ignore": Skip creation if environment already exists (default)
                - "error": Raise error if environment exists
                - "clear": Remove existing environment and create new one
        """
        if self.exists_env():
            if exists == "error":
                raise FileExistsError(
                    f"Virtual environment already exists at {self.env_path}"
                )
            elif exists == "ignore":
                logger.info(
                    f"Virtual environment already exists at {self.env_path}, skipping creation"
                )
                return
            elif exists == "clear":
                logger.info(f"Removing existing virtual environment at {self.env_path}")
                self.remove_env()
            else:
                raise ValueError(
                    f"Invalid exists option: {exists}. Must be one of: error, clear, ignore"
                )

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
        self,
        specs: list[str],
        pinned: dict[str, str],
        index_url: str | None = None,
        extra_index_url: str | list[str] | None = None,
        index_strategy: str | None = None,
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
            ]

            # Add index URL parameters
            if index_url:
                cmd += ["-i", index_url]
            if extra_index_url:
                cmd += (
                    ["--extra-index-url", extra_index_url]
                    if isinstance(extra_index_url, str)
                    else [
                        opt for v in extra_index_url for opt in ("--extra-index-url", v)
                    ]
                )
            if index_strategy:
                cmd += ["--index-strategy", index_strategy]

            cmd.extend(specs)
            try:
                result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.error(e.stderr)
                raise e
        # the temp file is automatically deleted here
        deps = [
            f"{m.group(1)}=={m.group(2)}"
            for line in result.stderr.splitlines()
            if (m := re.match(r"^\+ (\S+)==(\S+)$", line.strip()))
        ]
        return deps

    @staticmethod
    def _split_specs(
        specs: list[str],
        installed: dict[str, str],
        relax_names: frozenset[str] = frozenset(),
    ) -> tuple[list[str], list[str], dict[str, str]]:
        """
        Split the given requirement specs into:
        - keep： specs that need to be kept, e.g. git+github://xxx
        - to_resolve: specs that need to be passed to the resolver (unsatisfied ones)
        - pinned: already satisfied specs, used for constraint to lock their versions

        Package names in ``relax_names`` (host-aligned pins dropped by the
        retry path) are always sent to the resolver instead of being pinned
        back to the installed host version.
        """
        keep: list[str] = []
        to_resolve: list[str] = []
        pinned: dict[str, str] = {}

        for spec_str in specs:
            # skip git+xxx
            if is_vcs_url(spec_str):
                keep.append(spec_str)
                continue

            req = Requirement(spec_str)
            name = req.name.lower()
            if name in relax_names:
                to_resolve.append(spec_str)
                continue
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

        return keep, to_resolve, pinned

    def _filter_packages_not_installed(
        self,
        packages: list[str],
        index_url: str | None = None,
        extra_index_url: str | list[str] | None = None,
        index_strategy: str | None = None,
        relax_names: frozenset[str] = frozenset(),
    ) -> list[str]:
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
        keep, to_resolve, pinned = self._split_specs(packages, installed, relax_names)
        if not keep and not to_resolve:
            logger.debug("All requirement specifiers satisfied by system packages.")
            return []

        if to_resolve:
            resolved = self._resolve_install_plan(
                to_resolve, pinned, index_url, extra_index_url, index_strategy
            )
            logger.debug(f"Resolved install list: {resolved}")
            if not keep and not resolved:
                # no packages to install
                return []
        else:
            resolved = []

        final = keep.copy()
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

        Args:
            packages: List of package specifications
            **kwargs: Can include:
                - Pip configuration: index_url, extra_index_url, find_links, trusted_host,
                  no_build_isolation, log, skip_installed
                - Dynamic variables for #var# substitution: e.g., engine='vllm', mode='remote'
        """
        if not packages:
            return

        # Pop pip configuration parameters, remaining kwargs are for variable substitution
        log = kwargs.pop("log", False)
        skip_installed = kwargs.pop("skip_installed", SKIP_INSTALLED)
        index_url = kwargs.pop("index_url", None)
        extra_index_url = kwargs.pop("extra_index_url", None)
        index_strategy = kwargs.pop("index_strategy", None)

        # Process packages with variable substitution
        raw_packages = packages
        processed = self.process_packages(packages, **kwargs)
        if not processed:
            return
        self._install_cancelled = False

        def _do_install(
            install_list: list[str], relax_names: frozenset[str] = frozenset()
        ) -> None:
            uv_path = self._get_uv_path()

            if skip_installed:
                install_list = self._filter_packages_not_installed(
                    install_list,
                    index_url,
                    extra_index_url,
                    index_strategy,
                    relax_names,
                )
                if not install_list:
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
                ] + install_list
            else:
                cmd = [
                    uv_path,
                    "pip",
                    "install",
                    "-p",
                    str(self.env_path),
                    "--color=always",
                ] + install_list

            if index_url:
                cmd += ["-i", index_url]
            param_and_option = [
                (extra_index_url, "--extra-index-url"),
                ("find_links" in kwargs and kwargs["find_links"], "-f"),
                ("trusted_host" in kwargs and kwargs["trusted_host"], "--trusted-host"),
            ]
            for param, option in param_and_option:
                if param:
                    val = (
                        param
                        if not isinstance(param, bool)
                        else kwargs.get(
                            {"-f": "find_links", "--trusted-host": "trusted_host"}[
                                option
                            ]
                        )
                    )
                    if val:
                        cmd += (
                            [option, val]
                            if isinstance(val, str)
                            else [opt for v in val for opt in (option, v)]
                        )

            if index_strategy:
                cmd += ["--index-strategy", index_strategy]
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

        try:
            _do_install(processed)
        except subprocess.CalledProcessError:
            # An explicitly cancelled install also exits non-zero; never
            # start a second install in that case.
            if self._install_cancelled:
                raise
            # Host-aligned #system_*# pins can conflict with other requirements,
            # e.g. a model requires a newer engine whose dependencies exceed the
            # host-pinned version. Drop only those pins and retry once.
            pins = collect_system_pins(raw_packages) & set(processed)
            if not pins:
                raise
            # Re-process with the placeholders relaxed to bare names so that
            # only placeholder-derived pins are dropped; an explicit user/spec
            # pin that happens to spell the same version is kept.
            retry_list = self.process_packages(
                [relax_system_requirement(pkg) for pkg in raw_packages], **kwargs
            )
            # In skip_installed mode the relaxed names must not be pinned back
            # to the installed host versions by _split_specs.
            relaxed_names = frozenset(p.split("==", 1)[0].lower() for p in pins)
            logger.warning(
                "Package installation failed with host-aligned pins %s; "
                "retrying without them. The virtual environment may end up with "
                "versions different from the host environment.",
                sorted(pins),
            )
            _do_install(retry_list, relaxed_names)

    def cancel_install(self):
        if self._install_process and self._install_process.poll() is None:
            self._install_cancelled = True
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
