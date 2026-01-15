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

import ast
import importlib
import json
import operator
from abc import ABC, abstractmethod
from pathlib import Path

from packaging.markers import Marker, default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import Version

from .platform import (
    check_cuda_available,
    check_npu_available,
    get_cuda_arch,
    get_cuda_version,
)
from .utils import is_vcs_url


class VirtualEnvManager(ABC):
    @classmethod
    @abstractmethod
    def is_available(cls):
        pass

    def __init__(self, env_path: Path):
        self.env_path = env_path.resolve()

    @abstractmethod
    def exists_env(self) -> bool:
        pass

    @abstractmethod
    def create_env(
        self, python_path: Path | None = None, exists: str = "ignore"
    ) -> None:
        pass

    @abstractmethod
    def install_packages(self, packages: list[str], **kwargs):
        pass

    @staticmethod
    def process_packages(packages: list[str], **variables) -> list[str]:
        """
        Process a list of package names, replacing placeholders like #system_<package>#
        with the installed version of the corresponding package from the system environment.

        Example:
            "#system_torch#" -> "torch==2.1.0" (if torch 2.1.0 is installed)

        Args:
            packages (list[str]): A list of package names, which may include placeholders.
            **variables: Dynamic variables for marker substitution, e.g., engine='vllm'.

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
                    # Strip build metadata like "+cpu"
                    version = version.split("+")[0]
                except importlib.metadata.PackageNotFoundError:
                    raise RuntimeError(
                        f"System package '{real_pkg}' not found. Cannot resolve '{pkg}'."
                    )
                processed.append(f"{real_pkg}=={version}")
            else:
                processed.append(pkg)

        # apply extended syntax including:
        # - has_cuda: whether CUDA is available (bool)
        # - cuda_version: CUDA version string, e.g. "12.1" (str)
        # - cuda_arch: CUDA architecture string, e.g. "sm_80" (str)
        # - has_npu: whether an NPU is available (bool)
        # - #var#: dynamic variable substitution, e.g., #engine# == "vllm"
        processed = filter_requirements(processed, **variables)

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


def substitute_variables(marker_str: str, variables: dict) -> str:
    """
    Replace #var# placeholders in marker string with actual values.

    Example:
        substitute_variables('#engine# == "vllm"', {'engine': 'vllm'})
        -> '"vllm" == "vllm"'

    Args:
        marker_str: Marker expression, e.g., '#engine# == "vllm"'
        variables: Variable dict, e.g., {'engine': 'vllm', 'count': 10}

    Returns:
        Marker string with placeholders replaced
    """
    result = marker_str

    for var_name, var_value in variables.items():
        placeholder = f"#{var_name}#"
        if placeholder not in result:
            continue

        # Format value based on type
        if var_value is None:
            formatted = "None"
        elif isinstance(var_value, bool):
            # Boolean: use Python literal (True/False) without quotes
            formatted = str(var_value)  # True -> "True", False -> "False"
        elif isinstance(var_value, (int, float)):
            formatted = str(var_value)
        else:
            # String: escape and add quotes
            formatted = json.dumps(var_value)

        result = result.replace(placeholder, formatted)

    return result


def get_env() -> dict[str, str | bool]:
    env = default_environment().copy()
    # Your custom env vars here, e.g.:
    env.update(
        {
            "has_cuda": check_cuda_available(),
            "cuda_version": get_cuda_version(),
            "cuda_arch": get_cuda_arch(),
            "has_npu": check_npu_available(),
        }
    )
    return env


STANDARD_ENV_VARS = set(default_environment().keys())


def is_custom_marker(marker_str: str) -> bool:
    try:
        marker = Marker(marker_str)
    except Exception:
        return True

    def traverse_markers(node):
        if isinstance(node, tuple):
            env_var = node[0]
            if env_var not in STANDARD_ENV_VARS:
                return True
            return False
        elif isinstance(node, list):
            return any(traverse_markers(child) for child in node)
        return False

    return traverse_markers(marker._markers)


def eval_custom_marker(marker_str: str, env: dict) -> bool:
    ops = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.And: lambda a, b: a and b,
        ast.Or: lambda a, b: a or b,
        ast.Not: operator.not_,
    }

    def normalize_value(val):
        # Normalize for boolean
        if isinstance(val, str):
            if val.lower() == "true":
                return True
            if val.lower() == "false":
                return False

        # Normalize for version-like fields
        if isinstance(val, str):
            if val.count(".") >= 1 and all(
                part.isdigit() for part in val.split(".") if part
            ):
                return Version(val)

        return val

    def maybe_parse_cuda_arch(val):
        if isinstance(val, str) and val.startswith("sm_"):
            try:
                return int(val[3:])
            except ValueError:
                return val
        return val

    def _eval(node):
        if isinstance(node, ast.BoolOp):
            left = _eval(node.values[0])
            for right_node in node.values[1:]:
                right = _eval(right_node)
                op = ops[type(node.op)]
                left = op(left, right)
            return left

        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not _eval(node.operand)

        elif isinstance(node, ast.Compare):
            left = _eval(node.left)
            left = maybe_parse_cuda_arch(normalize_value(left))

            for op_node, right_expr in zip(node.ops, node.comparators):
                right = _eval(right_expr)
                right = maybe_parse_cuda_arch(normalize_value(right))

                op_func = ops[type(op_node)]
                if not op_func(left, right):
                    return False
                left = right  # for chained comparisons

            return True

        elif isinstance(node, ast.Name):
            return normalize_value(env.get(node.id))

        elif isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.Str):  # Python <3.8
            return node.s

        else:
            raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    tree = ast.parse(marker_str, mode="eval")
    return _eval(tree.body)


def filter_requirements(requirements: list[str], **variables) -> list[str]:
    """
    Filter requirements by evaluating markers in given env.
    If env is None, use get_env().

    Args:
        requirements: List of requirement strings
        **variables: Dynamic variables for #var# substitution, e.g., engine='vllm'
    """
    env = get_env()
    result = []
    for req_str in requirements:
        if is_vcs_url(req_str):
            result.append(req_str)
        elif ";" in req_str:
            req_part, marker_part = req_str.split(";", 1)
            marker_part = marker_part.strip()

            # Substitute #var# placeholders with actual values
            marker_part = substitute_variables(marker_part, variables)

            try:
                req = Requirement(req_str)
                if req.marker is None or req.marker.evaluate(env):
                    result.append(f"{req.name}{req.specifier}")
                    continue
            except InvalidRequirement:
                if is_custom_marker(marker_part):
                    if eval_custom_marker(marker_part, env):
                        req = Requirement(req_part.strip())
                        result.append(str(req))
                else:
                    raise
        else:
            req = Requirement(req_str.strip())
            result.append(str(req))

    return result
