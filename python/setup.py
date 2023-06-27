# Copyright 2022-2023 XProbe Inc.
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

import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from sysconfig import get_config_vars

import numpy as np
from Cython.Build import cythonize
from pkg_resources import parse_version
from setuptools import Extension, setup

try:
    import distutils.ccompiler

    if sys.platform != "win32":
        from numpy.distutils.ccompiler import CCompiler_compile

        distutils.ccompiler.CCompiler.compile = CCompiler_compile
except ImportError:
    pass

# From https://github.com/pandas-dev/pandas/pull/24274:
# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distuitls behaviour which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py
if sys.platform == "darwin":
    if "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
        current_system = platform.mac_ver()[0]
        python_target = get_config_vars().get(
            "MACOSX_DEPLOYMENT_TARGET", current_system
        )
        target_macos_version = "10.9"

        parsed_python_target = parse_version(python_target)
        parsed_current_system = parse_version(current_system)
        parsed_macos_version = parse_version(target_macos_version)
        if parsed_python_target <= parsed_macos_version <= parsed_current_system:
            os.environ["MACOSX_DEPLOYMENT_TARGET"] = target_macos_version


repo_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(repo_root)


cythonize_kw = dict(language_level=sys.version_info[0])
define_macros = []
if os.environ.get("CYTHON_TRACE"):
    define_macros.append(("CYTHON_TRACE_NOGIL", "1"))
    define_macros.append(("CYTHON_TRACE", "1"))
    cythonize_kw["compiler_directives"] = {"linetrace": True}

# Fixes Python 3.11 compatibility issue
#
# see also:
# - https://github.com/cython/cython/issues/4500
# - https://github.com/scoder/cython/commit/37f270155dbb5907435a1e83fdc90e403d776af5
if sys.version_info >= (3, 11):
    define_macros.append(("CYTHON_FAST_THREAD_STATE", "0"))

cy_extension_kw = {
    'define_macros': define_macros,
}

if "MSC" in sys.version:
    extra_compile_args = ["/std:c11", "/Ot", "/I" + os.path.join(repo_root, "misc")]
    cy_extension_kw["extra_compile_args"] = extra_compile_args
else:
    extra_compile_args = ["-O3"]
    if sys.platform != "darwin":
        # for macOS, we assume that C++ 11 is enabled by default
        extra_compile_args.append("-std=c++0x")
    cy_extension_kw["extra_compile_args"] = extra_compile_args


def _discover_pyx():
    exts = dict()
    for root, _, files in os.walk(os.path.join(repo_root, "xoscar"), followlinks=True):
        for fn in files:
            if not fn.endswith(".pyx"):
                continue
            full_fn = os.path.relpath(os.path.join(root, fn), repo_root)
            mod_name = full_fn.replace(".pyx", "").replace(os.path.sep, ".")
            exts[mod_name] = Extension(mod_name, [full_fn], include_dirs=[np.get_include()], **cy_extension_kw)
    return exts


extensions_dict = _discover_pyx()
cy_extensions = list(extensions_dict.values())

extensions = cythonize(cy_extensions, **cythonize_kw)


# Resolve path issue of versioneer
sys.path.append(repo_root)
versioneer = __import__("versioneer")


# build long description
def build_long_description():
    readme_path = os.path.join(os.path.dirname(os.path.abspath(repo_root)), "README.md")

    with open(readme_path, encoding="utf-8") as f:
        return f.read()


def build_cpp():
    source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    debug = int(os.environ.get("DEBUG", 0))
    cfg = "Debug" if debug else "Release"

    # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
    # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
    # from Python.
    output_directory = Path(source_dir) / "python" / "xoscar" / "collective" / "rendezvous"
    cmake_args = [
        f"-DLIBRARY_OUTPUT_DIRECTORY={output_directory}",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
        f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
    ]

    build_args = []
    # Adding CMake arguments set as environment variable
    # (needed e.g. to build for ARM OSx on conda-forge)
    if "CMAKE_ARGS" in os.environ:
        cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

    if sys.platform.startswith("darwin"):
        # Cross-compile support for macOS - respect ARCHFLAGS if set
        archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
        if archs:
            cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

    build_temp = Path(source_dir) / "build"
    if not build_temp.exists():
        build_temp.mkdir(parents=True)

    subprocess.run(
        ["cmake", source_dir, *cmake_args], cwd=build_temp, check=True
    )
    subprocess.run(
        ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
    )


setup_options = dict(
    version=versioneer.get_version(),
    ext_modules=extensions,
    long_description=build_long_description(),
    long_description_content_type="text/markdown",
)
setup(**setup_options)
build_cpp()
