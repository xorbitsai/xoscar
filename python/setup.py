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
import sysconfig
from distutils.command.build_ext import build_ext as _du_build_ext
from distutils.file_util import copy_file, move_file
from pathlib import Path

from sysconfig import get_config_vars

import numpy as np
from Cython.Build import cythonize
from packaging.version import Version
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib
from setuptools.extension import Library

try:
    import distutils.ccompiler

    if sys.platform != "win32":
        from numpy.distutils.ccompiler import CCompiler_compile

        distutils.ccompiler.CCompiler.compile = CCompiler_compile
except ImportError:
    pass

try:
    # Attempt to use Cython for building extensions, if available
    from Cython.Distutils.build_ext import build_ext as _build_ext

    # Additionally, assert that the compiler module will load
    # also. Ref #1229.
    __import__('Cython.Compiler.Main')
except ImportError:
    _build_ext = _du_build_ext

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

        parsed_python_target = Version(python_target)
        parsed_current_system = Version(current_system)
        parsed_macos_version = Version(target_macos_version)
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


# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

TARGET_TO_PLAT = {
    'x86': 'win32',
    'x64': 'win-amd64',
    'arm': 'win-arm32',
    'arm64': 'win-arm64',
}


# Copied from https://github.com/pypa/setuptools/blob/main/setuptools/_distutils/util.py#L50
def get_host_platform():
    """
    Return a string that identifies the current platform. Use this
    function to distinguish platform-specific build directories and
    platform-specific built distributions.
    """

    # This function initially exposed platforms as defined in Python 3.9
    # even with older Python versions when distutils was split out.
    # Now it delegates to stdlib sysconfig, but maintains compatibility.
    return sysconfig.get_platform()


def get_platform():
    if os.name == 'nt':
        target = os.environ.get('VSCMD_ARG_TGT_ARCH')
        return TARGET_TO_PLAT.get(target) or get_host_platform()
    return get_host_platform()


plat_specifier = ".{}-{}".format(get_platform(), sys.implementation.cache_tag)


def get_build_lib():
    return os.path.join("build", "lib" + plat_specifier)


def get_build_temp():
    return os.path.join("build", 'temp' + plat_specifier)


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class XoscarCmakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def finalize_options(self):
        """
        For python 3.12, the build_temp and build_lib dirs are temp dirs which are depended on your OS,
        which leads to that cannot find the copy directory during C++ compiled process.
        However, for Python < 3.12, these two dirs can be automatically located in the `build` directory of the project directory.
        Therefore, in order to be compatible with all Python versions,
        directly using fixed dirs here by coping source codes from `setuptools`.
        """
        self.build_temp = get_build_temp()
        self.build_lib = get_build_lib()
        super().finalize_options()

    def copy_extensions_to_source(self):
        build_py = self.get_finalized_command('build_py')
        for ext in self.extensions:
            if not isinstance(ext, XoscarCmakeExtension):
                fullname = self.get_ext_fullname(ext.name)
                filename = self.get_ext_filename(fullname)
                modpath = fullname.split('.')
                package = '.'.join(modpath[:-1])
                package_dir = build_py.get_package_dir(package)
                dest_filename = os.path.join(package_dir,
                                             os.path.basename(filename))
                src_filename = os.path.join(self.build_lib, filename)

                # Always copy, even if source is older than destination, to ensure
                # that the right extensions for the current Python/platform are
                # used.
                copy_file(
                    src_filename, dest_filename, verbose=self.verbose,
                    dry_run=self.dry_run
                )
                if ext._needs_stub:
                    self.write_stub(package_dir or os.curdir, ext, True)
            else:
                fullname = self.get_ext_fullname(ext.name)
                collective_dir = os.path.join("xoscar" , "collective")
                filename = self.get_ext_filename(fullname)
                src_dir = os.path.join(self.build_lib , collective_dir)
                src_filename = os.path.join(src_dir , filename)
                dest_filename = os.path.join(collective_dir,
                                                os.path.basename(filename))
                copy_file(
                    src_filename, dest_filename, verbose=self.verbose,
                    dry_run=self.dry_run
                )


    def build_extension(self, ext):
        # TODO: support windows compilation
        is_windows = sys.platform.startswith('win')
        bit_number = platform.architecture()[0]
        if isinstance(ext, XoscarCmakeExtension) and not (is_windows and bit_number=="32bit"):
            self.build_Cmake(ext)
        elif isinstance(ext, XoscarCmakeExtension) and (is_windows and bit_number=="32bit"):
            pass
        else:
            ext._convert_pyx_sources_to_lang()
            _compiler = self.compiler
            try:
                if isinstance(ext, Library):
                    self.compiler = self.shlib_compiler
                _build_ext.build_extension(self, ext)
                if ext._needs_stub:
                    build_lib = self.get_finalized_command('build_py').build_lib
                    self.write_stub(build_lib, ext)
            finally:
                self.compiler = _compiler

    def build_Cmake(self, ext: XoscarCmakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DBUILD_TMP_DIR={build_temp}",
            f"-DPYTHON_PATH={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.10",
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]



        subprocess.run(
            ["cmake", source_dir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )
        if sys.platform.startswith('win'):
            for file in os.listdir(self.build_lib):
                if file.startswith("xoscar_pygloo"):
                    src_filename = os.path.join(self.build_lib,
                                            os.path.basename(file))
                    dest_dir = os.path.join(self.build_lib,
                                                    "xoscar\\collective")
                    if not os.path.exists(dest_dir):
                        os.mkdir(dest_dir)
                    dest_filename = os.path.join(dest_dir,
                                            os.path.basename(file))
                    move_file(
                        src_filename, dest_filename, verbose=self.verbose,
                        dry_run=self.dry_run
                    )
                    libuv_filename = "xoscar\\collective\\uv.dll"
                    libuv_dest_filename = os.path.join(dest_dir, "uv.dll")
                    copy_file(
                        libuv_filename, libuv_dest_filename, verbose=self.verbose,
                        dry_run=self.dry_run
                    )


setup_options = dict(
    version=versioneer.get_version(),
    ext_modules=extensions + [XoscarCmakeExtension("xoscar_pygloo")],
    cmdclass={"build_ext": CMakeBuild},
    long_description=build_long_description(),
    long_description_content_type="text/markdown",
)
setup(**setup_options)
