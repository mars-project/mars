# Copyright 1999-2021 Alibaba Group Holding Ltd.
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
import shutil
import subprocess
import sys
import warnings
from sysconfig import get_config_var

from pkg_resources import parse_version
from setuptools import setup, Extension, Command

import numpy as np
from Cython.Build import cythonize
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.sdist import sdist

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
        target_macos_version = "10.9"
        parsed_macos_version = parse_version(target_macos_version)

        current_system = parse_version(platform.mac_ver()[0])
        python_target = parse_version(get_config_var("MACOSX_DEPLOYMENT_TARGET"))
        if python_target < parsed_macos_version <= current_system:
            os.environ["MACOSX_DEPLOYMENT_TARGET"] = target_macos_version


repo_root = os.path.dirname(os.path.abspath(__file__))


cythonize_kw = dict(language_level=sys.version_info[0])
cy_extension_kw = dict()
if os.environ.get("CYTHON_TRACE"):
    cy_extension_kw["define_macros"] = [
        ("CYTHON_TRACE_NOGIL", "1"),
        ("CYTHON_TRACE", "1"),
    ]
    cythonize_kw["compiler_directives"] = {"linetrace": True}

if "MSC" in sys.version:
    extra_compile_args = ["/Ot", "/I" + os.path.join(repo_root, "misc")]
    cy_extension_kw["extra_compile_args"] = extra_compile_args
else:
    extra_compile_args = ["-O3"]
    cy_extension_kw["extra_compile_args"] = extra_compile_args


def _discover_pyx():
    exts = dict()
    for root, _, files in os.walk(os.path.join(repo_root, "mars")):
        for fn in files:
            if not fn.endswith(".pyx"):
                continue
            full_fn = os.path.relpath(os.path.join(root, fn), repo_root)
            mod_name = full_fn.replace(".pyx", "").replace(os.path.sep, ".")
            exts[mod_name] = Extension(mod_name, [full_fn], **cy_extension_kw)
    return exts


cy_extension_kw["include_dirs"] = [np.get_include()]
extensions_dict = _discover_pyx()
cy_extensions = list(extensions_dict.values())

extensions = cythonize(cy_extensions, **cythonize_kw) + [
    Extension(
        "mars.lib.mmh3",
        ["mars/lib/mmh3_src/mmh3module.cpp", "mars/lib/mmh3_src/MurmurHash3.cpp"],
    )
]


class ExtraCommandMixin:
    _extra_pre_commands = []

    def run(self):
        [self.run_command(cmd) for cmd in self._extra_pre_commands]
        super().run()

    @classmethod
    def register_pre_command(cls, cmd):
        cls._extra_pre_commands.append(cmd)


class CustomInstall(ExtraCommandMixin, install):
    pass


class CustomDevelop(ExtraCommandMixin, develop):
    pass


class CustomSDist(ExtraCommandMixin, sdist):
    pass


class BuildWeb(Command):
    """build_web command"""

    user_options = []
    _web_src_path = "mars/services/web/ui"
    _web_dest_path = "mars/services/web/static/bundle.js"
    _commands = [
        ["npm", "install"],
        ["npm", "run", "bundle"],
    ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    @classmethod
    def run(cls):
        if int(os.environ.get("NO_WEB_UI", "0")):
            return

        npm_path = shutil.which("npm")
        web_src_path = os.path.join(repo_root, *cls._web_src_path.split("/"))
        web_dest_path = os.path.join(repo_root, *cls._web_dest_path.split("/"))

        if not os.path.exists(web_src_path):
            return
        elif npm_path is None:
            if not os.path.exists(web_dest_path):
                warnings.warn("Cannot find NPM, may affect displaying Mars Web")
            return

        replacements = {"npm": npm_path}
        cmd_errored = False
        for cmd in cls._commands:
            cmd = [replacements.get(c, c) for c in cmd]
            proc_result = subprocess.run(cmd, cwd=web_src_path)
            if proc_result.returncode != 0:
                warnings.warn(f'Failed when running `{" ".join(cmd)}`')
                cmd_errored = True
                break
        if not cmd_errored:
            assert os.path.exists(cls._web_dest_path)


CustomInstall.register_pre_command("build_web")
CustomDevelop.register_pre_command("build_web")
CustomSDist.register_pre_command("build_web")


# Resolve path issue of versioneer
sys.path.append(repo_root)
versioneer = __import__("versioneer")


setup_options = dict(
    version=versioneer.get_version(),
    ext_modules=extensions,
    cmdclass=versioneer.get_cmdclass(
        {
            "build_web": BuildWeb,
            "install": CustomInstall,
            "develop": CustomDevelop,
            "sdist": CustomSDist,
        }
    ),
)
setup(**setup_options)
