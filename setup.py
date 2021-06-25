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
import re
import sys
from sysconfig import get_config_var

from pkg_resources import parse_version
from setuptools import setup, Extension

import numpy as np
from Cython.Build import cythonize

try:
    import distutils.ccompiler
    if sys.platform != 'win32':
        from numpy.distutils.ccompiler import CCompiler_compile
        distutils.ccompiler.CCompiler.compile = CCompiler_compile
except ImportError:
    pass

# From https://github.com/pandas-dev/pandas/pull/24274:
# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distuitls behaviour which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py
if sys.platform == 'darwin':
    if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        target_macos_version = "10.9"
        parsed_macos_version = parse_version(target_macos_version)

        current_system = parse_version(platform.mac_ver()[0])
        python_target = parse_version(
            get_config_var('MACOSX_DEPLOYMENT_TARGET'))
        if python_target < parsed_macos_version <= current_system:
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = target_macos_version


repo_root = os.path.dirname(os.path.abspath(__file__))


def execfile(fname, globs, locs=None):
    locs = locs or globs
    exec(compile(open(fname).read(), fname, "exec"), globs, locs)


version_file_path = os.path.join(repo_root, 'mars', '_version.py')
version_ns = {'__file__': version_file_path}
execfile(version_file_path, version_ns)
version = version_ns['__version__']
# check version vs tag
if os.environ.get('GIT_TAG') and re.search(r'v\d', os.environ['GIT_TAG']) \
        and os.environ['GIT_TAG'] != 'v' + version:
    raise ValueError('Tag %r does not match source version %r'
                     % (os.environ['GIT_TAG'], version))


if os.path.exists(os.path.join(repo_root, '.git')):
    git_info = version_ns['get_git_info']()
    if git_info:
        with open(os.path.join(repo_root, 'mars', '.git-branch'), 'w') as git_file:
            git_file.write(' '.join(git_info))

cythonize_kw = dict(language_level=sys.version_info[0])
cy_extension_kw = dict()
if os.environ.get('CYTHON_TRACE'):
    cy_extension_kw['define_macros'] = [('CYTHON_TRACE_NOGIL', '1'), ('CYTHON_TRACE', '1')]
    cythonize_kw['compiler_directives'] = {'linetrace': True}

if 'MSC' in sys.version:
    extra_compile_args = ['/Ot', '/I' + os.path.join(repo_root, 'misc')]
    cy_extension_kw['extra_compile_args'] = extra_compile_args
else:
    extra_compile_args = ['-O3']
    cy_extension_kw['extra_compile_args'] = extra_compile_args


def _discover_pyx():
    exts = dict()
    for root, _, files in os.walk(os.path.join(repo_root, 'mars')):
        for fn in files:
            if not fn.endswith('.pyx'):
                continue
            full_fn = os.path.relpath(os.path.join(root, fn), repo_root)
            mod_name = full_fn.replace('.pyx', '').replace(os.path.sep, '.')
            exts[mod_name] = Extension(mod_name, [full_fn], **cy_extension_kw)
    return exts


cy_extension_kw['include_dirs'] = [np.get_include()]
extensions_dict = _discover_pyx()
cy_extensions = list(extensions_dict.values())

extensions = cythonize(cy_extensions, **cythonize_kw) + \
    [Extension('mars.lib.mmh3', ['mars/lib/mmh3_src/mmh3module.cpp', 'mars/lib/mmh3_src/MurmurHash3.cpp'])]


setup_options = dict(
    version=version,
    ext_modules=extensions,
)
setup(**setup_options)
