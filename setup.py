# Copyright 1999-2020 Alibaba Group Holding Ltd.
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
from distutils.cmd import Command
from distutils.spawn import find_executable
from distutils.sysconfig import get_config_var
from distutils.version import LooseVersion
from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.sdist import sdist

import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext


# From https://github.com/pandas-dev/pandas/pull/24274:
# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distuitls behaviour which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.py
if sys.platform == 'darwin':
    if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        current_system = LooseVersion(platform.mac_ver()[0])
        python_target = LooseVersion(
            get_config_var('MACOSX_DEPLOYMENT_TARGET'))
        if python_target < '10.9' and current_system >= '10.9':
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'


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

requirements = []
with open(os.path.join(repo_root, 'requirements.txt'), 'r') as f:
    requirements.extend(f.read().splitlines())

extra_requirements = []
with open(os.path.join(repo_root, 'requirements-extra.txt'), 'r') as f:
    extra_requirements.extend(f.read().splitlines())

dev_requirements = []
with open(os.path.join(repo_root, 'requirements-dev.txt'), 'r') as f:
    dev_requirements.extend(f.read().splitlines())

long_description = None
if os.path.exists(os.path.join(repo_root, 'README.rst')):
    with open(os.path.join(repo_root, 'README.rst')) as f:
        long_description = f.read()


if os.path.exists(os.path.join(repo_root, '.git')):
    git_info = version_ns['get_git_info']()
    if git_info:
        with open(os.path.join(repo_root, 'mars', '.git-branch'), 'w') as git_file:
            git_file.write('%s %s' % git_info)

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

cy_extension_kw['include_dirs'] = [np.get_include()]
cy_extensions = [
    Extension('mars.graph', ['mars/graph.pyx'], **cy_extension_kw),
    Extension('mars.optimizes.chunk_graph.fuse', ['mars/optimizes/chunk_graph/fuse.pyx'], **cy_extension_kw),
    Extension('mars._utils', ['mars/_utils.pyx'], **cy_extension_kw),
    Extension('mars.lib.gipc', ['mars/lib/gipc.pyx'], **cy_extension_kw),
    Extension('mars.actors.core', ['mars/actors/core.pyx'], **cy_extension_kw),
    Extension('mars.actors.distributor', ['mars/actors/distributor.pyx'], **cy_extension_kw),
    Extension('mars.actors.cluster', ['mars/actors/cluster.pyx'], **cy_extension_kw),
    Extension('mars.actors.pool.messages', ['mars/actors/pool/messages.pyx'], **cy_extension_kw),
    Extension('mars.actors.pool.utils', ['mars/actors/pool/utils.pyx'], **cy_extension_kw),
    Extension('mars.actors.pool.gevent_pool', ['mars/actors/pool/gevent_pool.pyx'], **cy_extension_kw),
    Extension('mars.serialize.core', ['mars/serialize/core.pyx'], **cy_extension_kw),
    Extension('mars.serialize.pbserializer', ['mars/serialize/pbserializer.pyx'], **cy_extension_kw),
    Extension('mars.serialize.jsonserializer', ['mars/serialize/jsonserializer.pyx'], **cy_extension_kw),
]

extensions = cythonize(cy_extensions, **cythonize_kw) + \
    [Extension('mars.lib.mmh3', ['mars/lib/mmh3_src/mmh3module.cpp', 'mars/lib/mmh3_src/MurmurHash3.cpp'])]


class BuildProto(Command):
    description = "Build protobuf file"
    user_options = []
    protoc_executable = None

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def _get_protoc_executable(self):
        cls = type(self)
        if cls.protoc_executable:
            return cls.protoc_executable

        cls.protoc_executable = find_executable('protoc') or os.environ.get('PROTOC')
        if cls.protoc_executable is None:
            sys.stderr.write('Cannot find protoc. Make sure it is installed. '
                             'You may point it with PROTOC environment variable.\n')
            sys.exit(1)
        return cls.protoc_executable

    def run(self):
        protoc_executable = find_executable('protoc') or os.environ.get('PROTOC')
        if protoc_executable is None:
            sys.stderr.write('Cannot find protoc. Make sure it is installed. '
                             'You may point it with PROTOC environment variable.\n')
            sys.exit(1)

        for root, _, files in os.walk(repo_root):
            for fn in files:
                if not fn.endswith('.proto'):
                    continue

                src_fn = os.path.join(root, fn)
                compiled_fn = src_fn.replace('.proto', '_pb2.py')
                rel_path = os.path.relpath(src_fn, repo_root)
                if not os.path.exists(compiled_fn) or \
                        os.path.getmtime(src_fn) > os.path.getmtime(compiled_fn):
                    sys.stdout.write('compiling protobuf %s\n' % rel_path)
                    subprocess.check_call([self._get_protoc_executable(), '--python_out=.', rel_path],
                                          cwd=repo_root)

                if fn == 'operand_type.proto':
                    op_globs = dict()
                    opcode_fn = os.path.join(repo_root, 'mars', 'opcodes.py')
                    if not os.path.exists(opcode_fn) or \
                            os.path.getmtime(src_fn) > os.path.getmtime(opcode_fn):
                        execfile(compiled_fn, op_globs)
                        OperandType = op_globs['OperandType']
                        with open(os.path.join(repo_root, 'mars', 'opcodes.py'), 'w') as opcode_file:
                            opcode_file.write('# Generated automatically from %s.  DO NOT EDIT!\n' % rel_path)
                            for val, desc in OperandType.DESCRIPTOR.values_by_number.items():
                                opcode_file.write('{0} = {1!r}\n'.format(desc.name, val))


class CustomBuildPy(build_py):
    def run(self):
        self.run_command('build_proto')
        build_py.run(self)


class CustomDevelop(develop):
    def run(self):
        self.run_command('build_proto')
        develop.run(self)


class CustomSDist(sdist):
    def run(self):
        self.run_command('build_proto')
        sdist.run(self)


setup_options = dict(
    name='pymars',
    version=version,
    description='MARS: a tensor-based unified framework for large-scale data computation.',
    long_description=long_description,
    author='Qin Xuye',
    author_email='qin@qinxuye.me',
    maintainer='Qin Xuye',
    maintainer_email='qin@qinxuye.me',
    url='http://github.com/mars-project/mars',
    license='Apache License 2.0',
    classifiers=[
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Software Development :: Libraries',
    ],
    packages=find_packages(exclude=('*.tests.*', '*.tests')),
    include_package_data=True,
    entry_points={'console_scripts': [
        'mars-scheduler = mars.scheduler.__main__:main',
        'mars-worker = mars.worker.__main__:main',
        'mars-web = mars.web.__main__:main',
    ]},
    python_requires='>=3.5',
    install_requires=requirements,
    cmdclass={'build_py': CustomBuildPy, 'sdist': CustomSDist, 'develop': CustomDevelop,
              'build_ext': build_ext, 'build_proto': BuildProto},
    ext_modules=extensions,
    extras_require={
        'distributed': extra_requirements,
        'dev': extra_requirements + dev_requirements,
    }
)
setup(**setup_options)
