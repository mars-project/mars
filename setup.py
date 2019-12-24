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
    Extension('mars.actors.core', ['mars/actors/core.pyx'], **cy_extension_kw),
    Extension('mars.actors.distributor', ['mars/actors/distributor.pyx'], **cy_extension_kw),
    Extension('mars.actors.cluster', ['mars/actors/cluster.pyx'], **cy_extension_kw),
    Extension('mars.actors.pool.messages', ['mars/actors/pool/messages.pyx'], **cy_extension_kw),
    Extension('mars.actors.pool.utils', ['mars/actors/pool/utils.pyx'], **cy_extension_kw),
    Extension('mars.actors.pool.aio_pool', ['mars/actors/pool/aio_pool.pyx'], **cy_extension_kw),
    Extension('mars.serialize.core', ['mars/serialize/core.pyx'], **cy_extension_kw),
    Extension('mars.serialize.pbserializer', ['mars/serialize/pbserializer.pyx'], **cy_extension_kw),
    Extension('mars.serialize.jsonserializer', ['mars/serialize/jsonserializer.pyx'], **cy_extension_kw),
]

extensions = cythonize(cy_extensions, **cythonize_kw) + \
    [Extension('mars.lib.mmh3', ['mars/lib/mmh3_src/mmh3module.cpp', 'mars/lib/mmh3_src/MurmurHash3.cpp'])]


build_protos_cmd = os.path.join(repo_root, 'bin', 'build-protos.py')
override_cmd_class = dict()
if os.path.exists(build_protos_cmd):
    class BuildProto(Command):
        description = "Build protobuf file"
        user_options = []
        protobuf_version = '3.6.0'

        def initialize_options(self):
            pass

        def finalize_options(self):
            pass

        def run(self):
            subprocess.check_call([sys.executable, build_protos_cmd, repo_root], env=os.environ.copy())


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

    override_cmd_class = {
        'build_py': CustomBuildPy, 'sdist': CustomSDist, 'develop': CustomDevelop,
        'build_ext': build_ext, 'build_proto': BuildProto}


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
    cmdclass=override_cmd_class,
    ext_modules=extensions,
    extras_require={
        'distributed': extra_requirements,
        'dev': extra_requirements + dev_requirements,
    }
)
setup(**setup_options)
