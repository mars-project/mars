# Copyright 1999-2017 Alibaba Group Holding Ltd.
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
import sys
from setuptools import setup, find_packages, Extension

import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext

repo_root = os.path.dirname(os.path.abspath(__file__))

try:
    execfile
except NameError:
    def execfile(fname, globs, locs=None):
        locs = locs or globs
        exec(compile(open(fname).read(), fname, "exec"), globs, locs)

version_file_path = os.path.join(repo_root, 'mars', '_version.py')
version_ns = {'__file__': version_file_path}
execfile(version_file_path, version_ns)

requirements = []
with open(os.path.join(repo_root, 'requirements.txt'), 'r') as f:
    requirements.extend(f.read().splitlines())


extra_requirements = []
with open(os.path.join(repo_root, 'requirements-extra.txt'), 'r') as f:
    extra_requirements.extend(f.read().splitlines())


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
extension_kw = dict()
if 'CYTHON_TRACE' in os.environ:
    extension_kw['define_macros'] = [('CYTHON_TRACE_NOGIL', '1'), ('CYTHON_TRACE', '1')]
    cythonize_kw['compiler_directives'] = {'linetrace': True}

if 'MSC' in sys.version:
    extra_compile_args = ['/Ot', '/I' + os.path.join(repo_root, 'misc')]
    extension_kw['extra_compile_args'] = extra_compile_args
else:
    extra_compile_args = ['-O3']
    extension_kw['extra_compile_args'] = extra_compile_args

extension_kw['include_dirs'] = [np.get_include()]
extensions = [
    Extension('mars.graph', ['mars/graph.pyx'], **extension_kw),
    Extension('mars.fuse', ['mars/fuse.pyx'], **extension_kw),
    Extension('mars.utils_c', ['mars/utils_c.pyx'], **extension_kw),
    Extension('mars.lib.gipc', ['mars/lib/gipc.pyx'], **extension_kw),
    Extension('mars.actors.core', ['mars/actors/core.pyx'], **extension_kw),
    Extension('mars.actors.distributor', ['mars/actors/distributor.pyx'], **extension_kw),
    Extension('mars.actors.cluster', ['mars/actors/cluster.pyx'], **extension_kw),
    Extension('mars.actors.pool.messages', ['mars/actors/pool/messages.pyx'], **extension_kw),
    Extension('mars.actors.pool.utils', ['mars/actors/pool/utils.pyx'], **extension_kw),
    Extension('mars.actors.pool.gevent_pool', ['mars/actors/pool/gevent_pool.pyx'], **extension_kw),
    Extension('mars.serialize.core', ['mars/serialize/core.pyx'], **extension_kw),
    Extension('mars.serialize.pbserializer', ['mars/serialize/pbserializer.pyx'], **extension_kw),
    Extension('mars.serialize.jsonserializer', ['mars/serialize/jsonserializer.pyx'], **extension_kw),
]


setup_options = dict(
    name='pymars',
    version=version_ns['__version__'],
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
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
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
    install_requires=requirements,
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extensions, **cythonize_kw),
    extras_require={'distributed': extra_requirements}
)
setup(**setup_options)
