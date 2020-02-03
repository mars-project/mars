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
import glob as local_glob
from gzip import GzipFile

from pyarrow import LocalFileSystem as ArrowLocalFileSystem
from pyarrow import HadoopFileSystem
try:
    import lz4
    import lz4.frame
except ImportError:  # pragma: no cover
    lz4 = None

from .compat import urlparse


compressions = {
    'gzip': lambda f: GzipFile(fileobj=f)
}

if lz4:
    compressions['lz4'] = lz4.frame.open


class LocalFileSystem(ArrowLocalFileSystem):
    _fs_instance = None

    @classmethod
    def get_instance(cls):
        if cls._fs_instance is None:
            cls._fs_instance = LocalFileSystem()
        return cls._fs_instance

    def stat(self, path):
        os_stat = os.stat(path)
        stat = dict(name=path, size=os_stat.st_size, created=os_stat.st_ctime)
        if os.path.isfile(path):
            stat['type'] = 'file'
        elif os.path.isdir(path):
            stat['type'] = 'directory'
        else:
            stat['type'] = 'other'
        return stat

    @staticmethod
    def glob(path):
        return local_glob.glob(path)


file_systems = {
    'file': LocalFileSystem,
    'hdfs': HadoopFileSystem
}


def _parse_from_path(uri):
    parsed_uri = urlparse(uri)
    options = dict()
    options['host'] = parsed_uri.netloc.rsplit("@", 1)[-1].rsplit(":", 1)[0]
    if parsed_uri.port:
        options["port"] = parsed_uri.port
    if parsed_uri.username:
        options["user"] = parsed_uri.username
    if parsed_uri.password:
        options["password"] = parsed_uri.password
    return options


def get_fs(path, storage_options):
    if os.path.exists(path) or local_glob.glob(path):
        scheme = 'file'
    else:
        scheme = urlparse(path).scheme
    if scheme == '' or len(scheme) == 1:  # len == 1 for windows
        scheme = 'file'
    if scheme == 'file':
        return file_systems[scheme].get_instance()
    else:
        options = _parse_from_path(path)
        storage_options = storage_options or dict()
        storage_options.update(options)
        return file_systems[scheme](**storage_options)


def open_file(path, mode='rb', compression=None, storage_options=None):
    fs = get_fs(path, storage_options)
    f = fs.open(path, mode=mode)

    if compression is not None:
        compress = compressions[compression]
        f = compress(f)

    return f


def glob(path, storage_options=None):
    if '*' in path:
        fs = get_fs(path, storage_options)
        return fs.glob(path)
    else:
        return [path]


def file_size(path, storage_options=None):
    fs = get_fs(path, storage_options)
    return fs.stat(path)['size']
