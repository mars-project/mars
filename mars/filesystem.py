# Copyright 1999-2018 Alibaba Group Holding Ltd.
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
from gzip import GzipFile

from pyarrow import LocalFileSystem as ArrowLocalFileSystem
try:
    import lz4, lz4.frame
except ImportError:
    lz4 = None

from .compat import urlparse


compressions = {
    'gzip': lambda f: GzipFile(fileobj=f)
}

if lz4:
    compressions['lz4'] = lz4.frame.open


class LocalFileSystem(ArrowLocalFileSystem):
    def __init__(self, **_):
        super(LocalFileSystem, self).__init__()

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


file_systems = {
    'file': LocalFileSystem
}


def get_fs(path, storage_options):
    parse_result = urlparse(path)
    scheme = parse_result.scheme or 'file'
    storage_options = storage_options or dict()
    return file_systems[scheme](**storage_options)


def open_file(path, mode='rb', compression=None, storage_options=None):
    fs = get_fs(path, storage_options)
    f = fs.open(path, mode=mode)

    if compression is not None:
        compress = compressions[compression]
        f = compress(f)

    return f


def file_size(path, storage_options=None):
    fs = get_fs(path, storage_options)
    return fs.stat(path)['size']
