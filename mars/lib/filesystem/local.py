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

import glob
import os
import shutil
from typing import List, Dict, Union, Tuple, Iterator, BinaryIO, TextIO

from ...utils import implements, stringify_path
from .base import FileSystem, path_type


class LocalFileSystem(FileSystem):

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = LocalFileSystem()
        return cls._instance

    @implements(FileSystem.cat)
    def cat(self, path: path_type):
        with self.open(path, 'rb') as f:
            return f.read()

    @implements(FileSystem.ls)
    def ls(self, path: path_type) -> List[path_type]:
        path = stringify_path(path)
        return sorted(os.path.join(path, x) for x in os.listdir(path))

    @implements(FileSystem.delete)
    def delete(self,
               path: path_type,
               recursive: bool = False):
        if os.path.isfile(path):
            os.remove(path)
        elif not recursive:
            os.rmdir(path)
        else:
            shutil.rmtree(path)

    @implements(FileSystem.rename)
    def rename(self,
               path: path_type,
               new_path: path_type):
        os.rename(path, new_path)

    @implements(FileSystem.stat)
    def stat(self, path: path_type) -> Dict:
        os_stat = os.stat(path)
        stat = dict(name=path, size=os_stat.st_size,
                    modified_time=os_stat.st_mtime)
        if os.path.isfile(path):
            stat['type'] = 'file'
        elif os.path.isdir(path):
            stat['type'] = 'directory'
        else:  # pragma: no cover
            stat['type'] = 'other'
        return stat

    @implements(FileSystem.mkdir)
    def mkdir(self,
              path: path_type,
              create_parents: bool = True):
        path = stringify_path(path)
        if create_parents:
            os.makedirs(path)
        else:
            os.mkdir(path)

    @implements(FileSystem.isdir)
    def isdir(self, path: path_type) -> bool:
        path = stringify_path(path)
        return os.path.isdir(path)

    @implements(FileSystem.isfile)
    def isfile(self, path: path_type) -> bool:
        path = stringify_path(path)
        return os.path.isfile(path)

    @implements(FileSystem._isfilestore)
    def _isfilestore(self) -> bool:
        return True

    @implements(FileSystem.exists)
    def exists(self, path: path_type):
        path = stringify_path(path)
        return os.path.exists(path)

    @implements(FileSystem.open)
    def open(self,
             path: path_type,
             mode: str = 'rb') -> Union[BinaryIO, TextIO]:
        path = stringify_path(path)
        return open(path, mode=mode)

    @implements(FileSystem.walk)
    def walk(self, path: path_type) -> Iterator[Tuple[str, List[str], List[str]]]:
        path = stringify_path(path)
        return os.walk(path)

    @implements(FileSystem.glob)
    def glob(self,
             path: path_type,
             recursive: bool = False) -> List[path_type]:
        path = stringify_path(path)
        return glob.glob(path, recursive=recursive)

    @property
    def pathsep(self) -> str:
        return os.path.sep
