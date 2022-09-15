# Copyright 2022 XProbe Inc.
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

from typing import List, Iterator, Tuple, Union, BinaryIO, TextIO, Dict

from fsspec import filesystem
from fsspec.core import stringify_path

from ...utils import implements
from .core import FileSystem
from .core import path_type


class FsSpecAdapter(FileSystem):
    def __init__(self, scheme: str, **kwargs):
        self._fs = filesystem(scheme, **kwargs)

    @implements(FileSystem.cat)
    def cat(self, path: path_type) -> bytes:
        return self._fs.cat_file(stringify_path(path))

    @implements(FileSystem.ls)
    def ls(self, path: path_type) -> List[path_type]:
        entries = []
        for entry in self._fs.ls(stringify_path(path), detail=False):
            if isinstance(entry, Dict):
                entries.append(entry.get("name"))
            elif isinstance(entry, str):
                entries.append(entry)
            else:  # pragma: no cover
                raise TypeError(f"Expect str or dict, but got {type(entry)}")
        return entries

    @implements(FileSystem.delete)
    def delete(self, path: path_type, recursive: bool = False):
        raise NotImplementedError

    @implements(FileSystem.stat)
    def stat(self, path: path_type) -> Dict:
        return self._fs.info(stringify_path(path))

    @implements(FileSystem.rename)
    def rename(self, path: path_type, new_path: path_type):
        raise NotImplementedError

    @implements(FileSystem.mkdir)
    def mkdir(self, path: path_type, create_parents: bool = True):
        raise NotImplementedError

    @implements(FileSystem.exists)
    def exists(self, path: path_type):
        return self._fs.exists(stringify_path(path))

    @implements(FileSystem.isdir)
    def isdir(self, path: path_type) -> bool:
        return self._fs.isdir(stringify_path(path))

    @implements(FileSystem.isfile)
    def isfile(self, path: path_type) -> bool:
        return self._fs.isfile(stringify_path(path))

    @implements(FileSystem._isfilestore)
    def _isfilestore(self) -> bool:
        raise NotImplementedError

    @implements(FileSystem.open)
    def open(self, path: path_type, mode: str = "rb") -> Union[BinaryIO, TextIO]:
        return self._fs.open(stringify_path(path), mode=mode)

    @implements(FileSystem.walk)
    def walk(self, path: path_type) -> Iterator[Tuple[str, List[str], List[str]]]:
        raise NotImplementedError

    @implements(FileSystem.glob)
    def glob(self, path: path_type, recursive: bool = False) -> List[path_type]:
        from ._glob import FileSystemGlob

        return FileSystemGlob(self).glob(stringify_path(path), recursive=recursive)
