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

from typing import List, Dict, Union, Tuple, Iterator, BinaryIO, TextIO

import pyarrow as pa
from pyarrow.fs import FileSystem as ArrowFileSystem, \
    LocalFileSystem as ArrowLocalFileSystem, \
    HadoopFileSystem as ArrowHadoopFileSystem, \
    FileSelector, FileInfo, FileType

from ...utils import implements
from .core import FileSystem, path_type


__all__ = ('ArrowBasedLocalFileSystem', 'HadoopFileSystem')


class ArrowBasedFileSystem(FileSystem):
    """
    FileSystem implemented with arrow fs API (>=2.0.0).
    """

    def __init__(self, arrow_fs: ArrowFileSystem, sequential_read=True):
        self._arrow_fs = arrow_fs
        # for open('rb'), open a sequential reading only or not
        self._sequential_read = sequential_read

    @implements(FileSystem.cat)
    def cat(self, path: path_type) -> bytes:
        file: pa.NativeFile = self._arrow_fs.open_input_stream(path)
        return file.read()

    @implements(FileSystem.ls)
    def ls(self, path: path_type) -> List[path_type]:
        file_selector: FileSelector = FileSelector(path)
        paths = []
        for file_info in self._arrow_fs.get_file_info(file_selector):
            paths.append(file_info.path)
        return paths

    def _get_file_info(self, path: path_type) -> FileInfo:
        file_info: FileInfo = self._arrow_fs.get_file_info([path])[0]
        return file_info

    @implements(FileSystem.delete)
    def delete(self,
               path: path_type,
               recursive: bool = False):
        info = self._get_file_info(path)
        if info.is_file:
            self._arrow_fs.delete_file(path)
        elif info.type == FileType.Directory:
            if not recursive and len(self.ls(path)) > 0:
                raise OSError(f"[Errno 66] Directory not empty: '{path}'")
            self._arrow_fs.delete_dir(path)
        else:  # pragma: no cover
            raise TypeError(f'path({path}) to delete '
                            f'must be a file or directory')

    @implements(FileSystem.rename)
    def rename(self,
               path: path_type,
               new_path: path_type):
        self._arrow_fs.move(path, new_path)

    @implements(FileSystem.stat)
    def stat(self, path: path_type) -> Dict:
        info = self._get_file_info(path)
        stat = dict(name=path, size=info.size, modified_time=info.mtime_ns / 1e9)
        if info.type == FileType.File:
            stat['type'] = 'file'
        elif info.type == FileType.Directory:
            stat['type'] = 'directory'
        else:  # pragma: no cover
            stat['type'] = 'other'
        return stat

    @implements(FileSystem.mkdir)
    def mkdir(self,
              path: path_type,
              create_parents: bool = True):
        self._arrow_fs.create_dir(path, recursive=create_parents)

    @implements(FileSystem.isdir)
    def isdir(self, path: path_type) -> bool:
        info = self._get_file_info(path)
        return info.type == FileType.Directory

    @implements(FileSystem.isfile)
    def isfile(self, path: path_type) -> bool:
        info = self._get_file_info(path)
        return info.is_file

    @implements(FileSystem._isfilestore)
    def _isfilestore(self) -> bool:
        return True

    @implements(FileSystem.exists)
    def exists(self, path: path_type):
        info = self._get_file_info(path)
        return info.type != FileType.NotFound

    @implements(FileSystem.open)
    def open(self,
             path: path_type,
             mode: str = 'rb') -> Union[BinaryIO, TextIO]:
        is_binary = mode.endswith('b')
        if not is_binary:  # pragma: no cover
            raise ValueError(f'mode can only be binary for '
                             f'arrow based filesystem, got {mode}')
        mode = mode.rstrip('b')
        if mode == 'w':
            return self._arrow_fs.open_output_stream(path)
        elif mode == 'r':
            if self._sequential_read:
                return self._arrow_fs.open_input_stream(path)
            else:
                return self._arrow_fs.open_input_file(path)
        elif mode == 'a':
            return self._arrow_fs.open_append_stream(path)

    @implements(FileSystem.walk)
    def walk(self, path: path_type) -> Iterator[Tuple[str, List[str], List[str]]]:
        q = [path]
        while q:
            curr = q.pop(0)
            file_selector: FileSelector = FileSelector(curr)
            dirs, files = [], []
            for info in self._arrow_fs.get_file_info(file_selector):
                if info.type == FileType.File:
                    files.append(info.base_name)
                elif info.type == FileType.Directory:
                    dirs.append(info.base_name)
                    q.append(info.path)
                else:  # pragma: no cover
                    continue
            yield curr, dirs, files

    @implements(FileSystem.glob)
    def glob(self,
             path: path_type,
             recursive: bool = False) -> List[path_type]:
        from ._glob import FileSystemGlob

        return FileSystemGlob(self).glob(path, recursive=recursive)


class ArrowBasedLocalFileSystem(ArrowBasedFileSystem):
    def __init__(self):
        super().__init__(ArrowLocalFileSystem(), sequential_read=False)

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ArrowBasedLocalFileSystem()
        return cls._instance


class HadoopFileSystem(ArrowBasedFileSystem):
    def __init__(self, host="default", port=0, user=None, kerb_ticket=None,
                 driver='libhdfs', extra_conf=None):
        assert driver == 'libhdfs'
        arrow_fs = ArrowHadoopFileSystem(host=host, port=port, user=user,
                                         kerb_ticket=kerb_ticket,
                                         extra_conf=extra_conf)
        super().__init__(arrow_fs)
