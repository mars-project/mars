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
from abc import ABC, abstractmethod
from urllib.parse import urlparse
from typing import Union, List, Dict, Tuple, BinaryIO, TextIO, Iterator

from ...utils import stringify_path

path_type = Union[str, os.PathLike]


class FileSystem(ABC):
    """
    Abstract filesystem interface
    """

    @abstractmethod
    def cat(self, path: path_type) -> bytes:
        """
        Return contents of file as a bytes object

        Parameters
        ----------
        path : str or path-like
            File path to read content from.

        Returns
        -------
        contents : bytes
        """

    @abstractmethod
    def ls(self, path: path_type) -> List[path_type]:
        """
        Return list of file paths

        Returns
        -------
        paths : list
        """

    @abstractmethod
    def delete(self,
               path: path_type,
               recursive: bool = False):
        """
        Delete the indicated file or directory

        Parameters
        ----------
        path : str
        recursive : bool, default False
            If True, also delete child paths for directories
        """

    def disk_usage(self, path: path_type) -> int:
        """
        Compute bytes used by all contents under indicated path in file tree

        Parameters
        ----------
        path : string
            Can be a file path or directory

        Returns
        -------
        usage : int
        """
        path = stringify_path(path)
        path_info = self.stat(path)
        if path_info['type'] == 'file':
            return path_info['size']

        total = 0
        for root, directories, files in self.walk(path):
            for child_path in files:
                abspath = self.path_join(root, child_path)
                total += self.stat(abspath)['size']

        return total

    def path_join(self, *args):
        return self.pathsep.join(args)

    def path_split(self, path):
        return path.rsplit(self.pathsep, 1)

    @abstractmethod
    def stat(self, path: path_type) -> Dict:
        """
        Information about a filesystem entry.

        Returns
        -------
        stat : dict
        """

    def rm(self,
           path: path_type,
           recursive: bool = False):
        """
        Alias for FileSystem.delete
        """
        return self.delete(path, recursive=recursive)

    def mv(self, path, new_path):
        """
        Alias for FileSystem.rename
        """
        return self.rename(path, new_path)

    @abstractmethod
    def rename(self,
               path: path_type,
               new_path: path_type):
        """
        Rename file, like UNIX mv command

        Parameters
        ----------
        path : string
            Path to alter
        new_path : string
            Path to move to
        """

    @abstractmethod
    def mkdir(self,
              path: path_type,
              create_parents: bool = True):
        """
        Create a directory.

        Parameters
        ----------
        path : str
            Path to the directory.
        create_parents : bool, default True
            If the parent directories don't exists create them as well.
        """

    @abstractmethod
    def exists(self, path: path_type):
        """
        Return True if path exists.

        Parameters
        ----------
        path : str
            Path to check.
        """

    @abstractmethod
    def isdir(self, path: path_type) -> bool:
        """
        Return True if path is a directory.

        Parameters
        ----------
        path : str
            Path to check.
        """

    @abstractmethod
    def isfile(self, path: path_type) -> bool:
        """
        Return True if path is a file.

        Parameters
        ----------
        path : str
            Path to check.
        """

    @abstractmethod
    def _isfilestore(self) -> bool:
        """
        Returns True if this FileSystem is a unix-style file store with
        directories.
        """

    @abstractmethod
    def open(self,
             path: path_type,
             mode: str = 'rb') -> Union[BinaryIO, TextIO]:
        """
        Open file for reading or writing.
        """

    @abstractmethod
    def walk(self, path: path_type) -> Iterator[Tuple[str, List[str], List[str]]]:
        """
        Directory tree generator.

        Parameters
        ----------
        path : str

        Returns
        -------
        generator
        """

    @abstractmethod
    def glob(self,
             path: path_type,
             recursive: bool = False) -> List[path_type]:
        """
        Return a list of paths matching a pathname pattern.

        Parameters
        ----------
        path : str
            Pattern may contain simple shell-style wildcards
        recursive : bool
            If recursive is true, the pattern '**' will match any files and
            zero or more directories and subdirectories.

        Returns
        -------
        paths : List
        """

    @property
    def pathsep(self) -> str:
        return '/'

    @staticmethod
    def parse_from_path(uri: str):
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
