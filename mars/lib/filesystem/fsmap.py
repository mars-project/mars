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

from collections.abc import MutableMapping
from urllib.parse import urlparse

from .local import LocalFileSystem


class FSMap(MutableMapping):
    """
    Wrap a FileSystem instance as a mutable wrapping.
    The keys of the mapping become files under the given root, and the
    values (which must be bytes) the contents of those files.

    Parameters
    ----------
    root: string
        prefix for all the files
    fs: FileSystem instance
    check: bool (=True)
        performs a touch at the location, to check for write access.
    """

    def __init__(self, root, fs, check=False, create=False):
        self.fs = fs
        self.root = self._get_path(fs, root)
        if create:
            if not self.fs.exists(root):
                self.fs.mkdir(root)
        if check:
            if not self.fs.exists(root):
                raise ValueError(
                    f"Path {root} does not exist. Create with the ``create=True`` keyword"
                )
            with self.fs.open(fs.pathsep.join([root, "a"]), 'w'):
                pass
            self.fs.rm(fs.pathsep.join([root, "a"]))

    @staticmethod
    def _get_path(fs, path):
        return path if isinstance(fs, LocalFileSystem) else urlparse(path).path

    @staticmethod
    def _normalize_path(fs, path, lstrip=False, rstrip=False):
        if fs.pathsep != '/':  # pragma: no cover
            path = path.replace('/', fs.pathsep)
        if lstrip:
            path = path.lstrip(fs.pathsep)
        if rstrip:
            path = path.rstrip(fs.pathsep)
        return path

    @staticmethod
    def _join_path(fs, paths):
        if fs.pathsep == '/':
            return '/'.join(paths)

        new_paths = []
        for i, path in enumerate(paths):
            path = FSMap._normalize_path(fs, path, lstrip=i > 0,
                                         rstrip=i < len(paths) - 1)
            new_paths.append(path)
        return fs.pathsep.join(new_paths)

    def clear(self):
        """Remove all keys below root - empties out mapping
        """
        try:
            self.fs.rm(self.root, True)
            self.fs.mkdir(self.root)
        except:  # noqa: E722  # pragma: no cover
            pass

    def _key_to_str(self, key):
        """Generate full path for the key"""
        if isinstance(key, (tuple, list)):
            key = str(tuple(key))
        else:
            key = str(key)
        return self._join_path(self.fs, [self.root, key]) if self.root else key

    def _str_to_key(self, s):
        """Strip path of to leave key name"""
        key = self._normalize_path(self.fs, s[len(self.root):], lstrip=True)
        if self.fs.pathsep != '/':  # pragma: no cover
            key = key.replace(self.fs.pathsep, '/')
        return key

    def __getitem__(self, key, default=None):
        """Retrieve data"""
        key = self._key_to_str(key)
        try:
            result = self.fs.cat(key)
        except:  # noqa: E722
            if default is not None:
                return default
            raise KeyError(key)
        return result

    def pop(self, key, default=None):
        result = self.__getitem__(key, default)
        try:
            del self[key]
        except KeyError:
            pass
        return result

    @staticmethod
    def _parent(fs, path):
        path = FSMap._get_path(fs, path.rstrip(fs.pathsep))
        if fs.pathsep in path:
            return path.rsplit(fs.pathsep, 1)[0]
        else:  # pragma: no cover
            return ''

    def __setitem__(self, key, value):
        """Store value in key"""
        key = self._key_to_str(key)
        try:
            self.fs.mkdir(self._parent(self.fs, key))
        except FileExistsError:
            pass
        with self.fs.open(key, "wb") as f:
            f.write(value)

    @staticmethod
    def _find(fs, path):
        out = set()
        for path, dirs, files in fs.walk(path):
            out.update(fs.pathsep.join([path, f]) for f in files)
        if fs.isfile(path) and path not in out:
            # walk works on directories, but find should also return [path]
            # when path happens to be a file
            out.add(path)
        return sorted(out)

    def __iter__(self):
        return (self._str_to_key(x) for x in self._find(self.fs, self.root))

    def __len__(self):
        return len(self._find(self.fs, self.root))

    def __delitem__(self, key):
        """Remove key"""
        try:
            self.fs.rm(self._key_to_str(key))
        except:  # noqa: E722
            raise KeyError

    def __contains__(self, key):
        """Does key exist in mapping?"""
        return self.fs.exists(self._key_to_str(key))
