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

import fnmatch
import os
import re

from .core import FileSystem


magic_check = re.compile('([*?[])')
magic_check_bytes = re.compile(b'([*?[])')


def has_magic(s):
    if isinstance(s, bytes):  # pragma: no cover
        match = magic_check_bytes.search(s)
    else:
        match = magic_check.search(s)
    return match is not None


def _ishidden(path):
    return path[0] in ('.', b'.'[0])


def _isrecursive(pattern):
    if isinstance(pattern, bytes):  # pragma: no cover
        return pattern == b'**'
    else:
        return pattern == '**'


class FileSystemGlob:
    def __init__(self, fs: FileSystem):
        self._fs = fs

    def glob(self, pathname, recursive=False):
        """Return a list of paths matching a pathname pattern.

        The pattern may contain simple shell-style wildcards a la
        fnmatch. However, unlike fnmatch, filenames starting with a
        dot are special cases that are not matched by '*' and '?'
        patterns.

        If recursive is true, the pattern '**' will match any files and
        zero or more directories and subdirectories.
        """
        return list(self.iglob(pathname, recursive=recursive))

    def iglob(self, pathname, recursive=False):
        """Return an iterator which yields the paths matching a pathname pattern.

        The pattern may contain simple shell-style wildcards a la
        fnmatch. However, unlike fnmatch, filenames starting with a
        dot are special cases that are not matched by '*' and '?'
        patterns.

        If recursive is true, the pattern '**' will match any files and
        zero or more directories and subdirectories.
        """
        it = self._iglob(pathname, recursive, False)
        if recursive and _isrecursive(pathname):  # pragma: no cover
            s = next(it)  # skip empty string
            assert not s
        return it

    def _iglob(self, pathname, recursive, dironly):
        dirname, basename = self._fs.path_split(
            pathname.replace(os.path.sep, '/'))
        if not has_magic(pathname):
            assert not dironly
            if basename:
                if self._fs.exists(pathname):
                    yield pathname
            else:  # pragma: no cover
                # Patterns ending with a slash should match only directories
                if self._fs.isdir(dirname):
                    yield pathname
            return
        if not dirname:  # pragma: no cover
            if recursive and _isrecursive(basename):
                yield from self._glob2(dirname, basename, dironly)
            else:
                yield from self._glob1(dirname, basename, dironly)
            return
        # `os.path.split()` returns the argument itself as a dirname if it is a
        # drive or UNC path.  Prevent an infinite recursion if a drive or UNC path
        # contains magic characters (i.e. r'\\?\C:').
        if dirname != pathname and has_magic(dirname):
            dirs = self._iglob(dirname, recursive, True)
        else:
            dirs = [dirname]
        if has_magic(basename):
            if recursive and _isrecursive(basename):
                glob_in_dir = self._glob2
            else:
                glob_in_dir = self._glob1
        else:
            glob_in_dir = self._glob0
        for dirname in dirs:
            for name in glob_in_dir(dirname, basename, dironly):
                yield self._fs.path_join(dirname, name)

    # These 2 helper functions non-recursively glob inside a literal directory.
    # They return a list of basenames.  _glob1 accepts a pattern while _glob0
    # takes a literal basename (so it only has to check for its existence).

    def _glob1(self, dirname, pattern, dironly):
        names = list(self._iterdir(dirname, dironly))
        if not _ishidden(pattern):
            names = (x for x in names if not _ishidden(x))
        return fnmatch.filter(names, pattern)

    def _glob0(self, dirname, basename, dironly):  # pragma: no cover
        if not basename:
            # `os.path.split()` returns an empty basename for paths ending with a
            # directory separator.  'q*x/' should match only directories.
            if self._fs.isdir(dirname):
                return [basename]
        else:
            if self._fs.exists(self._fs.path_join(dirname, basename)):
                return [basename]
        return []

    # Following functions are not public but can be used by third-party code.

    def glob0(self, dirname, pattern):  # pragma: no cover
        return self._glob0(dirname, pattern, False)

    def glob1(self, dirname, pattern):  # pragma: no cover
        return self._glob1(dirname, pattern, False)

    # This helper function recursively yields relative pathnames inside a literal
    # directory.

    def _glob2(self, dirname, pattern, dironly):  # pragma: no cover
        assert _isrecursive(pattern)
        yield pattern[:0]
        yield from self._rlistdir(dirname, dironly)

    # If dironly is false, yields all file names inside a directory.
    # If dironly is true, yields only directory names.
    def _iterdir(self, dirname, dironly):
        if not dirname:  # pragma: no cover
            if isinstance(dirname, bytes):
                dirname = bytes(os.curdir, 'ASCII')
            else:
                dirname = os.curdir
        for entry in self._fs.ls(dirname):
            if not dironly or self._fs.isdir(entry):
                yield self._fs.path_split(entry)[-1]

    # Recursively yields relative pathnames inside a literal directory.
    def _rlistdir(self, dirname, dironly):  # pragma: no cover
        names = list(self._iterdir(dirname, dironly))
        for x in names:
            if not _ishidden(x):
                yield x
                path = self._fs.path_join(dirname, x) if dirname else x
                for y in self._rlistdir(path, dironly):
                    yield self._fs.path_join(x, y)
