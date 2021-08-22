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

"""
Filename globbing utility, modified from python glob.

obviously，this implementation is not optimal, it will cause too many
oss requests. Lately, We can then convert the glob expression into
a regular expression, and then match the oss key list.
But before that, we need to figure out how to deal with magic char
in oss key, such like oss glob: oss://bucket/[key]/*, the key
oss://bucket/[key]/a exactly exists.

Notes:
    OSS need a bucket to specify the file or dir, the "**" patten is
    not supported. So _isrecursive(pattern) is removed.
"""

import fnmatch
import os
import re

from .common import oss_exists, oss_isdir, oss_scandir

__all__ = ["glob", "iglob", "escape"]


def glob(pathname, *, recursive=False):
    """Return a list of paths matching a pathname pattern.
    The pattern may contain simple shell-style wildcards a la
    fnmatch. However, unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns.
    """
    return list(iglob(pathname, recursive=recursive))


def iglob(pathname, *, recursive=False):
    """Return an iterator which yields the paths matching a pathname pattern.
    The pattern may contain simple shell-style wildcards like
    fnmatch. However, unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns.
    """
    it = _iglob(pathname, recursive, False)
    return it


def _iglob(pathname, recursive, dironly):
    dirname, basename = os.path.split(pathname)
    if not has_magic(pathname):
        assert not dironly
        if basename:
            if oss_exists(pathname):
                yield pathname
        else:
            # Patterns ending with a slash should match only directories
            if oss_isdir(dirname):
                yield pathname
        return
    # dirname will not be None in oss path.
    #  Prevent an infinite recursion if a drive or UNC path
    # contains magic characters (i.e. r'\\?\C:').
    if dirname != pathname and has_magic(dirname):
        dirs = _iglob(dirname, recursive, True)
    else:
        dirs = [dirname]
    if has_magic(basename):
        glob_in_dir = _glob1
    else:
        glob_in_dir = _glob0
    for dirname in dirs:
        for name in glob_in_dir(dirname, basename, dironly):
            yield os.path.join(dirname, name)


# These 2 helper functions non-recursively glob inside a literal directory.
# They return a list of basenames.  _glob1 accepts a pattern while _glob0
# takes a literal basename (so it only has to check for its existence).

def _glob1(dirname, pattern, dironly):
    names = list(_iterdir(dirname, dironly))
    if not _ishidden(pattern):
        names = (x for x in names if not _ishidden(x))
    return fnmatch.filter(names, pattern)


def _glob0(dirname, basename, dironly):
    if not basename:
        # `os.path.split()` returns an empty basename for paths ending with a
        # directory separator.  'q*x/' should match only directories.
        if oss_isdir(dirname):
            return [basename]
    else:
        if oss_exists(os.path.join(dirname, basename)):
            return [basename]
    return []


# If dironly is false, yields all file names inside a directory.
# If dironly is true, yields only directory names.
# An oss path must contain a dirname.
def _iterdir(dirname, dironly):
    for entry in oss_scandir(dirname):
        if not dironly or entry.is_dir():
            yield entry.name
    return


magic_check = re.compile('([*?[])')
magic_check_bytes = re.compile(b'([*?[])')


def has_magic(s):
    if isinstance(s, bytes):
        match = magic_check_bytes.search(s)
    else:
        match = magic_check.search(s)
    return match is not None


def _ishidden(path):
    return False


def escape(pathname):
    """Escape all special characters.
    """
    # Escaping is done by wrapping any of "*?[" between square brackets.
    # Metacharacters do not work in the drive part and shouldn't be escaped.
    drive, pathname = os.path.splitdrive(pathname)
    if isinstance(pathname, bytes):
        pathname = magic_check_bytes.sub(br'[\1]', pathname)
    else:
        pathname = magic_check.sub(r'[\1]', pathname)
    return drive + pathname
