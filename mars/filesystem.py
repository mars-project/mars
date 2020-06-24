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
import shutil
import pathlib
import glob as local_glob
from os.path import join as pjoin
from collections.abc import MutableMapping
from gzip import GzipFile
from urllib.parse import urlparse


try:
    from pyarrow import FileSystem, LocalFileSystem
    from pyarrow.util import implements
except ImportError:  # pragma: no cover
    def implements(f):
        def decorator(g):
            g.__doc__ = f.__doc__
            return g

        return decorator

    def _stringify_path(path):
        """
        Convert *path* to a string or unicode path if possible.
        """
        if isinstance(path, str):
            return path

        # checking whether path implements the filesystem protocol
        try:
            return path.__fspath__()  # new in python 3.6
        except AttributeError:
            # fallback pathlib ckeck for earlier python versions than 3.6
            if isinstance(path, pathlib.Path):
                return str(path)

        raise TypeError("not a path-like object")

    class FileSystem(object):
        """
        Abstract filesystem interface
        """

        def cat(self, path):
            """
            Return contents of file as a bytes object

            Returns
            -------
            contents : bytes
            """
            with self.open(path, 'rb') as f:
                return f.read()

        def ls(self, path):
            """
            Return list of file paths
            """
            raise NotImplementedError

        def delete(self, path, recursive=False):
            """
            Delete the indicated file or directory

            Parameters
            ----------
            path : string
            recursive : boolean, default False
                If True, also delete child paths for directories
            """
            raise NotImplementedError

        def disk_usage(self, path):
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
            path = _stringify_path(path)
            path_info = self.stat(path)
            if path_info['kind'] == 'file':
                return path_info['size']

            total = 0
            for root, directories, files in self.walk(path):
                for child_path in files:
                    abspath = self._path_join(root, child_path)
                    total += self.stat(abspath)['size']

            return total

        def _path_join(self, *args):
            return self.pathsep.join(args)

        def stat(self, path):
            """

            Returns
            -------
            stat : dict
            """
            raise NotImplementedError('FileSystem.stat')

        def rm(self, path, recursive=False):
            """
            Alias for FileSystem.delete
            """
            return self.delete(path, recursive=recursive)

        def mv(self, path, new_path):
            """
            Alias for FileSystem.rename
            """
            return self.rename(path, new_path)

        def rename(self, path, new_path):
            """
            Rename file, like UNIX mv command

            Parameters
            ----------
            path : string
                Path to alter
            new_path : string
                Path to move to
            """
            raise NotImplementedError('FileSystem.rename')

        def mkdir(self, path, create_parents=True):
            raise NotImplementedError

        def exists(self, path):
            raise NotImplementedError

        def isdir(self, path):
            """
            Return True if path is a directory
            """
            raise NotImplementedError

        def isfile(self, path):
            """
            Return True if path is a file
            """
            raise NotImplementedError

        def _isfilestore(self):
            """
            Returns True if this FileSystem is a unix-style file store with
            directories.
            """
            raise NotImplementedError

        def read_parquet(self, path, columns=None, metadata=None, schema=None,
                         use_threads=True, use_pandas_metadata=False):
            """
            Read Parquet data from path in file system. Can read from a single file
            or a directory of files

            Parameters
            ----------
            path : str
                Single file path or directory
            columns : List[str], optional
                Subset of columns to read
            metadata : pyarrow.parquet.FileMetaData
                Known metadata to validate files against
            schema : pyarrow.parquet.Schema
                Known schema to validate files against. Alternative to metadata
                argument
            use_threads : boolean, default True
                Perform multi-threaded column reads
            use_pandas_metadata : boolean, default False
                If True and file has custom pandas schema metadata, ensure that
                index columns are also loaded

            Returns
            -------
            table : pyarrow.Table
            """
            from pyarrow.parquet import ParquetDataset
            dataset = ParquetDataset(path, schema=schema, metadata=metadata,
                                     filesystem=self)
            return dataset.read(columns=columns, use_threads=use_threads,
                                use_pandas_metadata=use_pandas_metadata)

        def open(self, path, mode='rb'):
            """
            Open file for reading or writing
            """
            raise NotImplementedError

        @property
        def pathsep(self):
            return '/'

    class LocalFileSystem(FileSystem):

        _instance = None

        @classmethod
        def get_instance(cls):
            if cls._instance is None:
                cls._instance = LocalFileSystem()
            return cls._instance

        @implements(FileSystem.ls)
        def ls(self, path):
            path = _stringify_path(path)
            return sorted(pjoin(path, x) for x in os.listdir(path))

        @implements(FileSystem.mkdir)
        def mkdir(self, path, create_parents=True):
            path = _stringify_path(path)
            if create_parents:
                os.makedirs(path)
            else:
                os.mkdir(path)

        @implements(FileSystem.isdir)
        def isdir(self, path):
            path = _stringify_path(path)
            return os.path.isdir(path)

        @implements(FileSystem.isfile)
        def isfile(self, path):
            path = _stringify_path(path)
            return os.path.isfile(path)

        @implements(FileSystem._isfilestore)
        def _isfilestore(self):
            return True

        @implements(FileSystem.exists)
        def exists(self, path):
            path = _stringify_path(path)
            return os.path.exists(path)

        @implements(FileSystem.open)
        def open(self, path, mode='rb'):
            """
            Open file for reading or writing
            """
            path = _stringify_path(path)
            return open(path, mode=mode)

        @property
        def pathsep(self):
            return os.path.sep

        def walk(self, path):
            """
            Directory tree generator, see os.walk
            """
            path = _stringify_path(path)
            return os.walk(path)

try:
    from pyarrow import HadoopFileSystem
except ImportError:  # pragma: no cover
    HadoopFileSystem = object

try:
    import lz4
    import lz4.frame
except ImportError:  # pragma: no cover
    lz4 = None


compressions = {
    'gzip': lambda f: GzipFile(fileobj=f)
}

if lz4:
    compressions['lz4'] = lz4.frame.open

FileSystem = FileSystem
ArrowLocalFileSystem = LocalFileSystem


class LocalFileSystem(ArrowLocalFileSystem):
    _fs_instance = None

    @classmethod
    def get_instance(cls):
        if cls._fs_instance is None:
            cls._fs_instance = LocalFileSystem()
        return cls._fs_instance

    def delete(self, path, recursive=False):
        if os.path.isfile(path):
            os.remove(path)
        elif not recursive:
            os.rmdir(path)
        else:
            shutil.rmtree(path)

    if implements is not None:
        delete = implements(FileSystem.delete)(delete)

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
        if scheme == 'hdfs' and HadoopFileSystem is object:
            raise ImportError('Need to install pyarrow to connect to HDFS.')
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


class FSMap(MutableMapping):
    """Wrap a FileSystem instance as a mutable wrapping.
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
                    "Path %s does not exist. Create "
                    " with the ``create=True`` keyword" % root
                )
            with self.fs.open(fs.pathsep.join([root, "a"]), 'w'):
                pass
            self.fs.rm(fs.pathsep.join([root, "a"]))

    @staticmethod
    def _get_path(fs, path):
        return path if isinstance(fs, LocalFileSystem) else urlparse(path).path

    @staticmethod
    def _normalize_path(fs, path, lstrip=False, rstrip=False):
        if fs.pathsep != '/':
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
        except:  # noqa: E722
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
        if self.fs.pathsep != '/':
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
        else:
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
