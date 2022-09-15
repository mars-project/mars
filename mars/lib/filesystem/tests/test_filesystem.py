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

import glob as _glob
import os
import tempfile

import numpy as np
import pytest

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None

from ....tests.core import require_hadoop
from ....utils import lazy_import
from .. import glob, FileSystem, LocalFileSystem, FSMap

if pa is not None:
    from ..arrow import ArrowBasedLocalFileSystem, HadoopFileSystem
else:  # pragma: no cover
    ArrowBasedLocalFileSystem = None

fsspec_installed = lazy_import("fsspec") is not None


def test_path_parser():
    path = "hdfs://user:password@localhost:8080/test"
    parsed_result = FileSystem.parse_from_path(path)
    assert parsed_result["host"] == "localhost"
    assert parsed_result["port"] == 8080
    assert parsed_result["user"] == "user"
    assert parsed_result["password"] == "password"


def test_local_filesystem():
    local_fs1 = LocalFileSystem.get_instance()
    local_fs2 = LocalFileSystem.get_instance()
    assert local_fs1 is local_fs2

    with tempfile.TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "test")

        with open(file_path, "wb") as f:
            f.write(b"text for test")
        assert local_fs1.stat(tempdir)["type"] == "directory"
        assert local_fs1.stat(file_path)["type"] == "file"
        assert len(glob(tempdir + "*")) == 1


@pytest.mark.parametrize(
    "fs_type",
    [LocalFileSystem, ArrowBasedLocalFileSystem]
    if pa is not None
    else [LocalFileSystem],
)
def test_filesystems(fs_type):
    fs = fs_type.get_instance()

    with tempfile.TemporaryDirectory() as root:
        test1_dir = os.path.join(root, "test1")
        fs.mkdir(test1_dir, create_parents=False)
        test2_dir = os.path.join(root, "test2")
        sub_test2_dir = os.path.join(test2_dir, "sub_test2")
        fs.mkdir(sub_test2_dir)

        sub_test2_dir_stat = fs.stat(sub_test2_dir)
        assert sub_test2_dir_stat["type"] == "directory"
        assert sub_test2_dir_stat["name"] == sub_test2_dir
        assert fs.isdir(sub_test2_dir)

        test1_file = os.path.join(test1_dir, "test1")
        with fs.open(test1_file, "wb") as f:
            f.write(b"abc test")
        with fs.open(test1_file, "ab") as f:
            f.write(b"\nappend test")
        with fs.open(test1_file, "rb") as f:
            content = f.read()
            with open(test1_file, "rb") as f2:
                expected = f2.read()
                assert content == expected

        assert fs.cat(test1_file) == expected

        assert fs.isfile(test1_file)
        test1_file_stat = fs.stat(test1_file)
        assert test1_file_stat["type"] == "file"
        assert test1_file_stat["name"] == test1_file
        assert test1_file_stat["size"] == os.stat(test1_file).st_size
        np.testing.assert_almost_equal(
            test1_file_stat["modified_time"], os.stat(test1_file).st_mtime, decimal=6
        )

        walked = [
            (os.path.normpath(root), dirs, files) for root, dirs, files in fs.walk(root)
        ]
        expected = os.walk(root)
        assert sorted(walked) == sorted(expected)

        test2_file = os.path.join(sub_test2_dir, "test2")
        with fs.open(test2_file, "wb") as f:
            f.write(b"def test")

        for recursive in [False, True]:
            globs = [
                os.path.normpath(p)
                for p in fs.glob(os.path.join(root, "*"), recursive=recursive)
            ]
            expected = [
                os.path.normpath(p)
                for p in _glob.glob(os.path.join(root, "*"), recursive=recursive)
            ]
            assert sorted(globs) == sorted(expected)

        for path in [os.path.join(root, "*", "*"), test1_dir]:
            globs = [os.path.normpath(p) for p in fs.glob(path)]
            expected = [os.path.normpath(p) for p in _glob.glob(path)]
            assert sorted(globs) == sorted(expected)

        test1_new_file = os.path.join(test1_dir, "test1_new")
        fs.rename(test1_file, test1_new_file)
        test1_new_file2 = os.path.join(test1_dir, "test1_new2")
        fs.mv(test1_new_file, test1_new_file2)
        assert fs.exists(test1_new_file2)
        assert not fs.exists(test1_file)

        assert fs.disk_usage(test1_dir) > 0

        fs.delete(test2_file)
        assert not fs.exists(test2_file)

        assert fs._isfilestore()

        with pytest.raises(OSError):
            fs.delete(test1_dir)
        fs.delete(test1_dir, recursive=True)
        assert not fs.exists(test1_dir)


@require_hadoop
def test_hadoop_filesystem():
    fs = HadoopFileSystem(host="localhost", port=8020)

    test_dir = "/tmp/test/test_hadoop_fs"
    fs.mkdir(test_dir)
    test_file = f"{test_dir}/my_file.txt"
    test_file_content = b"text for text"
    with fs.open(test_file, "wb") as f:
        f.write(test_file_content)
    with fs.open(test_file, "rb") as f:
        assert test_file_content == f.read()
    # test file with hdfs:// prefix
    assert fs.exists(f"hdfs://{test_dir}")


def test_fsmap():
    fs = LocalFileSystem.get_instance()
    with tempfile.TemporaryDirectory() as root:
        fs_map = FSMap(root, fs, check=True)

        path = "/to/path/test_file"
        test_content = b"text for test"
        fs_map[path] = test_content
        assert fs_map[path] == test_content
        assert len(fs_map) == 1
        assert path in fs_map

        path2 = "/to/path2/test_file2"
        fs_map[path2] = test_content
        assert len(fs_map) == 2

        del fs_map[path]
        assert list(fs_map) == ["to/path2/test_file2"]

        path3 = "/to2/path3/test_file3"
        fs_map[path3] = test_content
        assert fs_map.pop(path3) == test_content
        assert fs_map.pop(path3, "fake_content") == "fake_content"
        with pytest.raises(KeyError):
            fs_map.pop("not_exist")

        fs_map.clear()
        assert len(fs_map) == 0

        # test root not exist
        with pytest.raises(ValueError):
            _ = FSMap(root + "/path2", fs, check=True)

        # create root
        fs_map = FSMap(root + "/path2", fs, create=True)
        assert len(fs_map) == 0


@pytest.mark.skipif(not fsspec_installed, reason="fsspec not installed")
def test_get_fs():
    from .. import get_fs, register_filesystem
    from ..fsspec_adapter import FsSpecAdapter

    class InMemoryFileSystemAdapter(FsSpecAdapter):
        def __init__(self, **kwargs):
            super().__init__("memory", **kwargs)

    register_filesystem("memory", InMemoryFileSystemAdapter)

    assert isinstance(get_fs("file://"), LocalFileSystem)
    assert isinstance(get_fs("memory://"), InMemoryFileSystemAdapter)

    try:
        get_fs("unknown://")
    except ValueError as e:
        assert "Unknown file system type" in e.__str__()
