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

import pytest

from ....utils import lazy_import

fsspec_installed = lazy_import("fsspec") is not None


@pytest.mark.skipif(not fsspec_installed, reason="fsspec not installed")
def test_fsspec_adapter():
    """
    Assuming the implementations follows fsspec strictly, we only need to test if the adapter
    works correctly.
    """
    from ..fsspec_adapter import FsSpecAdapter

    adapter = FsSpecAdapter(scheme="memory")

    fs = adapter._fs
    # generate directories and files as follows:
    # .
    # ├── dir
    # │   ├── bar.txt
    # │   └── subdir
    # │       └── baz.txt
    # └── foo.txt
    with fs.open("foo.txt", mode="wb") as f:
        f.write(str.encode("foo"))
    fs.mkdir("dir")
    fs.mkdirs("/dir/subdir")
    with fs.open("/dir/bar.txt", mode="wb") as f:
        f.write(str.encode("bar"))
    with fs.open("/dir/subdir/baz.txt", mode="wb") as f:
        f.write(str.encode("baz"))

    # open
    f = adapter.open("test.txt", mode="wb")
    f.write(str.encode("test"))
    f.close()

    # cat
    assert "test" == adapter.cat("test.txt").decode()
    try:
        adapter.cat("non-existent.txt")
        pytest.fail()
    except FileNotFoundError:
        pass

    # ls
    entries = adapter.ls("/")
    assert 3 == len(entries)
    assert "/test.txt" in entries
    assert "/foo.txt" in entries
    assert "/dir" in entries
    entries = adapter.ls("dir")
    assert 2 == len(entries)
    assert "/dir/bar.txt" in entries
    assert "/dir/subdir" in entries
    entries = adapter.ls("test.txt")
    assert 1 == len(entries)
    assert "/test.txt" in entries
    try:
        adapter.ls("non-existent.txt")
        pytest.fail()
    except FileNotFoundError:
        pass

    # stat
    stat = adapter.stat("test.txt")
    assert stat is not None
    assert stat["name"] == "/test.txt"
    assert stat["type"] == "file"
    stat = adapter.stat("dir")
    assert stat is not None
    assert stat["name"] == "/dir"
    assert stat["type"] == "directory"
    try:
        adapter.stat("non-existent.txt")
        pytest.fail()
    except FileNotFoundError:
        pass

    # exists
    assert adapter.exists("test.txt")
    assert not adapter.exists("non-existent.txt")

    # isdir
    assert adapter.isdir("dir")
    assert not adapter.isdir("test.txt")
    assert not adapter.isdir("non-existent.txt")

    # isfile
    assert adapter.isfile("test.txt")
    assert not adapter.isfile("dir")
    assert not adapter.isfile("non-existent.txt")

    # glob
    # the expected results come from built-in glob lib.
    expected = [
        "foo.txt",
        "dir",
        "dir/subdir",
        "dir/subdir/baz.txt",
        "dir/bar.txt",
        "test.txt",
    ].sort()
    actual = adapter.glob("**", recursive=True).sort()
    assert actual == expected
    expected = ["foo.txt"]
    actual = adapter.glob("**/foo.txt", recursive=True)
    assert actual == expected
    expected = ["dir/bar.txt"]
    actual = adapter.glob("**/bar.txt", recursive=True)
    assert actual == expected
    expected = ["dir/subdir/baz.txt"]
    actual = adapter.glob("**/baz.txt", recursive=True)
    assert actual == expected
    expected = ["dir/bar.txt", "dir/subdir/baz.txt"]
    actual = adapter.glob("**/ba[rz].txt", recursive=True)
    assert actual == expected
    actual = adapter.glob("**/ba?.txt", recursive=True)
    assert actual == expected
    expected = ["foo.txt", "test.txt", "dir"].sort()
    actual = adapter.glob("**", recursive=False).sort()
    assert actual == expected
    actual = adapter.glob("*", recursive=False).sort()
    assert actual == expected
    expected = ["foo.txt", "test.txt"].sort()
    actual = adapter.glob("*.txt", recursive=False).sort()
    assert actual == expected
