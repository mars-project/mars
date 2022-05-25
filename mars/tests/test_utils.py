# -*- coding: utf-8 -*-
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

import asyncio
import copy
import logging
import multiprocessing
import os
import random
import shutil
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from enum import Enum

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None
import pytest

from .. import dataframe as md
from .. import tensor as mt
from .. import utils
from ..core import tile, TileableGraph
from ..serialization.ray import register_ray_serializers
from .core import require_ray


def test_string_conversion():
    s = None
    assert utils.to_binary(s) is None
    assert utils.to_str(s) is None
    assert utils.to_text(s) is None

    s = "abcdefg"
    assert isinstance(utils.to_binary(s), bytes)
    assert utils.to_binary(s) == b"abcdefg"
    assert isinstance(utils.to_str(s), str)
    assert utils.to_str(s) == "abcdefg"
    assert isinstance(utils.to_text(s), str)
    assert utils.to_text(s) == "abcdefg"

    ustr = type("ustr", (str,), {})
    assert isinstance(utils.to_str(ustr(s)), str)
    assert utils.to_str(ustr(s)) == "abcdefg"

    s = b"abcdefg"
    assert isinstance(utils.to_binary(s), bytes)
    assert utils.to_binary(s) == b"abcdefg"
    assert isinstance(utils.to_str(s), str)
    assert utils.to_str(s) == "abcdefg"
    assert isinstance(utils.to_text(s), str)
    assert utils.to_text(s) == "abcdefg"

    ubytes = type("ubytes", (bytes,), {})
    assert isinstance(utils.to_binary(ubytes(s)), bytes)
    assert utils.to_binary(ubytes(s)) == b"abcdefg"

    s = "abcdefg"
    assert isinstance(utils.to_binary(s), bytes)
    assert utils.to_binary(s) == b"abcdefg"
    assert isinstance(utils.to_str(s), str)
    assert utils.to_str(s) == "abcdefg"
    assert isinstance(utils.to_text(s), str)
    assert utils.to_text(s) == "abcdefg"

    uunicode = type("uunicode", (str,), {})
    assert isinstance(utils.to_text(uunicode(s)), str)
    assert utils.to_text(uunicode(s)) == "abcdefg"

    with pytest.raises(TypeError):
        utils.to_binary(utils)
    with pytest.raises(TypeError):
        utils.to_str(utils)
    with pytest.raises(TypeError):
        utils.to_text(utils)


def test_tokenize():
    import shutil
    import tempfile

    class TestEnum(Enum):
        VAL1 = "val1"

    tempdir = tempfile.mkdtemp("mars_test_utils_")
    try:
        filename = os.path.join(tempdir, "test_npa.dat")
        mmp_array = np.memmap(filename, dtype=float, mode="w+", shape=(3, 4))
        mmp_array[:] = np.random.random((3, 4)).astype(float)
        mmp_array.flush()
        del mmp_array

        mmp_array1 = np.memmap(filename, dtype=float, shape=(3, 4))
        mmp_array2 = np.memmap(filename, dtype=float, shape=(3, 4))

        try:
            v = [
                1,
                2.3,
                "456",
                "789",
                b"101112",
                2147483649,
                None,
                np.ndarray,
                [912, "uvw"],
                np.arange(0, 10),
                np.array(10),
                np.array([b"\x01\x32\xff"]),
                np.int64,
                TestEnum.VAL1,
            ]
            copy_v = copy.deepcopy(v)
            assert utils.tokenize(v + [mmp_array1], ext_data=1234) == utils.tokenize(
                copy_v + [mmp_array2], ext_data=1234
            )
        finally:
            del mmp_array1, mmp_array2
    finally:
        shutil.rmtree(tempdir)

    v = {"a", "xyz", "uvw"}
    assert utils.tokenize(v) == utils.tokenize(copy.deepcopy(v))

    v = dict(x="abcd", y=98765)
    assert utils.tokenize(v) == utils.tokenize(copy.deepcopy(v))

    v = dict(x=dict(a=1, b=[1, 2, 3]), y=12345)
    assert utils.tokenize(v) == utils.tokenize(copy.deepcopy(v))

    # pandas relative
    if pd is not None:
        df = pd.DataFrame(
            [[utils.to_binary("测试"), utils.to_text("数据")]],
            index=["a"],
            columns=["中文", "data"],
        )
        v = [df, df.index, df.columns, df["data"], pd.Categorical(list("ABCD"))]
        assert utils.tokenize(v) == utils.tokenize(copy.deepcopy(v))

    class NonTokenizableCls:
        def __getstate__(self):
            raise SystemError

    with pytest.raises(TypeError):
        utils.tokenize(NonTokenizableCls())

    class CustomizedTokenize(object):
        def __mars_tokenize__(self):
            return id(type(self)), id(NonTokenizableCls)

    assert utils.tokenize(CustomizedTokenize()) == utils.tokenize(CustomizedTokenize())

    v = lambda x: x + 1
    assert utils.tokenize(v) == utils.tokenize(copy.deepcopy(v))

    def f(a, b):
        return np.add(a, b)

    assert utils.tokenize(f) == utils.tokenize(copy.deepcopy(f))

    partial_f = partial(f, 1, k=0)
    partial_f2 = partial(f, 1, k=1)
    assert utils.tokenize(partial_f) == utils.tokenize(copy.deepcopy(partial_f))
    assert utils.tokenize(partial_f) != utils.tokenize(partial_f2)


def test_lazy_import():
    old_sys_path = sys.path
    mock_mod = textwrap.dedent(
        """
        __version__ = '0.1.0b1'
        """.strip()
    )
    mock_mod2 = textwrap.dedent(
        """
        from mars.utils import lazy_import
        mock_mod = lazy_import("mock_mod")

        def get_version():
            return mock_mod.__version__
        """
    )

    temp_dir = tempfile.mkdtemp(prefix="mars-utils-test-")
    sys.path += [temp_dir]
    try:
        with open(os.path.join(temp_dir, "mock_mod.py"), "w") as outf:
            outf.write(mock_mod)
        with open(os.path.join(temp_dir, "mock_mod2.py"), "w") as outf:
            outf.write(mock_mod2)

        non_exist_mod = utils.lazy_import("non_exist_mod", locals=locals())
        assert non_exist_mod is None

        non_exist_mod1 = utils.lazy_import("non_exist_mod1", placeholder=True)
        with pytest.raises(AttributeError) as ex_data:
            non_exist_mod1.meth()
        assert "required" in str(ex_data.value)

        mod = utils.lazy_import(
            "mock_mod", globals=globals(), locals=locals(), rename="mod"
        )
        assert mod is not None
        assert mod.__version__ == "0.1.0b1"

        glob = globals().copy()
        mod = utils.lazy_import("mock_mod", globals=glob, locals=locals(), rename="mod")
        glob["mod"] = mod
        assert mod is not None
        assert mod.__version__ == "0.1.0b1"
        assert type(glob["mod"]).__name__ == "module"

        import mock_mod2 as mod2

        assert type(mod2.mock_mod).__name__ != "module"
        assert mod2.get_version() == "0.1.0b1"
        assert type(mod2.mock_mod).__name__ == "module"
    finally:
        shutil.rmtree(temp_dir)
        sys.path = old_sys_path
        sys.modules.pop("mock_mod", None)
        sys.modules.pop("mock_mod2", None)


def test_chunks_indexer():
    a = mt.ones((3, 4, 5), chunk_size=2)
    a = tile(a)

    assert a.chunk_shape == (2, 2, 3)

    with pytest.raises(ValueError):
        _ = a.cix[1]
    with pytest.raises(ValueError):
        _ = a.cix[1, :]

    chunk_key = a.cix[0, 0, 0].key
    expected = a.chunks[0].key
    assert chunk_key == expected

    # as chunks[9] and chunks[10] shares the same shape,
    #  their keys should be equal.
    chunk_key = a.cix[1, 1, 1].key
    expected = a.chunks[9].key
    assert chunk_key == expected

    chunk_key = a.cix[1, 1, 2].key
    expected = a.chunks[11].key
    assert chunk_key == expected

    chunk_key = a.cix[0, -1, -1].key
    expected = a.chunks[5].key
    assert chunk_key == expected

    chunk_key = a.cix[0, -1, -1].key
    expected = a.chunks[5].key
    assert chunk_key == expected

    chunk_keys = [c.key for c in a.cix[0, 0, :]]
    expected = [c.key for c in [a.cix[0, 0, 0], a.cix[0, 0, 1], a.cix[0, 0, 2]]]
    assert chunk_keys == expected

    chunk_keys = [c.key for c in a.cix[:, 0, :]]
    expected = [
        c.key
        for c in [
            a.cix[0, 0, 0],
            a.cix[0, 0, 1],
            a.cix[0, 0, 2],
            a.cix[1, 0, 0],
            a.cix[1, 0, 1],
            a.cix[1, 0, 2],
        ]
    ]
    assert chunk_keys == expected

    chunk_keys = [c.key for c in a.cix[:, :, :]]
    expected = [c.key for c in a.chunks]
    assert chunk_keys == expected


def test_require_not_none():
    @utils.require_not_none(1)
    def should_exist():
        pass

    assert should_exist is not None

    @utils.require_not_none(None)
    def should_not_exist():
        pass

    assert should_not_exist is None

    @utils.require_module("numpy.fft")
    def should_exist_np():
        pass

    assert should_exist_np is not None

    @utils.require_module("numpy.fft_error")
    def should_not_exist_np():
        pass

    assert should_not_exist_np is None


def test_type_dispatcher():
    dispatcher = utils.TypeDispatcher()

    type1 = type("Type1", (), {})
    type2 = type("Type2", (type1,), {})
    type3 = type("Type3", (), {})

    dispatcher.register(object, lambda x: "Object")
    dispatcher.register(type1, lambda x: "Type1")
    dispatcher.register("pandas.DataFrame", lambda x: "DataFrame")

    assert "Type1" == dispatcher(type2())
    assert "DataFrame" == dispatcher(pd.DataFrame())
    assert "Object" == dispatcher(type3())

    dispatcher.unregister(object)
    with pytest.raises(KeyError):
        dispatcher(type3())


def test_fixed_size_file_object():
    arr = [str(i).encode() * 20 for i in range(10)]
    bts = os.linesep.encode().join(arr)
    bio = BytesIO(bts)

    ref_bio = BytesIO(bio.read(100))
    bio.seek(0)
    ref_bio.seek(0)
    fix_bio = utils.FixedSizeFileObject(bio, 100)

    assert ref_bio.readline() == fix_bio.readline()
    assert ref_bio.tell() == fix_bio.tell()
    pos = ref_bio.tell() + 10
    assert ref_bio.seek(pos) == fix_bio.seek(pos)
    assert ref_bio.read(5) == fix_bio.read(5)
    assert ref_bio.readlines(25) == fix_bio.readlines(25)
    assert list(ref_bio) == list(fix_bio)


def test_timer():
    with utils.Timer() as timer:
        time.sleep(0.1)

    assert timer.duration >= 0.1


def test_quiet_stdio():
    old_stdout, old_stderr = sys.stdout, sys.stderr

    class _IOWrapper:
        def __init__(self, name=None):
            self.name = name
            self.content = ""

        @staticmethod
        def writable():
            return True

        def write(self, d):
            self.content += d
            return len(d)

    stdout_w = _IOWrapper("stdout")
    stderr_w = _IOWrapper("stderr")
    executor = ThreadPoolExecutor(1)
    try:
        sys.stdout = stdout_w
        sys.stderr = stderr_w

        with utils.quiet_stdio():
            with utils.quiet_stdio():
                assert sys.stdout.writable()
                assert sys.stderr.writable()

                print("LINE 1", end="\n")
                print("LINE 2", file=sys.stderr, end="\n")
                executor.submit(print, "LINE T").result()
            print("LINE 3", end="\n")

        print("LINE 1", end="\n")
        print("LINE 2", file=sys.stderr, end="\n")
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    assert stdout_w.content == "LINE T\nLINE 1\n"
    assert stderr_w.content == "LINE 2\n"


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.version_info[:2] < (3, 7),
    reason="asyncio task timeout detector is not supported on python versions below 3.7",
)
async def test_asyncio_task_timeout_detector():
    log_file_name = "test_asyncio_task_timeout_detector.log"
    try:
        os.environ["MARS_DEBUG_ASYNCIO_TASK_TIMEOUT_CHECK_INTERVAL"] = "1"
        p = multiprocessing.Process(
            target=_run_task_timeout_detector, args=(log_file_name,)
        )
        p.start()
        while p.is_alive():
            await asyncio.sleep(0.1)
        with open(log_file_name, "r") as f:
            detector_log = f.read()
            assert "timeout_func" in detector_log
    finally:
        os.environ.pop("MARS_DEBUG_ASYNCIO_TASK_TIMEOUT_CHECK_INTERVAL")
        if os.path.exists(log_file_name):
            os.remove(log_file_name)


def _run_task_timeout_detector(log_file_name):
    from ..utils import logger, register_asyncio_task_timeout_detector

    fh = logging.FileHandler(log_file_name)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    async def timeout_func():
        await asyncio.sleep(2)

    async def main():
        task = register_asyncio_task_timeout_detector()
        await asyncio.create_task(timeout_func())
        task.cancel()

    asyncio.run(main())


def test_module_placeholder():
    required_module = utils.ModulePlaceholder("required_module")

    with pytest.raises(AttributeError):
        required_module()
    with pytest.raises(AttributeError) as e:
        required_module.method()
    msg = e.value.args[0]
    assert msg == "required_module is required but not installed."


def test_merge_dict():
    from ..utils import merge_dict

    assert merge_dict({}, {1: 2}) == {1: 2}
    assert merge_dict({1: 2}, {}) == {1: 2}
    assert merge_dict(
        {"a": {1: 2}, "b": {2: 3}, "c": {1: {2: 3}}},
        {"a": {1: 3}, "b": {2: 3}, "c": {1: {2: 4}}},
    ) == {"a": {1: 3}, "b": {2: 3}, "c": {1: {2: 4}}}
    with pytest.raises(ValueError):
        merge_dict({"a": {1: 2}, "b": {2: 3}}, {"a": {1: 3}}, overwrite=False)


def test_flatten_dict_to_nested_dict():
    from ..utils import flatten_dict_to_nested_dict

    assert flatten_dict_to_nested_dict({}) == {}
    with pytest.raises(ValueError):
        flatten_dict_to_nested_dict({"a.b.c": 1, "a.b": 2})
    assert flatten_dict_to_nested_dict({"a.b.c": 1, "a.b.d": 2}) == {
        "a": {"b": {"c": 1, "d": 2}}
    }


def test_readable_size():
    assert utils.readable_size(32) == "32.00"
    assert utils.readable_size(14354) == "14.02K"
    assert utils.readable_size(14354000) == "13.69M"
    assert utils.readable_size(14354000000) == "13.37G"
    assert utils.readable_size(14354000000000) == "13.05T"


def test_estimate_pandas_size():
    df1 = pd.DataFrame(np.random.rand(50, 10))
    assert utils.estimate_pandas_size(df1) == sys.getsizeof(df1)

    df2 = pd.DataFrame(np.random.rand(1000, 10))
    assert utils.estimate_pandas_size(df2) == sys.getsizeof(df2)

    df3 = pd.DataFrame(
        {
            "A": np.random.choice(["abcd", "def", "gh"], size=(1000,)),
            "B": np.random.rand(1000),
            "C": np.random.rand(1000),
        }
    )
    assert utils.estimate_pandas_size(df3) != sys.getsizeof(df3)

    s1 = pd.Series(np.random.rand(1000))
    assert utils.estimate_pandas_size(s1) == sys.getsizeof(s1)

    from ..dataframe.arrays import ArrowStringArray

    array = ArrowStringArray(np.random.choice(["abcd", "def", "gh"], size=(1000,)))
    s2 = pd.Series(array)
    assert utils.estimate_pandas_size(s2) == sys.getsizeof(s2)

    s3 = pd.Series(np.random.choice(["abcd", "def", "gh"], size=(1000,)))
    assert utils.estimate_pandas_size(s3) != sys.getsizeof(s3)
    assert (
        pytest.approx(utils.estimate_pandas_size(s3) / sys.getsizeof(s3), abs=0.5) == 1
    )

    idx1 = pd.MultiIndex.from_arrays(
        [np.arange(0, 1000), np.random.choice(["abcd", "def", "gh"], size=(1000,))]
    )
    assert utils.estimate_pandas_size(idx1) == sys.getsizeof(idx1)

    string_idx = pd.Index(np.random.choice(["a", "bb", "cc"], size=(1000,)))
    assert utils.estimate_pandas_size(string_idx) != sys.getsizeof(string_idx)
    assert (
        pytest.approx(
            utils.estimate_pandas_size(string_idx) / sys.getsizeof(string_idx), abs=0.5
        )
        == 1
    )

    # dataframe with multi index
    idx2 = pd.MultiIndex.from_arrays(
        [np.arange(0, 1000), np.random.choice(["abcd", "def", "gh"], size=(1000,))]
    )
    df4 = pd.DataFrame(
        {
            "A": np.random.choice(["abcd", "def", "gh"], size=(1000,)),
            "B": np.random.rand(1000),
            "C": np.random.rand(1000),
        },
        index=idx2,
    )
    assert utils.estimate_pandas_size(df4) != sys.getsizeof(df4)
    assert (
        pytest.approx(utils.estimate_pandas_size(df4) / sys.getsizeof(df4), abs=0.5)
        == 1
    )

    # series with multi index
    idx3 = pd.MultiIndex.from_arrays(
        [
            np.random.choice(["a1", "a2", "a3"], size=(1000,)),
            np.random.choice(["abcd", "def", "gh"], size=(1000,)),
        ]
    )
    s4 = pd.Series(np.arange(1000), index=idx3)

    assert utils.estimate_pandas_size(s4) == sys.getsizeof(s4)


@require_ray
def test_web_serialize_lambda():
    register_ray_serializers()
    df = md.DataFrame(
        mt.random.rand(10_0000, 4, chunk_size=1_0000), columns=list("abcd")
    )
    r = df.apply(lambda x: x)
    graph = TileableGraph([r])
    s = utils.serialize_serializable(graph)
    f = utils.deserialize_serializable(s)
    assert isinstance(f, TileableGraph)


def test_get_func_token_values():
    from ..utils import get_func_token_values

    assert get_func_token_values(test_get_func_token_values) == [
        test_get_func_token_values.__code__.co_code
    ]
    captured_vars = [1, 2, 3]

    def closure_func(a, b):
        return captured_vars

    assert get_func_token_values(closure_func)[1][0] == captured_vars
    assert get_func_token_values(partial(closure_func, 1))[0][0] == 1
    assert get_func_token_values(partial(closure_func, 1))[-1][0] == captured_vars

    from .._utils import ceildiv

    assert get_func_token_values(ceildiv) == [ceildiv.__module__, ceildiv.__name__]

    class Func:
        def __call__(self, *args, **kwargs):
            pass

    func = Func()
    assert get_func_token_values(func) == [func]


@pytest.mark.parametrize("id_length", [0, 5, 32, 63])
def test_gen_random_id(id_length):
    rnd_id = utils.new_random_id(id_length)
    assert len(rnd_id) == id_length


@pytest.mark.asyncio
async def test_retry_callable():
    assert utils.retry_callable(lambda x: x)(1) == 1
    assert utils.retry_callable(lambda x: 0)(1) == 0

    class CustomException(BaseException):
        pass

    def f1(x):
        nonlocal num_retried
        num_retried += 1
        if num_retried == 3:
            return x
        raise CustomException

    num_retried = 0
    with pytest.raises(CustomException):
        utils.retry_callable(f1)(1)
    assert utils.retry_callable(f1, ex_type=CustomException)(1) == 1
    num_retried = 0
    with pytest.raises(CustomException):
        utils.retry_callable(f1, max_retries=2, ex_type=CustomException)(1)
    num_retried = 0
    assert utils.retry_callable(f1, max_retries=3, ex_type=CustomException)(1) == 1

    async def f2(x):
        return f1(x)

    num_retried = 0
    with pytest.raises(CustomException):
        await utils.retry_callable(f2)(1)
    assert await utils.retry_callable(f2, ex_type=CustomException)(1) == 1
