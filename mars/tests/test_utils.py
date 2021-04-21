# -*- coding: utf-8 -*-
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

import asyncio
import copy
import os
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

from mars import utils
from mars.tensor.fetch import TensorFetch
import mars.tensor as mt


def test_string_conversion():
    s = None
    assert utils.to_binary(s) is None
    assert utils.to_str(s) is None
    assert utils.to_text(s) is None

    s = 'abcdefg'
    assert isinstance(utils.to_binary(s), bytes)
    assert utils.to_binary(s) == b'abcdefg'
    assert isinstance(utils.to_str(s), str)
    assert utils.to_str(s) == 'abcdefg'
    assert isinstance(utils.to_text(s), str)
    assert utils.to_text(s) == u'abcdefg'

    ustr = type('ustr', (str,), {})
    assert isinstance(utils.to_str(ustr(s)), str)
    assert utils.to_str(ustr(s)) == 'abcdefg'

    s = b'abcdefg'
    assert isinstance(utils.to_binary(s), bytes)
    assert utils.to_binary(s) == b'abcdefg'
    assert isinstance(utils.to_str(s), str)
    assert utils.to_str(s) == 'abcdefg'
    assert isinstance(utils.to_text(s), str)
    assert utils.to_text(s) == u'abcdefg'

    ubytes = type('ubytes', (bytes,), {})
    assert isinstance(utils.to_binary(ubytes(s)), bytes)
    assert utils.to_binary(ubytes(s)) == b'abcdefg'

    s = u'abcdefg'
    assert isinstance(utils.to_binary(s), bytes)
    assert utils.to_binary(s) == b'abcdefg'
    assert isinstance(utils.to_str(s), str)
    assert utils.to_str(s) == 'abcdefg'
    assert isinstance(utils.to_text(s), str)
    assert utils.to_text(s) == u'abcdefg'

    uunicode = type('uunicode', (str,), {})
    assert isinstance(utils.to_text(uunicode(s)), str)
    assert utils.to_text(uunicode(s)) == u'abcdefg'

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
        VAL1 = 'val1'

    tempdir = tempfile.mkdtemp('mars_test_utils_')
    try:
        filename = os.path.join(tempdir, 'test_npa.dat')
        mmp_array = np.memmap(filename, dtype=float, mode='w+', shape=(3, 4))
        mmp_array[:] = np.random.random((3, 4)).astype(float)
        mmp_array.flush()
        del mmp_array

        mmp_array1 = np.memmap(filename, dtype=float, shape=(3, 4))
        mmp_array2 = np.memmap(filename, dtype=float, shape=(3, 4))

        try:
            v = [1, 2.3, '456', u'789', b'101112', 2147483649, None, np.ndarray,
                 [912, 'uvw'], np.arange(0, 10), np.array(10), np.array([b'\x01\x32\xff']),
                 np.int64, TestEnum.VAL1]
            copy_v = copy.deepcopy(v)
            assert (utils.tokenize(v + [mmp_array1], ext_data=1234)
                    == utils.tokenize(copy_v + [mmp_array2], ext_data=1234))
        finally:
            del mmp_array1, mmp_array2
    finally:
        shutil.rmtree(tempdir)

    v = {'a', 'xyz', 'uvw'}
    assert utils.tokenize(v) == utils.tokenize(copy.deepcopy(v))

    v = dict(x='abcd', y=98765)
    assert utils.tokenize(v) == utils.tokenize(copy.deepcopy(v))

    v = dict(x=dict(a=1, b=[1, 2, 3]), y=12345)
    assert utils.tokenize(v) == utils.tokenize(copy.deepcopy(v))

    # pandas relative
    if pd is not None:
        df = pd.DataFrame([[utils.to_binary('测试'), utils.to_text('数据')]],
                          index=['a'], columns=['中文', 'data'])
        v = [df, df.index, df.columns, df['data'], pd.Categorical(list('ABCD'))]
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


def test_build_tileable_graph():
    a = mt.ones((10, 10), chunk_size=8)
    b = mt.ones((10, 10), chunk_size=8)
    c = (a + 1) * 2 + b

    graph = utils.build_tileable_graph([c], set())
    assert len(graph) == 5
    a_data = next(n for n in graph if n.key == a.key)
    assert graph.count_successors(a_data) == 1
    assert graph.count_predecessors(a_data) == 0
    c_data = next(n for n in graph if n.key == c.key)
    assert graph.count_successors(c_data) == 0
    assert graph.count_predecessors(c_data) == 2

    graph = utils.build_tileable_graph([a, b, c], set())
    assert len(graph) == 5

    # test fetch replacement
    a = mt.ones((10, 10), chunk_size=8)
    b = mt.ones((10, 10), chunk_size=8)
    c = (a + 1) * 2 + b
    executed_keys = {a.key, b.key}

    graph = utils.build_tileable_graph([c], executed_keys)
    assert len(graph) == 5
    assert a.data not in graph
    assert b.data not in graph
    assert any(isinstance(n.op, TensorFetch) for n in graph)
    c_data = next(n for n in graph if n.key == c.key)
    assert graph.count_successors(c_data) == 0
    assert graph.count_predecessors(c_data) == 2

    executed_keys = {(a + 1).key}
    graph = utils.build_tileable_graph([c], executed_keys)
    assert any(isinstance(n.op, TensorFetch) for n in graph)
    assert len(graph) == 4

    executed_keys = {((a + 1) * 2).key}
    graph = utils.build_tileable_graph([c], executed_keys)
    assert any(isinstance(n.op, TensorFetch) for n in graph)
    assert len(graph) == 3

    executed_keys = {c.key}
    graph = utils.build_tileable_graph([c], executed_keys)
    assert any(isinstance(n.op, TensorFetch) for n in graph)
    assert len(graph) == 1


def test_blacklist_set():
    blset = utils.BlacklistSet(0.1)
    blset.update([1, 2])
    time.sleep(0.3)
    blset.add(3)

    with pytest.raises(KeyError):
        blset.remove(2)

    assert 1 not in blset
    assert 2 not in blset
    assert 3 in blset

    blset.add(4)
    time.sleep(0.3)
    blset.add(5)
    blset.add(6)

    assert {5, 6} == set(blset)


def test_lazy_import():
    old_sys_path = sys.path
    mock_mod = textwrap.dedent("""
    __version__ = '0.1.0b1'
    """.strip())

    temp_dir = tempfile.mkdtemp(prefix='mars-utils-test-')
    sys.path += [temp_dir]
    try:
        with open(os.path.join(temp_dir, 'test_mod.py'), 'w') as outf:
            outf.write(mock_mod)

        non_exist_mod = utils.lazy_import('non_exist_mod', locals=locals())
        assert non_exist_mod is None

        mod = utils.lazy_import(
            'test_mod', globals=globals(), locals=locals(), rename='mod')
        assert mod is not None
        assert mod.__version__ == '0.1.0b1'

        glob = globals().copy()
        mod = utils.lazy_import(
            'test_mod', globals=glob, locals=locals(), rename='mod')
        glob['mod'] = mod
        assert mod is not None
        assert mod.__version__ == '0.1.0b1'
        assert type(glob['mod']).__name__ == 'module'
    finally:
        shutil.rmtree(temp_dir)
        sys.path = old_sys_path


def test_chunks_indexer():
    a = mt.ones((3, 4, 5), chunk_size=2)
    a = a.tiles()

    assert a.chunk_shape == (2, 2, 3)

    with pytest.raises(ValueError):
        _ = a.cix[1]
    with pytest.raises(ValueError):
        _ = a.cix[1, :]

    chunk_key = a.cix[0, 0, 0].key
    expected = a.chunks[0].key
    assert chunk_key == expected

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
    expected = [c.key for c in [a.cix[0, 0, 0], a.cix[0, 0, 1], a.cix[0, 0, 2],
                                a.cix[1, 0, 0], a.cix[1, 0, 1], a.cix[1, 0, 2]]]
    assert chunk_keys == expected

    chunk_keys = [c.key for c in a.cix[:, :, :]]
    expected = [c.key for c in a.chunks]
    assert chunk_keys == expected


def test_check_chunks_unknown_shape():
    with pytest.raises(ValueError):
        a = mt.random.rand(10, chunk_size=5)
        mt.random.shuffle(a)
        a = a.tiles()
        utils.check_chunks_unknown_shape([a], ValueError)


def test_insert_reversed_tuple():
    assert utils.insert_reversed_tuple((), 9) == (9,)
    assert utils.insert_reversed_tuple((7, 4, 3, 1), 9) == (9, 7, 4, 3, 1)
    assert utils.insert_reversed_tuple((7, 4, 3, 1), 6) == (7, 6, 4, 3, 1)
    assert utils.insert_reversed_tuple((7, 4, 3, 1), 4) == (7, 4, 3, 1)
    assert utils.insert_reversed_tuple((7, 4, 3, 1), 0) == (7, 4, 3, 1, 0)


def test_require_not_none():
    @utils.require_not_none(1)
    def should_exist():
        pass

    assert should_exist is not None

    @utils.require_not_none(None)
    def should_not_exist():
        pass

    assert should_not_exist is None

    @utils.require_module('numpy.fft')
    def should_exist_np():
        pass

    assert should_exist_np is not None

    @utils.require_module('numpy.fft_error')
    def should_not_exist_np():
        pass

    assert should_not_exist_np is None


def test_type_dispatcher():
    dispatcher = utils.TypeDispatcher()

    type1 = type('Type1', (), {})
    type2 = type('Type2', (type1,), {})
    type3 = type('Type3', (), {})

    dispatcher.register(object, lambda x: 'Object')
    dispatcher.register(type1, lambda x: 'Type1')
    dispatcher.register('pandas.DataFrame', lambda x: 'DataFrame')

    assert 'Type1' == dispatcher(type2())
    assert 'DataFrame' == dispatcher(pd.DataFrame())
    assert 'Object' == dispatcher(type3())

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
            self.content = ''

        @staticmethod
        def writable():
            return True

        def write(self, d):
            self.content += d
            return len(d)

    stdout_w = _IOWrapper('stdout')
    stderr_w = _IOWrapper('stderr')
    executor = ThreadPoolExecutor(1)
    try:
        sys.stdout = stdout_w
        sys.stderr = stderr_w

        with utils.quiet_stdio():
            with utils.quiet_stdio():
                assert sys.stdout.writable()
                assert sys.stderr.writable()

                print('LINE 1', end='\n')
                print('LINE 2', file=sys.stderr, end='\n')
                executor.submit(print, 'LINE T').result()
            print('LINE 3', end='\n')

        print('LINE 1', end='\n')
        print('LINE 2', file=sys.stderr, end='\n')
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    assert stdout_w.content == 'LINE T\nLINE 1\n'
    assert stderr_w.content == 'LINE 2\n'


@pytest.mark.asyncio
@pytest.mark.parametrize('use_async', [False, True])
async def test_batch_decorator(use_async):
    def _wrap_async(func):
        async def _wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        _wrapped.__name__ = func.__name__
        return _wrapped if use_async else func

    class TestClass:
        def __init__(self):
            self.arg_list = []
            self.kwarg_list = []

        @utils.extensible
        @_wrap_async
        def method1(self, *args, **kwargs):
            raise NotImplementedError

        @method1.batch
        @_wrap_async
        def method1(self, args_list, kwargs_list):
            self.arg_list.extend(args_list)
            self.kwarg_list.extend(kwargs_list)
            return [len(self.kwarg_list)] * len(args_list)

        @utils.extensible
        @_wrap_async
        def method2(self, *args, **kwargs):
            self.arg_list.append(tuple(a - 1 for a in args))
            self.kwarg_list.append({k: v - 1 for k, v in kwargs.items()})
            return len(self.kwarg_list)

        @utils.extensible
        @_wrap_async
        def method3(self, *args, **kwargs):
            self.arg_list.append(tuple(a * 2 for a in args))
            self.kwarg_list.append({k: v * 2 for k, v in kwargs.items()})
            return len(self.kwarg_list)

        @method3.batch
        @_wrap_async
        def method3(self, args_list, kwargs_list):
            self.arg_list.extend([tuple(a * 2 + 1 for a in args) for args in args_list])
            self.kwarg_list.extend([{k: v * 2 + 1 for k, v in kwargs.items()}
                                    for kwargs in kwargs_list])
            return [len(self.kwarg_list)] * len(args_list)

    if use_async:
        assert asyncio.iscoroutinefunction(TestClass.method1)

    test_inst = TestClass()

    ret = test_inst.method1.batch(
        test_inst.method1.delay(12),
        test_inst.method1.delay(10))
    ret = await ret if use_async else ret
    assert ret == [2, 2]
    assert test_inst.arg_list == [(12,), (10,)]
    assert test_inst.kwarg_list == [{}, {}]

    test_inst.arg_list.clear()
    test_inst.kwarg_list.clear()

    ret = test_inst.method1(12, kwarg=34)
    ret = await ret if use_async else ret
    assert ret == 1
    assert test_inst.arg_list == [(12,)]
    assert test_inst.kwarg_list == [{'kwarg': 34}]

    if sys.version_info[:2] > (3, 6):
        test_inst = TestClass()
        ret = test_inst.method2.batch(
            test_inst.method2.delay(12, kwarg=34),
            test_inst.method2.delay(10, kwarg=33)
        )
        ret = await ret if use_async else ret
        assert ret == [1, 2]
        assert test_inst.arg_list == [(11,), (9,)]
        assert test_inst.kwarg_list == [{'kwarg': 33}, {'kwarg': 32}]

        test_inst = TestClass()
        ret = test_inst.method3(15, kwarg=56)
        ret = await ret if use_async else ret
        assert ret == 1
        ret = test_inst.method3.batch(
            test_inst.method3.delay(16, kwarg=57),
            test_inst.method3.delay(17, kwarg=58)
        )
        ret = await ret if use_async else ret
        assert ret == [3, 3]
        assert test_inst.arg_list == [(30,), (33,), (35,)]
        assert test_inst.kwarg_list == [{'kwarg': 112}, {'kwarg': 115}, {'kwarg': 117}]

        if use_async:
            test_inst = TestClass()
            ret = await test_inst.method2.batch(
                test_inst.method2.delay(12, kwarg=34),
                test_inst.method2.delay(10, kawarg=33))
            assert ret == [1, 2]
