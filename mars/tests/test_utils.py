# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import copy
import os
import shutil
import sys
import tempfile
import textwrap
import time
import unittest

import numpy as np
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from mars import utils
from mars.compat import Enum
from mars.tensor.expressions.fetch import TensorFetch
import mars.tensor as mt

try:
    unicode  # noqa F821
except NameError:
    unicode = str


class Test(unittest.TestCase):
    def testStringConversion(self):
        s = None
        self.assertIsNone(utils.to_binary(s))
        self.assertIsNone(utils.to_str(s))
        self.assertIsNone(utils.to_text(s))

        s = 'abcdefg'
        self.assertIsInstance(utils.to_binary(s), bytes)
        self.assertEqual(utils.to_binary(s), b'abcdefg')
        self.assertIsInstance(utils.to_str(s), str)
        self.assertEqual(utils.to_str(s), 'abcdefg')
        self.assertIsInstance(utils.to_text(s), unicode)
        self.assertEqual(utils.to_text(s), u'abcdefg')

        ustr = type('ustr', (str,), {})
        self.assertIsInstance(utils.to_str(ustr(s)), str)
        self.assertEqual(utils.to_str(ustr(s)), 'abcdefg')

        s = b'abcdefg'
        self.assertIsInstance(utils.to_binary(s), bytes)
        self.assertEqual(utils.to_binary(s), b'abcdefg')
        self.assertIsInstance(utils.to_str(s), str)
        self.assertEqual(utils.to_str(s), 'abcdefg')
        self.assertIsInstance(utils.to_text(s), unicode)
        self.assertEqual(utils.to_text(s), u'abcdefg')

        ubytes = type('ubytes', (bytes,), {})
        self.assertIsInstance(utils.to_binary(ubytes(s)), bytes)
        self.assertEqual(utils.to_binary(ubytes(s)), b'abcdefg')

        s = u'abcdefg'
        self.assertIsInstance(utils.to_binary(s), bytes)
        self.assertEqual(utils.to_binary(s), b'abcdefg')
        self.assertIsInstance(utils.to_str(s), str)
        self.assertEqual(utils.to_str(s), 'abcdefg')
        self.assertIsInstance(utils.to_text(s), unicode)
        self.assertEqual(utils.to_text(s), u'abcdefg')

        uunicode = type('uunicode', (unicode,), {})
        self.assertIsInstance(utils.to_text(uunicode(s)), unicode)
        self.assertEqual(utils.to_text(uunicode(s)), u'abcdefg')

        with self.assertRaises(TypeError):
            utils.to_binary(utils)
        with self.assertRaises(TypeError):
            utils.to_str(utils)
        with self.assertRaises(TypeError):
            utils.to_text(utils)

    def testTokenize(self):
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
                v = [1, 2.3, '456', u'789', b'101112', None, np.ndarray, [912, 'uvw'],
                     np.arange(0, 10), np.array(10), np.array([b'\x01\x32\xff']),
                     np.int64, TestEnum.VAL1]
                copy_v = copy.deepcopy(v)
                self.assertEqual(utils.tokenize(v + [mmp_array1], ext_data=1234),
                                 utils.tokenize(copy_v + [mmp_array2], ext_data=1234))
            finally:
                del mmp_array1, mmp_array2
        finally:
            shutil.rmtree(tempdir)

        v = {'a', 'xyz', 'uvw'}
        self.assertEqual(utils.tokenize(v), utils.tokenize(copy.deepcopy(v)))

        v = dict(x='abcd', y=98765)
        self.assertEqual(utils.tokenize(v), utils.tokenize(copy.deepcopy(v)))

        # pandas relative
        if pd is not None:
            df = pd.DataFrame([[utils.to_binary('测试'), utils.to_text('数据')]],
                              index=['a'], columns=['中文', 'data'])
            v = [df, df.index, df.columns, df['data']]
            self.assertEqual(utils.tokenize(v), utils.tokenize(copy.deepcopy(v)))

        non_tokenizable_cls = type('non_tokenizable_cls', (object,), {})
        with self.assertRaises(TypeError):
            utils.tokenize(non_tokenizable_cls())

        class CustomizedTokenize(object):
            def __mars_tokenize__(self):
                return id(type(self)), id(non_tokenizable_cls)

        self.assertEqual(utils.tokenize(CustomizedTokenize()),
                         utils.tokenize(CustomizedTokenize()))

    def testBuildGraph(self):
        a = mt.ones((10, 10), chunk_size=8)
        b = mt.ones((10, 10), chunk_size=8)
        c = (a + 1) * 2 + b

        graph = utils.build_graph([c])
        self.assertEqual(len(graph), 5)
        self.assertIn(a.data, graph)
        self.assertIn(b.data, graph)
        self.assertEqual(graph.count_successors(a.data), 1)
        self.assertEqual(graph.count_predecessors(a.data), 0)
        self.assertEqual(graph.count_successors(c.data), 0)
        self.assertEqual(graph.count_predecessors(c.data), 2)

        graph = utils.build_graph([a, b, c])
        self.assertEqual(len(graph), 5)

        graph = utils.build_graph([a, b, c], graph=graph)
        self.assertEqual(len(graph), 5)

        graph = utils.build_graph([c], tiled=True, compose=False)
        self.assertEqual(len(graph), 20)

        graph = utils.build_graph([c], tiled=True)
        self.assertEqual(len(graph), 12)

        # test fetch replacement
        a = mt.ones((10, 10), chunk_size=8)
        b = mt.ones((10, 10), chunk_size=8)
        c = (a + 1) * 2 + b
        executed_keys = [a.key, b.key]

        graph = utils.build_graph([c], executed_keys=executed_keys)
        self.assertEqual(len(graph), 5)
        self.assertNotIn(a.data, graph)
        self.assertNotIn(b.data, graph)
        self.assertTrue(any(isinstance(n.op, TensorFetch) for n in graph))
        self.assertEqual(graph.count_successors(c.data), 0)
        self.assertEqual(graph.count_predecessors(c.data), 1)

        executed_keys = [(a + 1).key]
        graph = utils.build_graph([c], executed_keys=executed_keys)
        self.assertTrue(any(isinstance(n.op, TensorFetch) for n in graph))
        self.assertEqual(len(graph), 4)

        executed_keys = [((a + 1) * 2).key]
        graph = utils.build_graph([c], executed_keys=executed_keys)
        self.assertTrue(any(isinstance(n.op, TensorFetch) for n in graph))
        self.assertEqual(len(graph), 3)

        executed_keys = [c.key]
        graph = utils.build_graph([c], executed_keys=executed_keys)
        self.assertTrue(any(isinstance(n.op, TensorFetch) for n in graph))
        self.assertEqual(len(graph), 1)

    def testKernelMode(self):
        from mars.config import option_context, options

        @utils.kernel_mode
        def wrapped():
            return utils.is_eager_mode()

        self.assertFalse(options.eager_mode)
        self.assertFalse(wrapped())

        with option_context({'eager_mode': True}):
            self.assertTrue(options.eager_mode)
            self.assertFalse(wrapped())

    def testBlacklistSet(self):
        blset = utils.BlacklistSet(0.1)
        blset.update([1, 2])
        time.sleep(0.3)
        blset.add(3)

        with self.assertRaises(KeyError):
            blset.remove(2)

        self.assertNotIn(1, blset)
        self.assertNotIn(2, blset)
        self.assertIn(3, blset)

        blset.add(4)
        time.sleep(0.3)
        blset.add(5)
        blset.add(6)

        self.assertSetEqual({5, 6}, set(blset))

    def testLazyImport(self):
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
            self.assertIsNone(non_exist_mod)

            mod = utils.lazy_import(
                'test_mod', globals=globals(), locals=locals(), rename='mod')
            self.assertIsNotNone(mod)
            self.assertEqual(mod.__version__, '0.1.0b1')

            glob = globals().copy()
            mod = utils.lazy_import(
                'test_mod', globals=glob, locals=locals(), rename='mod')
            glob['mod'] = mod
            self.assertIsNotNone(mod)
            self.assertEqual(mod.__version__, '0.1.0b1')
            self.assertEqual(type(glob['mod']).__name__, 'module')
        finally:
            shutil.rmtree(temp_dir)
            sys.path = old_sys_path
