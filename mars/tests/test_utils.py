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
import unittest

import numpy as np
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from mars import utils
from mars.tensor.expressions.datasource.core import TensorFetch
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
        v = (1, 2.3, '456', u'789', b'101112', None, np.ndarray,
             [912, 'uvw'], np.arange(0, 10), np.int64)
        self.assertEqual(utils.tokenize(v), utils.tokenize(copy.deepcopy(v)))

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
