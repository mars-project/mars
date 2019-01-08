#!/usr/bin/env python
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

import unittest

import numpy as np

from mars.tensor.expressions.datasource import ones, tensor
from mars.tensor.expressions.datasource.ones import TensorOnes
from mars.tensor.expressions.indexing import choose, unravel_index, nonzero
from mars.tensor.expressions.indexing.setitem import TensorIndexSetValue
from mars.tensor.expressions.merge.concatenate import TensorConcatenate
from mars.config import option_context


class Test(unittest.TestCase):
    def testBoolIndexing(self):
        t = ones((100, 200, 300))
        indexed = t[t < 2]
        self.assertEqual(len(indexed.shape), 1)
        self.assertTrue(np.isnan(indexed.shape[0]))

        t2 = ones((100, 200))
        indexed = t[t2 < 2]
        self.assertEqual(len(indexed.shape), 2)
        self.assertTrue(np.isnan(indexed.shape[0]))
        self.assertEqual(indexed.shape[1], 300)

        t3 = ones((101, 200))
        with self.assertRaises(IndexError) as cm:
            _ = t[t3 < 2]  # noqa: F841
        e = str(cm.exception)
        self.assertIn('along dimension 0', str(e))
        self.assertIn('dimension is 100 but corresponding boolean dimension is 101', str(e))

        t4 = ones((100, 201))
        with self.assertRaises(IndexError) as cm:
            _ = t[t4 < 2]  # noqa: F841
        e = str(cm.exception)
        self.assertIn('along dimension 1', str(e))
        self.assertIn('dimension is 200 but corresponding boolean dimension is 201', str(e))

    def testSlice(self):
        t = ones((100, 200, 300))
        t2 = t[10: 30, 199:, -30: 303]
        self.assertEqual(t2.shape, (20, 1, 30))

        t3 = t[10:90:4, 20:80:5]
        s1 = len(list(range(100))[10:90:4])
        s2 = len(list(range(200))[20:80:5])
        self.assertEqual(t3.shape, (s1, s2, 300))

    def testFancyIndexing(self):
        t = ones((100, 200, 300))
        t2 = t[[0, 1], [2, 3]]
        self.assertEqual(t2.shape, (2, 300))

        t3 = t[[[0, 1], [2, 3]], [4, 5]]
        self.assertEqual(t3.shape, (2, 2, 300))

        with self.assertRaises(IndexError) as cm:
            _ = t[[1, 2], [3, 4, 5]]  # noqa: F841
        e = str(cm.exception)
        self.assertEqual(e, 'shape mismatch: indexing arrays could not be broadcast '
                            'together with shapes (2,) (3,)')

        t = ones((100, 200, 300), chunk_size=10)
        t4 = t[:10, -10:, [13, 244, 151, 242, 34]].tiles()
        self.assertEqual(t4.shape, (10, 10, 5))
        self.assertEqual(t4.chunk_shape, (1, 1, 1))

        t5 = t[:10, -10:, [1, 10, 20, 33, 34, 200]].tiles()
        self.assertEqual(t5.shape, (10, 10, 6))
        self.assertEqual(t5.chunk_shape, (1, 1, 5))

    def testMixedIndexing(self):
        t = ones((100, 200, 300, 400))

        with self.assertRaises(IndexError):
            _ = t[ones((100, 200), dtype=float)]  # noqa: F841

        t2 = t[ones(100) < 2, ..., 20::101, 2]
        self.assertEqual(len(t2.shape), 3)
        self.assertTrue(np.isnan(t2.shape[0]))
        self.assertEqual(t2.shape[1:], (200, 3))

    def testBoolIndexingTiles(self):
        t = ones((100, 200, 300), chunk_size=30)
        indexed = t[t < 2]
        indexed.tiles()

        self.assertEqual(len(indexed.chunks), 280)
        self.assertEqual(indexed.chunks[0].index, (0,))
        self.assertEqual(indexed.chunks[20].index, (20,))
        self.assertIs(indexed.chunks[20].inputs[0], t.cix[(0, 2, 0)].data)
        self.assertIs(indexed.chunks[20].inputs[1], indexed.op.indexes[0].cix[0, 2, 0].data)

        t2 = ones((100, 200), chunk_size=30)
        indexed2 = t[t2 < 2]
        indexed2.tiles()

        self.assertEqual(len(indexed2.chunks), 280)
        self.assertEqual(len(indexed2.chunks[0].shape), 2)
        self.assertTrue(np.isnan(indexed2.chunks[0].shape[0]))
        self.assertEqual(indexed2.chunks[0].shape[1], 30)
        self.assertEqual(indexed2.chunks[20].inputs[0], t.cix[(0, 2, 0)].data)
        self.assertEqual(indexed2.chunks[20].inputs[1], indexed2.op.indexes[0].cix[0, 2].data)

    def testSliceTiles(self):
        t = ones((100, 200, 300), chunk_size=30)
        t2 = t[10: 40, 199:, -30: 303]
        t2.tiles()

        self.assertEqual(t2.chunk_shape, (2, 1, 1))
        self.assertEqual(t2.chunks[0].inputs[0], t.cix[0, -1, -1].data)
        self.assertEqual(t2.chunks[0].op.indexes, [slice(10, 30, 1), slice(19, 20, 1), slice(None)])
        self.assertEqual(t2.chunks[0].index, (0, 0, 0))
        self.assertEqual(t2.chunks[1].inputs[0], t.cix[1, -1, -1].data)
        self.assertEqual(t2.chunks[1].op.indexes, [slice(0, 10, 1), slice(19, 20, 1), slice(None)])
        self.assertEqual(t2.chunks[1].index, (1, 0, 0))

    def testIndicesIndexingTiles(self):
        t = ones((10, 20, 30), chunk_size=(2, 20, 30))
        t2 = t[3]
        t2.tiles()

        self.assertEqual(len(t2.chunks), 1)
        self.assertIs(t2.chunks[0].inputs[0], t.cix[1, 0, 0].data)
        self.assertEqual(t2.chunks[0].op.indexes[0], 1)

        t3 = t[4]
        t3.tiles()

        self.assertEqual(len(t3.chunks), 1)
        self.assertIs(t3.chunks[0].inputs[0], t.cix[2, 0, 0].data)
        self.assertEqual(t3.chunks[0].op.indexes[0], 0)

    def testMixedIndexingTiles(self):
        t = ones((100, 200, 300, 400), chunk_size=24)

        cmp = ones(400, chunk_size=24) < 2
        t2 = t[10:90:3, 5, ..., None, cmp]
        t2.tiles()

        self.assertEqual(t2.shape[:-1], (27, 300, 1))
        self.assertTrue(np.isnan(t2.shape[-1]))
        self.assertEqual(t2.chunk_shape, (4, 13, 1, 17))
        self.assertEqual(t2.chunks[0].op.indexes, [slice(10, 24, 3), 5, slice(None), None, cmp.cix[0,].data])

    def testSetItem(self):
        shape = (10, 20, 30, 40)
        t = ones(shape, chunk_size=5, dtype='i4')
        t[5:20:3, 5, ..., :-5] = 2.2

        self.assertIsInstance(t.op, TensorIndexSetValue)
        self.assertEqual(t.shape, shape)
        self.assertIsInstance(t.inputs[0].op.outputs[0].op, TensorOnes)

        t.tiles()
        self.assertIsInstance(t.chunks[0].op, TensorOnes)
        self.assertIsInstance(t.cix[1, 1, 0, 0].op, TensorIndexSetValue)
        self.assertEqual(t.cix[1, 1, 0, 0].op.value, 2)

        t2 = ones(shape, chunk_size=5, dtype='i4')
        shape = t2[5:20:3, 5, ..., :-5].shape
        t2[5:20:3, 5, ..., :-5] = ones(shape, chunk_size=4, dtype='i4') * 2

        t2.tiles()
        self.assertIsInstance(t2.chunks[0].op, TensorOnes)
        self.assertIsInstance(t2.cix[1, 1, 0, 0].op, TensorIndexSetValue)
        self.assertIsInstance(t2.cix[1, 1, 0, 0].op.value.op, TensorConcatenate)

        with self.assertRaises(ValueError):
            t[0, 0, 0, 0] = ones(2, chunk_size=10)

    def testChoose(self):
        with option_context() as options:
            options.tensor.chunk_size = 2

            choices = [[0, 1, 2, 3], [10, 11, 12, 13],
                       [20, 21, 22, 23], [30, 31, 32, 33]]
            a = choose([2, 3, 1, 0], choices)

            a.tiles()
            self.assertEqual(len(a.chunks), 2)
            self.assertIsInstance(a.chunks[0].op, type(a.op))
            self.assertEqual(len(a.chunks[0].inputs), 5)

    def testUnravelIndex(self):
        indices = tensor([22, 41, 37], chunk_size=1)
        t = unravel_index(indices, (7, 6))

        self.assertEqual(len(t), 2)

        [r.tiles() for r in t]

        self.assertEqual(len(t[0].chunks), 3)
        self.assertEqual(len(t[1].chunks), 3)

    def testNonzero(self):
        x = tensor([[1, 0, 0], [0, 2, 0], [1, 1, 0]], chunk_size=2)
        y = nonzero(x)

        self.assertEqual(len(y), 2)

    def testOperandKey(self):
        t = ones((10, 2), chunk_size=5)
        t_slice1 = t[:5]
        t_slice2 = t[5:]

        self.assertNotEqual(t_slice1.op.key, t_slice2.op.key)
