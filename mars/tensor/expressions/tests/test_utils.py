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
from numbers import Integral

import numpy as np

from mars.tensor.expressions.utils import normalize_chunk_sizes, broadcast_shape, \
    replace_ellipsis, calc_sliced_size, slice_split, decide_unify_split, unify_chunks, \
    split_index_into_chunks, decide_chunk_sizes
from mars.tensor.expressions.datasource import ones
from mars.config import option_context


class Test(unittest.TestCase):
    def testNormalizeChunks(self):
        self.assertEqual(((2, 2, 1), (2, 2, 2)), normalize_chunk_sizes((5, 6), (2, 2)))
        with self.assertRaises(ValueError):
            normalize_chunk_sizes((4, 6), ((2, 2, 1), (2, 2, 2)))
        self.assertEqual(((10, 10, 10), (5,)), normalize_chunk_sizes((30, 5), 10))

    def testBroadcastShape(self):
        self.assertEqual((4, 3), broadcast_shape((4, 3), (3,)))
        self.assertEqual((4, 3), broadcast_shape((4, 3), (4, 1)))
        self.assertEqual((3, 4, 2), broadcast_shape((3, 4, 2), (4, 2)))
        self.assertEqual((3, 4), broadcast_shape((3, 4), ()))

    def testReplaceIllipsis(self):
        index = (slice(2), Ellipsis, slice(1))
        self.assertEqual(replace_ellipsis(index, 5),
                         (slice(2), slice(None), slice(None), slice(None), slice(1)))

        index = (None, None, slice(2), Ellipsis)
        self.assertEqual(replace_ellipsis(index, 5),
                         (None, None, slice(2), slice(None), slice(None), slice(None), slice(None)))

        with self.assertRaises(IndexError):
            index = (None, None, Ellipsis, slice(2), Ellipsis)
            replace_ellipsis(replace_ellipsis(index, 5))

    def testCalcSlicedSize(self):
        sliceobj = slice(5)
        self.assertEqual(calc_sliced_size(6, sliceobj), 5)
        self.assertEqual(calc_sliced_size(5, sliceobj), 5)
        self.assertEqual(calc_sliced_size(4, sliceobj), 4)

        sliceobj = slice(-5)
        self.assertEqual(calc_sliced_size(6, sliceobj), 1)
        self.assertEqual(calc_sliced_size(5, sliceobj), 0)
        self.assertEqual(calc_sliced_size(4, sliceobj), 0)

        sliceobj = slice(3, 6)
        self.assertEqual(calc_sliced_size(3, sliceobj), 0)
        self.assertEqual(calc_sliced_size(5, sliceobj), 2)
        self.assertEqual(calc_sliced_size(6, sliceobj), 3)
        self.assertEqual(calc_sliced_size(7, sliceobj), 3)

        sliceobj = slice(-6, -3)
        self.assertEqual(calc_sliced_size(3, sliceobj), 0)
        self.assertEqual(calc_sliced_size(5, sliceobj), 2)
        self.assertEqual(calc_sliced_size(6, sliceobj), 3)
        self.assertEqual(calc_sliced_size(7, sliceobj), 3)

        sliceobj = slice(3, 8, 2)
        self.assertEqual(calc_sliced_size(10, sliceobj), 3)
        self.assertEqual(calc_sliced_size(7, sliceobj), 2)
        self.assertEqual(calc_sliced_size(8, sliceobj), 3)
        self.assertEqual(calc_sliced_size(4, sliceobj), 1)
        self.assertEqual(calc_sliced_size(3, sliceobj), 0)

        sliceobj = slice(8, 3, -2)
        self.assertEqual(calc_sliced_size(10, sliceobj), 3)
        self.assertEqual(calc_sliced_size(9, sliceobj), 3)
        self.assertEqual(calc_sliced_size(7, sliceobj), 2)
        self.assertEqual(calc_sliced_size(8, sliceobj), 2)
        self.assertEqual(calc_sliced_size(6, sliceobj), 1)
        self.assertEqual(calc_sliced_size(4, sliceobj), 0)
        self.assertEqual(calc_sliced_size(3, sliceobj), 0)

    def testSliceSplit(self):
        self.assertEqual(
            slice_split(slice(None), [60, 40]),
            {0: slice(None), 1: slice(None)}
        )
        self.assertEqual(
            slice_split(slice(0, 35), [20, 20, 20, 20, 20]),
            {0: slice(None), 1: slice(0, 15, 1)}
        )
        self.assertEqual(
            slice_split(slice(10, 35), [20, 10, 10, 10, 25, 25]),
            {0: slice(10, 20, 1), 1: slice(None), 2: slice(0, 5, 1)}
        )
        # step testing
        self.assertEqual(
            slice_split(slice(10, 41, 3), [15, 14, 13]),
            {0: slice(10, 15, 3), 1: slice(1, 14, 3), 2: slice(2, 12, 3)}
        )
        self.assertEqual(
            slice_split(slice(0, 100, 40), [20, 20, 20, 20, 20]),
            {0: slice(0, 20, 40), 2: slice(0, 20, 40), 4: slice(0, 20, 40)}
        )
        # single element
        self.assertEqual(
            slice_split(25, [20, 20, 20, 20, 20]),
            {1: 5}
        )
        # negative slicing
        self.assertEqual(
            slice_split(slice(100, 0, -3), [20, 20, 20, 20, 20]),
            {0: slice(-2, -20, -3), 1: slice(-1, -21, -3), 2: slice(-3, -21, -3), 3: slice(-2, -21, -3),
             4: slice(-1, -21, -3)}
        )
        self.assertEqual(
            slice_split(slice(100, 12, -3), [20, 20, 20, 20, 20]),
            {0: slice(-2, -8, -3), 1: slice(-1, -21, -3), 2: slice(-3, -21, -3), 3: slice(-2, -21, -3),
             4: slice(-1, -21, -3)}
        )
        self.assertEqual(
            slice_split(slice(100, -12, -3), [20, 20, 20, 20, 20]),
            {4: slice(-1, -12, -3)}
        )

    def testDecideUnifySplit(self):
        with self.assertRaises(ValueError):
            decide_unify_split((2, 7), (3, 8))
        with self.assertRaises(ValueError):
            decide_unify_split((2, 7), (np.nan, 8))
        self.assertEqual(decide_unify_split((1, 5, 1, 2), (2, 4, 3)), (1, 1, 4, 1, 2))
        self.assertEqual(decide_unify_split((1, 5, 1, 2), (2, 4, 3), (9,)), (1, 1, 4, 1, 2))

    def testUnifyChunks(self):
        t1 = ones((10, 8), chunks=3).tiles()
        t2 = ones((10, 8), chunks=2).tiles()

        new_t1, new_t2 = unify_chunks(t1, t2)
        self.assertEqual(new_t1.nsplits, ((2, 1, 1, 2, 2, 1, 1), (2, 1, 1, 2, 2)))
        self.assertIs(new_t1.inputs[0], t1.data)
        self.assertEqual(new_t2.nsplits, ((2, 1, 1, 2, 2, 1, 1), (2, 1, 1, 2, 2)))
        self.assertIs(new_t2.inputs[0], t2.data)

        t1 = ones((10, 8), chunks=4).tiles()
        t2 = ones((10, 8), chunks=2).tiles()

        new_t1, new_t2 = unify_chunks(t1, t2)
        self.assertEqual(new_t1.nsplits, ((2, 2, 2, 2, 2), (2, 2, 2, 2)))
        self.assertIs(new_t1.inputs[0], t1.data)
        self.assertIs(new_t2, t2)

        t1 = ones((10, 8), chunks=[4, 3]).tiles()
        t2 = ones((10, 8), chunks=2).tiles()

        new_t1, new_t2 = unify_chunks((t1, (0,)), (t2, (0,)))
        self.assertEqual(new_t1.nsplits, ((2, 2, 2, 2, 2), (3, 3, 2)))
        self.assertIs(new_t1.inputs[0], t1.data)
        self.assertIs(new_t2, t2)

        t1 = ones((10, 8), chunks=[4, 2]).tiles()
        t2 = ones((10, 8), chunks=[4, 4]).tiles()

        new_t1, new_t2 = unify_chunks((t1, (1, 0)), (t2, (1, 0)))
        self.assertIs(t1, new_t1)
        self.assertEqual(new_t2.nsplits, ((4, 4, 2), (2, 2, 2, 2)))

        t1 = ones((10, 8), chunks=2).tiles()
        t2 = ones(1, chunks=1).tiles()

        new_t1, new_t2 = unify_chunks((t1, (1, 0)), t2)
        self.assertIs(new_t1, t1)
        self.assertIs(new_t2, t2)

        t1 = ones((10, 8), chunks=2).tiles()
        t2 = ones(8, chunks=3).tiles()

        new_t1, new_t2 = unify_chunks((t1, (1, 0)), t2)
        self.assertEqual(new_t1.nsplits, ((2, 2, 2, 2, 2), (2, 1, 1, 2, 2)))
        self.assertEqual(new_t2.nsplits, ((2, 1, 1, 2, 2),))

    def testSplitIndexIntoChunks(self):
        splits = split_index_into_chunks([10, 20, 30], [5, 31, 21, 18])

        self.assertEqual(len(splits), 3)
        self.assertTrue(np.array_equal(splits[0], np.array([5])))
        self.assertTrue(np.array_equal(splits[1], np.array([11, 8])))
        self.assertTrue(np.array_equal(splits[2], np.array([1])))

        with self.assertRaises(IndexError):
            split_index_into_chunks([10, 20, 30], [100])

        splits = split_index_into_chunks([2, 2, 2, 2, 2, 1], [8, 10, 3, 1, 9, 10])

        self.assertEqual(len(splits), 6)
        self.assertTrue(np.array_equal(splits[0], np.array([1])))
        self.assertTrue(np.array_equal(splits[1], np.array([1])))
        self.assertTrue(np.array_equal(splits[2], np.array([])))
        self.assertTrue(np.array_equal(splits[3], np.array([])))
        self.assertTrue(np.array_equal(splits[4], np.array([0, 1])))
        self.assertTrue(np.array_equal(splits[5], np.array([0, 0])))

    def testDecideChunks(self):
        with option_context() as options:
            options.tensor.chunk_store_limit = 64

            itemsize = 1
            shape = (10, 20, 30)
            nsplit = decide_chunk_sizes(shape, None, itemsize)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))
            self.assertGreaterEqual(options.tensor.chunk_store_limit, itemsize * np.prod([np.max(a) for a in nsplit]))

            itemsize = 2
            shape = (20, 30, 40)
            nsplit = decide_chunk_sizes(shape, {1: 4}, itemsize)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))
            self.assertGreaterEqual(options.tensor.chunk_store_limit, itemsize * np.prod([np.max(a) for a in nsplit]))

            itemsize = 2
            shape = (20, 30, 40)
            nsplit = decide_chunk_sizes(shape, [2, 3], itemsize)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))
            self.assertGreaterEqual(options.tensor.chunk_store_limit, itemsize * np.prod([np.max(a) for a in nsplit]))

            itemsize = 2
            shape = (20, 30, 40)
            nsplit = decide_chunk_sizes(shape, [20, 3], itemsize)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))
            self.assertEqual(120, itemsize * np.prod([np.max(a) for a in nsplit]))  # 20 * 3 * 1 * 2 exceeds limitation

