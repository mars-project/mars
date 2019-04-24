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

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from mars.config import option_context
from mars.dataframe.core import IndexValue
from mars.dataframe.expressions.utils import decide_chunk_sizes, \
    split_monotonic_index_min_max, build_split_idx_to_origin_idx, parse_index


@unittest.skipIf(pd is None, 'pandas not installed')
class Test(unittest.TestCase):
    def testDecideChunks(self):
        with option_context() as options:
            options.tensor.chunk_store_limit = 64

            memory_usage = pd.Series([8, 22.2, 4, 2, 11.2], index=list('abcde'))

            shape = (10, 5)
            nsplit = decide_chunk_sizes(shape, None, memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            nsplit = decide_chunk_sizes(shape, {0: 4}, memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            nsplit = decide_chunk_sizes(shape, (2, 3), memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            nsplit = decide_chunk_sizes(shape, (10, 3), memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            options.tensor.chunk_store_limit = 20

            shape = (10, 5)
            nsplit = decide_chunk_sizes(shape, None, memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            nsplit = decide_chunk_sizes(shape, {1: 3}, memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            nsplit = decide_chunk_sizes(shape, (2, 3), memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

            nsplit = decide_chunk_sizes(shape, (10, 3), memory_usage)
            [self.assertTrue(all(isinstance(i, Integral) for i in ns)) for ns in nsplit]
            self.assertEqual(shape, tuple(sum(ns) for ns in nsplit))

    def testParseIndex(self):
        index = pd.Int64Index([])
        parsed_index = parse_index(index)
        self.assertIsInstance(parsed_index.value, IndexValue.Int64Index)
        pd.testing.assert_index_equal(index, parsed_index.to_pandas())

        index = pd.Int64Index([1, 2])
        parsed_index = parse_index(index)  # not parse data
        self.assertIsInstance(parsed_index.value, IndexValue.Int64Index)
        with self.assertRaises(AssertionError):
            pd.testing.assert_index_equal(index, parsed_index.to_pandas())

        parsed_index = parse_index(index, store_data=True)  # parse data
        self.assertIsInstance(parsed_index.value, IndexValue.Int64Index)
        pd.testing.assert_index_equal(index, parsed_index.to_pandas())

        index = pd.RangeIndex(0, 10, 3)
        parsed_index = parse_index(index)
        self.assertIsInstance(parsed_index.value, IndexValue.RangeIndex)
        pd.testing.assert_index_equal(index, parsed_index.to_pandas())

        index = pd.MultiIndex.from_arrays([[0, 1], ['a', 'b']])
        parsed_index = parse_index(index)  # not parse data
        self.assertIsInstance(parsed_index.value, IndexValue.MultiIndex)
        with self.assertRaises(AssertionError):
            pd.testing.assert_index_equal(index, parsed_index.to_pandas())

        parsed_index = parse_index(index, store_data=True)  # parse data
        self.assertIsInstance(parsed_index.value, IndexValue.MultiIndex)
        pd.testing.assert_index_equal(index, parsed_index.to_pandas())

    def testSplitMonotonicIndexMinMax(self):
        left_min_max = [[0, True, 3, True], [3, False, 5, False]]
        right_min_max = [[1, False, 3, True], [4, False, 6, True]]
        left_splits, right_splits = \
            split_monotonic_index_min_max(left_min_max, True, right_min_max, True)
        self.assertEqual(left_splits,
                         [[(0, True, 1, True), (1, False, 3, True)],
                          [(3, False, 4, True), (4, False, 5, False), (5, True, 6, True)]])
        self.assertEqual(right_splits,
                         [[(0, True, 1, True), (1, False, 3, True)],
                          [(3, False, 4, True), (4, False, 5, False), (5, True, 6, True)]])
        left_splits, right_splits = split_monotonic_index_min_max(right_min_max, False, left_min_max, False)
        self.assertEqual(list(reversed(left_splits)),
                         [[(0, True, 1, True), (1, False, 3, True)],
                          [(3, False, 4, True), (4, False, 5, False), (5, True, 6, True)]])
        self.assertEqual(list(reversed(right_splits)),
                         [[(0, True, 1, True), (1, False, 3, True)],
                          [(3, False, 4, True), (4, False, 5, False), (5, True, 6, True)]])

        left_min_max = [[2, True, 4, True], [8, True, 9, False]]
        right_min_max = [[1, False, 3, True], [4, False, 6, True]]
        left_splits, right_splits = \
            split_monotonic_index_min_max(left_min_max, True, right_min_max, True)
        self.assertEqual(left_splits,
                         [[(1, False, 2, False), (2, True, 3, True), (3, False, 4, True)],
                          [(4, False, 6, True), (8, True, 9, False)]])
        self.assertEqual(right_splits,
                         [[(1, False, 2, False), (2, True, 3, True)],
                          [(3, False, 4, True), (4, False, 6, True), (8, True, 9, False)]])

        left_min_max = [[1, False, 3, True], [4, False, 6, True], [10, True, 12, False], [13, True, 14, False]]
        right_min_max = [[2, True, 4, True], [5, True, 7, False]]
        left_splits, right_splits = \
            split_monotonic_index_min_max(left_min_max, True, right_min_max, True)
        self.assertEqual(left_splits,
                         [[(1, False, 2, False), (2, True, 3, True)],
                          [(3, False, 4, True), (4, False, 5, False), (5, True, 6, True)],
                          [(6, False, 7, False), (10, True, 12, False)],
                          [(13, True, 14, False)]])
        self.assertEqual(right_splits,
                         [[(1, False, 2, False), (2, True, 3, True), (3, False, 4, True)],
                          [(4, False, 5, False), (5, True, 6, True), (6, False, 7, False),
                           (10, True, 12, False), (13, True, 14, False)]])
        left_splits, right_splits = \
            split_monotonic_index_min_max(right_min_max, True, left_min_max, True)
        self.assertEqual(left_splits,
                         [[(1, False, 2, False), (2, True, 3, True), (3, False, 4, True)],
                          [(4, False, 5, False), (5, True, 6, True), (6, False, 7, False),
                           (10, True, 12, False), (13, True, 14, False)]])
        self.assertEqual(right_splits,
                         [[(1, False, 2, False), (2, True, 3, True)],
                          [(3, False, 4, True), (4, False, 5, False), (5, True, 6, True)],
                          [(6, False, 7, False), (10, True, 12, False)],
                          [(13, True, 14, False)]])

        # left min_max like ([.., .., 4 True], [4, False, ..., ...]
        # right min_max like ([..., ..., 4 False], [4, True, ..., ...]
        left_min_max = [[1, False, 4, True], [4, False, 6, True]]
        right_min_max = [[1, False, 4, False], [4, True, 6, True]]
        left_splits, right_splits = split_monotonic_index_min_max(
            left_min_max, True, right_min_max, True)
        self.assertEqual(left_splits,
                         [[(1, False, 4, False), (4, True, 4, True)], [(4, False, 6, True)]])
        self.assertEqual(right_splits,
                         [[(1, False, 4, False)], [(4, True, 4, True), (4, False, 6, True)]])

        # identical index
        left_min_max = [[1, False, 3, True], [4, False, 6, True]]
        right_min_max = [[1, False, 3, True], [4, False, 6, True]]
        left_splits, right_splits = \
            split_monotonic_index_min_max(left_min_max, True, right_min_max, True)
        self.assertEqual(left_splits, [[tuple(it)] for it in left_min_max])
        self.assertEqual(right_splits, [[tuple(it)] for it in left_min_max])

    def testBuildSplitIdxToOriginIdx(self):
        splits = [[(1, False, 2, False), (2, True, 3, True)], [(5, False, 6, True)]]
        res = build_split_idx_to_origin_idx(splits)

        self.assertEqual(res, {0: (0, 0), 1: (0, 1), 2: (1, 0)})

        splits = [[(5, False, 6, True)], [(1, False, 2, False), (2, True, 3, True)]]
        res = build_split_idx_to_origin_idx(splits, increase=False)

        self.assertEqual(res, {0: (1, 0), 1: (1, 1), 2: (0, 0)})
