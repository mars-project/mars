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

import itertools
import unittest

import numpy as np
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from mars.dataframe.core import IndexValue
from mars.dataframe.expressions.utils import split_monotonic_index_min_max, \
    build_split_idx_to_origin_idx
from mars.dataframe.expressions.datasource.dataframe import from_pandas
from mars.dataframe.expressions.arithmetic import add, DataFrameAdd
from mars.dataframe.expressions.arithmetic.core import DataFrameIndexAlignMap, \
    DataFrameIndexAlignReduce, DataFrameShuffleProxy
from mars.tests.core import TestBase


@unittest.skipIf(pd is None, 'pandas not installed')
class Test(TestBase):
    def testAddWithoutShuffle(self):
        # all the axes are monotonic
        # data1 with index split into [0...4], [5...9],
        # columns [3...7], [8...12]
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=np.arange(3, 13))
        df1 = from_pandas(data1, chunk_size=5)
        # data2 with index split into [6...11], [2, 5],
        # columns [4...9], [10, 13]
        data2 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(11, 1, -1),
                             columns=np.arange(4, 14))
        df2 = from_pandas(data2, chunk_size=6)

        df3 = add(df1, df2)

        # test df3's index and columns
        pd.testing.assert_index_equal(df3.columns.to_pandas(), (data1 + data2).columns)
        self.assertIsInstance(df3.index_value.value, IndexValue.Int64Index)
        pd.testing.assert_index_equal(df3.index_value.to_pandas(), pd.Int64Index([]))
        self.assertNotEqual(df3.index_value.key, df1.index_value.key)
        self.assertNotEqual(df3.index_value.key, df2.index_value.key)

        df3.tiles()

        data1_index_min_max = [(0, True, 4, True), (5, True, 9, True)]
        data1_columns_min_max = [[3, True, 7, True], [8, True, 12, True]]
        data2_index_min_max = [(2, True, 5, True), (6, True, 11, True)]
        data2_columns_min_max = [(4, True, 9, True), (10, True, 13, True)]

        left_index_splits, right_index_splits = split_monotonic_index_min_max(
            data1_index_min_max, True, data2_index_min_max, False)
        left_columns_splits, right_columns_splits = split_monotonic_index_min_max(
            data1_columns_min_max, True, data2_columns_min_max, True)

        left_index_idx_to_original_idx = build_split_idx_to_origin_idx(left_index_splits)
        right_index_idx_to_original_idx = build_split_idx_to_origin_idx(right_index_splits, False)
        left_columns_idx_to_original_idx = build_split_idx_to_origin_idx(left_columns_splits)
        right_columns_idx_to_original_idx = build_split_idx_to_origin_idx(right_columns_splits)

        self.assertEqual(df3.chunk_shape, (7, 7))
        for c in df3.chunks:
            self.assertIsInstance(c.op, DataFrameAdd)
            self.assertEqual(len(c.inputs), 2)
            idx = c.index
            # test the left side
            self.assertIsInstance(c.inputs[0].op, DataFrameIndexAlignMap)
            left_row_idx, left_row_inner_idx = left_index_idx_to_original_idx[idx[0]]
            left_col_idx, left_col_inner_idx = left_columns_idx_to_original_idx[idx[1]]
            self.assertIs(c.inputs[0].inputs[0], df1.cix[left_row_idx, left_col_idx].data)
            left_index_min_max = left_index_splits[left_row_idx][left_row_inner_idx]
            self.assertEqual(c.inputs[0].op.index_min, left_index_min_max[0])
            self.assertEqual(c.inputs[0].op.index_min_close, left_index_min_max[1])
            self.assertEqual(c.inputs[0].op.index_max, left_index_min_max[2])
            self.assertEqual(c.inputs[0].op.index_max_close, left_index_min_max[3])
            left_column_min_max = left_columns_splits[left_col_idx][left_col_inner_idx]
            self.assertEqual(c.inputs[0].op.column_min, left_column_min_max[0])
            self.assertEqual(c.inputs[0].op.column_min_close, left_column_min_max[1])
            self.assertEqual(c.inputs[0].op.column_max, left_column_min_max[2])
            self.assertEqual(c.inputs[0].op.column_max_close, left_column_min_max[3])
            # test the right side
            self.assertIsInstance(c.inputs[1].op, DataFrameIndexAlignMap)
            right_row_idx, right_row_inner_idx = right_index_idx_to_original_idx[idx[0]]
            right_col_idx, right_col_inner_idx = right_columns_idx_to_original_idx[idx[1]]
            self.assertIs(c.inputs[1].inputs[0], df2.cix[right_row_idx, right_col_idx].data)
            right_index_min_max = right_index_splits[right_row_idx][right_row_inner_idx]
            self.assertEqual(c.inputs[1].op.index_min, right_index_min_max[0])
            self.assertEqual(c.inputs[1].op.index_min_close, right_index_min_max[1])
            self.assertEqual(c.inputs[1].op.index_max, right_index_min_max[2])
            self.assertEqual(c.inputs[1].op.index_max_close, right_index_min_max[3])
            right_column_min_max = right_columns_splits[right_col_idx][right_col_inner_idx]
            self.assertEqual(c.inputs[1].op.column_min, right_column_min_max[0])
            self.assertEqual(c.inputs[1].op.column_min_close, right_column_min_max[1])
            self.assertEqual(c.inputs[1].op.column_max, right_column_min_max[2])
            self.assertEqual(c.inputs[1].op.column_max_close, right_column_min_max[3])

    def testAddWithOneShuffle(self):
        # only 1 axis is monotonic
        # data1 with index split into [0...4], [5...9],
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=5)
        # data2 with index split into [6...11], [2, 5],
        data2 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(11, 1, -1),
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = from_pandas(data2, chunk_size=6)

        df3 = add(df1, df2)

        # test df3's index and columns
        pd.testing.assert_index_equal(df3.columns.to_pandas(), (data1 + data2).columns)
        self.assertIsInstance(df3.index_value.value, IndexValue.Int64Index)
        pd.testing.assert_index_equal(df3.index_value.to_pandas(), pd.Int64Index([]))
        self.assertNotEqual(df3.index_value.key, df1.index_value.key)
        self.assertNotEqual(df3.index_value.key, df2.index_value.key)

        df3.tiles()

        data1_index_min_max = [(0, True, 4, True), (5, True, 9, True)]
        data2_index_min_max = [(2, True, 5, True), (6, True, 11, True)]

        left_index_splits, right_index_splits = split_monotonic_index_min_max(
            data1_index_min_max, True, data2_index_min_max, False)

        left_index_idx_to_original_idx = build_split_idx_to_origin_idx(left_index_splits)
        right_index_idx_to_original_idx = build_split_idx_to_origin_idx(right_index_splits, False)

        self.assertEqual(df3.chunk_shape, (7, 2))
        for c in df3.chunks:
            self.assertIsInstance(c.op, DataFrameAdd)
            self.assertEqual(len(c.inputs), 2)
            idx = c.index
            # test the left side
            self.assertIsInstance(c.inputs[0].op, DataFrameIndexAlignReduce)
            self.assertIsInstance(c.inputs[0].inputs[0].op, DataFrameShuffleProxy)
            left_row_idx, left_row_inner_idx = left_index_idx_to_original_idx[idx[0]]
            left_index_min_max = left_index_splits[left_row_idx][left_row_inner_idx]
            ics = [ic for ic in df1.chunks if ic.index[0] == left_row_idx]
            for j, ci, ic in zip(itertools.count(0), c.inputs[0].inputs[0].inputs, ics):
                self.assertIsInstance(ci.op, DataFrameIndexAlignMap)
                self.assertEqual(ci.index, (idx[0], j))
                self.assertEqual(ci.op.index_min, left_index_min_max[0])
                self.assertEqual(ci.op.index_min_close, left_index_min_max[1])
                self.assertEqual(ci.op.index_max, left_index_min_max[2])
                self.assertEqual(ci.op.index_max_close, left_index_min_max[3])
                self.assertTrue(ci.op.column_shuffle_size, 2)
                self.assertIs(ci.inputs[0], ic.data)
            # test the right side
            self.assertIsInstance(c.inputs[1].op, DataFrameIndexAlignReduce)
            self.assertIsInstance(c.inputs[1].inputs[0].op, DataFrameShuffleProxy)
            right_row_idx, right_row_inner_idx = right_index_idx_to_original_idx[idx[0]]
            right_index_min_max = right_index_splits[right_row_idx][right_row_inner_idx]
            ics = [ic for ic in df2.chunks if ic.index[0] == right_row_idx]
            for j, ci, ic in zip(itertools.count(0), c.inputs[1].inputs[0].inputs, ics):
                self.assertIsInstance(ci.op, DataFrameIndexAlignMap)
                self.assertEqual(ci.index, (idx[0], j))
                self.assertEqual(ci.op.index_min, right_index_min_max[0])
                self.assertEqual(ci.op.index_min_close, right_index_min_max[1])
                self.assertEqual(ci.op.index_max, right_index_min_max[2])
                self.assertEqual(ci.op.index_max_close, right_index_min_max[3])
                self.assertTrue(ci.op.column_shuffle_size, 2)
                self.assertIs(ci.inputs[0], ic.data)

        # make sure shuffle proxies' key are different
        proxy_keys = set()
        for i in range(df3.chunk_shape[0]):
            cs = [c for c in df3.chunks if c.index[0] == i]
            lps = {c.inputs[0].inputs[0].op.key for c in cs}
            self.assertEqual(len(lps), 1)
            proxy_keys.add(lps.pop())
            rps = {c.inputs[1].inputs[0].op.key for c in cs}
            self.assertEqual(len(rps), 1)
            proxy_keys.add(rps.pop())
        self.assertEqual(len(proxy_keys), 2 * df3.chunk_shape[0])

    def testAddWithAllShuffle(self):
        # no axis is monotonic
        data1 = pd.DataFrame(np.random.rand(10, 10), index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=5)
        data2 = pd.DataFrame(np.random.rand(10, 10), index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3],
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = from_pandas(data2, chunk_size=6)

        df3 = add(df1, df2)

        # test df3's index and columns
        pd.testing.assert_index_equal(df3.columns.to_pandas(), (data1 + data2).columns)
        self.assertIsInstance(df3.index_value.value, IndexValue.Int64Index)
        pd.testing.assert_index_equal(df3.index_value.to_pandas(), pd.Int64Index([]))
        self.assertNotEqual(df3.index_value.key, df1.index_value.key)
        self.assertNotEqual(df3.index_value.key, df2.index_value.key)

        df3.tiles()

        self.assertEqual(df3.chunk_shape, (2, 2))
        proxy_keys = set()
        for c in df3.chunks:
            self.assertIsInstance(c.op, DataFrameAdd)
            self.assertEqual(len(c.inputs), 2)
            # test left side
            self.assertIsInstance(c.inputs[0].op, DataFrameIndexAlignReduce)
            self.assertIsInstance(c.inputs[0].inputs[0].op, DataFrameShuffleProxy)
            proxy_keys.add(c.inputs[0].inputs[0].op.key)
            for ic, ci in zip(c.inputs[0].inputs[0].inputs, df1.chunks):
                self.assertIsInstance(ic.op, DataFrameIndexAlignMap)
                self.assertEqual(ic.op.index_shuffle_size, 2)
                self.assertEqual(ic.op.column_shuffle_size, 2)
                self.assertIs(ic.inputs[0], ci.data)
            # test right side
            self.assertIsInstance(c.inputs[1].op, DataFrameIndexAlignReduce)
            self.assertIsInstance(c.inputs[1].inputs[0].op, DataFrameShuffleProxy)
            proxy_keys.add(c.inputs[1].inputs[0].op.key)
            for ic, ci in zip(c.inputs[1].inputs[0].inputs, df2.chunks):
                self.assertIsInstance(ic.op, DataFrameIndexAlignMap)
                self.assertEqual(ic.op.index_shuffle_size, 2)
                self.assertEqual(ic.op.column_shuffle_size, 2)
                self.assertIs(ic.inputs[0], ci.data)

        self.assertEqual(len(proxy_keys), 2)
