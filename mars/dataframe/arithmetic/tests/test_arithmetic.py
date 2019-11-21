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

import itertools
import operator

import numpy as np
import pandas as pd

from mars.dataframe.core import IndexValue
from mars.dataframe.operands import ObjectType
from mars.dataframe.utils import hash_dtypes
from mars.dataframe.utils import split_monotonic_index_min_max, \
    build_split_idx_to_origin_idx, filter_index_value
from mars.dataframe.datasource.dataframe import from_pandas, DataFrameDataSource
from mars.dataframe.datasource.series import from_pandas as from_pandas_series, SeriesDataSource
from mars.dataframe.arithmetic import abs, DataFrameAbs, DataFrameAdd, DataFrameSubtract, \
    DataFrameFloorDiv, DataFrameTrueDiv
from mars.dataframe.align import DataFrameIndexAlignMap, \
    DataFrameIndexAlignReduce, DataFrameShuffleProxy
from mars.tests.core import TestBase, parameterized


binary_functions = dict(
    add=dict(func=operator.add, op=DataFrameAdd, func_name='add'),
    subtract=dict(func=operator.sub, op=DataFrameSubtract, func_name='sub'),
    floordiv=dict(func=operator.floordiv, op=DataFrameFloorDiv, func_name='floordiv'),
    truediv=dict(func=operator.truediv, op=DataFrameTrueDiv, func_name='truediv')
)


@parameterized(**binary_functions)
class TestBinary(TestBase):
    @property
    def rfunc_name(self):
        return 'r' + self.func_name

    def testWithoutShuffle(self):
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

        df3 = self.func(df1, df2)

        # test df3's index and columns
        pd.testing.assert_index_equal(df3.columns_value.to_pandas(), self.func(data1, data2).columns)
        self.assertTrue(df3.columns_value.should_be_monotonic)
        self.assertIsInstance(df3.index_value.value, IndexValue.Int64Index)
        self.assertTrue(df3.index_value.should_be_monotonic)
        pd.testing.assert_index_equal(df3.index_value.to_pandas(), pd.Int64Index([]))
        self.assertNotEqual(df3.index_value.key, df1.index_value.key)
        self.assertNotEqual(df3.index_value.key, df2.index_value.key)
        self.assertEqual(df3.shape[1], 11)  # columns is recorded, so we can get it

        df3.tiles()

        # test df3's index and columns after tiling
        pd.testing.assert_index_equal(df3.columns_value.to_pandas(), self.func(data1, data2).columns)
        self.assertTrue(df3.columns_value.should_be_monotonic)
        self.assertIsInstance(df3.index_value.value, IndexValue.Int64Index)
        self.assertTrue(df3.index_value.should_be_monotonic)
        pd.testing.assert_index_equal(df3.index_value.to_pandas(), pd.Int64Index([]))
        self.assertNotEqual(df3.index_value.key, df1.index_value.key)
        self.assertNotEqual(df3.index_value.key, df2.index_value.key)
        self.assertEqual(df3.shape[1], 11)  # columns is recorded, so we can get it

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
            self.assertIsInstance(c.op, self.op)
            self.assertEqual(len(c.inputs), 2)
            # test shape
            idx = c.index
            # test the left side
            self.assertIsInstance(c.inputs[0].op, DataFrameIndexAlignMap)
            left_row_idx, left_row_inner_idx = left_index_idx_to_original_idx[idx[0]]
            left_col_idx, left_col_inner_idx = left_columns_idx_to_original_idx[idx[1]]
            expect_df1_input = df1.cix[left_row_idx, left_col_idx].data
            self.assertIs(c.inputs[0].inputs[0], expect_df1_input)
            left_index_min_max = left_index_splits[left_row_idx][left_row_inner_idx]
            self.assertEqual(c.inputs[0].op.index_min, left_index_min_max[0])
            self.assertEqual(c.inputs[0].op.index_min_close, left_index_min_max[1])
            self.assertEqual(c.inputs[0].op.index_max, left_index_min_max[2])
            self.assertEqual(c.inputs[0].op.index_max_close, left_index_min_max[3])
            self.assertIsInstance(c.inputs[0].index_value.to_pandas(), type(data1.index))
            left_column_min_max = left_columns_splits[left_col_idx][left_col_inner_idx]
            self.assertEqual(c.inputs[0].op.column_min, left_column_min_max[0])
            self.assertEqual(c.inputs[0].op.column_min_close, left_column_min_max[1])
            self.assertEqual(c.inputs[0].op.column_max, left_column_min_max[2])
            self.assertEqual(c.inputs[0].op.column_max_close, left_column_min_max[3])
            expect_left_columns = filter_index_value(expect_df1_input.columns_value, left_column_min_max,
                                                     store_data=True)
            pd.testing.assert_index_equal(c.inputs[0].columns_value.to_pandas(), expect_left_columns.to_pandas())
            pd.testing.assert_index_equal(c.inputs[0].dtypes.index, expect_left_columns.to_pandas())
            # test the right side
            self.assertIsInstance(c.inputs[1].op, DataFrameIndexAlignMap)
            right_row_idx, right_row_inner_idx = right_index_idx_to_original_idx[idx[0]]
            right_col_idx, right_col_inner_idx = right_columns_idx_to_original_idx[idx[1]]
            expect_df2_input = df2.cix[right_row_idx, right_col_idx].data
            self.assertIs(c.inputs[1].inputs[0], expect_df2_input)
            right_index_min_max = right_index_splits[right_row_idx][right_row_inner_idx]
            self.assertEqual(c.inputs[1].op.index_min, right_index_min_max[0])
            self.assertEqual(c.inputs[1].op.index_min_close, right_index_min_max[1])
            self.assertEqual(c.inputs[1].op.index_max, right_index_min_max[2])
            self.assertEqual(c.inputs[1].op.index_max_close, right_index_min_max[3])
            self.assertIsInstance(c.inputs[1].index_value.to_pandas(), type(data2.index))
            right_column_min_max = right_columns_splits[right_col_idx][right_col_inner_idx]
            self.assertEqual(c.inputs[1].op.column_min, right_column_min_max[0])
            self.assertEqual(c.inputs[1].op.column_min_close, right_column_min_max[1])
            self.assertEqual(c.inputs[1].op.column_max, right_column_min_max[2])
            self.assertEqual(c.inputs[1].op.column_max_close, right_column_min_max[3])
            expect_right_columns = filter_index_value(expect_df2_input.columns_value, left_column_min_max,
                                                      store_data=True)
            pd.testing.assert_index_equal(c.inputs[1].columns_value.to_pandas(), expect_right_columns.to_pandas())
            pd.testing.assert_index_equal(c.inputs[1].dtypes.index, expect_right_columns.to_pandas())

    def testDataFrameAndSeriesWithAlignMap(self):
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=np.arange(3, 13))
        df1 = from_pandas(data1, chunk_size=5)
        s1 = df1[3]

        df2 = self.func(df1, s1)
        df2.tiles()

        self.assertEqual(df2.shape, (df1.shape[0], np.nan))
        self.assertEqual(df2.index_value.key, df1.index_value.key)

        data1_columns_min_max = [[3, True, 7, True], [8, True, 12, True]]
        data2_index_min_max = [(0, True, 4, True), (5, True, 9, True)]

        left_columns_splits, right_index_splits = split_monotonic_index_min_max(
            data1_columns_min_max, True, data2_index_min_max, True)

        left_columns_idx_to_original_idx = build_split_idx_to_origin_idx(left_columns_splits)
        right_index_idx_to_original_idx = build_split_idx_to_origin_idx(right_index_splits)

        self.assertEqual(df2.chunk_shape, (2, 7))
        for c in df2.chunks:
            self.assertIsInstance(c.op, self.op)
            self.assertEqual(len(c.inputs), 2)
            # test shape
            idx = c.index
            # test the left side (dataframe)
            self.assertIsInstance(c.inputs[0].op, DataFrameIndexAlignMap)
            left_col_idx, left_col_inner_idx = left_columns_idx_to_original_idx[idx[1]]
            expect_df1_input = df1.cix[idx[0], left_col_idx].data
            self.assertIs(c.inputs[0].inputs[0], expect_df1_input)
            left_column_min_max = left_columns_splits[left_col_idx][left_col_inner_idx]
            self.assertEqual(c.inputs[0].op.column_min, left_column_min_max[0])
            self.assertEqual(c.inputs[0].op.column_min_close, left_column_min_max[1])
            self.assertEqual(c.inputs[0].op.column_max, left_column_min_max[2])
            self.assertEqual(c.inputs[0].op.column_max_close, left_column_min_max[3])
            expect_left_columns = filter_index_value(expect_df1_input.columns_value, left_column_min_max,
                                                     store_data=True)
            pd.testing.assert_index_equal(c.inputs[0].columns_value.to_pandas(), expect_left_columns.to_pandas())
            pd.testing.assert_index_equal(c.inputs[0].dtypes.index, expect_left_columns.to_pandas())

            # test the right side (series)
            self.assertIsInstance(c.inputs[1].op, DataFrameIndexAlignMap)
            right_row_idx, right_row_inner_idx = right_index_idx_to_original_idx[idx[1]]
            expect_s1_input = s1.cix[(right_row_idx,)].data
            self.assertIs(c.inputs[1].inputs[0], expect_s1_input)
            right_index_min_max = right_index_splits[right_row_idx][right_row_inner_idx]
            self.assertEqual(c.inputs[1].op.index_min, right_index_min_max[0])
            self.assertEqual(c.inputs[1].op.index_min_close, right_index_min_max[1])
            self.assertEqual(c.inputs[1].op.index_max, right_index_min_max[2])
            self.assertEqual(c.inputs[1].op.index_max_close, right_index_min_max[3])
            self.assertIsInstance(c.inputs[1].index_value.to_pandas(), type(data1[3].index))

    def testDataFrameAndSeriesIdentical(self):
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=np.arange(10))
        df1 = from_pandas(data1, chunk_size=5)
        s1 = from_pandas_series(data1[3], chunk_size=5)

        df2 = self.func(df1, s1)
        df2.tiles()

        self.assertEqual(df2.shape, (10, 10))
        self.assertEqual(df2.index_value.key, df1.index_value.key)
        self.assertEqual(df2.columns_value.key, df1.columns_value.key)
        self.assertEqual(df2.columns_value.key, s1.index_value.key)

        self.assertEqual(df2.chunk_shape, (2, 2))
        for c in df2.chunks:
            self.assertIsInstance(c.op, self.op)
            self.assertEqual(len(c.inputs), 2)
            self.assertEqual(c.shape, (5, 5))
            self.assertEqual(c.index_value.key, df1.cix[c.index].index_value.key)
            self.assertEqual(c.index_value.key, df2.cix[c.index].index_value.key)
            self.assertEqual(c.columns_value.key, df1.cix[c.index].columns_value.key)
            self.assertEqual(c.columns_value.key, df2.cix[c.index].columns_value.key)
            pd.testing.assert_index_equal(c.columns_value.to_pandas(), df1.cix[c.index].columns_value.to_pandas())
            pd.testing.assert_index_equal(c.columns_value.to_pandas(), df2.cix[c.index].columns_value.to_pandas())
            pd.testing.assert_index_equal(c.dtypes.index, df1.cix[c.index].columns_value.to_pandas())

            # test the left side
            self.assertIsInstance(c.inputs[0].op, DataFrameDataSource)
            self.assertIs(c.inputs[0], df1.cix[c.index].data)
            # test the right side
            self.assertIsInstance(c.inputs[1].op, SeriesDataSource)
            self.assertIs(c.inputs[1], s1.cix[(c.index[1],)].data)

    def testDataFrameAndSeriesWithShuffle(self):
        data1 = pd.DataFrame(np.random.rand(10, 10),
                             index=[4, 9, 3, 2, 1, 5, 8, 6, 7, 10],
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=5)
        s1 = from_pandas_series(data1[10], chunk_size=6)

        df2 = self.func(df1, s1)

        # test df2's index and columns
        self.assertEqual(df2.shape, (df1.shape[0], np.nan))
        self.assertEqual(df2.index_value.key, df1.index_value.key)
        pd.testing.assert_index_equal(df2.columns_value.to_pandas(), pd.Int64Index([]))
        self.assertNotEqual(df2.columns_value.key, df1.columns_value.key)
        self.assertTrue(df2.columns_value.should_be_monotonic)

        df2.tiles()

        self.assertEqual(df2.chunk_shape, (2, 2))
        for c in df2.chunks:
            self.assertIsInstance(c.op, self.op)
            self.assertEqual(len(c.inputs), 2)
            idx = c.index
            # test the left side
            self.assertIsInstance(c.inputs[0].op, DataFrameIndexAlignReduce)
            expect_dtypes = pd.concat([hash_dtypes(ic.inputs[0].op.data.dtypes, 2)[c.index[1]]
                                       for ic in c.inputs[0].inputs[0].inputs])
            pd.testing.assert_series_equal(c.inputs[0].dtypes, expect_dtypes)
            pd.testing.assert_index_equal(c.inputs[0].columns_value.to_pandas(), c.inputs[0].dtypes.index)
            pd.testing.assert_index_equal(c.inputs[0].index_value.to_pandas(), c.index_value.to_pandas())
            self.assertIsInstance(c.inputs[0].inputs[0].op, DataFrameShuffleProxy)
            for j, ci, ic in zip(itertools.count(0), c.inputs[0].inputs[0].inputs, df1.cix[idx[0], :]):
                self.assertIsInstance(ci.op, DataFrameIndexAlignMap)
                self.assertEqual(ci.index, (idx[0], j))
                self.assertTrue(ci.op.column_shuffle_size, 2)
                shuffle_segments = ci.op.column_shuffle_segments
                expected_shuffle_segments = hash_dtypes(ic.data.dtypes, 2)
                self.assertEqual(len(shuffle_segments), len(expected_shuffle_segments))
                for ss, ess in zip(shuffle_segments, expected_shuffle_segments):
                    pd.testing.assert_series_equal(ss, ess)
                self.assertIs(ci.inputs[0], ic.data)

            # test the right side
            self.assertIsInstance(c.inputs[1].op, DataFrameIndexAlignReduce)
            self.assertEqual(c.inputs[1].op.object_type, ObjectType.series)
            self.assertIsInstance(c.inputs[1].inputs[0].op, DataFrameShuffleProxy)
            for j, ci, ic in zip(itertools.count(0), c.inputs[1].inputs[0].inputs, s1.chunks):
                self.assertIsInstance(ci.op, DataFrameIndexAlignMap)
                self.assertEqual(ci.index, (j,))
                self.assertTrue(ci.op.index_shuffle_size, 2)
                self.assertIs(ci.inputs[0], ic.data)

        # make sure shuffle proxies' key are different
        proxy_keys = set()
        for i in range(df2.chunk_shape[0]):
            cs = [c for c in df2.chunks if c.index[0] == i]
            lps = {c.inputs[0].inputs[0].op.key for c in cs}
            self.assertEqual(len(lps), 1)
            proxy_keys.add(lps.pop())
            rps = {c.inputs[1].inputs[0].op.key for c in cs}
            self.assertEqual(len(rps), 1)
            proxy_keys.add(rps.pop())
        self.assertEqual(len(proxy_keys), df2.chunk_shape[0] + 1)

    def testSeriesAndSeriesWithAlignMap(self):
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=np.arange(3, 13))
        df1 = from_pandas(data1, chunk_size=5)

        s1 = df1.iloc[4]
        s2 = df1[3]

        s3 = self.func(s1, s2)
        s3.tiles()

        self.assertEqual(s3.shape, (np.nan,))

        s1_index_min_max = [[3, True, 7, True], [8, True, 12, True]]
        s2_index_min_max = [(0, True, 4, True), (5, True, 9, True)]

        left_index_splits, right_index_splits = split_monotonic_index_min_max(
            s1_index_min_max, True, s2_index_min_max, True)

        left_index_idx_to_original_idx = build_split_idx_to_origin_idx(left_index_splits)
        right_index_idx_to_original_idx = build_split_idx_to_origin_idx(right_index_splits)

        self.assertEqual(s3.chunk_shape, (7,))
        for c in s3.chunks:
            self.assertIsInstance(c.op, self.op)
            self.assertEqual(len(c.inputs), 2)
            # test shape
            idx = c.index
            # test the left side (series)
            self.assertIsInstance(c.inputs[0].op, DataFrameIndexAlignMap)
            left_col_idx, left_col_inner_idx = left_index_idx_to_original_idx[idx[0]]
            expect_s1_input = s1.cix[(left_col_idx,)].data
            self.assertIs(c.inputs[0].inputs[0], expect_s1_input)
            left_index_min_max = left_index_splits[left_col_idx][left_col_inner_idx]
            self.assertEqual(c.inputs[0].op.index_min, left_index_min_max[0])
            self.assertEqual(c.inputs[0].op.index_min_close, left_index_min_max[1])
            self.assertEqual(c.inputs[0].op.index_max, left_index_min_max[2])
            self.assertEqual(c.inputs[0].op.index_max_close, left_index_min_max[3])
            self.assertIsInstance(c.inputs[0].index_value.to_pandas(), type(data1.iloc[4].index))
            expect_left_index = filter_index_value(expect_s1_input.index_value, left_index_min_max,
                                                   store_data=True)
            pd.testing.assert_index_equal(c.inputs[0].index_value.to_pandas(), expect_left_index.to_pandas())

            # test the right side (series)
            self.assertIsInstance(c.inputs[1].op, DataFrameIndexAlignMap)
            right_row_idx, right_row_inner_idx = right_index_idx_to_original_idx[idx[0]]
            expect_s2_input = s2.cix[(right_row_idx,)].data
            self.assertIs(c.inputs[1].inputs[0], expect_s2_input)
            right_index_min_max = right_index_splits[right_row_idx][right_row_inner_idx]
            self.assertEqual(c.inputs[1].op.index_min, right_index_min_max[0])
            self.assertEqual(c.inputs[1].op.index_min_close, right_index_min_max[1])
            self.assertEqual(c.inputs[1].op.index_max, right_index_min_max[2])
            self.assertEqual(c.inputs[1].op.index_max_close, right_index_min_max[3])
            self.assertIsInstance(c.inputs[1].index_value.to_pandas(), type(data1[3].index))
            expect_right_index = filter_index_value(expect_s2_input.index_value, right_index_min_max,
                                                    store_data=True)
            pd.testing.assert_index_equal(c.inputs[1].index_value.to_pandas(), expect_right_index.to_pandas())

    def testSeriesAndSeriesIdentical(self):
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=np.arange(10))
        s1 = from_pandas_series(data1[1], chunk_size=5)
        s2 = from_pandas_series(data1[3], chunk_size=5)

        s3 = self.func(s1, s2)
        s3.tiles()

        self.assertEqual(s3.shape, (10,))
        self.assertEqual(s3.index_value.key, s1.index_value.key)
        self.assertEqual(s3.index_value.key, s2.index_value.key)

        self.assertEqual(s3.chunk_shape, (2,))
        for c in s3.chunks:
            self.assertIsInstance(c.op, self.op)
            self.assertEqual(c.op.object_type, ObjectType.series)
            self.assertEqual(len(c.inputs), 2)
            self.assertEqual(c.shape, (5,))
            self.assertEqual(c.index_value.key, s1.cix[c.index].index_value.key)
            self.assertEqual(c.index_value.key, s2.cix[c.index].index_value.key)

            # test the left side
            self.assertIsInstance(c.inputs[0].op, SeriesDataSource)
            self.assertIs(c.inputs[0], s1.cix[c.index].data)
            # test the right side
            self.assertIsInstance(c.inputs[1].op, SeriesDataSource)
            self.assertIs(c.inputs[1], s2.cix[c.index].data)

    def testSeriesAndSeriesWithShuffle(self):
        data1 = pd.DataFrame(np.random.rand(10, 10),
                             index=[4, 9, 3, 2, 1, 5, 8, 6, 7, 10],
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        s1 = from_pandas_series(data1.iloc[4], chunk_size=5)
        s2 = from_pandas_series(data1[10], chunk_size=6)

        s3 = self.func(s1, s2)

        # test s3's index
        self.assertEqual(s3.shape, (np.nan,))
        self.assertNotEqual(s3.index_value.key, s1.index_value.key)
        self.assertNotEqual(s3.index_value.key, s2.index_value.key)
        pd.testing.assert_index_equal(s3.index_value.to_pandas(), pd.Int64Index([]))
        self.assertTrue(s3.index_value.should_be_monotonic)

        s3.tiles()

        self.assertEqual(s3.chunk_shape, (2,))
        for c in s3.chunks:
            self.assertIsInstance(c.op, self.op)
            self.assertEqual(len(c.inputs), 2)
            # test the left side
            self.assertIsInstance(c.inputs[0].op, DataFrameIndexAlignReduce)
            self.assertEqual(c.inputs[0].op.object_type, ObjectType.series)
            self.assertIsInstance(c.inputs[0].inputs[0].op, DataFrameShuffleProxy)
            for j, ci, ic in zip(itertools.count(0), c.inputs[0].inputs[0].inputs, s1.chunks):
                self.assertIsInstance(ci.op, DataFrameIndexAlignMap)
                self.assertEqual(ci.index, (j,))
                self.assertTrue(ci.op.index_shuffle_size, 2)
                self.assertIs(ci.inputs[0], ic.data)

            # test the right side
            self.assertIsInstance(c.inputs[1].op, DataFrameIndexAlignReduce)
            self.assertEqual(c.inputs[1].op.object_type, ObjectType.series)
            self.assertIsInstance(c.inputs[1].inputs[0].op, DataFrameShuffleProxy)
            for j, ci, ic in zip(itertools.count(0), c.inputs[1].inputs[0].inputs, s2.chunks):
                self.assertIsInstance(ci.op, DataFrameIndexAlignMap)
                self.assertEqual(ci.index, (j,))
                self.assertTrue(ci.op.index_shuffle_size, 2)
                self.assertIs(ci.inputs[0], ic.data)

        # make sure shuffle proxies' key are different
        proxy_keys = set()
        for c in s3.chunks:
            proxy_keys.add(c.inputs[0].inputs[0].op.key)
            proxy_keys.add(c.inputs[1].inputs[0].op.key)
        self.assertEqual(len(proxy_keys), 2)

    def testIdenticalIndexAndColumns(self):
        data1 = pd.DataFrame(np.random.rand(10, 10),
                             columns=np.arange(3, 13))
        df1 = from_pandas(data1, chunk_size=5)
        data2 = pd.DataFrame(np.random.rand(10, 10),
                             columns=np.arange(3, 13))
        df2 = from_pandas(data2, chunk_size=5)

        df3 = self.func(df1, df2)

        # test df3's index and columns
        pd.testing.assert_index_equal(df3.columns_value.to_pandas(), self.func(data1, data2).columns)
        self.assertFalse(df3.columns_value.should_be_monotonic)
        self.assertIsInstance(df3.index_value.value, IndexValue.RangeIndex)
        self.assertFalse(df3.index_value.should_be_monotonic)
        pd.testing.assert_index_equal(df3.index_value.to_pandas(), pd.RangeIndex(0, 10))
        self.assertEqual(df3.index_value.key, df1.index_value.key)
        self.assertEqual(df3.index_value.key, df2.index_value.key)
        self.assertEqual(df3.shape, (10, 10))  # columns is recorded, so we can get it

        df3.tiles()

        self.assertEqual(df3.chunk_shape, (2, 2))
        for c in df3.chunks:
            self.assertIsInstance(c.op, self.op)
            self.assertEqual(len(c.inputs), 2)
            self.assertEqual(c.shape, (5, 5))
            self.assertEqual(c.index_value.key, df1.cix[c.index].index_value.key)
            self.assertEqual(c.index_value.key, df2.cix[c.index].index_value.key)
            self.assertEqual(c.columns_value.key, df1.cix[c.index].columns_value.key)
            self.assertEqual(c.columns_value.key, df2.cix[c.index].columns_value.key)
            pd.testing.assert_index_equal(c.columns_value.to_pandas(), df1.cix[c.index].columns_value.to_pandas())
            pd.testing.assert_index_equal(c.columns_value.to_pandas(), df2.cix[c.index].columns_value.to_pandas())
            pd.testing.assert_index_equal(c.dtypes.index, df1.cix[c.index].columns_value.to_pandas())

            # test the left side
            self.assertIs(c.inputs[0], df1.cix[c.index].data)
            # test the right side
            self.assertIs(c.inputs[1], df2.cix[c.index].data)

    def testWithOneShuffle(self):
        # only 1 axis is monotonic
        # data1 with index split into [0...4], [5...9],
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=5)
        # data2 with index split into [6...11], [2, 5],
        data2 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(11, 1, -1),
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = from_pandas(data2, chunk_size=6)

        df3 = self.func(df1, df2)

        # test df3's index and columns
        pd.testing.assert_index_equal(df3.columns_value.to_pandas(), self.func(data1, data2).columns)
        self.assertTrue(df3.columns_value.should_be_monotonic)
        self.assertIsInstance(df3.index_value.value, IndexValue.Int64Index)
        self.assertTrue(df3.index_value.should_be_monotonic)
        pd.testing.assert_index_equal(df3.index_value.to_pandas(), pd.Int64Index([]))
        self.assertNotEqual(df3.index_value.key, df1.index_value.key)
        self.assertNotEqual(df3.index_value.key, df2.index_value.key)
        self.assertEqual(df3.shape[1], 12)  # columns is recorded, so we can get it

        df3.tiles()

        data1_index_min_max = [(0, True, 4, True), (5, True, 9, True)]
        data2_index_min_max = [(2, True, 5, True), (6, True, 11, True)]

        left_index_splits, right_index_splits = split_monotonic_index_min_max(
            data1_index_min_max, True, data2_index_min_max, False)

        left_index_idx_to_original_idx = build_split_idx_to_origin_idx(left_index_splits)
        right_index_idx_to_original_idx = build_split_idx_to_origin_idx(right_index_splits, False)

        self.assertEqual(df3.chunk_shape, (7, 2))
        for c in df3.chunks:
            self.assertIsInstance(c.op, self.op)
            self.assertEqual(len(c.inputs), 2)
            idx = c.index
            # test the left side
            self.assertIsInstance(c.inputs[0].op, DataFrameIndexAlignReduce)
            expect_dtypes = pd.concat([hash_dtypes(ic.inputs[0].op.data.dtypes, 2)[c.index[1]]
                                       for ic in c.inputs[0].inputs[0].inputs])
            pd.testing.assert_series_equal(c.inputs[0].dtypes, expect_dtypes)
            pd.testing.assert_index_equal(c.inputs[0].columns_value.to_pandas(), c.inputs[0].dtypes.index)
            self.assertIsInstance(c.inputs[0].index_value.to_pandas(), type(data1.index))
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
                self.assertIsInstance(ci.index_value.to_pandas(), type(data1.index))
                self.assertTrue(ci.op.column_shuffle_size, 2)
                shuffle_segments = ci.op.column_shuffle_segments
                expected_shuffle_segments = hash_dtypes(ic.data.dtypes, 2)
                self.assertEqual(len(shuffle_segments), len(expected_shuffle_segments))
                for ss, ess in zip(shuffle_segments, expected_shuffle_segments):
                    pd.testing.assert_series_equal(ss, ess)
                self.assertIs(ci.inputs[0], ic.data)
            # test the right side
            self.assertIsInstance(c.inputs[1].op, DataFrameIndexAlignReduce)
            expect_dtypes = pd.concat([hash_dtypes(ic.inputs[0].op.data.dtypes, 2)[c.index[1]]
                                       for ic in c.inputs[1].inputs[0].inputs])
            pd.testing.assert_series_equal(c.inputs[1].dtypes, expect_dtypes)
            pd.testing.assert_index_equal(c.inputs[1].columns_value.to_pandas(), c.inputs[1].dtypes.index)
            self.assertIsInstance(c.inputs[1].index_value.to_pandas(), type(data1.index))
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
                shuffle_segments = ci.op.column_shuffle_segments
                expected_shuffle_segments = hash_dtypes(ic.data.dtypes, 2)
                self.assertEqual(len(shuffle_segments), len(expected_shuffle_segments))
                for ss, ess in zip(shuffle_segments, expected_shuffle_segments):
                    pd.testing.assert_series_equal(ss, ess)
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

    def testWithAllShuffle(self):
        # no axis is monotonic
        data1 = pd.DataFrame(np.random.rand(10, 10), index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=5)
        data2 = pd.DataFrame(np.random.rand(10, 10), index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3],
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = from_pandas(data2, chunk_size=6)

        df3 = self.func(df1, df2)

        # test df3's index and columns
        pd.testing.assert_index_equal(df3.columns_value.to_pandas(), self.func(data1, data2).columns)
        self.assertTrue(df3.columns_value.should_be_monotonic)
        self.assertIsInstance(df3.index_value.value, IndexValue.Int64Index)
        self.assertTrue(df3.index_value.should_be_monotonic)
        pd.testing.assert_index_equal(df3.index_value.to_pandas(), pd.Int64Index([]))
        self.assertNotEqual(df3.index_value.key, df1.index_value.key)
        self.assertNotEqual(df3.index_value.key, df2.index_value.key)
        self.assertEqual(df3.shape[1], 12)  # columns is recorded, so we can get it

        df3.tiles()

        self.assertEqual(df3.chunk_shape, (2, 2))
        proxy_keys = set()
        for c in df3.chunks:
            self.assertIsInstance(c.op, self.op)
            self.assertEqual(len(c.inputs), 2)
            # test left side
            self.assertIsInstance(c.inputs[0].op, DataFrameIndexAlignReduce)
            expect_dtypes = pd.concat([hash_dtypes(ic.inputs[0].op.data.dtypes, 2)[c.index[1]]
                                       for ic in c.inputs[0].inputs[0].inputs if ic.index[0] == 0])
            pd.testing.assert_series_equal(c.inputs[0].dtypes, expect_dtypes)
            pd.testing.assert_index_equal(c.inputs[0].columns_value.to_pandas(), c.inputs[0].dtypes.index)
            self.assertIsInstance(c.inputs[0].index_value.to_pandas(), type(data1.index))
            self.assertIsInstance(c.inputs[0].inputs[0].op, DataFrameShuffleProxy)
            proxy_keys.add(c.inputs[0].inputs[0].op.key)
            for ic, ci in zip(c.inputs[0].inputs[0].inputs, df1.chunks):
                self.assertIsInstance(ic.op, DataFrameIndexAlignMap)
                self.assertEqual(ic.op.index_shuffle_size, 2)
                self.assertIsInstance(ic.index_value.to_pandas(), type(data1.index))
                self.assertEqual(ic.op.column_shuffle_size, 2)
                self.assertIsNotNone(ic.columns_value)
                shuffle_segments = ic.op.column_shuffle_segments
                expected_shuffle_segments = hash_dtypes(ci.data.dtypes, 2)
                self.assertEqual(len(shuffle_segments), len(expected_shuffle_segments))
                for ss, ess in zip(shuffle_segments, expected_shuffle_segments):
                    pd.testing.assert_series_equal(ss, ess)
                self.assertIs(ic.inputs[0], ci.data)
            # test right side
            self.assertIsInstance(c.inputs[1].op, DataFrameIndexAlignReduce)
            expect_dtypes = pd.concat([hash_dtypes(ic.inputs[0].op.data.dtypes, 2)[c.index[1]]
                                       for ic in c.inputs[1].inputs[0].inputs if ic.index[0] == 0])
            pd.testing.assert_series_equal(c.inputs[1].dtypes, expect_dtypes)
            pd.testing.assert_index_equal(c.inputs[1].columns_value.to_pandas(), c.inputs[1].dtypes.index)
            self.assertIsInstance(c.inputs[0].index_value.to_pandas(), type(data1.index))
            self.assertIsInstance(c.inputs[1].inputs[0].op, DataFrameShuffleProxy)
            proxy_keys.add(c.inputs[1].inputs[0].op.key)
            for ic, ci in zip(c.inputs[1].inputs[0].inputs, df2.chunks):
                self.assertIsInstance(ic.op, DataFrameIndexAlignMap)
                self.assertEqual(ic.op.index_shuffle_size, 2)
                self.assertIsInstance(ic.index_value.to_pandas(), type(data1.index))
                self.assertEqual(ic.op.column_shuffle_size, 2)
                self.assertIsNotNone(ic.columns_value)
                shuffle_segments = ic.op.column_shuffle_segments
                expected_shuffle_segments = hash_dtypes(ci.data.dtypes, 2)
                self.assertEqual(len(shuffle_segments), len(expected_shuffle_segments))
                for ss, ess in zip(shuffle_segments, expected_shuffle_segments):
                    pd.testing.assert_series_equal(ss, ess)
                self.assertIs(ic.inputs[0], ci.data)

        self.assertEqual(len(proxy_keys), 2)

        data4 = pd.DataFrame(np.random.rand(10, 10), index=np.random.randint(-100, 100, size=(10,)),
                             columns=[np.random.bytes(10) for _ in range(10)])
        df4 = from_pandas(data4, chunk_size=3)

        data5 = pd.DataFrame(np.random.rand(10, 10), index=np.random.randint(-100, 100, size=(10,)),
                             columns=[np.random.bytes(10) for _ in range(10)])
        df5 = from_pandas(data5, chunk_size=3)

        df6 = self.func(df4, df5)

        # test df6's index and columns
        pd.testing.assert_index_equal(df6.columns_value.to_pandas(), self.func(data4, data5).columns)
        self.assertTrue(df6.columns_value.should_be_monotonic)
        self.assertIsInstance(df6.index_value.value, IndexValue.Int64Index)
        self.assertTrue(df6.index_value.should_be_monotonic)
        pd.testing.assert_index_equal(df6.index_value.to_pandas(), pd.Int64Index([]))
        self.assertNotEqual(df6.index_value.key, df4.index_value.key)
        self.assertNotEqual(df6.index_value.key, df5.index_value.key)
        self.assertEqual(df6.shape[1], 20)  # columns is recorded, so we can get it

        df6.tiles()

        self.assertEqual(df6.chunk_shape, (4, 4))
        proxy_keys = set()
        for c in df6.chunks:
            self.assertIsInstance(c.op, self.op)
            self.assertEqual(len(c.inputs), 2)
            # test left side
            self.assertIsInstance(c.inputs[0].op, DataFrameIndexAlignReduce)
            expect_dtypes = pd.concat([hash_dtypes(ic.inputs[0].op.data.dtypes, 4)[c.index[1]]
                                       for ic in c.inputs[0].inputs[0].inputs if ic.index[0] == 0])
            pd.testing.assert_series_equal(c.inputs[0].dtypes, expect_dtypes)
            pd.testing.assert_index_equal(c.inputs[0].columns_value.to_pandas(), c.inputs[0].dtypes.index)
            self.assertIsInstance(c.inputs[0].index_value.to_pandas(), type(data1.index))
            self.assertIsInstance(c.inputs[0].inputs[0].op, DataFrameShuffleProxy)
            proxy_keys.add(c.inputs[0].inputs[0].op.key)
            for ic, ci in zip(c.inputs[0].inputs[0].inputs, df4.chunks):
                self.assertIsInstance(ic.op, DataFrameIndexAlignMap)
                self.assertEqual(ic.op.index_shuffle_size, 4)
                self.assertIsInstance(ic.index_value.to_pandas(), type(data1.index))
                self.assertEqual(ic.op.column_shuffle_size, 4)
                self.assertIsNotNone(ic.columns_value)
                shuffle_segments = ic.op.column_shuffle_segments
                expected_shuffle_segments = hash_dtypes(ci.data.dtypes, 4)
                self.assertEqual(len(shuffle_segments), len(expected_shuffle_segments))
                for ss, ess in zip(shuffle_segments, expected_shuffle_segments):
                    pd.testing.assert_series_equal(ss, ess)
                self.assertIs(ic.inputs[0], ci.data)
            # test right side
            self.assertIsInstance(c.inputs[1].op, DataFrameIndexAlignReduce)
            expect_dtypes = pd.concat([hash_dtypes(ic.inputs[0].op.data.dtypes, 4)[c.index[1]]
                                       for ic in c.inputs[1].inputs[0].inputs if ic.index[0] == 0])
            pd.testing.assert_series_equal(c.inputs[1].dtypes, expect_dtypes)
            pd.testing.assert_index_equal(c.inputs[1].columns_value.to_pandas(), c.inputs[1].dtypes.index)
            self.assertIsInstance(c.inputs[0].index_value.to_pandas(), type(data1.index))
            self.assertIsInstance(c.inputs[1].inputs[0].op, DataFrameShuffleProxy)
            proxy_keys.add(c.inputs[1].inputs[0].op.key)
            for ic, ci in zip(c.inputs[1].inputs[0].inputs, df5.chunks):
                self.assertIsInstance(ic.op, DataFrameIndexAlignMap)
                self.assertEqual(ic.op.index_shuffle_size, 4)
                self.assertIsInstance(ic.index_value.to_pandas(), type(data1.index))
                self.assertEqual(ic.op.column_shuffle_size, 4)
                self.assertIsNotNone(ic.columns_value)
                shuffle_segments = ic.op.column_shuffle_segments
                expected_shuffle_segments = hash_dtypes(ci.data.dtypes, 4)
                self.assertEqual(len(shuffle_segments), len(expected_shuffle_segments))
                for ss, ess in zip(shuffle_segments, expected_shuffle_segments):
                    pd.testing.assert_series_equal(ss, ess)
                self.assertIs(ic.inputs[0], ci.data)

        self.assertEqual(len(proxy_keys), 2)

    def testWithoutShuffleAndWithOneChunk(self):
        # only 1 axis is monotonic
        # data1 with index split into [0...4], [5...9],
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=(5, 10))
        # data2 with index split into [6...11], [2, 5],
        data2 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(11, 1, -1),
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = from_pandas(data2, chunk_size=(6, 10))

        df3 = self.func(df1, df2)

        # test df3's index and columns
        pd.testing.assert_index_equal(df3.columns_value.to_pandas(), self.func(data1, data2).columns)
        self.assertTrue(df3.columns_value.should_be_monotonic)
        self.assertIsInstance(df3.index_value.value, IndexValue.Int64Index)
        self.assertTrue(df3.index_value.should_be_monotonic)
        pd.testing.assert_index_equal(df3.index_value.to_pandas(), pd.Int64Index([]))
        self.assertNotEqual(df3.index_value.key, df1.index_value.key)
        self.assertNotEqual(df3.index_value.key, df2.index_value.key)
        self.assertEqual(df3.shape[1], 12)  # columns is recorded, so we can get it

        df3.tiles()

        data1_index_min_max = [(0, True, 4, True), (5, True, 9, True)]
        data2_index_min_max = [(2, True, 5, True), (6, True, 11, True)]

        left_index_splits, right_index_splits = split_monotonic_index_min_max(
            data1_index_min_max, True, data2_index_min_max, False)

        left_index_idx_to_original_idx = build_split_idx_to_origin_idx(left_index_splits)
        right_index_idx_to_original_idx = build_split_idx_to_origin_idx(right_index_splits, False)

        self.assertEqual(df3.chunk_shape, (7, 1))
        for c in df3.chunks:
            self.assertIsInstance(c.op, self.op)
            self.assertEqual(len(c.inputs), 2)
            # test shape
            idx = c.index
            # test the left side
            self.assertIsInstance(c.inputs[0].op, DataFrameIndexAlignMap)
            left_row_idx, left_row_inner_idx = left_index_idx_to_original_idx[idx[0]]
            expect_df1_input = df1.cix[left_row_idx, 0].data
            self.assertIs(c.inputs[0].inputs[0], expect_df1_input)
            left_index_min_max = left_index_splits[left_row_idx][left_row_inner_idx]
            self.assertEqual(c.inputs[0].op.index_min, left_index_min_max[0])
            self.assertEqual(c.inputs[0].op.index_min_close, left_index_min_max[1])
            self.assertEqual(c.inputs[0].op.index_max, left_index_min_max[2])
            self.assertEqual(c.inputs[0].op.index_max_close, left_index_min_max[3])
            self.assertIsInstance(c.inputs[0].index_value.to_pandas(), type(data1.index))
            self.assertEqual(c.inputs[0].op.column_min, expect_df1_input.columns_value.min_val)
            self.assertEqual(c.inputs[0].op.column_min_close, expect_df1_input.columns_value.min_val_close)
            self.assertEqual(c.inputs[0].op.column_max, expect_df1_input.columns_value.max_val)
            self.assertEqual(c.inputs[0].op.column_max_close, expect_df1_input.columns_value.max_val_close)
            expect_left_columns = expect_df1_input.columns_value
            pd.testing.assert_index_equal(c.inputs[0].columns_value.to_pandas(), expect_left_columns.to_pandas())
            pd.testing.assert_index_equal(c.inputs[0].dtypes.index, expect_left_columns.to_pandas())
            # test the right side
            self.assertIsInstance(c.inputs[1].op, DataFrameIndexAlignMap)
            right_row_idx, right_row_inner_idx = right_index_idx_to_original_idx[idx[0]]
            expect_df2_input = df2.cix[right_row_idx, 0].data
            self.assertIs(c.inputs[1].inputs[0], expect_df2_input)
            right_index_min_max = right_index_splits[right_row_idx][right_row_inner_idx]
            self.assertEqual(c.inputs[1].op.index_min, right_index_min_max[0])
            self.assertEqual(c.inputs[1].op.index_min_close, right_index_min_max[1])
            self.assertEqual(c.inputs[1].op.index_max, right_index_min_max[2])
            self.assertEqual(c.inputs[1].op.index_max_close, right_index_min_max[3])
            self.assertIsInstance(c.inputs[1].index_value.to_pandas(), type(data2.index))
            self.assertEqual(c.inputs[1].op.column_min, expect_df2_input.columns_value.min_val)
            self.assertEqual(c.inputs[1].op.column_min_close, expect_df2_input.columns_value.min_val_close)
            self.assertEqual(c.inputs[1].op.column_max, expect_df2_input.columns_value.max_val)
            self.assertEqual(c.inputs[1].op.column_max_close, expect_df2_input.columns_value.max_val_close)
            expect_right_columns = expect_df2_input.columns_value
            pd.testing.assert_index_equal(c.inputs[1].columns_value.to_pandas(), expect_right_columns.to_pandas())
            pd.testing.assert_index_equal(c.inputs[1].dtypes.index, expect_right_columns.to_pandas())

    def testBothOneChunk(self):
        # no axis is monotonic, but 1 chunk for all axes
        data1 = pd.DataFrame(np.random.rand(10, 10), index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=10)
        data2 = pd.DataFrame(np.random.rand(10, 10), index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3],
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = from_pandas(data2, chunk_size=10)

        df3 = self.func(df1, df2)

        # test df3's index and columns
        pd.testing.assert_index_equal(df3.columns_value.to_pandas(), self.func(data1, data2).columns)
        self.assertTrue(df3.columns_value.should_be_monotonic)
        self.assertIsInstance(df3.index_value.value, IndexValue.Int64Index)
        self.assertTrue(df3.index_value.should_be_monotonic)
        pd.testing.assert_index_equal(df3.index_value.to_pandas(), pd.Int64Index([]))
        self.assertNotEqual(df3.index_value.key, df1.index_value.key)
        self.assertNotEqual(df3.index_value.key, df2.index_value.key)
        self.assertEqual(df3.shape[1], 12)  # columns is recorded, so we can get it

        df3.tiles()

        self.assertEqual(df3.chunk_shape, (1, 1))
        for c in df3.chunks:
            self.assertIsInstance(c.op, self.op)
            self.assertEqual(len(c.inputs), 2)
            # test the left side
            self.assertIs(c.inputs[0], df1.chunks[0].data)
            # test the right side
            self.assertIs(c.inputs[1], df2.chunks[0].data)

    def testWithShuffleAndOneChunk(self):
        # no axis is monotonic
        data1 = pd.DataFrame(np.random.rand(10, 10), index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=(5, 10))
        data2 = pd.DataFrame(np.random.rand(10, 10), index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3],
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = from_pandas(data2, chunk_size=(6, 10))

        df3 = self.func(df1, df2)

        # test df3's index and columns
        pd.testing.assert_index_equal(df3.columns_value.to_pandas(), self.func(data1, data2).columns)
        self.assertTrue(df3.columns_value.should_be_monotonic)
        self.assertIsInstance(df3.index_value.value, IndexValue.Int64Index)
        self.assertTrue(df3.index_value.should_be_monotonic)
        pd.testing.assert_index_equal(df3.index_value.to_pandas(), pd.Int64Index([]))
        self.assertNotEqual(df3.index_value.key, df1.index_value.key)
        self.assertNotEqual(df3.index_value.key, df2.index_value.key)
        self.assertEqual(df3.shape[1], 12)  # columns is recorded, so we can get it

        df3.tiles()

        self.assertEqual(df3.chunk_shape, (2, 1))
        proxy_keys = set()
        for c in df3.chunks:
            self.assertIsInstance(c.op, self.op)
            self.assertEqual(len(c.inputs), 2)
            # test left side
            self.assertIsInstance(c.inputs[0].op, DataFrameIndexAlignReduce)
            expect_dtypes = pd.concat([ic.inputs[0].op.data.dtypes
                                       for ic in c.inputs[0].inputs[0].inputs if ic.index[0] == 0])
            pd.testing.assert_series_equal(c.inputs[0].dtypes, expect_dtypes)
            pd.testing.assert_index_equal(c.inputs[0].columns_value.to_pandas(), c.inputs[0].dtypes.index)
            self.assertIsInstance(c.inputs[0].index_value.to_pandas(), type(data1.index))
            self.assertIsInstance(c.inputs[0].inputs[0].op, DataFrameShuffleProxy)
            proxy_keys.add(c.inputs[0].inputs[0].op.key)
            for ic, ci in zip(c.inputs[0].inputs[0].inputs, df1.chunks):
                self.assertIsInstance(ic.op, DataFrameIndexAlignMap)
                self.assertEqual(ic.op.index_shuffle_size, 2)
                self.assertIsInstance(ic.index_value.to_pandas(), type(data1.index))
                self.assertEqual(ic.op.column_min, ci.columns_value.min_val)
                self.assertEqual(ic.op.column_min_close, ci.columns_value.min_val_close)
                self.assertEqual(ic.op.column_max, ci.columns_value.max_val)
                self.assertEqual(ic.op.column_max_close, ci.columns_value.max_val_close)
                self.assertIsNone(ic.op.column_shuffle_size, None)
                self.assertIsNotNone(ic.columns_value)
                self.assertIs(ic.inputs[0], ci.data)
            # test right side
            self.assertIsInstance(c.inputs[1].op, DataFrameIndexAlignReduce)
            expect_dtypes = pd.concat([ic.inputs[0].op.data.dtypes
                                       for ic in c.inputs[1].inputs[0].inputs if ic.index[0] == 0])
            pd.testing.assert_series_equal(c.inputs[1].dtypes, expect_dtypes)
            pd.testing.assert_index_equal(c.inputs[1].columns_value.to_pandas(), c.inputs[1].dtypes.index)
            self.assertIsInstance(c.inputs[0].index_value.to_pandas(), type(data1.index))
            self.assertIsInstance(c.inputs[1].inputs[0].op, DataFrameShuffleProxy)
            proxy_keys.add(c.inputs[1].inputs[0].op.key)
            for ic, ci in zip(c.inputs[1].inputs[0].inputs, df2.chunks):
                self.assertIsInstance(ic.op, DataFrameIndexAlignMap)
                self.assertEqual(ic.op.index_shuffle_size, 2)
                self.assertIsInstance(ic.index_value.to_pandas(), type(data1.index))
                self.assertIsNone(ic.op.column_shuffle_size)
                self.assertEqual(ic.op.column_min, ci.columns_value.min_val)
                self.assertEqual(ic.op.column_min_close, ci.columns_value.min_val_close)
                self.assertEqual(ic.op.column_max, ci.columns_value.max_val)
                self.assertEqual(ic.op.column_max_close, ci.columns_value.max_val_close)
                self.assertIsNone(ic.op.column_shuffle_size, None)
                self.assertIsNotNone(ic.columns_value)
                self.assertIs(ic.inputs[0], ci.data)

        self.assertEqual(len(proxy_keys), 2)

    def testOnSameDataFrame(self):
        data = pd.DataFrame(np.random.rand(10, 10), index=np.random.randint(-100, 100, size=(10,)),
                            columns=[np.random.bytes(10) for _ in range(10)])
        df = from_pandas(data, chunk_size=3)
        df2 = self.func(df, df)

        # test df2's index and columns
        pd.testing.assert_index_equal(df2.columns_value.to_pandas(), self.func(data, data).columns)
        self.assertFalse(df2.columns_value.should_be_monotonic)
        self.assertIsInstance(df2.index_value.value, IndexValue.Int64Index)
        self.assertFalse(df2.index_value.should_be_monotonic)
        pd.testing.assert_index_equal(df2.index_value.to_pandas(), pd.Int64Index([]))
        self.assertEqual(df2.index_value.key, df.index_value.key)
        self.assertEqual(df2.columns_value.key, df.columns_value.key)
        self.assertEqual(df2.shape[1], 10)

        df2.tiles()

        self.assertEqual(df2.chunk_shape, df.chunk_shape)
        for c in df2.chunks:
            self.assertIsInstance(c.op, self.op)
            self.assertEqual(len(c.inputs), 2)
            # test the left side
            self.assertIs(c.inputs[0], df.cix[c.index].data)
            # test the right side
            self.assertIs(c.inputs[1], df.cix[c.index].data)

    def testDataFrameAndScalar(self):
        data = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                            columns=np.arange(3, 13))
        df = from_pandas(data, chunk_size=5)
        # test operator with scalar
        result = self.func(df, 1)
        result2 = getattr(df, self.func_name)(1)

        # test reverse operator with scalar
        result3 = getattr(df, self.rfunc_name)(1)
        result4 = self.func(df, 1)
        result5 = self.func(1, df)
        pd.testing.assert_index_equal(result.columns_value.to_pandas(), data.columns)
        self.assertIsInstance(result.index_value.value, IndexValue.Int64Index)

        pd.testing.assert_index_equal(result2.columns_value.to_pandas(), data.columns)
        self.assertIsInstance(result2.index_value.value, IndexValue.Int64Index)

        pd.testing.assert_index_equal(result3.columns_value.to_pandas(), data.columns)
        self.assertIsInstance(result3.index_value.value, IndexValue.Int64Index)

        pd.testing.assert_index_equal(result4.columns_value.to_pandas(), data.columns)
        self.assertIsInstance(result4.index_value.value, IndexValue.Int64Index)

        pd.testing.assert_index_equal(result5.columns_value.to_pandas(), data.columns)
        self.assertIsInstance(result5.index_value.value, IndexValue.Int64Index)

        # test NotImplemented, use other's rfunc instead
        class TestRFunc:
            pass

        setattr(TestRFunc, '__%s__' % self.rfunc_name, lambda *_: 1)
        other = TestRFunc()
        ret = self.func(df, other)
        self.assertEqual(ret, 1)

    def testSeriesAndScalar(self):
        data = pd.Series(range(10), index=[1, 3, 4, 2, 9, 10, 33, 23, 999, 123])
        s1 = from_pandas_series(data, chunk_size=3)
        r = getattr(s1, self.func_name)(456)
        r.tiles()

        self.assertEqual(r.index_value.key, s1.index_value.key)
        self.assertEqual(r.chunk_shape, s1.chunk_shape)

        for cr in r.chunks:
            cs = s1.cix[cr.index]
            self.assertEqual(cr.index_value.key, cs.index_value.key)
            self.assertIsInstance(cr.op, self.op)
            self.assertEqual(len(cr.inputs), 1)
            self.assertIsInstance(cr.inputs[0].op, SeriesDataSource)
            self.assertEqual(cr.op.rhs, 456)

        r = getattr(s1, self.rfunc_name)(789)
        r.tiles()

        self.assertEqual(r.index_value.key, s1.index_value.key)
        self.assertEqual(r.chunk_shape, s1.chunk_shape)

        for cr in r.chunks:
            cs = s1.cix[cr.index]
            self.assertEqual(cr.index_value.key, cs.index_value.key)
            self.assertIsInstance(cr.op, self.op)
            self.assertEqual(len(cr.inputs), 1)
            self.assertIsInstance(cr.inputs[0].op, SeriesDataSource)
            self.assertEqual(cr.op.lhs, 789)

    def testCheckInputs(self):
        data = pd.DataFrame(np.random.rand(10, 3))
        df = from_pandas(data)

        with self.assertRaises(ValueError):
            _ = df + np.random.rand(5, 3)

        with self.assertRaises(ValueError):
            _ = df + np.random.rand(10)

        with self.assertRaises(ValueError):
            _ = df + np.random.rand(10, 3, 2)

        data = pd.Series(np.random.rand(10))
        series = from_pandas_series(data)

        with self.assertRaises(ValueError):
            _ = series + np.random.rand(5, 3)

        with self.assertRaises(ValueError):
            _ = series + np.random.rand(5)


class TestUnary(TestBase):
    def testAbs(self):
        data1 = pd.DataFrame(np.random.rand(10, 10), index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=(5, 10))

        df2 = abs(df1)

        # test df2's index and columns
        pd.testing.assert_index_equal(df2.columns_value.to_pandas(), df1.columns_value.to_pandas())
        self.assertIsInstance(df2.index_value.value, IndexValue.Int64Index)
        self.assertEqual(df2.shape, (10, 10))

        df2.tiles()

        self.assertEqual(df2.chunk_shape, (2, 1))
        for c2, c1 in zip(df2.chunks, df1.chunks):
            self.assertIsInstance(c2.op, DataFrameAbs)
            self.assertEqual(len(c2.inputs), 1)
            # compare with input chunks
            self.assertEqual(c2.index, c1.index)
            pd.testing.assert_index_equal(c2.columns_value.to_pandas(), c1.columns_value.to_pandas())
            pd.testing.assert_index_equal(c2.index_value.to_pandas(), c1.index_value.to_pandas())
