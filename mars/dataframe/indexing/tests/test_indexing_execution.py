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

import contextlib
import os
import tempfile

import numpy as np
import pandas as pd

import mars.dataframe as md
import mars.tensor as mt
from mars.dataframe.datasource.read_csv import DataFrameReadCSV
from mars.dataframe.datasource.read_sql import DataFrameReadSQL
from mars.executor import register, Executor
from mars.session import new_session
from mars.tests.core import TestBase, ExecutorForTest


class Test(TestBase):
    def setUp(self):
        super().setUp()
        self.executor = self._create_test_context()[1]

    def testSetIndex(self):
        df1 = pd.DataFrame([[1, 3, 3], [4, 2, 6], [7, 8, 9]],
                           index=['a1', 'a2', 'a3'], columns=['x', 'y', 'z'])
        df2 = md.DataFrame(df1, chunk_size=2)

        expected = df1.set_index('y', drop=True)
        df3 = df2.set_index('y', drop=True)
        pd.testing.assert_frame_equal(
            expected, self.executor.execute_dataframe(df3, concat=True)[0])

        expected = df1.set_index('y', drop=False)
        df4 = df2.set_index('y', drop=False)
        pd.testing.assert_frame_equal(
            expected, self.executor.execute_dataframe(df4, concat=True)[0])

    def testILocGetItem(self):
        df1 = pd.DataFrame([[1, 3, 3], [4, 2, 6], [7, 8, 9]],
                           index=['a1', 'a2', 'a3'], columns=['x', 'y', 'z'])
        df2 = md.DataFrame(df1, chunk_size=2)

        # plain index
        expected = df1.iloc[1]
        df3 = df2.iloc[1]
        pd.testing.assert_series_equal(
            expected, self.executor.execute_dataframe(df3, concat=True, check_series_name=False)[0])

        # plain index on axis 1
        expected = df1.iloc[:2, 1]
        df4 = df2.iloc[:2, 1]
        pd.testing.assert_series_equal(
            expected, self.executor.execute_dataframe(df4, concat=True)[0])

        # slice index
        expected = df1.iloc[:, 2:4]
        df5 = df2.iloc[:, 2:4]
        pd.testing.assert_frame_equal(
            expected, self.executor.execute_dataframe(df5, concat=True)[0])

        # plain fancy index
        expected = df1.iloc[[0], [0, 1, 2]]
        df6 = df2.iloc[[0], [0, 1, 2]]
        pd.testing.assert_frame_equal(
            expected, self.executor.execute_dataframe(df6, concat=True)[0])

        # plain fancy index with shuffled order
        expected = df1.iloc[[0], [1, 2, 0]]
        df7 = df2.iloc[[0], [1, 2, 0]]
        pd.testing.assert_frame_equal(
            expected, self.executor.execute_dataframe(df7, concat=True)[0])

        # fancy index
        expected = df1.iloc[[1, 2], [0, 1, 2]]
        df8 = df2.iloc[[1, 2], [0, 1, 2]]
        pd.testing.assert_frame_equal(
            expected, self.executor.execute_dataframe(df8, concat=True)[0])

        # fancy index with shuffled order
        expected = df1.iloc[[2, 1], [1, 2, 0]]
        df9 = df2.iloc[[2, 1], [1, 2, 0]]
        pd.testing.assert_frame_equal(
            expected, self.executor.execute_dataframe(df9, concat=True)[0])

        # one fancy index
        expected = df1.iloc[[2, 1]]
        df10 = df2.iloc[[2, 1]]
        pd.testing.assert_frame_equal(
            expected, self.executor.execute_dataframe(df10, concat=True)[0])

        # plain index
        expected = df1.iloc[1, 2]
        df11 = df2.iloc[1, 2]
        self.assertEqual(
            expected, self.executor.execute_dataframe(df11, concat=True)[0])

        # bool index array
        expected = df1.iloc[[True, False, True], [2, 1]]
        df12 = df2.iloc[[True, False, True], [2, 1]]
        pd.testing.assert_frame_equal(
            expected, self.executor.execute_dataframe(df12, concat=True)[0])

        # bool index array on axis 1
        expected = df1.iloc[[2, 1], [True, False, True]]
        df14 = df2.iloc[[2, 1], [True, False, True]]
        pd.testing.assert_frame_equal(
            expected, self.executor.execute_dataframe(df14, concat=True)[0])

        # bool index
        expected = df1.iloc[[True, False, True], [2, 1]]
        df13 = df2.iloc[md.Series([True, False, True], chunk_size=1), [2, 1]]
        pd.testing.assert_frame_equal(
            expected, self.executor.execute_dataframe(df13, concat=True)[0])

        # test Series
        data = pd.Series(np.arange(10))
        series = md.Series(data, chunk_size=3).iloc[:3]
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series, concat=True)[0], data.iloc[:3])

        series = md.Series(data, chunk_size=3).iloc[4]
        self.assertEqual(
            self.executor.execute_dataframe(series, concat=True)[0], data.iloc[4])

        series = md.Series(data, chunk_size=3).iloc[[2, 3, 4, 9]]
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series, concat=True)[0], data.iloc[[2, 3, 4, 9]])

        series = md.Series(data, chunk_size=3).iloc[[4, 3, 9, 2]]
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series, concat=True)[0], data.iloc[[4, 3, 9, 2]])

        series = md.Series(data).iloc[5:]
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series, concat=True)[0], data.iloc[5:])

        # bool index array
        selection = np.random.RandomState(0).randint(2, size=10, dtype=bool)
        series = md.Series(data).iloc[selection]
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series, concat=True)[0], data.iloc[selection])

        # bool index
        series = md.Series(data).iloc[md.Series(selection, chunk_size=4)]
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series, concat=True)[0], data.iloc[selection])

    def testILocSetItem(self):
        df1 = pd.DataFrame([[1, 3, 3], [4, 2, 6], [7, 8, 9]],
                           index=['a1', 'a2', 'a3'], columns=['x', 'y', 'z'])
        df2 = md.DataFrame(df1, chunk_size=2)

        # plain index
        expected = df1
        expected.iloc[1] = 100
        df2.iloc[1] = 100
        pd.testing.assert_frame_equal(
            expected, self.executor.execute_dataframe(df2, concat=True)[0])

        # slice index
        expected.iloc[:, 2:4] = 1111
        df2.iloc[:, 2:4] = 1111
        pd.testing.assert_frame_equal(
            expected, self.executor.execute_dataframe(df2, concat=True)[0])

        # plain fancy index
        expected.iloc[[0], [0, 1, 2]] = 2222
        df2.iloc[[0], [0, 1, 2]] = 2222
        pd.testing.assert_frame_equal(
            expected, self.executor.execute_dataframe(df2, concat=True)[0])

        # fancy index
        expected.iloc[[1, 2], [0, 1, 2]] = 3333
        df2.iloc[[1, 2], [0, 1, 2]] = 3333
        pd.testing.assert_frame_equal(
            expected, self.executor.execute_dataframe(df2, concat=True)[0])

        # plain index
        expected.iloc[1, 2] = 4444
        df2.iloc[1, 2] = 4444
        pd.testing.assert_frame_equal(
            expected, self.executor.execute_dataframe(df2, concat=True)[0])

        # test Series
        data = pd.Series(np.arange(10))
        series = md.Series(data, chunk_size=3)
        series.iloc[:3] = 1
        data.iloc[:3] = 1
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series, concat=True)[0], data)

        series.iloc[4] = 2
        data.iloc[4] = 2
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series, concat=True)[0], data)

        series.iloc[[2, 3, 4, 9]] = 3
        data.iloc[[2, 3, 4, 9]] = 3
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series, concat=True)[0], data)

        series.iloc[5:] = 4
        data.iloc[5:] = 4
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series, concat=True)[0], data)

    def testLocGetItem(self):
        rs = np.random.RandomState(0)
        # index and columns are labels
        raw1 = pd.DataFrame(rs.randint(10, size=(5, 4)),
                            index=['a1', 'a2', 'a3', 'a4', 'a5'],
                            columns=['a', 'b', 'c', 'd'])
        # columns are labels
        raw2 = raw1.copy()
        raw2.reset_index(inplace=True, drop=True)
        # columns are non unique and monotonic
        raw3 = raw1.copy()
        raw3.columns = ['a', 'b', 'b', 'd']
        # columns are non unique and non monotonic
        raw4 = raw1.copy()
        raw4.columns = ['b', 'a', 'b', 'd']
        # index that is timestamp
        raw5 = raw1.copy()
        raw5.index = pd.date_range('2020-1-1', periods=5)

        df1 = md.DataFrame(raw1, chunk_size=2)
        df2 = md.DataFrame(raw2, chunk_size=2)
        df3 = md.DataFrame(raw3, chunk_size=2)
        df4 = md.DataFrame(raw4, chunk_size=2)
        df5 = md.DataFrame(raw5, chunk_size=2)

        df = df2.loc[3, 'b']
        result = self.executor.execute_tensor(df, concat=True)[0]
        expected = raw2.loc[3, 'b']
        self.assertEqual(result, expected)

        df = df1.loc['a3', 'b']
        result = self.executor.execute_tensor(df, concat=True, check_shape=False)[0]
        expected = raw1.loc['a3', 'b']
        self.assertEqual(result, expected)

        df = df2.loc[1:4, 'b':'d']
        result = self.executor.execute_dataframe(df, concat=True)[0]
        expected = raw2.loc[1:4, 'b': 'd']
        pd.testing.assert_frame_equal(result, expected)

        df = df2.loc[:4, 'b':]
        result = self.executor.execute_dataframe(df, concat=True)[0]
        expected = raw2.loc[:4, 'b':]
        pd.testing.assert_frame_equal(result, expected)

        # slice on axis index whose index_value does not have value
        df = df1.loc['a2':'a4', 'b':]
        result = self.executor.execute_dataframe(df, concat=True)[0]
        expected = raw1.loc['a2':'a4', 'b':]
        pd.testing.assert_frame_equal(result, expected)

        df = df2.loc[:, 'b']
        result = self.executor.execute_dataframe(df, concat=True)[0]
        expected = raw2.loc[:, 'b']
        pd.testing.assert_series_equal(result, expected)

        # 'b' is non-unique
        df = df3.loc[:, 'b']
        result = self.executor.execute_dataframe(df, concat=True)[0]
        expected = raw3.loc[:, 'b']
        pd.testing.assert_frame_equal(result, expected)

        # 'b' is non-unique, and non-monotonic
        df = df4.loc[:, 'b']
        result = self.executor.execute_dataframe(df, concat=True)[0]
        expected = raw4.loc[:, 'b']
        pd.testing.assert_frame_equal(result, expected)

        # label on axis 0
        df = df1.loc['a2', :]
        result = self.executor.execute_dataframe(df, concat=True)[0]
        expected = raw1.loc['a2', :]
        pd.testing.assert_series_equal(result, expected)

        # label-based fancy index
        df = df2.loc[[3, 0, 1], ['c', 'a', 'd']]
        result = self.executor.execute_dataframe(df, concat=True)[0]
        expected = raw2.loc[[3, 0, 1], ['c', 'a', 'd']]
        pd.testing.assert_frame_equal(result, expected)

        # label-based fancy index, asc sorted
        df = df2.loc[[0, 1, 3], ['a', 'c', 'd']]
        result = self.executor.execute_dataframe(df, concat=True)[0]
        expected = raw2.loc[[0, 1, 3], ['a', 'c', 'd']]
        pd.testing.assert_frame_equal(result, expected)

        # label-based fancy index in which non-unique exists
        selection = rs.randint(2, size=(5,), dtype=bool)
        df = df3.loc[selection, ['b', 'a', 'd']]
        result = self.executor.execute_dataframe(df, concat=True)[0]
        expected = raw3.loc[selection, ['b', 'a', 'd']]
        pd.testing.assert_frame_equal(result, expected)

        df = df3.loc[md.Series(selection), ['b', 'a', 'd']]
        result = self.executor.execute_dataframe(df, concat=True)[0]
        expected = raw3.loc[selection, ['b', 'a', 'd']]
        pd.testing.assert_frame_equal(result, expected)

        # label-based fancy index on index
        # whose index_value does not have value
        df = df1.loc[['a3', 'a1'], ['b', 'a', 'd']]
        result = self.executor.execute_dataframe(df, concat=True, check_nsplits=False)[0]
        expected = raw1.loc[['a3', 'a1'], ['b', 'a', 'd']]
        pd.testing.assert_frame_equal(result, expected)

        # get timestamp by str
        df = df5.loc['20200101']
        result = self.executor.execute_dataframe(df, concat=True, check_series_name=False)[0]
        expected = raw5.loc['20200101']
        pd.testing.assert_series_equal(result, expected)

        # get timestamp by str, return scalar
        df = df5.loc['2020-1-1', 'c']
        result = self.executor.execute_dataframe(df, concat=True)[0]
        expected = raw5.loc['2020-1-1', 'c']
        self.assertEqual(result, expected)

    def testDataFrameGetitem(self):
        data = pd.DataFrame(np.random.rand(10, 5), columns=['c1', 'c2', 'c3', 'c4', 'c5'])
        df = md.DataFrame(data, chunk_size=2)
        data2 = data.copy()
        data2.index = pd.date_range('2020-1-1', periods=10)
        mdf = md.DataFrame(data2, chunk_size=3)

        series1 = df['c2']
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series1, concat=True)[0], data['c2'])

        series2 = df['c5']
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series2, concat=True)[0], data['c5'])

        df1 = df[['c1', 'c2', 'c3']]
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df1, concat=True)[0], data[['c1', 'c2', 'c3']])

        df2 = df[['c3', 'c2', 'c1']]
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df2, concat=True)[0], data[['c3', 'c2', 'c1']])

        df3 = df[['c1']]
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df3, concat=True)[0], data[['c1']])

        df4 = df[['c3', 'c1', 'c2', 'c1']]
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df4, concat=True)[0], data[['c3', 'c1', 'c2', 'c1']])

        df5 = df[np.array(['c1', 'c2', 'c3'])]
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df5, concat=True)[0], data[['c1', 'c2', 'c3']])

        df6 = df[['c3', 'c2', 'c1']]
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df6, concat=True)[0], data[['c3', 'c2', 'c1']])

        df7 = df[1:7:2]
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df7, concat=True)[0], data[1:7:2])

        series3 = df['c1'][0]
        self.assertEqual(
            self.executor.execute_dataframe(series3, concat=True)[0], data['c1'][0])

        df8 = mdf[3:7]
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df8, concat=True)[0], data2[3:7])

        df9 = mdf['2020-1-2': '2020-1-5']
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df9, concat=True)[0], data2['2020-1-2': '2020-1-5'])

    def testDataFrameGetitemBool(self):
        data = pd.DataFrame(np.random.rand(10, 5), columns=['c1', 'c2', 'c3', 'c4', 'c5'])
        df = md.DataFrame(data, chunk_size=2)

        mask_data = data.c1 > 0.5
        mask = md.Series(mask_data, chunk_size=2)

        # getitem by mars series
        self.assertEqual(
            self.executor.execute_dataframe(df[mask], concat=True)[0].shape, data[mask_data].shape)
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df[mask], concat=True)[0], data[mask_data])

        # getitem by pandas series
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df[mask_data], concat=True)[0], data[mask_data])

        # getitem by mars series with alignment but no shuffle
        mask_data = pd.Series([True, True, True, False, False, True, True, False, False, True],
                              index=range(9, -1, -1))
        mask = md.Series(mask_data, chunk_size=2)
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df[mask], concat=True)[0], data[mask_data])

        # getitem by mars series with shuffle alignment
        mask_data = pd.Series([True, True, True, False, False, True, True, False, False, True],
                              index=[0, 3, 6, 2, 9, 8, 5, 7, 1, 4])
        mask = md.Series(mask_data, chunk_size=2)
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df[mask], concat=True)[0].sort_index(), data[mask_data])

        # getitem by mars series with shuffle alignment and extra element
        mask_data = pd.Series([True, True, True, False, False, True, True, False, False, True, False],
                              index=[0, 3, 6, 2, 9, 8, 5, 7, 1, 4, 10])
        mask = md.Series(mask_data, chunk_size=2)
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df[mask], concat=True)[0].sort_index(), data[mask_data])

        # getitem by DataFrame with all bool columns
        r = df[df > 0.5]
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_frame_equal(result, data[data > 0.5])

    def testDataFrameGetitemUsingAttr(self):
        data = pd.DataFrame(np.random.rand(10, 5), columns=['c1', 'c2', 'key', 'dtypes', 'size'])
        df = md.DataFrame(data, chunk_size=2)

        series1 = df.c2
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series1, concat=True)[0], data.c2)

        # accessing column using attribute shouldn't overwrite existing attributes
        self.assertEqual(df.key, getattr(getattr(df, '_data'), '_key'))
        self.assertEqual(df.size, data.size)
        pd.testing.assert_series_equal(df.dtypes, data.dtypes)

        # accessing non-existing attributes should trigger exception
        with self.assertRaises(AttributeError):
            _ = df.zzz  # noqa: F841

    def testSeriesGetitem(self):
        data = pd.Series(np.random.rand(10))
        series = md.Series(data)
        self.assertEqual(
            self.executor.execute_dataframe(series[1], concat=True)[0], data[1])

        data = pd.Series(np.random.rand(10), name='a')
        series = md.Series(data, chunk_size=4)

        for i in range(10):
            series1 = series[i]
            self.assertEqual(
                self.executor.execute_dataframe(series1, concat=True)[0], data[i])

        series2 = series[[0, 1, 2, 3, 4]]
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series2, concat=True)[0], data[[0, 1, 2, 3, 4]])

        series3 = series[[4, 3, 2, 1, 0]]
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series3, concat=True)[0], data[[4, 3, 2, 1, 0]])

        series4 = series[[1, 2, 3, 2, 1, 0]]
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series4, concat=True)[0], data[[1, 2, 3, 2, 1, 0]])
        #
        index = ['i' + str(i) for i in range(20)]
        data = pd.Series(np.random.rand(20), index=index, name='a')
        series = md.Series(data, chunk_size=3)

        for idx in index:
            series1 = series[idx]
            self.assertEqual(
                self.executor.execute_dataframe(series1, concat=True)[0], data[idx])

        selected = ['i1', 'i2', 'i3', 'i4', 'i5']
        series2 = series[selected]
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series2, concat=True)[0], data[selected])

        selected = ['i4', 'i7', 'i0', 'i1', 'i5']
        series3 = series[selected]
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series3, concat=True)[0], data[selected])

        selected = ['i0', 'i1', 'i5', 'i4', 'i0', 'i1']
        series4 = series[selected]
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series4, concat=True)[0], data[selected])

        selected = ['i0']
        series5 = series[selected]
        pd.testing.assert_series_equal(
            self.executor.execute_dataframe(series5, concat=True)[0], data[selected])

    def testHead(self):
        data = pd.DataFrame(np.random.rand(10, 5), columns=['c1', 'c2', 'c3', 'c4', 'c5'])
        df = md.DataFrame(data, chunk_size=2)

        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df.head(), concat=True)[0], data.head())
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df.head(3), concat=True)[0], data.head(3))
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df.head(-3), concat=True)[0], data.head(-3))
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df.head(8), concat=True)[0], data.head(8))
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df.head(-8), concat=True)[0], data.head(-8))
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df.head(13), concat=True)[0], data.head(13))
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df.head(-13), concat=True)[0], data.head(-13))

    def testTail(self):
        data = pd.DataFrame(np.random.rand(10, 5), columns=['c1', 'c2', 'c3', 'c4', 'c5'])
        df = md.DataFrame(data, chunk_size=2)

        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df.tail(), concat=True)[0], data.tail())
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df.tail(3), concat=True)[0], data.tail(3))
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df.tail(-3), concat=True)[0], data.tail(-3))
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df.tail(8), concat=True)[0], data.tail(8))
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df.tail(-8), concat=True)[0], data.tail(-8))
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df.tail(13), concat=True)[0], data.tail(13))
        pd.testing.assert_frame_equal(
            self.executor.execute_dataframe(df.tail(-13), concat=True)[0], data.tail(-13))

    def testAt(self):
        data = pd.DataFrame(np.random.rand(10, 5), columns=['c' + str(i) for i in range(5)],
                            index=['i' + str(i) for i in range(10)])
        df = md.DataFrame(data, chunk_size=3)
        data2 = data.copy()
        data2.index = np.arange(10)
        df2 = md.DataFrame(data2, chunk_size=3)

        with self.assertRaises(ValueError):
            _ = df.at[['i3, i4'], 'c1']

        result = self.executor.execute_dataframe(df.at['i3', 'c1'], concat=True)[0]
        self.assertEqual(result, data.at['i3', 'c1'])

        result = self.executor.execute_dataframe(df['c1'].at['i2'], concat=True)[0]
        self.assertEqual(result, data['c1'].at['i2'])

        result = self.executor.execute_dataframe(df2.at[3, 'c2'], concat=True)[0]
        self.assertEqual(result, data2.at[3, 'c2'])

        result = self.executor.execute_dataframe(df2.loc[3].at['c2'], concat=True)[0]
        self.assertEqual(result, data2.loc[3].at['c2'])

    def testIAt(self):
        data = pd.DataFrame(np.random.rand(10, 5), columns=['c' + str(i) for i in range(5)],
                            index=['i' + str(i) for i in range(10)])
        df = md.DataFrame(data, chunk_size=3)

        with self.assertRaises(ValueError):
            _ = df.iat[[1, 2], 3]

        result = self.executor.execute_dataframe(df.iat[3, 4], concat=True)[0]
        self.assertEqual(result, data.iat[3, 4])

        result = self.executor.execute_dataframe(df.iloc[:, 2].iat[3], concat=True)[0]
        self.assertEqual(result, data.iloc[:, 2].iat[3])

    def testSetitem(self):
        data = pd.DataFrame(np.random.rand(10, 5), columns=['c' + str(i) for i in range(5)],
                            index=['i' + str(i) for i in range(10)])
        data2 = np.random.rand(10)
        df = md.DataFrame(data, chunk_size=3)

        df['c3'] = df['c3'] + 1
        df['c10'] = 10
        df[4] = mt.tensor(data2, chunk_size=4)

        result = self.executor.execute_dataframe(df, concat=True)[0]
        expected = data.copy()
        expected['c3'] = expected['c3'] + 1
        expected['c10'] = 10
        expected[4] = data2
        pd.testing.assert_frame_equal(result, expected)

    def testResetIndexExecution(self):
        data = pd.DataFrame([('bird',    389.0),
                             ('bird',     24.0),
                             ('mammal',   80.5),
                             ('mammal', np.nan)],
                            index=['falcon', 'parrot', 'lion', 'monkey'],
                            columns=('class', 'max_speed'))
        df = md.DataFrame(data)
        df2 = df.reset_index()
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        expected = data.reset_index()
        pd.testing.assert_frame_equal(result, expected)

        df = md.DataFrame(data, chunk_size=2)
        df2 = df.reset_index()
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        expected = data.reset_index()
        pd.testing.assert_frame_equal(result, expected)

        df = md.DataFrame(data, chunk_size=1)
        df2 = df.reset_index(drop=True)
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        expected = data.reset_index(drop=True)
        pd.testing.assert_frame_equal(result, expected)

        index = pd.MultiIndex.from_tuples([('bird', 'falcon'),
                                           ('bird', 'parrot'),
                                           ('mammal', 'lion'),
                                           ('mammal', 'monkey')],
                                          names=['class', 'name'])
        data = pd.DataFrame([('bird',    389.0),
                             ('bird',     24.0),
                             ('mammal',   80.5),
                             ('mammal', np.nan)],
                            index=index,
                            columns=('type', 'max_speed'))
        df = md.DataFrame(data, chunk_size=1)
        df2 = df.reset_index(level='class')
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        expected = data.reset_index(level='class')
        pd.testing.assert_frame_equal(result, expected)

        columns = pd.MultiIndex.from_tuples([('speed', 'max'), ('species', 'type')])
        data.columns = columns
        df = md.DataFrame(data, chunk_size=2)
        df2 = df.reset_index(level='class', col_level=1, col_fill='species')
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        expected = data.reset_index(level='class', col_level=1, col_fill='species')
        pd.testing.assert_frame_equal(result, expected)

        # Test Series

        s = pd.Series([1, 2, 3, 4], name='foo',
                      index=pd.Index(['a', 'b', 'c', 'd'], name='idx'))

        series = md.Series(s)
        s2 = series.reset_index(name='bar')
        result = self.executor.execute_dataframe(s2, concat=True)[0]
        expected = s.reset_index(name='bar')
        pd.testing.assert_frame_equal(result, expected)

        series = md.Series(s, chunk_size=2)
        s2 = series.reset_index(drop=True)
        result = self.executor.execute_dataframe(s2, concat=True)[0]
        expected = s.reset_index(drop=True)
        pd.testing.assert_series_equal(result, expected)

        # Test Unknown shape
        sess = new_session()
        data1 = pd.DataFrame(np.random.rand(10, 3), index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9])
        df1 = md.DataFrame(data1, chunk_size=5)
        data2 = pd.DataFrame(np.random.rand(10, 3), index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3])
        df2 = md.DataFrame(data2, chunk_size=6)
        df = (df1 + df2).reset_index()
        result = sess.run(df)
        pd.testing.assert_index_equal(result.index, pd.RangeIndex(12))
        # Inconsistent with Pandas when input dataframe's shape is unknown.
        result = result.sort_values(by=result.columns[0])
        expected = (data1 + data2).reset_index()
        np.testing.assert_array_equal(result.to_numpy(), expected.to_numpy())

        data1 = pd.Series(np.random.rand(10,), index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9])
        series1 = md.Series(data1, chunk_size=3)
        data2 = pd.Series(np.random.rand(10,), index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3])
        series2 = md.Series(data2, chunk_size=3)
        df = (series1 + series2).reset_index()
        result = sess.run(df)
        pd.testing.assert_index_equal(result.index, pd.RangeIndex(12))
        # Inconsistent with Pandas when input dataframe's shape is unknown.
        result = result.sort_values(by=result.columns[0])
        expected = (data1 + data2).reset_index()
        np.testing.assert_array_equal(result.to_numpy(), expected.to_numpy())

        # case from https://github.com/mars-project/mars/issues/1286
        data = pd.DataFrame(np.random.rand(10, 3), columns=list('abc'))
        df = md.DataFrame(data, chunk_size=3)

        r = df.sort_values('a').reset_index(drop=True)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = data.sort_values('a').reset_index(drop=True)
        pd.testing.assert_frame_equal(result, expected)

    def testSetColumns(self):
        rs = np.random.RandomState(0)
        raw = pd.DataFrame(rs.rand(10, 4))

        df = md.DataFrame(raw, chunk_size=3)
        df.columns = ['a', 'b', 'c', 'd']

        result = self.executor.execute_dataframe(df, concat=True)[0]
        expected = raw.copy()
        expected.columns = ['a', 'b', 'c', 'd']
        pd.testing.assert_frame_equal(result, expected)

        with self.assertRaises(ValueError):
            df.columns = ['1', '2', '3']

    def testRename(self):
        rs = np.random.RandomState(0)
        raw = pd.DataFrame(rs.rand(10, 4), columns=['A', 'B', 'C', 'D'])
        df = md.DataFrame(raw, chunk_size=3)

        with self.assertWarns(Warning):
            df.rename(str, errors='raise')

        with self.assertRaises(NotImplementedError):
            df.rename({"A": "a", "B": "b"}, axis=1, copy=False)

        r = df.rename(str)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      raw.rename(str))

        r = df.rename({"A": "a", "B": "b"}, axis=1)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      raw.rename({"A": "a", "B": "b"}, axis=1))

        df.rename({"A": "a", "B": "b"}, axis=1, inplace=True)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(df, concat=True)[0],
                                      raw.rename({"A": "a", "B": "b"}, axis=1))

        raw = pd.DataFrame(rs.rand(10, 4),
                           columns=pd.MultiIndex.from_tuples((('A', 'C'), ('A', 'D'), ('B', 'E'), ('B', 'F'))))
        df = md.DataFrame(raw, chunk_size=3)

        r = df.rename({"C": "a", "D": "b"}, level=1, axis=1)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      raw.rename({"C": "a", "D": "b"}, level=1, axis=1))

        raw = pd.Series(rs.rand(10), name='series')
        series = md.Series(raw, chunk_size=3)

        r = series.rename('new_series')
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       raw.rename('new_series'))

        r = series.rename(lambda x: 2 ** x)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       raw.rename(lambda x: 2 ** x))

        with self.assertRaises(TypeError):
            series.name = {1: 10, 2: 20}

        series.name = 'new_series'
        pd.testing.assert_series_equal(self.executor.execute_dataframe(series, concat=True)[0],
                                       raw.rename('new_series'))

        raw = pd.MultiIndex.from_frame(pd.DataFrame(rs.rand(10, 2), columns=['A', 'B']))
        idx = md.Index(raw)

        r = idx.rename(['C', 'D'])
        pd.testing.assert_index_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      raw.rename(['C', 'D']))

    def testInsert(self):
        rs = np.random.RandomState(0)
        raw = pd.DataFrame(rs.rand(10, 4), columns=['A', 'B', 'C', 'D'])

        with self.assertRaises(ValueError):
            tensor = mt.tensor(rs.rand(10, 10), chunk_size=4)
            df = md.DataFrame(raw.copy(deep=True), chunk_size=3)
            df.insert(4, 'E', tensor)

        df = md.DataFrame(raw.copy(deep=True), chunk_size=3)
        df.insert(4, 'E', 0)
        raw_dup = raw.copy(deep=True)
        raw_dup.insert(4, 'E', 0)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(df, concat=True)[0],
                                      raw_dup)

        raw_tensor = rs.rand(10)
        tensor = mt.tensor(raw_tensor, chunk_size=4)
        df = md.DataFrame(raw.copy(deep=True), chunk_size=3)
        df.insert(4, 'E', tensor)
        raw_dup = raw.copy(deep=True)
        raw_dup.insert(4, 'E', raw_tensor)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(df, concat=True)[0],
                                      raw_dup)

    @staticmethod
    @contextlib.contextmanager
    def _inject_execute_data_source(limit, op_cls):
        def _execute_data_source(ctx, op):
            op_cls.execute(ctx, op)
            result = ctx[op.outputs[0].key]
            if len(result) > limit:
                raise RuntimeError('have data more than expected')  # pragma: no cover

        try:
            register(op_cls, _execute_data_source)
            yield
        finally:
            del Executor._op_runners[op_cls]

    def testOptimizedHeadTail(self):
        import sqlalchemy as sa

        with tempfile.TemporaryDirectory() as tempdir:
            executor = ExecutorForTest(storage=self.executor.storage)

            filename = os.path.join(tempdir, 'test_head.csv')
            rs = np.random.RandomState(0)
            pd_df = pd.DataFrame({'a': rs.randint(1000, size=(100,)).astype(np.int64),
                                  'b': rs.randint(1000, size=(100,)).astype(np.int64),
                                  'c': ['sss' for _ in range(100)],
                                  'd': ['eeee' for _ in range(100)]})
            pd_df.to_csv(filename, index=False)

            size = os.path.getsize(filename)
            chunk_bytes = size / 3

            df = md.read_csv(filename, chunk_bytes=chunk_bytes)

            # test DataFrame.head
            r = df.head(3)

            with self._inject_execute_data_source(3, DataFrameReadCSV):
                result = executor.execute_tileables([r])[0]
                expected = pd_df.head(3)
                pd.testing.assert_frame_equal(result, expected)

            # test DataFrame.tail
            r = df.tail(3)

            result = executor.execute_tileables([r])[0]
            expected = pd_df.tail(3)
            pd.testing.assert_frame_equal(result.reset_index(drop=True),
                                          expected.reset_index(drop=True))

            # test head more than 1 chunk
            r = df.head(99)

            result = executor.execute_tileables([r])[0]
            result.reset_index(drop=True, inplace=True)
            expected = pd_df.head(99)
            pd.testing.assert_frame_equal(result, expected)

            # test Series.tail more than 1 chunk
            r = df.tail(99)

            result = executor.execute_tileables([r])[0]
            expected = pd_df.tail(99)
            pd.testing.assert_frame_equal(result.reset_index(drop=True),
                                          expected.reset_index(drop=True))

            filename = os.path.join(tempdir, 'test_sql.db')
            conn = sa.create_engine('sqlite:///' + filename)
            pd_df.to_sql('test_sql', conn)

            df = md.read_sql('test_sql', conn, index_col='index', chunk_size=20)

            # test DataFrame.head
            r = df.head(3)

            with self._inject_execute_data_source(3, DataFrameReadSQL):
                result = executor.execute_tileables([r])[0]
                result.index.name = None
                expected = pd_df.head(3)
                pd.testing.assert_frame_equal(result, expected)
