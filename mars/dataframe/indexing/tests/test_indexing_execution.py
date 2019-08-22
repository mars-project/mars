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

import numpy as np
import pandas as pd

import mars.dataframe as md
from mars.executor import Executor
from mars.tests.core import TestBase
from mars.dataframe.datasource.dataframe import from_pandas


class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.executor = Executor()

    def testSetIndex(self):
        df1 = pd.DataFrame([[1, 3, 3], [4, 2, 6], [7, 8, 9]],
                           index=['a1', 'a2', 'a3'], columns=['x', 'y', 'z'])
        df2 = from_pandas(df1, chunk_size=2)

        expected = df1.set_index('y', drop=True)
        df3 = df2.set_index('y', drop=True)
        result = self.executor.execute_dataframe(df3, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        expected = df1.set_index('y', drop=False)
        df4 = df2.set_index('y', drop=False)
        result = self.executor.execute_dataframe(df4, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

    def testILocGetItem(self):
        df1 = pd.DataFrame([[1, 3, 3], [4, 2, 6], [7, 8, 9]],
                           index=['a1', 'a2', 'a3'], columns=['x', 'y', 'z'])
        df2 = from_pandas(df1, chunk_size=2)

        # plain index
        expected = df1.iloc[1]
        df3 = df2.iloc[1]
        result = self.executor.execute_dataframe(df3, concat=True)[0]
        pd.testing.assert_series_equal(expected, result)

        # slice index
        expected = df1.iloc[:, 2:4]
        df4 = df2.iloc[:, 2:4]
        result = self.executor.execute_dataframe(df4, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # plain fancy index
        expected = df1.iloc[[0], [0, 1, 2]]
        df5 = df2.iloc[[0], [0, 1, 2]]
        result = self.executor.execute_dataframe(df5, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # fancy index
        expected = df1.iloc[[1, 2], [0, 1, 2]]
        df6 = df2.iloc[[1, 2], [0, 1, 2]]
        result = self.executor.execute_dataframe(df6, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # plain index
        expected = df1.iloc[1, 2]
        df7 = df2.iloc[1, 2]
        result = self.executor.execute_dataframe(df7, concat=True)[0]
        self.assertEqual(expected, result)

    def testILocSetItem(self):
        df1 = pd.DataFrame([[1,3,3], [4,2,6], [7, 8, 9]],
                            index=['a1', 'a2', 'a3'], columns=['x', 'y', 'z'])
        df2 = from_pandas(df1, chunk_size=2)

        # plain index
        expected = df1
        expected.iloc[1] = 100
        df2.iloc[1] = 100
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # slice index
        expected.iloc[:, 2:4] = 1111
        df2.iloc[:, 2:4] = 1111
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # plain fancy index
        expected.iloc[[0], [0, 1, 2]] = 2222
        df2.iloc[[0], [0, 1, 2]] = 2222
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # fancy index
        expected.iloc[[1, 2], [0, 1, 2]] = 3333
        df2.iloc[[1, 2], [0, 1, 2]] = 3333
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # plain index
        expected.iloc[1, 2] = 4444
        df2.iloc[1, 2] = 4444
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

    def testDataFrameGetitem(self):
        data = pd.DataFrame(np.random.rand(10, 5), columns=['c1', 'c2', 'c3', 'c4', 'c5'])
        df = md.DataFrame(data, chunk_size=2)

        series1 = df['c2']
        pd.testing.assert_series_equal(series1.execute(), data['c2'])

        series2 = df['c5']
        pd.testing.assert_series_equal(series2.execute(), data['c5'])

        df1 = df[['c1', 'c2', 'c3']]
        pd.testing.assert_frame_equal(df1.execute(), data[['c1', 'c2', 'c3']])

        df2 = df[['c3', 'c2', 'c1']]
        pd.testing.assert_frame_equal(df2.execute(), data[['c3', 'c2', 'c1']])

        df3 = df[['c1']]
        pd.testing.assert_frame_equal(df3.execute(), data[['c1']])

        df4 = df[['c3', 'c1', 'c2', 'c1']]
        pd.testing.assert_frame_equal(df4.execute(), data[['c3', 'c1', 'c2', 'c1']])

        series3 = df['c1'][0]
        self.assertEqual(series3.execute(), data['c1'][0])

    def testSeriesGetitem(self):
        data = pd.Series(np.random.rand(10))
        series = md.Series(data)
        self.assertEqual(series[1].execute(), data[1])

        data = pd.Series(np.random.rand(10), name='a')
        series = md.Series(data, chunk_size=4)

        for i in range(10):
            series1 = series[i]
            self.assertEqual(series1.execute(), data[i])

        series2 = series[[0, 1, 2, 3, 4]]
        pd.testing.assert_series_equal(series2.execute(), data[[0, 1, 2, 3, 4]])

        series3 = series[[4, 3, 2, 1, 0]]
        pd.testing.assert_series_equal(series3.execute(), data[[4, 3, 2, 1, 0]])

        series4 = series[[1, 2, 3, 2, 1, 0]]
        pd.testing.assert_series_equal(series4.execute(), data[[1, 2, 3, 2, 1, 0]])
        #
        index = ['i' + str(i) for i in range(20)]
        data = pd.Series(np.random.rand(20), index=index, name='a')
        series = md.Series(data, chunk_size=3)

        for idx in index:
            series1 = series[idx]
            self.assertEqual(series1.execute(), data[idx])

        selected = ['i1', 'i2', 'i3', 'i4', 'i5']
        series2 = series[selected]
        pd.testing.assert_series_equal(series2.execute(), data[selected])

        selected = ['i4', 'i7', 'i0', 'i1', 'i5']
        series3 = series[selected]
        pd.testing.assert_series_equal(series3.execute(), data[selected])

        selected = ['i0', 'i1', 'i5', 'i4', 'i0', 'i1']
        series4 = series[selected]
        pd.testing.assert_series_equal(series4.execute(), data[selected])

        selected = ['i0']
        series5 = series[selected]
        pd.testing.assert_series_equal(series5.execute(), data[selected])
