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

import numpy as np
import pandas as pd

from mars.dataframe.base import to_gpu, to_cpu, df_reset_index, series_reset_index
from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df
from mars.dataframe.datasource.series import from_pandas as from_pandas_series
from mars.session import new_session
from mars.tests.core import TestBase, require_cudf, ExecutorForTest
from mars.utils import lazy_import

cudf = lazy_import('cudf', globals=globals())


class Test(TestBase):
    def setUp(self):
        super().setUp()
        self.executor = ExecutorForTest()

    @require_cudf
    def testToGPUExecution(self):
        pdf = pd.DataFrame(np.random.rand(20, 30), index=np.arange(20, 0, -1))
        df = from_pandas_df(pdf, chunk_size=(13, 21))
        cdf = to_gpu(df)

        res = self.executor.execute_dataframe(cdf, concat=True)[0]
        self.assertIsInstance(res, cudf.DataFrame)
        pd.testing.assert_frame_equal(res.to_pandas(), pdf)

        pseries = pdf.iloc[:, 0]
        series = from_pandas_series(pseries)
        cseries = series.to_gpu()

        res = self.executor.execute_dataframe(cseries, concat=True)[0]
        self.assertIsInstance(res, cudf.Series)
        pd.testing.assert_series_equal(res.to_pandas(), pseries)

    @require_cudf
    def testToCPUExecution(self):
        pdf = pd.DataFrame(np.random.rand(20, 30), index=np.arange(20, 0, -1))
        df = from_pandas_df(pdf, chunk_size=(13, 21))
        cdf = to_gpu(df)
        df2 = to_cpu(cdf)

        res = self.executor.execute_dataframe(df2, concat=True)[0]
        self.assertIsInstance(res, pd.DataFrame)
        pd.testing.assert_frame_equal(res, pdf)

        pseries = pdf.iloc[:, 0]
        series = from_pandas_series(pseries, chunk_size=(13, 21))
        cseries = to_gpu(series)
        series2 = to_cpu(cseries)

        res = self.executor.execute_dataframe(series2, concat=True)[0]
        self.assertIsInstance(res, pd.Series)
        pd.testing.assert_series_equal(res, pseries)

    def testRechunkExecution(self):
        data = pd.DataFrame(np.random.rand(8, 10))
        df = from_pandas_df(pd.DataFrame(data), chunk_size=3)
        df2 = df.rechunk((3, 4))
        res = self.executor.execute_dataframe(df2, concat=True)[0]
        pd.testing.assert_frame_equal(data, res)

        data = pd.DataFrame(np.random.rand(10, 10), index=np.random.randint(-100, 100, size=(10,)),
                            columns=[np.random.bytes(10) for _ in range(10)])
        df = from_pandas_df(data)
        df2 = df.rechunk(5)
        res = self.executor.execute_dataframe(df2, concat=True)[0]
        pd.testing.assert_frame_equal(data, res)

        # test Series rechunk execution.
        data = pd.Series(np.random.rand(10,))
        series = from_pandas_series(data)
        series2 = series.rechunk(3)
        res = self.executor.execute_dataframe(series2, concat=True)[0]
        pd.testing.assert_series_equal(data, res)

        series2 = series.rechunk(1)
        res = self.executor.execute_dataframe(series2, concat=True)[0]
        pd.testing.assert_series_equal(data, res)

    def testResetIndex(self):
        data = pd.DataFrame([('bird',    389.0),
                             ('bird',     24.0),
                             ('mammal',   80.5),
                             ('mammal', np.nan)],
                            index=['falcon', 'parrot', 'lion', 'monkey'],
                            columns=('class', 'max_speed'))
        df = from_pandas_df(data)
        df2 = df_reset_index(df)
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        expected = data.reset_index()
        pd.testing.assert_frame_equal(result, expected)

        df = from_pandas_df(data, chunk_size=2)
        df2 = df_reset_index(df)
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        expected = data.reset_index()
        pd.testing.assert_frame_equal(result, expected)

        df = from_pandas_df(data, chunk_size=1)
        df2 = df_reset_index(df, drop=True)
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
        df = from_pandas_df(data, chunk_size=1)
        df2 = df_reset_index(df, level='class')
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        expected = data.reset_index(level='class')
        pd.testing.assert_frame_equal(result, expected)

        columns = pd.MultiIndex.from_tuples([('speed', 'max'), ('species', 'type')])
        df = from_pandas_df(data, chunk_size=2)
        df2 = df_reset_index(df, level='class', col_level=1, col_fill='species')
        data.columns = columns
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        expected = data.reset_index(level='class', col_level=1, col_fill='species')
        pd.testing.assert_frame_equal(result, expected)

        # Test Series

        s = pd.Series([1, 2, 3, 4], name='foo',
                      index=pd.Index(['a', 'b', 'c', 'd'], name='idx'))

        series = from_pandas_series(s)
        s2 = series_reset_index(series, name='bar')
        result = self.executor.execute_dataframe(s2, concat=True)[0]
        expected = s.reset_index(name='bar')
        pd.testing.assert_frame_equal(result, expected)

        series = from_pandas_series(s, chunk_size=2)
        s2 = series_reset_index(series, drop=True)
        result = self.executor.execute_dataframe(s2, concat=True)[0]
        expected = s.reset_index(drop=True)
        pd.testing.assert_series_equal(result, expected)

        # Test Unknown shape
        sess = new_session()
        data1 = pd.DataFrame(np.random.rand(10, 3), index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9])
        df1 = from_pandas_df(data1, chunk_size=5)
        data2 = pd.DataFrame(np.random.rand(10, 3), index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3])
        df2 = from_pandas_df(data2, chunk_size=6)
        df = (df1 + df2).reset_index()
        result = sess.run(df)
        pd.testing.assert_index_equal(result.index, pd.RangeIndex(12))
        # Inconsistent with Pandas when input dataframe's shape is unknown.
        result = result.sort_values(by=result.columns[0])
        expected = (data1 + data2).reset_index()
        np.testing.assert_array_equal(result.to_numpy(), expected.to_numpy())

        data1 = pd.Series(np.random.rand(10,), index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9])
        series1 = from_pandas_series(data1, chunk_size=3)
        data2 = pd.Series(np.random.rand(10,), index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3])
        series2 = from_pandas_series(data2, chunk_size=3)
        df = (series1 + series2).reset_index()
        result = sess.run(df)
        pd.testing.assert_index_equal(result.index, pd.RangeIndex(12))
        # Inconsistent with Pandas when input dataframe's shape is unknown.
        result = result.sort_values(by=result.columns[0])
        expected = (data1 + data2).reset_index()
        np.testing.assert_array_equal(result.to_numpy(), expected.to_numpy())
