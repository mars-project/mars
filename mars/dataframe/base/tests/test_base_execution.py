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

import random

import numpy as np
import pandas as pd

from mars.config import options
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

    def testResetIndexExecution(self):
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

    def testSeriesMapExecution(self):
        raw = pd.Series(np.arange(10))
        s = from_pandas_series(raw, chunk_size=7)

        with self.assertRaises(ValueError):
            # cannot infer dtype, the inferred is int,
            # but actually it is float
            # just due to nan
            s.map({5: 10})

        r = s.map({5: 10}, dtype=float)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.map({5: 10})
        pd.testing.assert_series_equal(result, expected)

        r = s.map({i: 10 + i for i in range(7)}, dtype=float)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.map({i: 10 + i for i in range(7)})
        pd.testing.assert_series_equal(result, expected)

        r = s.map({5: 10}, dtype=float, na_action='ignore')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.map({5: 10}, na_action='ignore')
        pd.testing.assert_series_equal(result, expected)

        # dtype can be inferred
        r = s.map({5: 10.})
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.map({5: 10.})
        pd.testing.assert_series_equal(result, expected)

        r = s.map(lambda x: x + 1, dtype=int)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.map(lambda x: x + 1)
        pd.testing.assert_series_equal(result, expected)

        def f(x: int) -> float:
            return x + 1.

        # dtype can be inferred for function
        r = s.map(f)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.map(lambda x: x + 1.)
        pd.testing.assert_series_equal(result, expected)

        # test arg is a md.Series
        raw2 = pd.Series([10], index=[5])
        s2 = from_pandas_series(raw2)

        r = s.map(s2, dtype=float)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.map(raw2)
        pd.testing.assert_series_equal(result, expected)

        # test arg is a md.Series, and dtype can be inferred
        raw2 = pd.Series([10.], index=[5])
        s2 = from_pandas_series(raw2)

        r = s.map(s2)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.map(raw2)
        pd.testing.assert_series_equal(result, expected)

        # test str
        raw = pd.Series(['a', 'b', 'c', 'd'])
        s = from_pandas_series(raw, chunk_size=2)

        r = s.map({'c': 'e'})
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.map({'c': 'e'})
        pd.testing.assert_series_equal(result, expected)

    def testDescribeExecution(self):
        s_raw = pd.Series(np.random.rand(10))

        # test one chunk
        series = from_pandas_series(s_raw, chunk_size=10)

        r = series.describe()
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s_raw.describe()
        pd.testing.assert_series_equal(result, expected)

        r = series.describe(percentiles=[])
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s_raw.describe(percentiles=[])
        pd.testing.assert_series_equal(result, expected)

        # test multi chunks
        series = from_pandas_series(s_raw, chunk_size=3)

        r = series.describe()
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s_raw.describe()
        pd.testing.assert_series_equal(result, expected)

        r = series.describe(percentiles=[])
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s_raw.describe(percentiles=[])
        pd.testing.assert_series_equal(result, expected)

        df_raw = pd.DataFrame(np.random.rand(10, 4), columns=list('abcd'))
        df_raw['e'] = np.random.randint(100, size=10)

        # test one chunk
        df = from_pandas_df(df_raw, chunk_size=10)

        r = df.describe()
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = df_raw.describe()
        pd.testing.assert_frame_equal(result, expected)

        r = series.describe(percentiles=[], include=np.float64)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s_raw.describe(percentiles=[], include=np.float64)
        pd.testing.assert_series_equal(result, expected)

        # test multi chunks
        df = from_pandas_df(df_raw, chunk_size=3)

        r = df.describe()
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = df_raw.describe()
        pd.testing.assert_frame_equal(result, expected)

        r = df.describe(percentiles=[], include=np.float64)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = df_raw.describe(percentiles=[], include=np.float64)
        pd.testing.assert_frame_equal(result, expected)

        with self.assertRaises(ValueError):
            df.describe(percentiles=[1.1])

    def testDataFrameFillNAExecution(self):
        df_raw = pd.DataFrame(np.nan, index=range(0, 20), columns=list('ABCDEFGHIJ'))
        for _ in range(20):
            df_raw.iloc[random.randint(0, 19), random.randint(0, 9)] = random.randint(0, 99)

        value_df_raw = pd.DataFrame(np.random.randint(0, 100, (10, 7)).astype(np.float32),
                                    columns=list('ABCDEFG'))

        # test DataFrame single chunk with numeric fill
        df = from_pandas_df(df_raw)
        r = df.fillna(1)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = df_raw.fillna(1)
        pd.testing.assert_frame_equal(result, expected)

        # test DataFrame single chunk with value as single chunk
        df = from_pandas_df(df_raw)
        value_df = from_pandas_df(value_df_raw)
        r = df.fillna(value_df)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = df_raw.fillna(value_df_raw)
        pd.testing.assert_frame_equal(result, expected)

        # test chunked with numeric fill
        df = from_pandas_df(df_raw, chunk_size=3)
        r = df.fillna(1)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = df_raw.fillna(1)
        pd.testing.assert_frame_equal(result, expected)

        # test inplace tile
        df = from_pandas_df(df_raw, chunk_size=3)
        df.fillna(1, inplace=True)
        result = self.executor.execute_dataframe(df, concat=True)[0]
        expected = df_raw.fillna(1)
        pd.testing.assert_frame_equal(result, expected)

        # test forward fill in axis=0 without limit
        df = from_pandas_df(df_raw, chunk_size=3)
        r = df.fillna(method='pad')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = df_raw.fillna(method='pad')
        pd.testing.assert_frame_equal(result, expected)

        # test backward fill in axis=0 without limit
        df = from_pandas_df(df_raw, chunk_size=3)
        r = df.fillna(method='backfill')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = df_raw.fillna(method='backfill')
        pd.testing.assert_frame_equal(result, expected)

        # test forward fill in axis=1 without limit
        df = from_pandas_df(df_raw, chunk_size=3)
        r = df.ffill(axis=1)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = df_raw.ffill(axis=1)
        pd.testing.assert_frame_equal(result, expected)

        # test backward fill in axis=1 without limit
        df = from_pandas_df(df_raw, chunk_size=3)
        r = df.bfill(axis=1)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = df_raw.bfill(axis=1)
        pd.testing.assert_frame_equal(result, expected)

        # test fill with dataframe
        df = from_pandas_df(df_raw, chunk_size=3)
        value_df = from_pandas_df(value_df_raw, chunk_size=4)
        r = df.fillna(value_df)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = df_raw.fillna(value_df_raw)
        pd.testing.assert_frame_equal(result, expected)

        # test fill with series
        value_series_raw = pd.Series(np.random.randint(0, 100, (10,)).astype(np.float32),
                                     index=list('ABCDEFGHIJ'))
        df = from_pandas_df(df_raw, chunk_size=3)
        value_series = from_pandas_series(value_series_raw, chunk_size=4)
        r = df.fillna(value_series)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = df_raw.fillna(value_series_raw)
        pd.testing.assert_frame_equal(result, expected)

    def testSeriesFillNAExecution(self):
        series_raw = pd.Series(np.nan, index=range(20))
        for _ in range(3):
            series_raw.iloc[random.randint(0, 19)] = random.randint(0, 99)
        value_series_raw = pd.Series(np.random.randint(0, 100, (10,)).astype(np.float32))

        series = from_pandas_series(series_raw)
        r = series.fillna(1)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = series_raw.fillna(1)
        pd.testing.assert_series_equal(result, expected)

        # test DataFrame single chunk with value as single chunk
        series = from_pandas_series(series_raw)
        value_series = from_pandas_series(value_series_raw)
        r = series.fillna(value_series)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = series_raw.fillna(value_series_raw)
        pd.testing.assert_series_equal(result, expected)

        # test chunked with numeric fill
        series = from_pandas_series(series_raw, chunk_size=3)
        r = series.fillna(1)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = series_raw.fillna(1)
        pd.testing.assert_series_equal(result, expected)

        # test inplace tile
        series = from_pandas_series(series_raw, chunk_size=3)
        series.fillna(1, inplace=True)
        result = self.executor.execute_dataframe(series, concat=True)[0]
        expected = series_raw.fillna(1)
        pd.testing.assert_series_equal(result, expected)

        # test forward fill in axis=0 without limit
        series = from_pandas_series(series_raw, chunk_size=3)
        r = series.fillna(method='pad')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = series_raw.fillna(method='pad')
        pd.testing.assert_series_equal(result, expected)

        # test backward fill in axis=0 without limit
        series = from_pandas_series(series_raw, chunk_size=3)
        r = series.fillna(method='backfill')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = series_raw.fillna(method='backfill')
        pd.testing.assert_series_equal(result, expected)

        # test fill with series
        series = from_pandas_series(series_raw, chunk_size=3)
        value_df = from_pandas_series(value_series_raw, chunk_size=4)
        r = series.fillna(value_df)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = series_raw.fillna(value_series_raw)
        pd.testing.assert_series_equal(result, expected)

    def testDataFrameApplyExecute(self):
        cols = [chr(ord('A') + i) for i in range(10)]
        df_raw = pd.DataFrame(dict((c, [i ** 2 for i in range(20)]) for c in cols))

        old_chunk_store_limit = options.chunk_store_limit
        try:
            options.chunk_store_limit = 20

            df = from_pandas_df(df_raw, chunk_size=5)

            r = df.apply('ffill')
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.apply('ffill')
            pd.testing.assert_frame_equal(result, expected)

            r = df.apply(np.sqrt)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.apply(np.sqrt)
            pd.testing.assert_frame_equal(result, expected)

            r = df.apply(lambda x: pd.Series([1, 2]))
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.apply(lambda x: pd.Series([1, 2]))
            pd.testing.assert_frame_equal(result, expected)

            r = df.apply(np.sum, axis='index')
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.apply(np.sum, axis='index')
            pd.testing.assert_series_equal(result, expected)

            r = df.apply(np.sum, axis='columns')
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.apply(np.sum, axis='columns')
            pd.testing.assert_series_equal(result, expected)

            r = df.apply(lambda x: [1, 2], axis=1)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.apply(lambda x: [1, 2], axis=1)
            pd.testing.assert_series_equal(result, expected)

            r = df.apply(lambda x: pd.Series([1, 2], index=['foo', 'bar']), axis=1)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.apply(lambda x: pd.Series([1, 2], index=['foo', 'bar']), axis=1)
            pd.testing.assert_frame_equal(result, expected)

            r = df.apply(lambda x: [1, 2], axis=1, result_type='expand')
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.apply(lambda x: [1, 2], axis=1, result_type='expand')
            pd.testing.assert_frame_equal(result, expected)

            r = df.apply(lambda x: list(range(10)), axis=1, result_type='reduce')
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.apply(lambda x: list(range(10)), axis=1, result_type='reduce')
            pd.testing.assert_series_equal(result, expected)

            r = df.apply(lambda x: list(range(10)), axis=1, result_type='broadcast')
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.apply(lambda x: list(range(10)), axis=1, result_type='broadcast')
            pd.testing.assert_frame_equal(result, expected)
        finally:
            options.chunk_store_limit = old_chunk_store_limit

    def testSeriesApplyExecute(self):
        idxes = [chr(ord('A') + i) for i in range(20)]
        s_raw = pd.Series([i ** 2 for i in range(20)], index=idxes)

        series = from_pandas_series(s_raw, chunk_size=5)
        r = series.apply('add', args=(1,))
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s_raw.apply('add', args=(1,))
        pd.testing.assert_series_equal(result, expected)

        r = series.apply(np.sqrt)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s_raw.apply(np.sqrt)
        pd.testing.assert_series_equal(result, expected)

        r = series.apply('sqrt')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s_raw.apply('sqrt')
        pd.testing.assert_series_equal(result, expected)

        r = series.apply(lambda x: [x, x + 1], convert_dtype=False)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s_raw.apply(lambda x: [x, x + 1], convert_dtype=False)
        pd.testing.assert_series_equal(result, expected)
