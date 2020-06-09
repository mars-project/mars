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
from collections import OrderedDict
from distutils.version import LooseVersion

import numpy as np
import pandas as pd

from mars.config import options, option_context
from mars.dataframe.base import to_gpu, to_cpu, df_reset_index, series_reset_index, cut
from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df
from mars.dataframe.datasource.series import from_pandas as from_pandas_series
from mars.dataframe.datasource.index import from_pandas as from_pandas_index
from mars.session import new_session
from mars.tensor import tensor
from mars.tests.core import TestBase, require_cudf
from mars.utils import lazy_import

cudf = lazy_import('cudf', globals=globals())


class Test(TestBase):
    def setUp(self):
        super().setUp()
        self.ctx, self.executor = self._create_test_context()

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

        # test index rechunk execution
        data = pd.Index(np.random.rand(10,))
        index = from_pandas_index(data)
        index2 = index.rechunk(3)
        res = self.executor.execute_dataframe(index2, concat=True)[0]
        pd.testing.assert_index_equal(data, res)

        index2 = index.rechunk(1)
        res = self.executor.execute_dataframe(index2, concat=True)[0]
        pd.testing.assert_index_equal(data, res)

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
        data.columns = columns
        df = from_pandas_df(data, chunk_size=2)
        df2 = df_reset_index(df, level='class', col_level=1, col_fill='species')
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

        # test input dataframe which has unknown shape
        with self.ctx:
            df = from_pandas_df(df_raw, chunk_size=3)
            df2 = df[df['a'] < 0.5]
            r = df2.describe()

            result = self.executor.execute_tileables([r])[0]
            expected = df_raw[df_raw['a'] < 0.5].describe()
            pd.testing.assert_frame_equal(result, expected)

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

            r = df.apply(['sum', 'max'])
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.apply(['sum', 'max'])
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

        r = series.apply(['sum', 'max'])
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s_raw.apply(['sum', 'max'])
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

    def testTransformExecute(self):
        cols = [chr(ord('A') + i) for i in range(10)]
        df_raw = pd.DataFrame(dict((c, [i ** 2 for i in range(20)]) for c in cols))

        idx_vals = [chr(ord('A') + i) for i in range(20)]
        s_raw = pd.Series([i ** 2 for i in range(20)], index=idx_vals)

        def rename_fn(f, new_name):
            f.__name__ = new_name
            return f

        old_chunk_store_limit = options.chunk_store_limit
        try:
            options.chunk_store_limit = 20

            # DATAFRAME CASES
            df = from_pandas_df(df_raw, chunk_size=5)

            # test transform scenarios on data frames
            r = df.transform(lambda x: list(range(len(x))))
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.transform(lambda x: list(range(len(x))))
            pd.testing.assert_frame_equal(result, expected)

            r = df.transform(lambda x: list(range(len(x))), axis=1)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.transform(lambda x: list(range(len(x))), axis=1)
            pd.testing.assert_frame_equal(result, expected)

            r = df.transform(['cumsum', 'cummax', lambda x: x + 1])
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.transform(['cumsum', 'cummax', lambda x: x + 1])
            pd.testing.assert_frame_equal(result, expected)

            fn_dict = OrderedDict([
                ('A', 'cumsum'),
                ('D', ['cumsum', 'cummax']),
                ('F', lambda x: x + 1),
            ])
            r = df.transform(fn_dict)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.transform(fn_dict)
            pd.testing.assert_frame_equal(result, expected)

            # test agg scenarios on series
            r = df.transform(lambda x: x.iloc[:-1], _call_agg=True)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.agg(lambda x: x.iloc[:-1])
            pd.testing.assert_frame_equal(result, expected)

            r = df.transform(lambda x: x.iloc[:-1], axis=1, _call_agg=True)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.agg(lambda x: x.iloc[:-1], axis=1)
            pd.testing.assert_frame_equal(result, expected)

            fn_list = [rename_fn(lambda x: x.iloc[1:].reset_index(drop=True), 'f1'),
                       lambda x: x.iloc[:-1].reset_index(drop=True)]
            r = df.transform(fn_list, _call_agg=True)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.agg(fn_list)
            pd.testing.assert_frame_equal(result, expected)

            r = df.transform(lambda x: x.sum(), _call_agg=True)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.agg(lambda x: x.sum())
            pd.testing.assert_series_equal(result, expected)

            fn_dict = OrderedDict([
                ('A', rename_fn(lambda x: x.iloc[1:].reset_index(drop=True), 'f1')),
                ('D', [rename_fn(lambda x: x.iloc[1:].reset_index(drop=True), 'f1'),
                       lambda x: x.iloc[:-1].reset_index(drop=True)]),
                ('F', lambda x: x.iloc[:-1].reset_index(drop=True)),
            ])
            r = df.transform(fn_dict, _call_agg=True)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = df_raw.agg(fn_dict)
            pd.testing.assert_frame_equal(result, expected)

            # SERIES CASES
            series = from_pandas_series(s_raw, chunk_size=5)

            # test transform scenarios on series
            r = series.transform(lambda x: x + 1)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = s_raw.transform(lambda x: x + 1)
            pd.testing.assert_series_equal(result, expected)

            r = series.transform(['cumsum', lambda x: x + 1])
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = s_raw.transform(['cumsum', lambda x: x + 1])
            pd.testing.assert_frame_equal(result, expected)
        finally:
            options.chunk_store_limit = old_chunk_store_limit

    def testStringMethodExecution(self):
        s = pd.Series(['s1,s2', 'ef,', 'dd', np.nan])
        s2 = pd.concat([s, s, s])

        series = from_pandas_series(s, chunk_size=2)
        series2 = from_pandas_series(s2, chunk_size=2)

        # test getitem
        r = series.str[:3]
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s.str[:3]
        pd.testing.assert_series_equal(result, expected)

        # test split, expand=False
        r = series.str.split(',', n=2)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s.str.split(',', n=2)
        pd.testing.assert_series_equal(result, expected)

        # test split, expand=True
        r = series.str.split(',', expand=True, n=1)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s.str.split(',', expand=True, n=1)
        pd.testing.assert_frame_equal(result, expected)

        # test rsplit
        r = series.str.rsplit(',', expand=True, n=1)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s.str.rsplit(',', expand=True, n=1)
        pd.testing.assert_frame_equal(result, expected)

        # test cat all data
        r = series2.str.cat(sep='/', na_rep='e')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s2.str.cat(sep='/', na_rep='e')
        self.assertEqual(result, expected)

        # test cat list
        r = series.str.cat(['a', 'b', np.nan, 'c'])
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s.str.cat(['a', 'b', np.nan, 'c'])
        pd.testing.assert_series_equal(result, expected)

        # test cat series
        r = series.str.cat(series.str.capitalize(), join='outer')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s.str.cat(s.str.capitalize(), join='outer')
        pd.testing.assert_series_equal(result, expected)

        # test extractall
        r = series.str.extractall(r"(?P<letter>[ab])(?P<digit>\d)")
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s.str.extractall(r"(?P<letter>[ab])(?P<digit>\d)")
        pd.testing.assert_frame_equal(result, expected)

        # test extract, expand=False
        r = series.str.extract(r'[ab](\d)', expand=False)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s.str.extract(r'[ab](\d)', expand=False)
        pd.testing.assert_series_equal(result, expected)

        # test extract, expand=True
        r = series.str.extract(r'[ab](\d)', expand=True)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s.str.extract(r'[ab](\d)', expand=True)
        pd.testing.assert_frame_equal(result, expected)

    def testDatetimeMethodExecution(self):
        # test datetime
        s = pd.Series([pd.Timestamp('2020-1-1'),
                       pd.Timestamp('2020-2-1'),
                       np.nan])
        series = from_pandas_series(s, chunk_size=2)

        r = series.dt.year
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s.dt.year
        pd.testing.assert_series_equal(result, expected)

        r = series.dt.strftime('%m-%d-%Y')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s.dt.strftime('%m-%d-%Y')
        pd.testing.assert_series_equal(result, expected)

        # test timedelta
        s = pd.Series([pd.Timedelta('1 days'),
                       pd.Timedelta('3 days'),
                       np.nan])
        series = from_pandas_series(s, chunk_size=2)

        r = series.dt.days
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s.dt.days
        pd.testing.assert_series_equal(result, expected)

    def testSeriesIsin(self):
        # one chunk in multiple chunks
        a = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = pd.Series([2, 1, 9, 3])
        sa = from_pandas_series(a, chunk_size=10)
        sb = from_pandas_series(b, chunk_size=2)

        result = self.executor.execute_dataframe(sa.isin(sb), concat=True)[0]
        expected = a.isin(b)
        pd.testing.assert_series_equal(result, expected)

        # multiple chunk in one chunks
        a = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = pd.Series([2, 1, 9, 3])
        sa = from_pandas_series(a, chunk_size=2)
        sb = from_pandas_series(b, chunk_size=4)

        result = self.executor.execute_dataframe(sa.isin(sb), concat=True)[0]
        expected = a.isin(b)
        pd.testing.assert_series_equal(result, expected)

        # multiple chunk in multiple chunks
        a = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = pd.Series([2, 1, 9, 3])
        sa = from_pandas_series(a, chunk_size=2)
        sb = from_pandas_series(b, chunk_size=2)

        result = self.executor.execute_dataframe(sa.isin(sb), concat=True)[0]
        expected = a.isin(b)
        pd.testing.assert_series_equal(result, expected)

        a = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = pd.Series([2, 1, 9, 3])
        sa = from_pandas_series(a, chunk_size=2)

        result = self.executor.execute_dataframe(sa.isin(b), concat=True)[0]
        expected = a.isin(b)
        pd.testing.assert_series_equal(result, expected)

        a = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = np.array([2, 1, 9, 3])
        sa = from_pandas_series(a, chunk_size=2)
        sb = tensor(b, chunk_size=3)

        result = self.executor.execute_dataframe(sa.isin(sb), concat=True)[0]
        expected = a.isin(b)
        pd.testing.assert_series_equal(result, expected)

        a = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = {2, 1, 9, 3}  # set
        sa = from_pandas_series(a, chunk_size=2)

        result = self.executor.execute_dataframe(sa.isin(b), concat=True)[0]
        expected = a.isin(b)
        pd.testing.assert_series_equal(result, expected)

    def testCheckNA(self):
        df_raw = pd.DataFrame(np.nan, index=range(0, 20), columns=list('ABCDEFGHIJ'))
        for _ in range(20):
            df_raw.iloc[random.randint(0, 19), random.randint(0, 9)] = random.randint(0, 99)

        df = from_pandas_df(df_raw, chunk_size=4)

        pd.testing.assert_frame_equal(self.executor.execute_dataframe(df.isna(), concat=True)[0],
                                      df_raw.isna())
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(df.notna(), concat=True)[0],
                                      df_raw.notna())

        series_raw = pd.Series(np.nan, index=range(20))
        for _ in range(3):
            series_raw.iloc[random.randint(0, 19)] = random.randint(0, 99)

        series = from_pandas_series(series_raw, chunk_size=4)

        pd.testing.assert_series_equal(self.executor.execute_dataframe(series.isna(), concat=True)[0],
                                       series_raw.isna())
        pd.testing.assert_series_equal(self.executor.execute_dataframe(series.notna(), concat=True)[0],
                                       series_raw.notna())

    def testDropNA(self):
        # dataframe cases
        df_raw = pd.DataFrame(np.nan, index=range(0, 20), columns=list('ABCDEFGHIJ'))
        for _ in range(30):
            df_raw.iloc[random.randint(0, 19), random.randint(0, 9)] = random.randint(0, 99)
        for rowid in range(random.randint(1, 5)):
            row = random.randint(0, 19)
            for idx in range(0, 10):
                df_raw.iloc[row, idx] = random.randint(0, 99)

        # only one chunk in columns, can run dropna directly
        r = from_pandas_df(df_raw, chunk_size=(4, 10)).dropna()
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.dropna())

        # multiple chunks in columns, count() will be called first
        r = from_pandas_df(df_raw, chunk_size=4).dropna()
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.dropna())

        r = from_pandas_df(df_raw, chunk_size=4).dropna(how='all')
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.dropna(how='all'))

        r = from_pandas_df(df_raw, chunk_size=4).dropna(subset=list('ABFI'))
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.dropna(subset=list('ABFI')))

        r = from_pandas_df(df_raw, chunk_size=4).dropna(how='all', subset=list('BDHJ'))
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.dropna(how='all', subset=list('BDHJ')))

        r = from_pandas_df(df_raw, chunk_size=4)
        r.dropna(how='all', inplace=True)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      df_raw.dropna(how='all'))

        # series cases
        series_raw = pd.Series(np.nan, index=range(20))
        for _ in range(10):
            series_raw.iloc[random.randint(0, 19)] = random.randint(0, 99)

        r = from_pandas_series(series_raw, chunk_size=4).dropna()
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       series_raw.dropna())

        r = from_pandas_series(series_raw, chunk_size=4)
        r.dropna(inplace=True)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       series_raw.dropna())

    def testCutExecution(self):
        rs = np.random.RandomState(0)
        raw = rs.random(15) * 1000
        s = pd.Series(raw, index=['i{}'.format(i) for i in range(15)])
        bins = [10, 100, 500]
        ii = pd.interval_range(10, 500, 3)
        labels = ['a', 'b']

        t = tensor(raw, chunk_size=4)
        series = from_pandas_series(s, chunk_size=4)
        iii = from_pandas_index(ii, chunk_size=2)

        # cut on Series
        r = cut(series, bins)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_series_equal(result, pd.cut(s, bins))

        r, b = cut(series, bins, retbins=True)
        r_result = self.executor.execute_dataframe(r, concat=True)[0]
        b_result = self.executor.execute_tensor(b, concat=True)[0]
        r_expected, b_expected = pd.cut(s, bins, retbins=True)
        pd.testing.assert_series_equal(r_result, r_expected)
        np.testing.assert_array_equal(b_result, b_expected)

        # cut on tensor
        r = cut(t, bins)
        # result and expected is array whose dtype is CategoricalDtype
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = pd.cut(raw, bins)
        self.assertEqual(len(result), len(expected))
        for r, e in zip(result, expected):
            np.testing.assert_equal(r, e)

        # one chunk
        r = cut(s, tensor(bins, chunk_size=2), right=False, include_lowest=True)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_series_equal(result, pd.cut(s, bins, right=False, include_lowest=True))

        # test labels
        r = cut(t, bins, labels=labels)
        # result and expected is array whose dtype is CategoricalDtype
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = pd.cut(raw, bins, labels=labels)
        self.assertEqual(len(result), len(expected))
        for r, e in zip(result, expected):
            np.testing.assert_equal(r, e)

        r = cut(t, bins, labels=False)
        # result and expected is array whose dtype is CategoricalDtype
        result = self.executor.execute_tensor(r, concat=True)[0]
        expected = pd.cut(raw, bins, labels=False)
        np.testing.assert_array_equal(result, expected)

        # test labels which is tensor
        labels_t = tensor(['a', 'b'], chunk_size=1)
        r = cut(raw, bins, labels=labels_t, include_lowest=True)
        # result and expected is array whose dtype is CategoricalDtype
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = pd.cut(raw, bins, labels=labels, include_lowest=True)
        self.assertEqual(len(result), len(expected))
        for r, e in zip(result, expected):
            np.testing.assert_equal(r, e)

        # test labels=False
        r, b = cut(raw, ii, labels=False, retbins=True)
        # result and expected is array whose dtype is CategoricalDtype
        r_result = self.executor.execute_tileable(r, concat=True)[0]
        b_result = self.executor.execute_tileable(b, concat=True)[0]
        r_expected, b_expected = pd.cut(raw, ii, labels=False, retbins=True)
        for r, e in zip(r_result, r_expected):
            np.testing.assert_equal(r, e)
        pd.testing.assert_index_equal(b_result, b_expected)

        # test bins which is md.IntervalIndex
        r, b = cut(series, iii, labels=tensor(labels, chunk_size=1), retbins=True)
        r_result = self.executor.execute_dataframe(r, concat=True)[0]
        b_result = self.executor.execute_dataframe(b, concat=True)[0]
        r_expected, b_expected = pd.cut(s, ii, labels=labels, retbins=True)
        pd.testing.assert_series_equal(r_result, r_expected)
        pd.testing.assert_index_equal(b_result, b_expected)

        # test duplicates
        bins2 = [0, 2, 4, 6, 10, 10]
        r, b = cut(s, bins2, labels=False, retbins=True,
                   right=False, duplicates='drop')
        r_result = self.executor.execute_dataframe(r, concat=True)[0]
        b_result = self.executor.execute_tensor(b, concat=True)[0]
        r_expected, b_expected = pd.cut(s, bins2, labels=False, retbins=True,
                                        right=False, duplicates='drop')
        pd.testing.assert_series_equal(r_result, r_expected)
        np.testing.assert_array_equal(b_result, b_expected)

        ctx, executor = self._create_test_context(self.executor)
        with ctx:
            # test integer bins
            r = cut(series, 3)
            result = executor.execute_dataframes([r])[0]
            pd.testing.assert_series_equal(result, pd.cut(s, 3))

            r, b = cut(series, 3, right=False, retbins=True)
            r_result, b_result = executor.execute_dataframes([r, b])
            r_expected, b_expected = pd.cut(s, 3, right=False, retbins=True)
            pd.testing.assert_series_equal(r_result, r_expected)
            np.testing.assert_array_equal(b_result, b_expected)

            # test min max same
            s2 = pd.Series([1.1] * 15)
            r = cut(s2, 3)
            result = executor.execute_dataframes([r])[0]
            pd.testing.assert_series_equal(result, pd.cut(s2, 3))

            # test inf exist
            s3 = s2.copy()
            s3[-1] = np.inf
            with self.assertRaises(ValueError):
                executor.execute_dataframes([cut(s3, 3)])

    def testShiftExecution(self):
        # test dataframe
        rs = np.random.RandomState(0)
        raw = pd.DataFrame(rs.randint(1000, size=(10, 8)),
                           columns=['col' + str(i + 1) for i in range(8)])

        df = from_pandas_df(raw, chunk_size=5)

        for periods in (2, -2, 6, -6):
            for axis in (0, 1):
                for fill_value in (None, 0, 1.):
                    r = df.shift(periods=periods, axis=axis,
                                 fill_value=fill_value)

                    try:
                        result = self.executor.execute_dataframe(r, concat=True)[0]
                        expected = raw.shift(periods=periods, axis=axis,
                                             fill_value=fill_value)
                        pd.testing.assert_frame_equal(result, expected)
                    except AssertionError as e:  # pragma: no cover
                        raise AssertionError(
                            'Failed when periods: {}, axis: {}, fill_value: {}'.format(
                                periods, axis, fill_value
                            )) from e

        raw2 = raw.copy()
        raw2.index = pd.date_range('2020-1-1', periods=10)
        raw2.columns = pd.date_range('2020-3-1', periods=8)

        df2 = from_pandas_df(raw2, chunk_size=5)

        # test freq not None
        for periods in (2, -2):
            for axis in (0, 1):
                for fill_value in (None, 0, 1.):
                    r = df2.shift(periods=periods, freq='D', axis=axis,
                                  fill_value=fill_value)

                    try:
                        result = self.executor.execute_dataframe(r, concat=True)[0]
                        expected = raw2.shift(periods=periods, freq='D', axis=axis,
                                              fill_value=fill_value)
                        pd.testing.assert_frame_equal(result, expected)
                    except AssertionError as e:  # pragma: no cover
                        raise AssertionError(
                            'Failed when periods: {}, axis: {}, fill_value: {}'.format(
                                periods, axis, fill_value
                            )) from e

        # test tshift
        r = df2.tshift(periods=1)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw2.tshift(periods=1)
        pd.testing.assert_frame_equal(result, expected)

        with self.assertRaises(ValueError):
            _ = df.tshift(periods=1)

        # test series
        s = raw.iloc[:, 0]

        series = from_pandas_series(s, chunk_size=5)
        for periods in (0, 2, -2, 6, -6):
            for fill_value in (None, 0, 1.):
                r = series.shift(periods=periods, fill_value=fill_value)

                try:
                    result = self.executor.execute_dataframe(r, concat=True)[0]
                    expected = s.shift(periods=periods, fill_value=fill_value)
                    pd.testing.assert_series_equal(result, expected)
                except AssertionError as e:  # pragma: no cover
                    raise AssertionError(
                        'Failed when periods: {}, fill_value: {}'.format(
                            periods, fill_value
                        )) from e

        s2 = raw2.iloc[:, 0]

        # test freq not None
        series2 = from_pandas_series(s2, chunk_size=5)
        for periods in (2, -2):
            for fill_value in (None, 0, 1.):
                r = series2.shift(periods=periods, freq='D', fill_value=fill_value)

                try:
                    result = self.executor.execute_dataframe(r, concat=True)[0]
                    expected = s2.shift(periods=periods, freq='D', fill_value=fill_value)
                    pd.testing.assert_series_equal(result, expected)
                except AssertionError as e:  # pragma: no cover
                    raise AssertionError(
                        'Failed when periods: {}, fill_value: {}'.format(
                            periods, fill_value
                        )) from e

    def testDiffExecution(self):
        rs = np.random.RandomState(0)
        raw = pd.DataFrame(rs.randint(1000, size=(10, 8)),
                           columns=['col' + str(i + 1) for i in range(8)])

        raw1 = raw.copy()
        if LooseVersion(pd.__version__) >= '1.0.0':
            raw1['col4'] = raw1['col4'] < 400

        r = from_pandas_df(raw1, chunk_size=(10, 5)).diff(-1)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      raw1.diff(-1))

        r = from_pandas_df(raw1, chunk_size=5).diff(-1)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      raw1.diff(-1))

        r = from_pandas_df(raw, chunk_size=(5, 8)).diff(1, axis=1)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      raw.diff(1, axis=1))

        r = from_pandas_df(raw, chunk_size=5).diff(1, axis=1)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      raw.diff(1, axis=1))

        # test series
        s = raw.iloc[:, 0]
        s1 = s.copy()
        if LooseVersion(pd.__version__) >= '1.0.0':
            s1 = s1 < 400

        r = from_pandas_series(s, chunk_size=10).diff(-1)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       s.diff(-1))

        r = from_pandas_series(s, chunk_size=5).diff(-1)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       s.diff(-1))

        r = from_pandas_series(s1, chunk_size=5).diff(1)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       s1.diff(1))

    def testValueCountsExecution(self):
        rs = np.random.RandomState(0)
        s = pd.Series(rs.randint(5, size=100))
        s[rs.randint(100)] = np.nan

        ctx, executor = self._create_test_context(self.executor)

        # test 1 chunk
        series = from_pandas_series(s, chunk_size=100)

        r = series.value_counts()
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       s.value_counts())

        r = series.value_counts(bins=5, normalize=True)
        with ctx:
            pd.testing.assert_series_equal(executor.execute_dataframes([r])[0],
                                           s.value_counts(bins=5, normalize=True))

        # test multi chunks
        series = from_pandas_series(s, chunk_size=30)

        r = series.value_counts()
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       s.value_counts())

        r = series.value_counts(normalize=True)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       s.value_counts(normalize=True))

        with ctx:
            # test bins and normalize
            r = series.value_counts(bins=5, normalize=True)
            pd.testing.assert_series_equal(executor.execute_dataframes([r])[0],
                                           s.value_counts(bins=5, normalize=True))

    def testAsType(self):
        rs = np.random.RandomState(0)
        raw = pd.DataFrame(rs.randint(1000, size=(20, 8)),
                           columns=['c' + str(i + 1) for i in range(8)])
        # single chunk
        df = from_pandas_df(raw)
        r = df.astype('int32')

        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.astype('int32')
        pd.testing.assert_frame_equal(expected, result)

        # multiply chunks
        df = from_pandas_df(raw, chunk_size=6)
        r = df.astype('int32')

        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.astype('int32')
        pd.testing.assert_frame_equal(expected, result)

        # dict type
        df = from_pandas_df(raw, chunk_size=5)
        r = df.astype({'c1': 'int32', 'c2': 'float', 'c8': 'str'})

        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.astype({'c1': 'int32', 'c2': 'float', 'c8': 'str'})
        pd.testing.assert_frame_equal(expected, result)

        # test series
        s = pd.Series(rs.randint(5, size=20))
        series = from_pandas_series(s)
        r = series.astype('int32')

        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s.astype('int32')
        pd.testing.assert_series_equal(result, expected)

        # multiply chunks
        series = from_pandas_series(s, chunk_size=6)
        r = series.astype('str')

        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s.astype('str')
        pd.testing.assert_series_equal(result, expected)

        # test category
        raw = pd.DataFrame(rs.randint(3, size=(20, 8)),
                           columns=['c' + str(i + 1) for i in range(8)])

        df = from_pandas_df(raw)
        r = df.astype('category')

        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.astype('category')
        pd.testing.assert_frame_equal(expected, result)

        df = from_pandas_df(raw)
        r = df.astype({'c1': 'category', 'c8': 'int32', 'c4': 'str'})

        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.astype({'c1': 'category', 'c8': 'int32', 'c4': 'str'})
        pd.testing.assert_frame_equal(expected, result)

        df = from_pandas_df(raw, chunk_size=5)
        r = df.astype('category')

        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.astype('category')
        pd.testing.assert_frame_equal(expected, result)

        df = from_pandas_df(raw, chunk_size=3)
        r = df.astype({'c1': 'category', 'c8': 'int32', 'c4': 'str'})

        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.astype({'c1': 'category', 'c8': 'int32', 'c4': 'str'})
        pd.testing.assert_frame_equal(expected, result)

        df = from_pandas_df(raw, chunk_size=6)
        r = df.astype({'c1': 'category', 'c5': 'float', 'c2': 'int32',
                       'c7': pd.CategoricalDtype([1, 3, 4, 2]),
                       'c4': pd.CategoricalDtype([1, 3, 2])})
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.astype({'c1': 'category', 'c5': 'float', 'c2': 'int32',
                               'c7': pd.CategoricalDtype([1, 3, 4, 2]),
                               'c4': pd.CategoricalDtype([1, 3, 2])})
        pd.testing.assert_frame_equal(expected, result)

        df = from_pandas_df(raw, chunk_size=8)
        r = df.astype({'c2': 'category'})
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.astype({'c2': 'category'})
        pd.testing.assert_frame_equal(expected, result)

        # test series category
        raw = pd.Series(np.random.choice(['a', 'b', 'c'], size=(10,)))
        series = from_pandas_series(raw, chunk_size=4)
        result = self.executor.execute_dataframe(series.astype('category'), concat=True)[0]
        expected = raw.astype('category')
        pd.testing.assert_series_equal(expected, result)

        series = from_pandas_series(raw, chunk_size=3)
        result = self.executor.execute_dataframe(
            series.astype(pd.CategoricalDtype(['a', 'c', 'b']), copy=False), concat=True)[0]
        expected = raw.astype(pd.CategoricalDtype(['a', 'c', 'b']),  copy=False)
        pd.testing.assert_series_equal(expected, result)

        series = from_pandas_series(raw, chunk_size=6)
        result = self.executor.execute_dataframe(
            series.astype(pd.CategoricalDtype(['a', 'c', 'b', 'd'])), concat=True)[0]
        expected = raw.astype(pd.CategoricalDtype(['a', 'c', 'b', 'd']))
        pd.testing.assert_series_equal(expected, result)

    def testDrop(self):
        # test dataframe drop
        rs = np.random.RandomState(0)
        raw = pd.DataFrame(rs.randint(1000, size=(20, 8)),
                           columns=['c' + str(i + 1) for i in range(8)])

        df = from_pandas_df(raw, chunk_size=3)

        columns = ['c2', 'c4', 'c5', 'c6']
        index = [3, 6, 7]
        r = df.drop(columns=columns, index=index)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      raw.drop(columns=columns, index=index))

        idx_series = from_pandas_series(pd.Series(index))
        r = df.drop(idx_series)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      raw.drop(pd.Series(index)))

        df.drop(columns, axis=1, inplace=True)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(df, concat=True)[0],
                                      raw.drop(columns, axis=1))

        del df['c3']
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(df, concat=True)[0],
                                      raw.drop(columns + ['c3'], axis=1))

        ps = df.pop('c8')
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(df, concat=True)[0],
                                      raw.drop(columns + ['c3', 'c8'], axis=1))
        pd.testing.assert_series_equal(self.executor.execute_dataframe(ps, concat=True)[0],
                                       raw['c8'])

        # test series drop
        raw = pd.Series(rs.randint(1000, size=(20,)))

        series = from_pandas_series(raw, chunk_size=3)

        r = series.drop(index=index)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       raw.drop(index=index))

        # test index drop
        ser = pd.Series(range(20))
        rs.shuffle(ser)
        raw = pd.Index(ser)

        idx = from_pandas_index(raw)

        r = idx.drop(index)
        pd.testing.assert_index_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                      raw.drop(index))

    def testDropDuplicates(self):
        # test dataframe drop
        rs = np.random.RandomState(0)
        raw = pd.DataFrame(rs.randint(1000, size=(20, 5)),
                           columns=['c' + str(i + 1) for i in range(5)],
                           index=['i' + str(j) for j in range(20)])
        duplicate_lines = rs.randint(1000, size=5)
        for i in [1, 3, 10, 11, 15]:
            raw.iloc[i] = duplicate_lines

        with option_context({'combine_size': 2}):
            # test dataframe
            for chunk_size in [(8, 3), (20, 5)]:
                df = from_pandas_df(raw, chunk_size=chunk_size)
                if chunk_size[0] < len(raw):
                    methods = ['tree', 'subset_tree', 'shuffle']
                else:
                    # 1 chunk
                    methods = [None]
                for method in methods:
                    for subset in [None, 'c1', ['c1', 'c2']]:
                        for keep in ['first', 'last', False]:
                            for ignore_index in [True, False]:
                                try:
                                    r = df.drop_duplicates(method=method, subset=subset,
                                                           keep=keep, ignore_index=ignore_index)
                                    result = self.executor.execute_dataframe(r, concat=True)[0]
                                    try:
                                        expected = raw.drop_duplicates(subset=subset,
                                                                       keep=keep, ignore_index=ignore_index)
                                    except TypeError:
                                        # ignore_index is supported in pandas 1.0
                                        expected = raw.drop_duplicates(subset=subset,
                                                                       keep=keep)
                                        if ignore_index:
                                            expected.reset_index(drop=True, inplace=True)

                                    pd.testing.assert_frame_equal(result, expected)
                                except Exception as e:  # pragma: no cover
                                    raise AssertionError('failed when method={}, subset={}, '
                                                         'keep={}, ignore_index={}'.format(
                                        method, subset, keep, ignore_index)) from e

            # test series and index
            s = raw['c3']
            ind = pd.Index(s)

            for tp, obj in [('series', s), ('index', ind)]:
                for chunk_size in [8, 20]:
                    to_m = from_pandas_series if tp == 'series' else from_pandas_index
                    mobj = to_m(obj, chunk_size=chunk_size)
                    if chunk_size < len(obj):
                        methods = ['tree', 'shuffle']
                    else:
                        # 1 chunk
                        methods = [None]
                    for method in methods:
                        for keep in ['first', 'last', False]:
                            try:
                                r = mobj.drop_duplicates(method=method, keep=keep)
                                result = self.executor.execute_dataframe(r, concat=True)[0]
                                expected = obj.drop_duplicates(keep=keep)

                                cmp = pd.testing.assert_series_equal \
                                    if tp == 'series' else pd.testing.assert_index_equal
                                cmp(result, expected)
                            except Exception as e:  # pragma: no cover
                                raise AssertionError('failed when method={}, keep={}'.format(
                                    method, keep)) from e

            # test inplace
            series = from_pandas_series(s, chunk_size=11)
            series.drop_duplicates(inplace=True)
            result = self.executor.execute_dataframe(series, concat=True)[0]
            expected = s.drop_duplicates()
            pd.testing.assert_series_equal(result, expected)
