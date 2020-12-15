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

import unittest

import pandas as pd
import numpy as np
try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None

import mars.dataframe as md
from mars.config import option_context
from mars.dataframe import CustomReduction, NamedAgg
from mars.dataframe.base import to_gpu
from mars.tests.core import TestBase, parameterized, ExecutorForTest, \
    require_cudf, require_cupy
from mars.utils import lazy_import

cp = lazy_import('cupy', rename='cp', globals=globals())


reduction_functions = dict(
    sum=dict(func_name='sum', has_min_count=True),
    prod=dict(func_name='prod', has_min_count=True),
    min=dict(func_name='min', has_min_count=False),
    max=dict(func_name='max', has_min_count=False),
    mean=dict(func_name='mean', has_min_count=False),
    var=dict(func_name='var', has_min_count=False),
    std=dict(func_name='std', has_min_count=False),
    sem=dict(func_name='sem', has_min_count=False),
    skew=dict(func_name='skew', has_min_count=False),
    kurt=dict(func_name='kurt', has_min_count=False),
)


@parameterized(**reduction_functions)
class TestReduction(TestBase):
    def setUp(self):
        self.executor = ExecutorForTest()

    def compute(self, data, **kwargs):
        return getattr(data, self.func_name)(**kwargs)

    def testSeriesReduction(self):
        data = pd.Series(np.random.randint(0, 8, (10,)), index=[str(i) for i in range(10)], name='a')
        r = self.compute(md.Series(data))
        self.assertAlmostEqual(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.Series(data, chunk_size=6))
        self.assertAlmostEqual(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.Series(data, chunk_size=3))
        self.assertAlmostEqual(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.Series(data, chunk_size=4), axis='index')
        self.assertAlmostEqual(
            self.compute(data, axis='index'), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.Series(data, chunk_size=4), axis='index')
        self.assertAlmostEqual(
            self.compute(data, axis='index'), self.executor.execute_dataframe(r, concat=True)[0])

        data = pd.Series(np.random.rand(20), name='a')
        data[0] = 0.1  # make sure not all elements are NAN
        data[data > 0.5] = np.nan
        r = self.compute(md.Series(data, chunk_size=3))
        self.assertAlmostEqual(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.Series(data, chunk_size=3), skipna=False)
        self.assertTrue(
            np.isnan(self.executor.execute_dataframe(r, concat=True)[0]))

        if self.has_min_count:
            r = self.compute(md.Series(data, chunk_size=3), skipna=False, min_count=2)
            self.assertTrue(
                np.isnan(self.executor.execute_dataframe(r, concat=True)[0]))

            r = self.compute(md.Series(data, chunk_size=3), min_count=1)
            self.assertAlmostEqual(
                self.compute(data, min_count=1),
                self.executor.execute_dataframe(r, concat=True)[0])

            reduction_df5 = self.compute(md.Series(data, chunk_size=3), min_count=21)
            self.assertTrue(
                np.isnan(self.executor.execute_dataframe(reduction_df5, concat=True)[0]))

    def testSeriesLevelReduction(self):
        rs = np.random.RandomState(0)
        idx = pd.MultiIndex.from_arrays([
            [str(i) for i in range(100)], rs.choice(['A', 'B'], size=(100,))
        ], names=['a', 'b'])
        data = pd.Series(rs.randint(0, 8, size=(100,)), index=idx)

        r = self.compute(md.Series(data, chunk_size=13), level=1)
        pd.testing.assert_series_equal(
            self.compute(data, level=1).sort_index(),
            self.executor.execute_dataframe(r, concat=True)[0].sort_index())

        # test null
        data = pd.Series(rs.rand(100), name='a', index=idx)
        idx_df = idx.to_frame()
        data[data > 0.5] = np.nan
        data[int(idx_df[idx_df.b == 'A'].iloc[0, 0])] = 0.1
        data[int(idx_df[idx_df.b == 'B'].iloc[0, 0])] = 0.1

        r = self.compute(md.Series(data, chunk_size=13), level=1)
        pd.testing.assert_series_equal(
            self.compute(data, level=1).sort_index(),
            self.executor.execute_dataframe(r, concat=True)[0].sort_index())

        r = self.compute(md.Series(data, chunk_size=13), level=1, skipna=False)
        pd.testing.assert_series_equal(
            self.compute(data, level=1, skipna=False).sort_index(),
            self.executor.execute_dataframe(r, concat=True)[0].sort_index())

        if self.has_min_count:
            r = self.compute(md.Series(data, chunk_size=13), min_count=1, level=1)
            pd.testing.assert_series_equal(
                self.compute(data, min_count=1, level=1).sort_index(),
                self.executor.execute_dataframe(r, concat=True)[0].sort_index())

    def testDataFrameReduction(self):
        data = pd.DataFrame(np.random.rand(20, 10))
        r = self.compute(md.DataFrame(data))
        pd.testing.assert_series_equal(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.DataFrame(data, chunk_size=3))
        pd.testing.assert_series_equal(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.DataFrame(data, chunk_size=6), axis='index', numeric_only=True)
        pd.testing.assert_series_equal(
            self.compute(data, axis='index', numeric_only=True),
            self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.DataFrame(data, chunk_size=3), axis=1)
        pd.testing.assert_series_equal(
            self.compute(data, axis=1),
            self.executor.execute_dataframe(r, concat=True)[0])

        # test null
        np_data = np.random.rand(20, 10)
        np_data[np_data > 0.6] = np.nan
        data = pd.DataFrame(np_data)

        r = self.compute(md.DataFrame(data, chunk_size=3))
        pd.testing.assert_series_equal(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.DataFrame(data, chunk_size=3), skipna=False)
        pd.testing.assert_series_equal(
            self.compute(data, skipna=False), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.DataFrame(data, chunk_size=3), skipna=False)
        pd.testing.assert_series_equal(
            self.compute(data, skipna=False), self.executor.execute_dataframe(r, concat=True)[0])

        if self.has_min_count:
            r = self.compute(md.DataFrame(data, chunk_size=3), min_count=15)
            pd.testing.assert_series_equal(
                self.compute(data, min_count=15),
                self.executor.execute_dataframe(r, concat=True)[0])

            r = self.compute(md.DataFrame(data, chunk_size=3), min_count=3)
            pd.testing.assert_series_equal(
                self.compute(data, min_count=3),
                self.executor.execute_dataframe(r, concat=True)[0])

            r = self.compute(md.DataFrame(data, chunk_size=3), axis=1, min_count=3)
            pd.testing.assert_series_equal(
                self.compute(data, axis=1, min_count=3),
                self.executor.execute_dataframe(r, concat=True)[0])

            r = self.compute(md.DataFrame(data, chunk_size=3), axis=1, min_count=8)
            pd.testing.assert_series_equal(
                self.compute(data, axis=1, min_count=8),
                self.executor.execute_dataframe(r, concat=True)[0])

        # test numeric_only
        data = pd.DataFrame(np.random.rand(10, 10), index=np.random.randint(-100, 100, size=(10,)),
                            columns=[np.random.bytes(10) for _ in range(10)])
        r = self.compute(md.DataFrame(data, chunk_size=2))
        pd.testing.assert_series_equal(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.DataFrame(data, chunk_size=6), axis='index', numeric_only=True)
        pd.testing.assert_series_equal(
            self.compute(data, axis='index', numeric_only=True),
            self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.DataFrame(data, chunk_size=3), axis='columns')
        pd.testing.assert_series_equal(
            self.compute(data, axis='columns'),
            self.executor.execute_dataframe(r, concat=True)[0])

        data_dict = dict((str(i), np.random.rand(10)) for i in range(10))
        data_dict['string'] = [str(i) for i in range(10)]
        data_dict['bool'] = np.random.choice([True, False], (10,))
        data = pd.DataFrame(data_dict)
        r = self.compute(md.DataFrame(data, chunk_size=3), axis='index', numeric_only=True)
        pd.testing.assert_series_equal(
            self.compute(data, axis='index', numeric_only=True),
            self.executor.execute_dataframe(r, concat=True)[0])

        data1 = pd.DataFrame(np.random.rand(10, 10), columns=[str(i) for i in range(10)])
        data2 = pd.DataFrame(np.random.rand(10, 10), columns=[str(i) for i in range(10)])
        df = md.DataFrame(data1, chunk_size=5) + md.DataFrame(data2, chunk_size=6)
        r = self.compute(df)
        pd.testing.assert_series_equal(
            self.compute(data1 + data2).sort_index(),
            self.executor.execute_dataframe(r, concat=True)[0].sort_index())

    def testDataFrameLevelReduction(self):
        idx = pd.MultiIndex.from_arrays([
            [str(i) for i in range(100)], np.random.choice(['A', 'B'], size=(100,))
        ], names=['a', 'b'])
        data = pd.DataFrame(np.random.rand(100, 10), index=idx)

        r = self.compute(md.DataFrame(data, chunk_size=13), level=1)
        pd.testing.assert_frame_equal(
            self.compute(data, level=1).sort_index(),
            self.executor.execute_dataframe(r, concat=True)[0].sort_index())

        r = self.compute(md.DataFrame(data, chunk_size=13), level=1, numeric_only=True)
        pd.testing.assert_frame_equal(
            self.compute(data, numeric_only=True, level=1).sort_index(),
            self.executor.execute_dataframe(r, concat=True)[0].sort_index())

        # test null
        data = pd.DataFrame(np.random.rand(100, 10), index=idx)
        data[data > 0.6] = np.nan

        r = self.compute(md.DataFrame(data, chunk_size=13), level=1)
        pd.testing.assert_frame_equal(
            self.compute(data, level=1).sort_index(),
            self.executor.execute_dataframe(r, concat=True)[0].sort_index())

        r = self.compute(md.DataFrame(data, chunk_size=13), level=1, skipna=False)
        pd.testing.assert_frame_equal(
            self.compute(data, level=1, skipna=False).sort_index(),
            self.executor.execute_dataframe(r, concat=True)[0].sort_index())

        if self.has_min_count:
            r = self.compute(md.DataFrame(data, chunk_size=13), level=1, min_count=10)
            pd.testing.assert_frame_equal(
                self.compute(data, level=1, min_count=10).sort_index(),
                self.executor.execute_dataframe(r, concat=True)[0].sort_index())

        # behavior of 'skew', 'kurt' differs for cases with and without level
        if self.func_name not in ('skew', 'kurt'):
            data_dict = dict((str(i), np.random.rand(100)) for i in range(10))
            data_dict['string'] = [str(i) for i in range(100)]
            data_dict['bool'] = np.random.choice([True, False], (100,))
            data = pd.DataFrame(data_dict, index=idx)

            r = self.compute(md.DataFrame(data, chunk_size=13), level=1, numeric_only=True)
            pd.testing.assert_frame_equal(
                self.compute(data, level=1, numeric_only=True).sort_index(),
                self.executor.execute_dataframe(r, concat=True)[0].sort_index())


@require_cudf
@require_cupy
class TestGPUReduction(TestBase):
    def testGPUExecution(self):
        df_raw = pd.DataFrame(np.random.rand(30, 3), columns=list('abc'))
        df = to_gpu(md.DataFrame(df_raw, chunk_size=6))

        r = df.sum()
        res = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_series_equal(res.to_pandas(), df_raw.sum())

        r = df.kurt()
        res = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_series_equal(res.to_pandas(), df_raw.kurt())

        r = df.agg(['sum', 'var'])
        res = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_frame_equal(res.to_pandas(), df_raw.agg(['sum', 'var']))

        s_raw = pd.Series(np.random.rand(30))
        s = to_gpu(md.Series(s_raw, chunk_size=6))

        r = s.sum()
        res = self.executor.execute_dataframe(r, concat=True)[0]
        self.assertAlmostEqual(res, s_raw.sum())

        r = s.kurt()
        res = self.executor.execute_dataframe(r, concat=True)[0]
        self.assertAlmostEqual(res, s_raw.kurt())

        r = s.agg(['sum', 'var'])
        res = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_series_equal(res.to_pandas(), s_raw.agg(['sum', 'var']))

        s_raw = pd.Series(np.random.randint(0, 3, size=(30,))
                          * np.random.randint(0, 5, size=(30,)))
        s = to_gpu(md.Series(s_raw, chunk_size=6))

        r = s.unique()
        res = self.executor.execute_dataframe(r, concat=True)[0]
        np.testing.assert_array_equal(cp.asnumpy(res).sort(), s_raw.unique().sort())


bool_reduction_functions = dict(
    all=dict(func_name='all'),
    any=dict(func_name='any'),
)


@parameterized(**bool_reduction_functions)
class TestBoolReduction(TestBase):
    def setUp(self):
        self.executor = ExecutorForTest()

    def compute(self, data, **kwargs):
        return getattr(data, self.func_name)(**kwargs)

    def testSeriesReduction(self):
        data = pd.Series(np.random.rand(10) > 0.5, index=[str(i) for i in range(10)], name='a')
        r = self.compute(md.Series(data))
        self.assertEqual(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.Series(data, chunk_size=6))
        self.assertAlmostEqual(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.Series(data, chunk_size=3))
        self.assertAlmostEqual(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.Series(data, chunk_size=4), axis='index')
        self.assertAlmostEqual(
            self.compute(data, axis='index'), self.executor.execute_dataframe(r, concat=True)[0])

        # test null
        data = pd.Series(np.random.rand(20), name='a')
        data[0] = 0.1  # make sure not all elements are NAN
        data[data > 0.5] = np.nan
        r = self.compute(md.Series(data, chunk_size=3))
        self.assertEqual(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.Series(data, chunk_size=3), skipna=False)
        self.assertTrue(
            self.executor.execute_dataframe(r, concat=True)[0])

    def testSeriesLevelReduction(self):
        idx = pd.MultiIndex.from_arrays([
            [str(i) for i in range(100)], np.random.choice(['A', 'B'], size=(100,))
        ], names=['a', 'b'])
        data = pd.Series(np.random.randint(0, 8, size=(100,)), index=idx)

        r = self.compute(md.Series(data, chunk_size=13), level=1)
        pd.testing.assert_series_equal(
            self.compute(data, level=1).sort_index(),
            self.executor.execute_dataframe(r, concat=True)[0].sort_index())

        # test null
        data = pd.Series(np.random.rand(100), name='a', index=idx)
        idx_df = idx.to_frame()
        data[data > 0.5] = np.nan
        data[int(idx_df[idx_df.b == 'A'].iloc[0, 0])] = 0.1
        data[int(idx_df[idx_df.b == 'B'].iloc[0, 0])] = 0.1

        r = self.compute(md.Series(data, chunk_size=13), level=1)
        pd.testing.assert_series_equal(
            self.compute(data, level=1).sort_index(),
            self.executor.execute_dataframe(r, concat=True)[0].sort_index())

        r = self.compute(md.Series(data, chunk_size=13), level=1, skipna=False)
        pd.testing.assert_series_equal(
            self.compute(data, level=1, skipna=False).sort_index(),
            self.executor.execute_dataframe(r, concat=True)[0].sort_index())

    def testDataFrameReduction(self):
        data = pd.DataFrame(np.random.rand(20, 10))
        data.iloc[:, :5] = data.iloc[:, :5] > 0.5
        r = self.compute(md.DataFrame(data))
        pd.testing.assert_series_equal(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.DataFrame(data, chunk_size=3))
        pd.testing.assert_series_equal(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.DataFrame(data, chunk_size=6), axis='index', bool_only=True)
        pd.testing.assert_series_equal(
            self.compute(data, axis='index', bool_only=True),
            self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.DataFrame(data, chunk_size=3), axis=1)
        pd.testing.assert_series_equal(
            self.compute(data, axis=1),
            self.executor.execute_dataframe(r, concat=True)[0])

        # test null
        np_data = np.random.rand(20, 10)
        np_data[np_data > 0.6] = np.nan
        data = pd.DataFrame(np_data)

        r = self.compute(md.DataFrame(data, chunk_size=3))
        pd.testing.assert_series_equal(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.DataFrame(data, chunk_size=3), skipna=False)
        pd.testing.assert_series_equal(
            self.compute(data, skipna=False), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.DataFrame(data, chunk_size=3), skipna=False)
        pd.testing.assert_series_equal(
            self.compute(data, skipna=False), self.executor.execute_dataframe(r, concat=True)[0])

        # test bool_only
        data = pd.DataFrame(np.random.rand(10, 10), index=np.random.randint(-100, 100, size=(10,)),
                            columns=[np.random.bytes(10) for _ in range(10)])
        data.iloc[:, :5] = data.iloc[:, :5] > 0.5
        data.iloc[:5, 5:] = data.iloc[:5, 5:] > 0.5
        r = self.compute(md.DataFrame(data, chunk_size=2))
        pd.testing.assert_series_equal(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.DataFrame(data, chunk_size=6), axis='index', bool_only=True)
        pd.testing.assert_series_equal(
            self.compute(data, axis='index', bool_only=True),
            self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.DataFrame(data, chunk_size=3), axis='columns')
        pd.testing.assert_series_equal(
            self.compute(data, axis='columns'),
            self.executor.execute_dataframe(r, concat=True)[0])

        data_dict = dict((str(i), np.random.rand(10)) for i in range(10))
        data_dict['string'] = [str(i) for i in range(10)]
        data_dict['bool'] = np.random.choice([True, False], (10,))
        data = pd.DataFrame(data_dict)
        r = self.compute(md.DataFrame(data, chunk_size=3), axis='index', bool_only=True)
        pd.testing.assert_series_equal(
            self.compute(data, axis='index', bool_only=True),
            self.executor.execute_dataframe(r, concat=True)[0])

    def testDataFrameLevelReduction(self):
        idx = pd.MultiIndex.from_arrays([
            [str(i) for i in range(100)], np.random.choice(['A', 'B'], size=(100,))
        ], names=['a', 'b'])
        data = pd.DataFrame(np.random.rand(100, 10), index=idx)
        data.iloc[:, :5] = data.iloc[:, :5] > 0.5

        r = self.compute(md.DataFrame(data, chunk_size=13), level=1)
        pd.testing.assert_frame_equal(
            self.compute(data, level=1).sort_index(),
            self.executor.execute_dataframe(r, concat=True)[0].sort_index())

        # test null
        data = pd.DataFrame(np.random.rand(100, 10), index=idx)
        data[data > 0.6] = np.nan

        r = self.compute(md.DataFrame(data, chunk_size=13), level=1)
        pd.testing.assert_frame_equal(
            self.compute(data, level=1).sort_index(),
            self.executor.execute_dataframe(r, concat=True)[0].sort_index())

        r = self.compute(md.DataFrame(data, chunk_size=13), level=1, skipna=False)
        pd.testing.assert_frame_equal(
            self.compute(data, level=1, skipna=False).sort_index(),
            self.executor.execute_dataframe(r, concat=True)[0].sort_index())

        # test bool_only
        # bool_only not supported when level specified


class TestCount(TestBase):
    def setUp(self):
        self.executor = ExecutorForTest()

    def testSeriesCount(self):
        array = np.random.rand(10)
        array[[2, 7, 9]] = np.nan
        data = pd.Series(array)
        series = md.Series(data)

        result = self.executor.execute_dataframe(series.count(), concat=True)[0]
        expected = data.count()
        self.assertEqual(result, expected)

        series2 = md.Series(data, chunk_size=1)

        result = self.executor.execute_dataframe(series2.count(), concat=True)[0]
        expected = data.count()
        self.assertEqual(result, expected)

        series2 = md.Series(data, chunk_size=3)

        result = self.executor.execute_dataframe(series2.count(), concat=True)[0]
        expected = data.count()
        self.assertEqual(result, expected)

    def testDataFrameCount(self):
        data = pd.DataFrame({
            "Person": ["John", "Myla", "Lewis", "John", "Myla"],
            "Age": [24., np.nan, 21., 33, 26],
            "Single": [False, True, True, True, False]})
        df = md.DataFrame(data)

        result = self.executor.execute_dataframe(df.count(), concat=True)[0]
        expected = data.count()
        pd.testing.assert_series_equal(result, expected)

        result = self.executor.execute_dataframe(df.count(axis='columns'), concat=True)[0]
        expected = data.count(axis='columns')
        pd.testing.assert_series_equal(result, expected)

        df2 = md.DataFrame(data, chunk_size=2)

        result = self.executor.execute_dataframe(df2.count(), concat=True)[0]
        expected = data.count()
        pd.testing.assert_series_equal(result, expected)

        result = self.executor.execute_dataframe(df2.count(axis='columns'), concat=True)[0]
        expected = data.count(axis='columns')
        pd.testing.assert_series_equal(result, expected)

        df3 = md.DataFrame(data, chunk_size=3)

        result = self.executor.execute_dataframe(df3.count(numeric_only=True), concat=True)[0]
        expected = data.count(numeric_only=True)
        pd.testing.assert_series_equal(result, expected)

        result = self.executor.execute_dataframe(df3.count(axis='columns', numeric_only=True), concat=True)[0]
        expected = data.count(axis='columns', numeric_only=True)
        pd.testing.assert_series_equal(result, expected)

    def testNunique(self):
        data1 = pd.Series(np.random.randint(0, 5, size=(20,)))

        series = md.Series(data1)
        result = self.executor.execute_dataframe(series.nunique(), concat=True)[0]
        expected = data1.nunique()
        self.assertEqual(result, expected)

        series = md.Series(data1, chunk_size=6)
        result = self.executor.execute_dataframe(series.nunique(), concat=True)[0]
        expected = data1.nunique()
        self.assertEqual(result, expected)

        # test dropna
        data2 = data1.copy()
        data2[[2, 9, 18]] = np.nan

        series = md.Series(data2)
        result = self.executor.execute_dataframe(series.nunique(), concat=True)[0]
        expected = data2.nunique()
        self.assertEqual(result, expected)

        series = md.Series(data2, chunk_size=3)
        result = self.executor.execute_dataframe(series.nunique(dropna=False), concat=True)[0]
        expected = data2.nunique(dropna=False)
        self.assertEqual(result, expected)

        # test dataframe
        data1 = pd.DataFrame(np.random.randint(0, 6, size=(20, 20)),
                             columns=['c' + str(i) for i in range(20)])
        df = md.DataFrame(data1)
        result = self.executor.execute_dataframe(df.nunique(), concat=True)[0]
        expected = data1.nunique()
        pd.testing.assert_series_equal(result, expected)

        df = md.DataFrame(data1, chunk_size=6)
        result = self.executor.execute_dataframe(df.nunique(), concat=True)[0]
        expected = data1.nunique()
        pd.testing.assert_series_equal(result, expected)

        df = md.DataFrame(data1)
        result = self.executor.execute_dataframe(df.nunique(axis=1), concat=True)[0]
        expected = data1.nunique(axis=1)
        pd.testing.assert_series_equal(result, expected)

        df = md.DataFrame(data1, chunk_size=3)
        result = self.executor.execute_dataframe(df.nunique(axis=1), concat=True)[0]
        expected = data1.nunique(axis=1)
        pd.testing.assert_series_equal(result, expected)

        # test dropna
        data2 = data1.copy()
        data2.iloc[[2, 9, 18], [2, 9, 18]] = np.nan

        df = md.DataFrame(data2)
        result = self.executor.execute_dataframe(df.nunique(), concat=True)[0]
        expected = data2.nunique()
        pd.testing.assert_series_equal(result, expected)

        df = md.DataFrame(data2, chunk_size=3)
        result = self.executor.execute_dataframe(df.nunique(dropna=False), concat=True)[0]
        expected = data2.nunique(dropna=False)
        pd.testing.assert_series_equal(result, expected)

        df = md.DataFrame(data1, chunk_size=3)
        result = self.executor.execute_dataframe(df.nunique(axis=1), concat=True)[0]
        expected = data1.nunique(axis=1)
        pd.testing.assert_series_equal(result, expected)

    @unittest.skipIf(pa is None, 'pyarrow not installed')
    def testUseArrowDtypeNUnique(self):
        with option_context({'dataframe.use_arrow_dtype': True, 'combine_size': 2}):
            rs = np.random.RandomState(0)
            data1 = pd.DataFrame({'a': rs.random(10),
                                  'b': [f's{i}' for i in rs.randint(100, size=10)]})
            data1['c'] = data1['b'].copy()
            data1['d'] = data1['b'].copy()
            data1['e'] = data1['b'].copy()

            df = md.DataFrame(data1, chunk_size=(3, 2))
            r = df.nunique(axis=0)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = data1.nunique(axis=0)
            pd.testing.assert_series_equal(result, expected)

            r = df.nunique(axis=1)
            result = self.executor.execute_dataframe(r, concat=True)[0]
            expected = data1.nunique(axis=1)
            pd.testing.assert_series_equal(result, expected)

    def testUnique(self):
        data1 = pd.Series(np.random.randint(0, 5, size=(20,)))

        series = md.Series(data1)
        result = self.executor.execute_dataframe(series.unique(), concat=True)[0]
        expected = data1.unique()
        np.testing.assert_array_equal(result, expected)

        series = md.Series(data1, chunk_size=6)
        result = self.executor.execute_dataframe(series.unique(), concat=True)[0]
        expected = data1.unique()
        np.testing.assert_array_equal(result, expected)

        data2 = pd.Series([pd.Timestamp('20200101'), ] * 5 +
                          [pd.Timestamp('20200202')] +
                          [pd.Timestamp('20020101')] * 9)
        series = md.Series(data2)
        result = self.executor.execute_dataframe(series.unique(), concat=True)[0]
        expected = data2.unique()
        np.testing.assert_array_equal(result, expected)

        series = md.Series(data2, chunk_size=6)
        result = self.executor.execute_dataframe(series.unique(), concat=True)[0]
        expected = data2.unique()
        np.testing.assert_array_equal(result, expected)


cum_reduction_functions = dict(
    cummax=dict(func_name='cummax'),
    cummin=dict(func_name='cummin'),
    cumprod=dict(func_name='cumprod'),
    cumsum=dict(func_name='cumsum'),
)


@parameterized(**cum_reduction_functions)
class TestCumReduction(TestBase):
    def setUp(self):
        self.executor = ExecutorForTest()

    def compute(self, data, **kwargs):
        return getattr(data, self.func_name)(**kwargs)

    def testSeriesCumReduction(self):
        data = pd.Series(np.random.rand(20), index=[str(i) for i in range(20)], name='a')
        r = self.compute(md.Series(data))
        pd.testing.assert_series_equal(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.Series(data, chunk_size=6))
        pd.testing.assert_series_equal(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.Series(data, chunk_size=3))
        pd.testing.assert_series_equal(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.Series(data, chunk_size=4), axis='index')
        pd.testing.assert_series_equal(
            self.compute(data, axis='index'), self.executor.execute_dataframe(r, concat=True)[0])

        data = pd.Series(np.random.rand(20), name='a')
        data[0] = 0.1  # make sure not all elements are NAN
        data[data > 0.5] = np.nan
        r = self.compute(md.Series(data, chunk_size=3))
        pd.testing.assert_series_equal(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.Series(data, chunk_size=3), skipna=False)
        pd.testing.assert_series_equal(
            self.compute(data, skipna=False), self.executor.execute_dataframe(r, concat=True)[0])

    def testDataFrameCumReduction(self):
        data = pd.DataFrame(np.random.rand(20, 10))
        r = self.compute(md.DataFrame(data))
        pd.testing.assert_frame_equal(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.DataFrame(data, chunk_size=3))
        pd.testing.assert_frame_equal(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.DataFrame(data, chunk_size=3), axis=1)
        pd.testing.assert_frame_equal(
            self.compute(data, axis=1),
            self.executor.execute_dataframe(r, concat=True)[0])

        # test null
        np_data = np.random.rand(20, 10)
        np_data[np_data > 0.6] = np.nan
        data = pd.DataFrame(np_data)

        r = self.compute(md.DataFrame(data, chunk_size=3))
        pd.testing.assert_frame_equal(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.DataFrame(data, chunk_size=3), skipna=False)
        pd.testing.assert_frame_equal(
            self.compute(data, skipna=False), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.DataFrame(data, chunk_size=3), skipna=False)
        pd.testing.assert_frame_equal(
            self.compute(data, skipna=False), self.executor.execute_dataframe(r, concat=True)[0])

        # test numeric_only
        data = pd.DataFrame(np.random.rand(10, 10), index=np.random.randint(-100, 100, size=(10,)),
                            columns=[np.random.bytes(10) for _ in range(10)])
        r = self.compute(md.DataFrame(data, chunk_size=2))
        pd.testing.assert_frame_equal(
            self.compute(data), self.executor.execute_dataframe(r, concat=True)[0])

        r = self.compute(md.DataFrame(data, chunk_size=3), axis='columns')
        pd.testing.assert_frame_equal(
            self.compute(data, axis='columns'),
            self.executor.execute_dataframe(r, concat=True)[0])


class TestAggregate(TestBase):
    def setUp(self):
        self.executor = ExecutorForTest()

    def testDataFrameAggregate(self):
        all_aggs = ['sum', 'prod', 'min', 'max', 'count', 'size',
                    'mean', 'var', 'std', 'sem', 'skew', 'kurt']
        data = pd.DataFrame(np.random.rand(20, 20))

        df = md.DataFrame(data)
        result = df.agg(all_aggs)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                      data.agg(all_aggs))

        result = df.agg('size')
        self.assertEqual(self.executor.execute_dataframe(result)[0], data.agg('size'))

        for func in (a for a in all_aggs if a != 'size'):
            result = df.agg(func)
            pd.testing.assert_series_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                           data.agg(func))

            result = df.agg(func, axis=1)
            pd.testing.assert_series_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                           data.agg(func, axis=1))

        df = md.DataFrame(data, chunk_size=3)

        # will redirect to transform
        result = df.agg(['cumsum', 'cummax'])
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                      data.agg(['cumsum', 'cummax']))

        result = df.agg('size')
        self.assertEqual(self.executor.execute_dataframe(result)[0], data.agg('size'))

        for func in (a for a in all_aggs if a != 'size'):
            result = df.agg(func)
            pd.testing.assert_series_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                           data.agg(func))

            result = df.agg(func, axis=1)
            pd.testing.assert_series_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                           data.agg(func, axis=1))

        result = df.agg(['sum'])
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                      data.agg(['sum']))

        result = df.agg(all_aggs)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                      data.agg(all_aggs))

        result = df.agg(all_aggs, axis=1)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                      data.agg(all_aggs, axis=1))

        result = df.agg({0: ['sum', 'min', 'var'], 9: ['mean', 'var', 'std']})
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                      data.agg({0: ['sum', 'min', 'var'], 9: ['mean', 'var', 'std']}))

        result = df.agg(sum_0=NamedAgg(0, 'sum'), min_0=NamedAgg(0, 'min'),
                        mean_9=NamedAgg(9, 'mean'))
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                      data.agg(sum_0=NamedAgg(0, 'sum'), min_0=NamedAgg(0, 'min'),
                                               mean_9=NamedAgg(9, 'mean')))

    def testSeriesAggregate(self):
        all_aggs = ['sum', 'prod', 'min', 'max', 'count', 'size',
                    'mean', 'var', 'std', 'sem', 'skew', 'kurt']
        data = pd.Series(np.random.rand(20), index=[str(i) for i in range(20)], name='a')
        series = md.Series(data)

        result = series.agg(all_aggs)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                       data.agg(all_aggs))

        for func in all_aggs:
            result = series.agg(func)
            self.assertAlmostEqual(self.executor.execute_dataframe(result, concat=True)[0],
                                   data.agg(func))

        series = md.Series(data, chunk_size=3)

        for func in all_aggs:
            result = series.agg(func)
            self.assertAlmostEqual(self.executor.execute_dataframe(result, concat=True)[0],
                                   data.agg(func))

        result = series.agg(all_aggs)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                       data.agg(all_aggs))

        result = series.agg({'col_sum': 'sum', 'col_count': 'count'})
        pd.testing.assert_series_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                       data.agg({'col_sum': 'sum', 'col_count': 'count'}))

        result = series.agg(col_var='var', col_skew='skew')
        pd.testing.assert_series_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                       data.agg(col_var='var', col_skew='skew'))

    def testAggregateStrCat(self):
        agg_fun = lambda x: x.str.cat(sep='_', na_rep='NA')

        rs = np.random.RandomState(0)
        raw_df = pd.DataFrame({'a': rs.choice(['A', 'B', 'C'], size=(100,)),
                               'b': rs.choice([None, 'alfa', 'bravo', 'charlie'], size=(100,))})

        mdf = md.DataFrame(raw_df, chunk_size=13)

        r = mdf.agg(agg_fun)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(r, concat=True)[0],
                                       raw_df.agg(agg_fun))

        raw_series = pd.Series(rs.choice([None, 'alfa', 'bravo', 'charlie'], size=(100,)))

        ms = md.Series(raw_series, chunk_size=13)

        r = ms.agg(agg_fun)
        self.assertEqual(self.executor.execute_dataframe(r, concat=True)[0],
                         raw_series.agg(agg_fun))


class MockReduction1(CustomReduction):
    def agg(self, v1):
        return v1.sum()


class MockReduction2(CustomReduction):
    def pre(self, value):
        return value + 1, value ** 2

    def agg(self, v1, v2):
        return v1.sum(), v2.prod()

    def post(self, v1, v2):
        return v1 + v2


class TestCustomAggregate(TestBase):
    def setUp(self):
        self.executor = ExecutorForTest()

    def testDataFrameAggregate(self):
        data = pd.DataFrame(np.random.rand(30, 20))

        df = md.DataFrame(data)
        result = df.agg(MockReduction1())
        pd.testing.assert_series_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                       data.agg(MockReduction1()))

        result = df.agg(MockReduction2())
        pd.testing.assert_series_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                       data.agg(MockReduction2()))

        df = md.DataFrame(data, chunk_size=5)
        result = df.agg(MockReduction2())
        pd.testing.assert_series_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                       data.agg(MockReduction2()))

        result = df.agg(MockReduction2())
        pd.testing.assert_series_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                       data.agg(MockReduction2()))

    def testSeriesAggregate(self):
        data = pd.Series(np.random.rand(20))

        s = md.Series(data)
        result = s.agg(MockReduction1())
        self.assertEqual(self.executor.execute_dataframe(result, concat=True)[0],
                         data.agg(MockReduction1()))

        result = s.agg(MockReduction2())
        self.assertEqual(self.executor.execute_dataframe(result, concat=True)[0],
                         data.agg(MockReduction2()))

        s = md.Series(data, chunk_size=5)
        result = s.agg(MockReduction2())
        self.assertAlmostEqual(self.executor.execute_dataframe(result, concat=True)[0],
                               data.agg(MockReduction2()))

        result = s.agg(MockReduction2())
        self.assertAlmostEqual(self.executor.execute_dataframe(result, concat=True)[0],
                               data.agg(MockReduction2()))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
