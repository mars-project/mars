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

from mars.tests.core import ExecutorForTest, TestBase
from mars.tensor import tensor
from mars.dataframe import Series, DataFrame


class Test(TestBase):
    def setUp(self) -> None:
        super().setUp()
        self.executor = ExecutorForTest('numpy')

    def testSeriesQuantileExecution(self):
        raw = pd.Series(np.random.rand(10), name='a')
        a = Series(raw, chunk_size=3)

        # q = 0.5, scalar
        r = a.quantile()
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.quantile()

        self.assertEqual(result, expected)

        # q is a list
        r = a.quantile([0.3, 0.7])
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.quantile([0.3, 0.7])

        pd.testing.assert_series_equal(result, expected)

        # test interpolation
        r = a.quantile([0.3, 0.7], interpolation='midpoint')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.quantile([0.3, 0.7], interpolation='midpoint')

        pd.testing.assert_series_equal(result, expected)

        ctx, executor = self._create_test_context(self.executor)
        with ctx:
            q = tensor([0.3, 0.7])

            # q is a tensor
            r = a.quantile(q)
            result = executor.execute_dataframes([r])[0]
            expected = raw.quantile([0.3, 0.7])

            pd.testing.assert_series_equal(result, expected)

    def testDataFrameQuantileExecution(self):
        raw = pd.DataFrame({'a': np.random.rand(10),
                            'b': np.random.randint(1000, size=10),
                            'c': np.random.rand(10),
                            'd': [np.random.bytes(10) for _ in range(10)],
                            'e': [pd.Timestamp(f'201{i}') for i in range(10)],
                            'f': [pd.Timedelta(f'{i} days') for i in range(10)]
                            },
                           index=pd.RangeIndex(1, 11))
        df = DataFrame(raw, chunk_size=3)

        # q = 0.5, axis = 0, series
        r = df.quantile()
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.quantile()

        pd.testing.assert_series_equal(result, expected)

        # q = 0.5, axis = 1, series
        r = df.quantile(axis=1)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.quantile(axis=1)

        pd.testing.assert_series_equal(result, expected)

        # q is a list, axis = 0, dataframe
        r = df.quantile([0.3, 0.7])
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.quantile([0.3, 0.7])

        pd.testing.assert_frame_equal(result, expected)

        # q is a list, axis = 1, dataframe
        r = df.quantile([0.3, 0.7], axis=1)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.quantile([0.3, 0.7], axis=1)

        pd.testing.assert_frame_equal(result, expected)

        # test interpolation
        r = df.quantile([0.3, 0.7], interpolation='midpoint')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw.quantile([0.3, 0.7], interpolation='midpoint')

        pd.testing.assert_frame_equal(result, expected)

        ctx, executor = self._create_test_context(self.executor)
        with ctx:
            q = tensor([0.3, 0.7])

            # q is a tensor
            r = df.quantile(q)
            result = executor.execute_dataframes([r])[0]
            expected = raw.quantile([0.3, 0.7])

            pd.testing.assert_frame_equal(result, expected)

        # test numeric_only
        raw2 = pd.DataFrame({'a': np.random.rand(10),
                             'b': np.random.randint(1000, size=10),
                             'c': np.random.rand(10),
                             'd': [pd.Timestamp(f'201{i}') for i in range(10)],
                             }, index=pd.RangeIndex(1, 11))
        df2 = DataFrame(raw2, chunk_size=3)

        r = df2.quantile([0.3, 0.7], numeric_only=False)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw2.quantile([0.3, 0.7], numeric_only=False)

        pd.testing.assert_frame_equal(result, expected)

        r = df2.quantile(numeric_only=False)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw2.quantile(numeric_only=False)

        pd.testing.assert_series_equal(result, expected)

    def testDataFrameCorr(self):
        rs = np.random.RandomState(0)
        raw = rs.rand(20, 10)
        raw = pd.DataFrame(np.where(raw > 0.4, raw, np.nan), columns=list('ABCDEFGHIJ'))
        raw['k'] = pd.Series(['aaa'] * 20)

        df = DataFrame(raw)

        result = df.corr()
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                      raw.corr())

        result = df.corr(method='kendall')
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                      raw.corr(method='kendall'))

        df = DataFrame(raw, chunk_size=6)

        with self.assertRaises(Exception):
            self.executor.execute_dataframe(df.corr(method='kendall'), concat=True)

        result = df.corr()
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                      raw.corr())

        result = df.corr(min_periods=7)
        pd.testing.assert_frame_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                      raw.corr(min_periods=7))

    def testDataFrameCorrWith(self):
        rs = np.random.RandomState(0)
        raw_df = rs.rand(20, 10)
        raw_df = pd.DataFrame(np.where(raw_df > 0.4, raw_df, np.nan), columns=list('ABCDEFGHIJ'))
        raw_df2 = rs.rand(20, 10)
        raw_df2 = pd.DataFrame(np.where(raw_df2 > 0.4, raw_df2, np.nan), columns=list('ACDEGHIJKL'))
        raw_s = rs.rand(20)
        raw_s = pd.Series(np.where(raw_s > 0.4, raw_s, np.nan))
        raw_s2 = rs.rand(10)
        raw_s2 = pd.Series(np.where(raw_s2 > 0.4, raw_s2, np.nan), index=raw_df2.columns)

        df = DataFrame(raw_df)
        df2 = DataFrame(raw_df2)

        result = df.corrwith(df2)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                       raw_df.corrwith(raw_df2))

        result = df.corrwith(df2, axis=1)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                       raw_df.corrwith(raw_df2, axis=1))

        result = df.corrwith(df2, method='kendall')
        pd.testing.assert_series_equal(self.executor.execute_dataframe(result, concat=True)[0],
                                       raw_df.corrwith(raw_df2, method='kendall'))

        df = DataFrame(raw_df, chunk_size=4)
        df2 = DataFrame(raw_df2, chunk_size=6)
        s = Series(raw_s, chunk_size=5)
        s2 = Series(raw_s2, chunk_size=5)

        with self.assertRaises(Exception):
            self.executor.execute_dataframe(df.corrwith(df2, method='kendall'), concat=True)

        result = df.corrwith(df2)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(result, concat=True)[0].sort_index(),
                                       raw_df.corrwith(raw_df2).sort_index())

        result = df.corrwith(df2, axis=1)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(result, concat=True)[0].sort_index(),
                                       raw_df.corrwith(raw_df2, axis=1).sort_index())

        result = df.corrwith(s)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(result, concat=True)[0].sort_index(),
                                       raw_df.corrwith(raw_s).sort_index())

        result = df.corrwith(s2, axis=1)
        pd.testing.assert_series_equal(self.executor.execute_dataframe(result, concat=True)[0].sort_index(),
                                       raw_df.corrwith(raw_s2, axis=1).sort_index())

    def testSeriesCorr(self):
        rs = np.random.RandomState(0)
        raw = rs.rand(20)
        raw = pd.Series(np.where(raw > 0.4, raw, np.nan))
        raw2 = rs.rand(20)
        raw2 = pd.Series(np.where(raw2 > 0.4, raw2, np.nan))

        s = Series(raw)
        s2 = Series(raw2)

        result = s.corr(s2)
        self.assertEqual(self.executor.execute_dataframe(result, concat=True)[0],
                         raw.corr(raw2))

        result = s.corr(s2, method='kendall')
        self.assertEqual(self.executor.execute_dataframe(result, concat=True)[0],
                         raw.corr(raw2, method='kendall'))

        result = s.autocorr(2)
        self.assertEqual(self.executor.execute_dataframe(result, concat=True)[0],
                         raw.autocorr(2))

        s = Series(raw, chunk_size=6)
        s2 = Series(raw2, chunk_size=4)

        with self.assertRaises(Exception):
            self.executor.execute_dataframe(s.corr(s2, method='kendall'), concat=True)

        result = s.corr(s2)
        self.assertAlmostEqual(self.executor.execute_dataframe(result, concat=True)[0],
                               raw.corr(raw2))

        result = s.corr(s2, min_periods=7)
        self.assertAlmostEqual(self.executor.execute_dataframe(result, concat=True)[0],
                               raw.corr(raw2, min_periods=7))

        result = s.autocorr(2)
        self.assertAlmostEqual(self.executor.execute_dataframe(result, concat=True)[0],
                               raw.autocorr(2))
