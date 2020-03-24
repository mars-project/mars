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

import numpy as np
import pandas as pd

from mars.tests.core import ExecutorForTest
from mars.dataframe import DataFrame, Series
from mars.session import new_session


class Test(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.executor = ExecutorForTest('numpy')

    def testSortValuesExecution(self):
        df = pd.DataFrame(np.random.rand(100, 10), columns=['a' + str(i) for i in range(10)])

        # test one chunk
        mdf = DataFrame(df)
        result = self.executor.execute_dataframe(mdf.sort_values('a0'), concat=True)[0]
        expected = df.sort_values('a0')

        pd.testing.assert_frame_equal(result, expected)

        result = self.executor.execute_dataframe(mdf.sort_values(['a6', 'a7'], ascending=False), concat=True)[0]
        expected = df.sort_values(['a6', 'a7'], ascending=False)

        pd.testing.assert_frame_equal(result, expected)

        # test psrs
        mdf = DataFrame(df, chunk_size=10)
        result = self.executor.execute_dataframe(mdf.sort_values('a0'), concat=True)[0]
        expected = df.sort_values('a0')

        pd.testing.assert_frame_equal(result, expected)

        result = self.executor.execute_dataframe(mdf.sort_values(['a3', 'a4']), concat=True)[0]
        expected = df.sort_values(['a3', 'a4'])

        pd.testing.assert_frame_equal(result, expected)

        # test ascending=False
        result = self.executor.execute_dataframe(mdf.sort_values(['a0', 'a1'], ascending=False), concat=True)[0]
        expected = df.sort_values(['a0', 'a1'], ascending=False)

        pd.testing.assert_frame_equal(result, expected)

        result = self.executor.execute_dataframe(mdf.sort_values(['a7'], ascending=False), concat=True)[0]
        expected = df.sort_values(['a7'], ascending=False)

        pd.testing.assert_frame_equal(result, expected)

        # test rechunk
        mdf = DataFrame(df, chunk_size=3)
        result = self.executor.execute_dataframe(mdf.sort_values('a0'), concat=True)[0]
        expected = df.sort_values('a0')

        pd.testing.assert_frame_equal(result, expected)

        result = self.executor.execute_dataframe(mdf.sort_values(['a3', 'a4']), concat=True)[0]
        expected = df.sort_values(['a3', 'a4'])

        pd.testing.assert_frame_equal(result, expected)

        # test other types
        raw = pd.DataFrame({'a': np.random.rand(10),
                            'b': np.random.randint(1000, size=10),
                            'c': np.random.rand(10),
                            'd': [np.random.bytes(10) for _ in range(10)],
                            'e': [pd.Timestamp('201{}'.format(i)) for i in range(10)],
                            'f': [pd.Timedelta('{} days'.format(i)) for i in range(10)]
                            },)
        mdf = DataFrame(raw, chunk_size=3)

        for label in raw.columns:
            result = self.executor.execute_dataframe(mdf.sort_values(label), concat=True)[0]
            expected = raw.sort_values(label)
            pd.testing.assert_frame_equal(result, expected)

        result = self.executor.execute_dataframe(mdf.sort_values(['a', 'b', 'e'], ascending=False), concat=True)[0]
        expected = raw.sort_values(['a', 'b', 'e'], ascending=False)

        pd.testing.assert_frame_equal(result, expected)

        # test nan
        df = pd.DataFrame({
            'col1': ['A', 'A', 'B', 'B', 'D', 'C'],
            'col2': [2, 1, 9, np.nan, 7, 4],
            'col3': [0, 1, 9, 4, 2, 3],
        })
        mdf = DataFrame(df)
        result = self.executor.execute_dataframe(mdf.sort_values(['col2']), concat=True)[0]
        expected = df.sort_values(['col2'])

        pd.testing.assert_frame_equal(result, expected)

        mdf = DataFrame(df, chunk_size=3)
        result = self.executor.execute_dataframe(mdf.sort_values(['col2']), concat=True)[0]
        expected = df.sort_values(['col2'])

        pd.testing.assert_frame_equal(result, expected)

        # test ignore_index
        executor = ExecutorForTest(storage=new_session().context)

        df = pd.DataFrame(np.random.rand(10, 3), columns=['a' + str(i) for i in range(3)])

        mdf = DataFrame(df, chunk_size=3)
        result = executor.execute_dataframe(
            mdf.sort_values(['a0', 'a1'], ignore_index=True), concat=True)[0]
        try:  # for python3.5
            expected = df.sort_values(['a0', 'a1'], ignore_index=True)
        except TypeError:
            expected = df.sort_values(['a0', 'a1'])
            expected.index = pd.RangeIndex(len(expected))

        pd.testing.assert_frame_equal(result, expected)

        # test inplace
        mdf = DataFrame(df)
        mdf.sort_values('a0', inplace=True)
        result = self.executor.execute_dataframe(mdf, concat=True)[0]
        df.sort_values('a0', inplace=True)

        pd.testing.assert_frame_equal(result, df)

        # test Sereis.sort_values
        raw = pd.Series(np.random.rand(10))
        series = Series(raw)
        result = self.executor.execute_dataframe(series.sort_values(), concat=True)[0]
        expected = raw.sort_values()

        pd.testing.assert_series_equal(result, expected)

        series = Series(raw, chunk_size=3)
        result = self.executor.execute_dataframe(series.sort_values(), concat=True)[0]
        expected = raw.sort_values()

        pd.testing.assert_series_equal(result, expected)

        series = Series(raw, chunk_size=2)
        result = self.executor.execute_dataframe(series.sort_values(ascending=False), concat=True)[0]
        expected = raw.sort_values(ascending=False)

        pd.testing.assert_series_equal(result, expected)

    def testSortIndexExecution(self):
        raw = pd.DataFrame(np.random.rand(100, 20), index=np.random.rand(100))

        mdf = DataFrame(raw)
        result = self.executor.execute_dataframe(mdf.sort_index(), concat=True)[0]
        expected = raw.sort_index()
        pd.testing.assert_frame_equal(result, expected)

        mdf = DataFrame(raw)
        mdf.sort_index(inplace=True)
        result = self.executor.execute_dataframe(mdf, concat=True)[0]
        expected = raw.sort_index()
        pd.testing.assert_frame_equal(result, expected)

        mdf = DataFrame(raw, chunk_size=30)
        result = self.executor.execute_dataframe(mdf.sort_index(), concat=True)[0]
        expected = raw.sort_index()
        pd.testing.assert_frame_equal(result, expected)

        mdf = DataFrame(raw, chunk_size=20)
        result = self.executor.execute_dataframe(mdf.sort_index(ascending=False), concat=True)[0]
        expected = raw.sort_index(ascending=False)
        pd.testing.assert_frame_equal(result, expected)

        executor = ExecutorForTest(storage=new_session().context)

        mdf = DataFrame(raw, chunk_size=10)
        result = executor.execute_dataframe(mdf.sort_index(ignore_index=True), concat=True)[0]
        try:  # for python3.5
            expected = raw.sort_index(ignore_index=True)
        except TypeError:
            expected = raw.sort_index()
            expected.index = pd.RangeIndex(len(expected))
        pd.testing.assert_frame_equal(result, expected)

        # test axis=1
        raw = pd.DataFrame(np.random.rand(10, 10), columns=np.random.rand(10))

        mdf = DataFrame(raw)
        result = self.executor.execute_dataframe(mdf.sort_index(axis=1), concat=True)[0]
        expected = raw.sort_index(axis=1)
        pd.testing.assert_frame_equal(result, expected)

        mdf = DataFrame(raw, chunk_size=3)
        result = self.executor.execute_dataframe(mdf.sort_index(axis=1), concat=True)[0]
        expected = raw.sort_index(axis=1)
        pd.testing.assert_frame_equal(result, expected)

        mdf = DataFrame(raw, chunk_size=4)
        result = self.executor.execute_dataframe(mdf.sort_index(axis=1, ascending=False), concat=True)[0]
        expected = raw.sort_index(axis=1, ascending=False)
        pd.testing.assert_frame_equal(result, expected)

        mdf = DataFrame(raw, chunk_size=4)
        executor = ExecutorForTest(storage=new_session().context)

        result = executor.execute_dataframe(mdf.sort_index(axis=1, ignore_index=True), concat=True)[0]
        try:  # for python3.5
            expected = raw.sort_index(axis=1, ignore_index=True)
        except TypeError:
            expected = raw.sort_index(axis=1)
            expected.index = pd.RangeIndex(len(expected))
        pd.testing.assert_frame_equal(result, expected)

        # test series
        raw = pd.Series(np.random.rand(10, ), index=np.random.rand(10))

        series = Series(raw)
        result = self.executor.execute_dataframe(series.sort_index(), concat=True)[0]
        expected = raw.sort_index()
        pd.testing.assert_series_equal(result, expected)

        series = Series(raw, chunk_size=2)
        result = self.executor.execute_dataframe(series.sort_index(), concat=True)[0]
        expected = raw.sort_index()
        pd.testing.assert_series_equal(result, expected)

        series = Series(raw, chunk_size=3)
        result = self.executor.execute_dataframe(series.sort_index(ascending=False), concat=True)[0]
        expected = raw.sort_index(ascending=False)
        pd.testing.assert_series_equal(result, expected)
