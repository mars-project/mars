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

import operator

import numpy as np
import pandas as pd

from mars.executor import Executor
from mars.tests.core import TestBase, parameterized
from mars.tensor.datasource import array as from_array
from mars.dataframe.datasource.dataframe import from_pandas
from mars.dataframe.datasource.series import from_pandas as from_pandas_series
from mars.dataframe.arithmetic import abs


binary_functions = dict(
    add=dict(func=operator.add, func_name='add'),
    subtract=dict(func=operator.sub, func_name='sub'),
    floordiv=dict(func=operator.floordiv, func_name='floordiv'),
    truediv=dict(func=operator.truediv, func_name='truediv')
)


@parameterized(**binary_functions)
class TestBinary(TestBase):
    def setUp(self):
        self.executor = Executor()

    @property
    def rfunc_name(self):
        return 'r' + self.func_name

    def testWithoutShuffleExecution(self):
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

        expected = self.func(data1, data2)
        result = self.executor.execute_dataframe(df3, concat=True)[0]

        pd.testing.assert_frame_equal(expected, result)

    def testWithOneShuffleExecution(self):
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

        expected = self.func(data1, data2)
        result = self.executor.execute_dataframe(df3, concat=True)[0]

        pd.testing.assert_frame_equal(expected, result)

        # only 1 axis is monotonic
        # data1 with columns split into [0...4], [5...9],
        data1 = pd.DataFrame(np.random.rand(10, 10), index=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
                             columns=np.arange(10))
        df1 = from_pandas(data1, chunk_size=5)
        # data2 with columns split into [6...11], [2, 5],
        data2 = pd.DataFrame(np.random.rand(10, 10), index=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2],
                             columns=np.arange(11, 1, -1))
        df2 = from_pandas(data2, chunk_size=6)

        df3 = self.func(df1, df2)

        expected = self.func(data1, data2)
        result = self.executor.execute_dataframe(df3, concat=True)[0]

        pd.testing.assert_frame_equal(expected, result)

    def testWithAllShuffleExecution(self):
        # no axis is monotonic
        data1 = pd.DataFrame(np.random.rand(10, 10), index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=5)
        data2 = pd.DataFrame(np.random.rand(10, 10), index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3],
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = from_pandas(data2, chunk_size=6)

        df3 = self.func(df1, df2)

        expected = self.func(data1, data2)
        result = self.executor.execute_dataframe(df3, concat=True)[0]

        pd.testing.assert_frame_equal(expected, result)

    def testBothWithOneChunk(self):
        # only 1 axis is monotonic
        # data1 with index split into [0...4], [5...9],
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=10)
        # data2 with index split into [6...11], [2, 5],
        data2 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(11, 1, -1),
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = from_pandas(data2, chunk_size=10)

        df3 = self.func(df1, df2)

        expected = self.func(data1, data2)
        result = self.executor.execute_dataframe(df3, concat=True)[0]

        pd.testing.assert_frame_equal(expected, result)

        # only 1 axis is monotonic
        # data1 with columns split into [0...4], [5...9],
        data1 = pd.DataFrame(np.random.rand(10, 10), index=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
                             columns=np.arange(10))
        df1 = from_pandas(data1, chunk_size=10)
        # data2 with columns split into [6...11], [2, 5],
        data2 = pd.DataFrame(np.random.rand(10, 10), index=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2],
                             columns=np.arange(11, 1, -1))
        df2 = from_pandas(data2, chunk_size=10)

        df3 = self.func(df1, df2)

        expected = self.func(data1, data2)
        result = self.executor.execute_dataframe(df3, concat=True)[0]

        pd.testing.assert_frame_equal(expected, result)

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

        expected = self.func(data1, data2)
        result = self.executor.execute_dataframe(df3, concat=True)[0]

        pd.testing.assert_frame_equal(expected, result)

        # only 1 axis is monotonic
        # data1 with columns split into [0...4], [5...9],
        data1 = pd.DataFrame(np.random.rand(10, 10), index=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
                             columns=np.arange(10))
        df1 = from_pandas(data1, chunk_size=(10, 5))
        # data2 with columns split into [6...11], [2, 5],
        data2 = pd.DataFrame(np.random.rand(10, 10), index=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2],
                             columns=np.arange(11, 1, -1))
        df2 = from_pandas(data2, chunk_size=(10, 6))

        df3 = self.func(df1, df2)

        expected = self.func(data1, data2)
        result = self.executor.execute_dataframe(df3, concat=True)[0]

        pd.testing.assert_frame_equal(expected, result)

    def testWithShuffleAndWithOneChunk(self):
        # only 1 axis is monotonic
        # data1 with index split into [0...4], [5...9],
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=(10, 5))
        # data2 with index split into [6...11], [2, 5],
        data2 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(11, 1, -1),
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = from_pandas(data2, chunk_size=(10, 6))

        df3 = self.func(df1, df2)

        expected = self.func(data1, data2)
        result = self.executor.execute_dataframe(df3, concat=True)[0]

        pd.testing.assert_frame_equal(expected, result)

        # only 1 axis is monotonic
        # data1 with columns split into [0...4], [5...9],
        data1 = pd.DataFrame(np.random.rand(10, 10), index=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7],
                             columns=np.arange(10))
        df1 = from_pandas(data1, chunk_size=(5, 10))
        # data2 with columns split into [6...11], [2, 5],
        data2 = pd.DataFrame(np.random.rand(10, 10), index=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2],
                             columns=np.arange(11, 1, -1))
        df2 = from_pandas(data2, chunk_size=(6, 10))

        df3 = self.func(df1, df2)

        expected = self.func(data1, data2)
        result = self.executor.execute_dataframe(df3, concat=True)[0]

        pd.testing.assert_frame_equal(expected, result)

    def testSameIndex(self):
        data = pd.DataFrame(np.random.rand(10, 10), index=np.random.randint(0, 2, size=(10,)),
                            columns=['c' + str(i) for i in range(10)])
        df = from_pandas(data, chunk_size=3)
        df2 = self.func(df, df)

        expected = self.func(data, data)
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        series = from_pandas_series(data.iloc[0], chunk_size=3)
        df3 = self.func(df, series)

        expected = self.func(data, data.iloc[0])
        result = self.executor.execute_dataframe(df3, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        series = from_pandas_series(data.iloc[:, 0], chunk_size=3)
        df4 = getattr(df, self.func_name)(series, axis=0)

        expected = getattr(data, self.func_name)(data.iloc[:, 0], axis=0)
        result = self.executor.execute_dataframe(df4, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

    def testChained(self):
        data1 = pd.DataFrame(np.random.rand(10, 10))
        df1 = from_pandas(data1, chunk_size=5)
        data2 = pd.DataFrame(np.random.rand(10, 10))
        df2 = from_pandas(data2, chunk_size=6)

        df3 = self.func(df1, df2)

        data4 = pd.DataFrame(np.random.rand(10, 10))
        df4 = from_pandas(data4, chunk_size=6)

        df5 = self.func(df3, df4)

        result = self.executor.execute_dataframe(df5, concat=True)[0]
        expected = self.func(self.func(data1, data2), data4)

        pd.testing.assert_frame_equal(expected, result)

    def testRfunc(self):
        data1 = pd.DataFrame(np.random.rand(10, 10))
        df1 = from_pandas(data1, chunk_size=5)
        data2 = pd.DataFrame(np.random.rand(10, 10))
        df2 = from_pandas(data2, chunk_size=6)
        df3 = getattr(df1, self.rfunc_name)(df2)
        result = self.executor.execute_dataframe(df3, concat=True)[0]
        expected = self.func(data2, data1)
        pd.testing.assert_frame_equal(expected, result)

        data3 = pd.DataFrame(np.random.rand(10, 10))
        df4 = from_pandas(data3, chunk_size=5)
        df5 = getattr(df4, self.rfunc_name)(1)
        result = self.executor.execute_dataframe(df5, concat=True)[0]
        expected2 = self.func(1, data3)
        pd.testing.assert_frame_equal(expected2, result)

    def testWithMultiForms(self):
        # test multiple forms
        # such as self+other, self.add(other), add(self,other)
        data1 = pd.DataFrame(np.random.rand(10, 10))
        df1 = from_pandas(data1, chunk_size=5)
        data2 = pd.DataFrame(np.random.rand(10, 10))
        df2 = from_pandas(data2, chunk_size=6)

        expected = self.func(data1, data2)
        result = self.executor.execute_dataframe(self.func(df1, df2), concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)
        result = self.executor.execute_dataframe(self.func(df1, df2), concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)
        result = self.executor.execute_dataframe(getattr(df1, self.func_name)(df2), concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)
        result = self.executor.execute_dataframe(getattr(df1, self.rfunc_name)(df2), concat=True)[0]
        pd.testing.assert_frame_equal(self.func(data2, data1), result)

    def testDataframeAndScalar(self):
        # test dataframe and scalar
        pdf = pd.DataFrame(np.random.rand(10, 10))
        df = from_pandas(pdf, chunk_size=2)
        expected = self.func(pdf, 1)
        result = self.executor.execute_dataframe(self.func(df, 1), concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)
        result2 = self.executor.execute_dataframe(self.func(df, 1), concat=True)[0]
        pd.testing.assert_frame_equal(expected, result2)
        result3 = self.executor.execute_dataframe(getattr(df, self.func_name)(1), concat=True)[0]
        pd.testing.assert_frame_equal(expected, result3)

        # test scalar and dataframe
        result4 = self.executor.execute_dataframe(self.func(df, 1), concat=True)[0]
        pd.testing.assert_frame_equal(expected, result4)

        expected2 = self.func(1, pdf)
        result5 = self.executor.execute_dataframe(self.func(1, df), concat=True)[0]
        pd.testing.assert_frame_equal(expected2, result5)

        result6 = self.executor.execute_dataframe(getattr(df, self.rfunc_name)(1), concat=True)[0]
        pd.testing.assert_frame_equal(expected2, result6)

    def testWithShuffleOnStringIndex(self):
        # no axis is monotonic, and the index values are strings.
        data1 = pd.DataFrame(np.random.rand(10, 10), index=[str(x) for x in [0, 10, 2, 3, 4, 5, 6, 7, 8, 9]],
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=5)
        data2 = pd.DataFrame(np.random.rand(10, 10), index=[str(x) for x in [11, 1, 2, 5, 7, 6, 8, 9, 10, 3]],
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = from_pandas(data2, chunk_size=6)

        df3 = self.func(df1, df2)

        expected = self.func(data1, data2)
        result = self.executor.execute_dataframe(df3, concat=True)[0]

        pd.testing.assert_frame_equal(expected, result)

    def testDataframeAndSeries(self):
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        data2 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(11, 1, -1),
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])

        s1 = from_pandas_series(data2[1], chunk_size=(6,))

        # operate on single-column dataframe and series
        df1 = from_pandas(data1[[1]], chunk_size=(5, 5))
        r1 = getattr(df1, self.func_name)(s1, axis='index')

        expected = getattr(data1[[1]], self.func_name)(data2[1], axis='index')
        result = self.executor.execute_dataframe(r1, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # operate on dataframe and series without shuffle
        df2 = from_pandas(data1, chunk_size=(5, 5))
        r2 = getattr(df2, self.func_name)(s1, axis='index')

        expected = getattr(data1, self.func_name)(data2[1], axis='index')
        result = self.executor.execute_dataframe(r2, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # operate on dataframe and series with shuffle
        df3 = from_pandas(data1, chunk_size=(5, 5))
        r3 = getattr(df3, self.func_name)(s1, axis='columns')

        expected = getattr(data1, self.func_name)(data2[1], axis='columns')
        result = self.executor.execute_dataframe(r3, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # test both one chunk, axis=0
        pdf = pd.DataFrame({'ca': [1, 3, 2], 'cb': [360, 180, 2]}, index=[1, 2, 3])
        df = from_pandas(pdf)
        series = pd.Series([0, 1, 2], index=[1, 2, 3])
        mars_series = from_pandas_series(series)
        result = self.executor.execute_dataframe(getattr(df, self.func_name)(mars_series, axis=0), concat=True)[0]
        expected = getattr(pdf, self.func_name)(series, axis=0)
        pd.testing.assert_frame_equal(expected, result)

        # test different number of chunks, axis=0
        pdf = pd.DataFrame({'ca': [1, 3, 2], 'cb': [360, 180, 2]}, index=[1, 2, 3])
        df = from_pandas(pdf, chunk_size=1)
        series = pd.Series([0, 1, 2], index=[1, 2, 3])
        mars_series = from_pandas_series(series)
        result = self.executor.execute_dataframe(getattr(df, self.func_name)(mars_series, axis=0), concat=True)[0]
        expected = getattr(pdf, self.func_name)(series, axis=0)
        pd.testing.assert_frame_equal(expected, result)

        # test with row shuffle, axis=0
        pdf = pd.DataFrame({'ca': [1, 3, 2], 'cb': [360, 180, 2]}, index=[2, 1, 3])
        df = from_pandas(pdf, chunk_size=1)
        series = pd.Series([0, 1, 2], index=[3, 1, 2])
        mars_series = from_pandas_series(series)
        result = self.executor.execute_dataframe(getattr(df, self.func_name)(mars_series, axis=0), concat=True)[0]
        expected = getattr(pdf, self.func_name)(series, axis=0).reindex([3, 1, 2])
        # modify the order of rows
        result = result.reindex(index=[3, 1, 2])
        pd.testing.assert_frame_equal(expected, result)

        # test both one chunk, axis=1
        pdf = pd.DataFrame({1: [1, 3, 2], 2: [360, 180, 2], 3: [1, 2, 3]}, index=['ra', 'rb', 'rc'])
        df = from_pandas(pdf)
        series = pd.Series([0, 1, 2], index=[1, 2, 3])
        mars_series = from_pandas_series(series)
        result = self.executor.execute_dataframe(getattr(df, self.func_name)(mars_series, axis=1), concat=True)[0]
        expected = getattr(pdf, self.func_name)(series, axis=1)
        pd.testing.assert_frame_equal(expected, result)

        # test different number of chunks, axis=1
        pdf = pd.DataFrame({1: [1, 3, 2], 2: [360, 180, 2], 3: [1, 2, 3]}, index=['ra', 'rb', 'rc'])
        df = from_pandas(pdf, chunk_size=1)
        series = pd.Series([0, 1, 2], index=[1, 2, 3])
        mars_series = from_pandas_series(series)
        result = self.executor.execute_dataframe(getattr(df, self.func_name)(mars_series, axis=1), concat=True)[0]
        expected = getattr(pdf, self.func_name)(series, axis=1)
        pd.testing.assert_frame_equal(expected, result)

        # test with row shuffle, axis=1
        pdf = pd.DataFrame({1: [1, 3, 2], 3: [1, 2, 3], 2: [360, 180, 2]}, index=['ra', 'rb', 'rc'])
        df = from_pandas(pdf, chunk_size=1)
        series = pd.Series([0, 1, 2], index=[3, 1, 2])
        mars_series = from_pandas_series(series)
        result = self.executor.execute_dataframe(getattr(df, self.func_name)(mars_series, axis=1), concat=True)[0]
        expected = getattr(pdf, self.func_name)(series, axis=1)
        # modify the order of columns
        result = result[[1, 2, 3]]
        pd.testing.assert_frame_equal(expected, result)

    def testSeries(self):
        # only one chunk
        s1 = pd.Series(np.arange(10) + 1)
        s2 = pd.Series(np.arange(10) + 1)
        r = self.func(from_pandas_series(s1, chunk_size=10), from_pandas_series(s2, chunk_size=10))
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = self.func(s1, s2)
        pd.testing.assert_series_equal(expected, result)

        # same index
        s1 = pd.Series(np.arange(10) + 1)
        s2 = pd.Series(np.arange(10) + 1)
        r = self.func(from_pandas_series(s1, chunk_size=4), from_pandas_series(s2, chunk_size=6))
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = self.func(s1, s2)
        pd.testing.assert_series_equal(expected, result)

        # no shuffle
        s1 = pd.Series(np.arange(10) + 1, index=range(10))
        s2 = pd.Series(np.arange(10) + 1, index=range(10, 0, -1))
        r = self.func(from_pandas_series(s1, chunk_size=4), from_pandas_series(s2, chunk_size=6))
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = self.func(s1, s2)
        pd.testing.assert_series_equal(expected, result)

        # shuffle
        s1 = pd.Series(np.arange(10) + 1, index=np.random.permutation(range(10)))
        s2 = pd.Series(np.arange(10) + 1, index=np.random.permutation(range(10, 0, -1)))
        r = self.func(from_pandas_series(s1, chunk_size=4), from_pandas_series(s2, chunk_size=6))
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = self.func(s1, s2)
        pd.testing.assert_series_equal(expected, result)

        # operate with scalar
        s1 = pd.Series(np.arange(10) + 1, index=np.random.permutation(range(10)))
        r = self.func(from_pandas_series(s1, chunk_size=4), 4)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = self.func(s1, 4)
        pd.testing.assert_series_equal(expected, result)

        # reverse with scalar
        s1 = pd.Series(np.arange(10) + 1, index=np.random.permutation(range(10)))
        r = self.func(4, from_pandas_series(s1, chunk_size=4))
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = self.func(4, s1)
        pd.testing.assert_series_equal(expected, result)

    def testWithPlainValue(self):
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=6)
        s1 = df1[2]

        r = getattr(df1, self.func_name)([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], axis=0)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = getattr(data1, self.func_name)([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], axis=0)
        pd.testing.assert_frame_equal(expected, result)

        r = getattr(df1, self.func_name)((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), axis=0)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = getattr(data1, self.func_name)((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), axis=0)
        pd.testing.assert_frame_equal(expected, result)

        r = getattr(s1, self.func_name)([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = getattr(data1[2], self.func_name)([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        pd.testing.assert_series_equal(expected, result)

        r = getattr(s1, self.func_name)((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = getattr(data1[2], self.func_name)((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
        pd.testing.assert_series_equal(expected, result)

        # specify index, not the default range index
        data1 = pd.DataFrame(np.random.rand(10, 7), index=np.arange(5, 15),
                             columns=[4, 1, 3, 2, 5, 6, 7])
        df1 = from_pandas(data1, chunk_size=6)
        s1 = df1[2]

        r = getattr(df1, self.func_name)(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), axis=0)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = getattr(data1, self.func_name)(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), axis=0)
        pd.testing.assert_frame_equal(expected, result)

        r = getattr(df1, self.func_name)(from_array(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])), axis=0)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = getattr(data1, self.func_name)(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), axis=0)
        pd.testing.assert_frame_equal(expected, result)

        r = getattr(s1, self.func_name)(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = getattr(data1[2], self.func_name)(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        pd.testing.assert_series_equal(expected, result)

        r = getattr(s1, self.func_name)(from_array(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = getattr(data1[2], self.func_name)(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        pd.testing.assert_series_equal(expected, result)


class TestUnary(TestBase):
    def setUp(self):
        super(TestUnary, self).setUp()
        self.executor = Executor()

    def testAbs(self):
        data1 = pd.DataFrame(np.random.uniform(low=-1, high=1, size=(10, 10)))
        df1 = from_pandas(data1, chunk_size=5)

        result = self.executor.execute_dataframe(abs(df1), concat=True)[0]
        expected = data1.abs()
        pd.testing.assert_frame_equal(expected, result)
