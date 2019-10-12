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

import unittest

import numpy as np
import pandas as pd

from mars.executor import Executor
from mars.tests.core import TestBase
from mars.tensor.datasource import array as from_array
from mars.dataframe.datasource.dataframe import from_pandas
from mars.dataframe.datasource.series import from_pandas as from_pandas_series
from mars.dataframe.arithmetic import abs, add, radd, floordiv, rfloordiv, truediv, rtruediv


@unittest.skipIf(pd is None, 'pandas not installed')
class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.executor = Executor()

    def testAddWithoutShuffleExecution(self):
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

        df3 = add(df1, df2)

        expected = data1 + data2
        result = self.executor.execute_dataframe(df3, concat=True)[0]

        pd.testing.assert_frame_equal(expected, result)

    def testAddWithOneShuffleExecution(self):
        # only 1 axis is monotonic
        # data1 with index split into [0...4], [5...9],
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=5)
        # data2 with index split into [6...11], [2, 5],
        data2 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(11, 1, -1),
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = from_pandas(data2, chunk_size=6)

        df3 = add(df1, df2)

        expected = data1 + data2
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

        df3 = add(df1, df2)

        expected = data1 + data2
        result = self.executor.execute_dataframe(df3, concat=True)[0]

        pd.testing.assert_frame_equal(expected, result)

    def testAddWithAllShuffleExecution(self):
        # no axis is monotonic
        data1 = pd.DataFrame(np.random.rand(10, 10), index=[0, 10, 2, 3, 4, 5, 6, 7, 8, 9],
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=5)
        data2 = pd.DataFrame(np.random.rand(10, 10), index=[11, 1, 2, 5, 7, 6, 8, 9, 10, 3],
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = from_pandas(data2, chunk_size=6)

        df3 = add(df1, df2)

        expected = data1 + data2
        result = self.executor.execute_dataframe(df3, concat=True)[0]

        pd.testing.assert_frame_equal(expected, result)

    def testAddBothWithOneChunk(self):
        # only 1 axis is monotonic
        # data1 with index split into [0...4], [5...9],
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=10)
        # data2 with index split into [6...11], [2, 5],
        data2 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(11, 1, -1),
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = from_pandas(data2, chunk_size=10)

        df3 = add(df1, df2)

        expected = data1 + data2
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

        df3 = add(df1, df2)

        expected = data1 + data2
        result = self.executor.execute_dataframe(df3, concat=True)[0]

        pd.testing.assert_frame_equal(expected, result)

    def testAddWithoutShuffleAndWithOneChunk(self):
        # only 1 axis is monotonic
        # data1 with index split into [0...4], [5...9],
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=(5, 10))
        # data2 with index split into [6...11], [2, 5],
        data2 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(11, 1, -1),
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = from_pandas(data2, chunk_size=(6, 10))

        df3 = add(df1, df2)

        expected = data1 + data2
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

        df3 = add(df1, df2)

        expected = data1 + data2
        result = self.executor.execute_dataframe(df3, concat=True)[0]

        pd.testing.assert_frame_equal(expected, result)

    def testAddWithShuffleAndWithOneChunk(self):
        # only 1 axis is monotonic
        # data1 with index split into [0...4], [5...9],
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=(10, 5))
        # data2 with index split into [6...11], [2, 5],
        data2 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(11, 1, -1),
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = from_pandas(data2, chunk_size=(10, 6))

        df3 = add(df1, df2)

        expected = data1 + data2
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

        df3 = add(df1, df2)

        expected = data1 + data2
        result = self.executor.execute_dataframe(df3, concat=True)[0]

        pd.testing.assert_frame_equal(expected, result)

    def testAddWithAdded(self):
        data1 = pd.DataFrame(np.random.rand(10, 10))
        df1 = from_pandas(data1, chunk_size=5)
        data2 = pd.DataFrame(np.random.rand(10, 10))
        df2 = from_pandas(data2, chunk_size=6)

        df3 = add(df1, df2)

        data4 = pd.DataFrame(np.random.rand(10, 10))
        df4 = from_pandas(data4, chunk_size=6)

        df5 = add(df3, df4)

        result = self.executor.execute_dataframe(df5, concat=True)[0]
        expected = data1 + data2 + data4

        pd.testing.assert_frame_equal(expected, result)

    def testRadd(self):
        data1 = pd.DataFrame(np.random.rand(10, 10))
        df1 = from_pandas(data1, chunk_size=5)
        data2 = pd.DataFrame(np.random.rand(10, 10))
        df2 = from_pandas(data2, chunk_size=6)
        df3 = df1.radd(df2)
        result = self.executor.execute_dataframe(df3, concat=True)[0]
        expected = data1 + data2
        pd.testing.assert_frame_equal(expected, result)

        data3 = pd.DataFrame(np.random.rand(10, 10))
        df4 = from_pandas(data3, chunk_size=5)
        df5 = df4.radd(1)
        result = self.executor.execute_dataframe(df5, concat=True)[0]
        expected2 = data3 + 1
        pd.testing.assert_frame_equal(expected2, result)

    def testAddWithMultiForms(self):
        # test multiple forms of add
        # such as self+other, self.add(other), add(self,other)
        data1 = pd.DataFrame(np.random.rand(10, 10))
        df1 = from_pandas(data1, chunk_size=5)
        data2 = pd.DataFrame(np.random.rand(10, 10))
        df2 = from_pandas(data2, chunk_size=6)

        expected = data1 + data2
        result = self.executor.execute_dataframe(df1 + df2, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)
        result = self.executor.execute_dataframe(add(df1, df2), concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)
        result = self.executor.execute_dataframe(df1.add(df2), concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)
        result = self.executor.execute_dataframe(df1.radd(df2), concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

    def testAddDataframeAndScalar(self):
        # test dataframe + scalar
        pdf = pd.DataFrame(np.random.rand(10, 10))
        df = from_pandas(pdf, chunk_size=2)
        expected = pdf + 1
        result = self.executor.execute_dataframe(add(df, 1), concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)
        result2 = self.executor.execute_dataframe(df + 1, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result2)
        result3 = self.executor.execute_dataframe(df.add(1), concat=True)[0]
        pd.testing.assert_frame_equal(expected, result3)

        # test scalar + dataframe
        result4 = self.executor.execute_dataframe(radd(df, 1), concat=True)[0]
        pd.testing.assert_frame_equal(expected, result4)

        expected2 = 1 + pdf
        result5 = self.executor.execute_dataframe(1 + df, concat=True)[0]
        pd.testing.assert_frame_equal(expected2, result5)

        result6 = self.executor.execute_dataframe(df.radd(1), concat=True)[0]
        pd.testing.assert_frame_equal(expected2, result6)

    def testAddWithShuffleOnStringIndex(self):
        # no axis is monotonic, and the index values are strings.
        data1 = pd.DataFrame(np.random.rand(10, 10), index=[str(x) for x in [0, 10, 2, 3, 4, 5, 6, 7, 8, 9]],
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=5)
        data2 = pd.DataFrame(np.random.rand(10, 10), index=[str(x) for x in [11, 1, 2, 5, 7, 6, 8, 9, 10, 3]],
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])
        df2 = from_pandas(data2, chunk_size=6)

        df3 = add(df1, df2)

        expected = data1 + data2
        result = self.executor.execute_dataframe(df3, concat=True)[0]

        pd.testing.assert_frame_equal(expected, result)

    def testAddDataframeAndSeries(self):
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        data2 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(11, 1, -1),
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])

        s1 = from_pandas_series(data2[1], chunk_size=(6,))

        # add single-column dataframe to series
        df1 = from_pandas(data1[[1]], chunk_size=(5, 5))
        r1 = add(df1, s1, axis='index')

        expected = data1[[1]].add(data2[1], axis='index')
        result = self.executor.execute_dataframe(r1, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # add dataframe to series without shuffle
        df2 = from_pandas(data1, chunk_size=(5, 5))
        r2 = add(df2, s1, axis='index')

        expected = data1.add(data2[1], axis='index')
        result = self.executor.execute_dataframe(r2, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # add dataframe to series with shuffle
        df3 = from_pandas(data1, chunk_size=(5, 5))
        r3 = add(df3, s1, axis='columns')

        expected = data1.add(data2[1], axis='columns')
        result = self.executor.execute_dataframe(r3, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # test both one chunk, axis=0
        pdf = pd.DataFrame({'ca': [1, 3, 2], 'cb': [360, 180, 2]}, index=[1, 2, 3])
        df = from_pandas(pdf)
        series = pd.Series([0, 1, 2], index=[1, 2, 3])
        mars_series = from_pandas_series(series)
        result = self.executor.execute_dataframe(df.add(mars_series, axis=0), concat=True)[0]
        expected = pdf.add(series, axis=0)
        pd.testing.assert_frame_equal(expected, result)

        # test different number of chunks, axis=0
        pdf = pd.DataFrame({'ca': [1, 3, 2], 'cb': [360, 180, 2]}, index=[1, 2, 3])
        df = from_pandas(pdf, chunk_size=1)
        series = pd.Series([0, 1, 2], index=[1, 2, 3])
        mars_series = from_pandas_series(series)
        result = self.executor.execute_dataframe(df.add(mars_series, axis=0), concat=True)[0]
        expected = pdf.add(series, axis=0)
        pd.testing.assert_frame_equal(expected, result)

        # test with row shuffle, axis=0
        pdf = pd.DataFrame({'ca': [1, 3, 2], 'cb': [360, 180, 2]}, index=[2, 1, 3])
        df = from_pandas(pdf, chunk_size=1)
        series = pd.Series([0, 1, 2], index=[3, 1, 2])
        mars_series = from_pandas_series(series)
        result = self.executor.execute_dataframe(df.add(mars_series, axis=0), concat=True)[0]
        expected = pdf.add(series, axis=0).reindex([3, 1, 2])
        # modify the order of rows
        result = result.reindex(index=[3, 1, 2])
        pd.testing.assert_frame_equal(expected, result)

        # test both one chunk, axis=1
        pdf = pd.DataFrame({1: [1, 3, 2], 2: [360, 180, 2], 3: [1, 2, 3]}, index=['ra', 'rb', 'rc'])
        df = from_pandas(pdf)
        series = pd.Series([0, 1, 2], index=[1, 2, 3])
        mars_series = from_pandas_series(series)
        result = self.executor.execute_dataframe(df.add(mars_series, axis=1), concat=True)[0]
        expected = pdf.add(series, axis=1)
        pd.testing.assert_frame_equal(expected, result)

        # test different number of chunks, axis=1
        pdf = pd.DataFrame({1: [1, 3, 2], 2: [360, 180, 2], 3: [1, 2, 3]}, index=['ra', 'rb', 'rc'])
        df = from_pandas(pdf, chunk_size=1)
        series = pd.Series([0, 1, 2], index=[1, 2, 3])
        mars_series = from_pandas_series(series)
        result = self.executor.execute_dataframe(df.add(mars_series, axis=1), concat=True)[0]
        expected = pdf.add(series, axis=1)
        pd.testing.assert_frame_equal(expected, result)

        # test with row shuffle, axis=1
        pdf = pd.DataFrame({1: [1, 3, 2], 3: [1, 2, 3], 2: [360, 180, 2]}, index=['ra', 'rb', 'rc'])
        df = from_pandas(pdf, chunk_size=1)
        series = pd.Series([0, 1, 2], index=[3, 1, 2])
        mars_series = from_pandas_series(series)
        result = self.executor.execute_dataframe(df.add(mars_series, axis=1), concat=True)[0]
        expected = pdf.add(series, axis=1)
        # modify the order of columns
        result = result[[1, 2, 3]]
        pd.testing.assert_frame_equal(expected, result)

    def testDataframeFloorDiv(self):
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        data2 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(11, 1, -1),
                             columns=[5, 9, 12, 3, 11, 10, 6, 4, 1, 2])

        s1 = from_pandas_series(data2[1], chunk_size=(6,))

        # div single-column dataframe with series
        df1 = from_pandas(data1[[1]], chunk_size=(5, 5))
        r1 = truediv(df1, s1, axis='index')

        expected = data1[[1]].truediv(data2[1], axis='index')
        result = self.executor.execute_dataframe(r1, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # div dataframe with series without shuffle
        df2 = from_pandas(data1, chunk_size=(5, 5))
        r2 = truediv(df2, s1, axis='index')

        expected = data1.truediv(data2[1], axis='index')
        result = self.executor.execute_dataframe(r2, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # div dataframe with series with shuffle
        df3 = from_pandas(data1, chunk_size=(5, 5))
        r3 = rtruediv(df3, s1, axis='columns')

        expected = data1.rtruediv(data2[1], axis='columns')
        result = self.executor.execute_dataframe(r3, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # div dataframe with scalar
        df4 = from_pandas(data1, chunk_size=(5, 5))
        r4 = truediv(df4, 4, axis='columns')

        expected = data1.truediv(4, axis='columns')
        result = self.executor.execute_dataframe(r4, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # rdiv dataframe with scalar
        df5 = from_pandas(data1, chunk_size=(5, 5))
        r5 = rtruediv(df5, 4, axis='columns')

        expected = data1.rtruediv(4, axis='columns')
        result = self.executor.execute_dataframe(r5, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # test both one chunk, axis=0
        pdf = pd.DataFrame({'ca': [1, 3, 2], 'cb': [360, 180, 2]}, index=[1, 2, 3])
        df = from_pandas(pdf)
        series = pd.Series([0, 1, 2], index=[1, 2, 3])
        mars_series = from_pandas_series(series)
        result = self.executor.execute_dataframe(df.rtruediv(mars_series, axis=0), concat=True)[0]
        expected = pdf.rtruediv(series, axis=0)
        pd.testing.assert_frame_equal(expected, result)

        # test different number of chunks, axis=0
        pdf = pd.DataFrame({'ca': [1, 3, 2], 'cb': [360, 180, 2]}, index=[1, 2, 3])
        df = from_pandas(pdf, chunk_size=1)
        series = pd.Series([0, 1, 2], index=[1, 2, 3])
        mars_series = from_pandas_series(series)
        result = self.executor.execute_dataframe(df.truediv(mars_series, axis=0), concat=True)[0]
        expected = pdf.truediv(series, axis=0)
        pd.testing.assert_frame_equal(expected, result)

        # test with row shuffle, axis=0
        pdf = pd.DataFrame({'ca': [1, 3, 2], 'cb': [360, 180, 2]}, index=[2, 1, 3])
        df = from_pandas(pdf, chunk_size=1)
        series = pd.Series([0, 1, 2], index=[3, 1, 2])
        mars_series = from_pandas_series(series)
        result = self.executor.execute_dataframe(df.rtruediv(mars_series, axis=0), concat=True)[0]
        expected = pdf.rtruediv(series, axis=0).reindex([3, 1, 2])
        # modify the order of rows
        result = result.reindex(index=[3, 1, 2])
        pd.testing.assert_frame_equal(expected, result)

        # test both one chunk, axis=1
        pdf = pd.DataFrame({1: [1, 3, 2], 2: [360, 180, 2], 3: [1, 2, 3]}, index=['ra', 'rb', 'rc'])
        df = from_pandas(pdf)
        series = pd.Series([0, 1, 2], index=[1, 2, 3])
        mars_series = from_pandas_series(series)
        result = self.executor.execute_dataframe(df.truediv(mars_series, axis=1), concat=True)[0]
        expected = pdf.truediv(series, axis=1)
        pd.testing.assert_frame_equal(expected, result)

        # test different number of chunks(axis = 1)
        pdf = pd.DataFrame({1: [1, 3, 2], 2: [360, 180, 2], 3: [1, 2, 3]}, index=['ra', 'rb', 'rc'])
        df = from_pandas(pdf, chunk_size=1)
        series = pd.Series([0, 1, 2], index=[1, 2, 3])
        mars_series = from_pandas_series(series)
        result = self.executor.execute_dataframe(df.rtruediv(mars_series, axis=1), concat=True)[0]
        expected = pdf.rtruediv(series, axis=1)
        pd.testing.assert_frame_equal(expected, result)

        # test with row shuffle, axis=1
        pdf = pd.DataFrame({1: [1, 3, 2], 3: [1, 2, 3], 2: [360, 180, 2]}, index=['ra', 'rb', 'rc'])
        df = from_pandas(pdf, chunk_size=1)
        series = pd.Series([0, 1, 2], index=[3, 1, 2])
        mars_series = from_pandas_series(series)
        result = self.executor.execute_dataframe(df.truediv(mars_series, axis=1), concat=True)[0]
        expected = pdf.truediv(series, axis=1)
        # modify the order of columns
        result = result[[1, 2, 3]]
        pd.testing.assert_frame_equal(expected, result)

        # test with floor div
        pdf = pd.DataFrame({1: [1, 3, 2, 4, 3, 5, 4, 3, 2],
                            2: [360, 180, 2, 1, 4, 3, 5, 1, 100],
                            3: [1, 2, 3, 9, 8, 9, 2, 4, 4]},
                           index=['ra', 'rb', 'rc', 'rd', 're', 'rf', 'rg', 'rh', 'ri'])
        df = from_pandas(pdf)
        series = pd.Series([9, 3, 7], index=[3, 1, 2])
        mars_series = from_pandas_series(series)

        result = self.executor.execute_dataframe(floordiv(df, mars_series, axis=1), concat=True)[0]
        expected = pdf.floordiv(series, axis=1)
        pd.testing.assert_frame_equal(expected, result)

        result = self.executor.execute_dataframe(rfloordiv(df, mars_series, axis=1), concat=True)[0]
        expected = pdf.rfloordiv(series, axis=1)
        pd.testing.assert_frame_equal(expected, result)

    def testAddSeries(self):
        # only one chunk
        s1 = pd.Series(range(10))
        s2 = pd.Series(range(10))
        r = from_pandas_series(s1, chunk_size=10) + from_pandas_series(s2, chunk_size=10)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s1 + s2
        pd.testing.assert_series_equal(expected, result)

        # same index
        s1 = pd.Series(range(10))
        s2 = pd.Series(range(10))
        r = from_pandas_series(s1, chunk_size=4) + from_pandas_series(s2, chunk_size=6)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s1 + s2
        pd.testing.assert_series_equal(expected, result)

        # no shuffle
        s1 = pd.Series(range(10), index=range(10))
        s2 = pd.Series(range(10), index=range(10, 0, -1))
        r = from_pandas_series(s1, chunk_size=4) + from_pandas_series(s2, chunk_size=6)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s1 + s2
        pd.testing.assert_series_equal(expected, result)

        # shuffle
        s1 = pd.Series(range(10), index=np.random.permutation(range(10)))
        s2 = pd.Series(range(10), index=np.random.permutation(range(10, 0, -1)))
        r = from_pandas_series(s1, chunk_size=4) + from_pandas_series(s2, chunk_size=6)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s1 + s2
        pd.testing.assert_series_equal(expected, result)

        # add with scalar
        s1 = pd.Series(range(10), index=np.random.permutation(range(10)))
        r = from_pandas_series(s1, chunk_size=4) + 4
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = s1 + 4
        pd.testing.assert_series_equal(expected, result)

        # radd with scalar
        s1 = pd.Series(range(10), index=np.random.permutation(range(10)))
        r = 4 + from_pandas_series(s1, chunk_size=4)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = 4 + s1
        pd.testing.assert_series_equal(expected, result)

    def testAddWithPlainValue(self):
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=6)
        s1 = df1[2]

        r = df1.add([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], axis=0)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = data1.add([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], axis=0)
        pd.testing.assert_frame_equal(expected, result)

        r = df1.add((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), axis=0)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = data1.add((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), axis=0)
        pd.testing.assert_frame_equal(expected, result)

        r = df1.add(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), axis=0)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = data1.add(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), axis=0)
        pd.testing.assert_frame_equal(expected, result)

        r = df1.add(from_array(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])), axis=0)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = data1.add(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), axis=0)
        pd.testing.assert_frame_equal(expected, result)

        r = s1.add([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = data1[2].add([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        pd.testing.assert_series_equal(expected, result)

        r = s1.add((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = data1[2].add((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
        pd.testing.assert_series_equal(expected, result)

        r = s1.add(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = data1[2].add(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        pd.testing.assert_series_equal(expected, result)

        r = s1.add(from_array(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = data1[2].add(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        pd.testing.assert_series_equal(expected, result)

    @unittest.expectedFailure
    def testAddWithPlainValueUnaligned(self):
        # When adding dataframe with a sequence value, pandas treats the sequence
        # as a series using the index_value of the dataframe.
        #
        # In mars we cannot do such things because the index_value is not stored.
        # We also cannot split the sequence using the nsplits of the dataframe since
        # in many cases the shape of the dataframe chunks is np.nan.
        #
        # We record this case as `expectedFailure`.
        data1 = pd.DataFrame(np.random.rand(10, 10), index=np.arange(10),
                             columns=[4, 1, 3, 2, 10, 5, 9, 8, 6, 7])
        df1 = from_pandas(data1, chunk_size=6)

        r = df1.add([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], axis=1)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = data1.add([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], axis=0)
        pd.testing.assert_frame_equal(expected, result)

    def testAbs(self):
        data1 = pd.DataFrame(np.random.uniform(low=-1, high=1, size=(10, 10)))
        df1 = from_pandas(data1, chunk_size=5)

        result = self.executor.execute_dataframe(abs(df1), concat=True)[0]
        expected = data1.abs()
        pd.testing.assert_frame_equal(expected, result)
