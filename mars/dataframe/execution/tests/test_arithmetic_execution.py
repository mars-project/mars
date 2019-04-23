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
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from mars.executor import Executor
from mars.tests.core import TestBase
from mars.dataframe.expressions.datasource.dataframe import from_pandas
from mars.dataframe.expressions.arithmetic import add


@unittest.skipIf(pd is None, 'pandas not installed')
class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.executor = Executor()

    @staticmethod
    def _concat(index, dtypes, results):
        empty = pd.DataFrame(index=index)
        for c, d in zip(dtypes.index, dtypes):
            empty[c] = pd.Series(dtype=d)
        for df in results:
            empty.loc[df.index, df.columns] = df
        return empty

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

        graph = df3.build_graph(tiled=True)
        results = self.executor.execute_graph(graph, keys=[c.key for c in df3.chunks])

        expected = data1 + data2
        result = self._concat(expected.index, expected.dtypes, results)

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

        graph = df3.build_graph(tiled=True)
        results = self.executor.execute_graph(graph, keys=[c.key for c in df3.chunks])

        expected = data1 + data2
        result = self._concat(expected.index, expected.dtypes, results)

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

        graph = df3.build_graph(tiled=True)
        results = self.executor.execute_graph(graph, keys=[c.key for c in df3.chunks])

        expected = data1 + data2
        result = self._concat(expected.index, expected.dtypes, results)

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

        graph = df3.build_graph(tiled=True, compose=False)
        results = self.executor.execute_graph(graph, keys=[c.key for c in df3.chunks])

        expected = data1 + data2
        result = self._concat(expected.index, expected.dtypes, results)

        pd.testing.assert_frame_equal(expected, result)
