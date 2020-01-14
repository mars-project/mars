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
from mars.tensor import tensor
from mars.dataframe import Series, DataFrame
from mars.context import LocalContext


class Test(unittest.TestCase):
    def setUp(self):
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

        this = self

        class MockSession:
            def __init__(self):
                self.executor = this.executor

        ctx = LocalContext(MockSession())
        executor = ExecutorForTest('numpy', storage=ctx)
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
                            'e': [pd.Timestamp('201{}'.format(i)) for i in range(10)],
                            'f': [pd.Timedelta('{} days'.format(i)) for i in range(10)]
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

        this = self

        class MockSession:
            def __init__(self):
                self.executor = this.executor

        ctx = LocalContext(MockSession())
        executor = ExecutorForTest('numpy', storage=ctx)
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
                             'd': [pd.Timestamp('201{}'.format(i)) for i in range(10)],
                            },
                           index=pd.RangeIndex(1, 11))
        df2 = DataFrame(raw2, chunk_size=3)

        r = df2.quantile([0.3, 0.7], numeric_only=False)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw2.quantile([0.3, 0.7], numeric_only=False)

        pd.testing.assert_frame_equal(result, expected)

        r = df2.quantile(numeric_only=False)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = raw2.quantile(numeric_only=False)

        pd.testing.assert_series_equal(result, expected)
