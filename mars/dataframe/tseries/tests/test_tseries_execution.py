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

import pandas as pd

from mars.dataframe import to_datetime, Series, DataFrame, Index
from mars.tensor import tensor
from mars.tests.core import TestBase, ExecutorForTest


class Test(TestBase):
    def setUp(self):
        super().setUp()
        self.executor = ExecutorForTest()

    def testToDatetimeExecution(self):
        # scalar
        r = to_datetime(1490195805, unit='s')

        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = pd.to_datetime(1490195805, unit='s')
        self.assertEqual(pd.to_datetime(result.item()), expected)

        # test list like
        raw = ['3/11/2000', '3/12/2000', '3/13/2000']
        t = tensor(raw, chunk_size=2)
        r = to_datetime(t, infer_datetime_format=True)

        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = pd.to_datetime(raw, infer_datetime_format=True)
        pd.testing.assert_index_equal(result, expected)

        # test series
        raw_series = pd.Series(raw)
        s = Series(raw_series, chunk_size=2)
        r = to_datetime(s)

        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = pd.to_datetime(raw_series)
        pd.testing.assert_series_equal(result, expected)

        # test DataFrame
        raw_df = pd.DataFrame({'year': [2015, 2016],
                               'month': [2, 3],
                               'day': [4, 5]})
        df = DataFrame(raw_df, chunk_size=(1, 2))
        r = to_datetime(df)

        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = pd.to_datetime(raw_df)
        pd.testing.assert_series_equal(result, expected)

        # test Index
        raw_index = pd.Index([1, 2, 3])
        s = Index(raw_index, chunk_size=2)
        r = to_datetime(s)

        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = pd.to_datetime(raw_index)
        pd.testing.assert_index_equal(result, expected)

        # test raises == 'ignore'
        raw = ['13000101']
        r = to_datetime(raw, format='%Y%m%d', errors='ignore')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = pd.to_datetime(raw, format='%Y%m%d', errors='ignore')
        pd.testing.assert_index_equal(result, expected)

        # test unit
        r = to_datetime([1490195805], unit='s')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = pd.to_datetime([1490195805], unit='s')
        pd.testing.assert_index_equal(result, expected)

        # test origin
        r = to_datetime([1, 2, 3], unit='D', origin=pd.Timestamp('1960-01-01'))
        result = self.executor.execute_dataframe(r, concat=True)[0]
        expected = pd.to_datetime([1, 2, 3], unit='D', origin=pd.Timestamp('1960-01-01'))
        pd.testing.assert_index_equal(result, expected)
