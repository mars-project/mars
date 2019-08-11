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

import pandas as pd

from mars.executor import Executor
from mars.tests.core import TestBase
from mars.dataframe.datasource.dataframe import from_pandas


class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.executor = Executor()

    def testSetIndex(self):
        df1 = pd.DataFrame([[1,3,3], [4,2,6], [7, 8, 9]],
                           index=['a1', 'a2', 'a3'], columns=['x', 'y', 'z'])
        df2 = from_pandas(df1, chunk_size=2)

        expected = df1.set_index('y', drop=True)
        df3 = df2.set_index('y', drop=True)
        result = self.executor.execute_dataframe(df3, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        expected = df1.set_index('y', drop=False)
        df4 = df2.set_index('y', drop=False)
        result = self.executor.execute_dataframe(df4, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

    def testILocGetItem(self):
        df1 = pd.DataFrame([[1,3,3], [4,2,6], [7, 8, 9]],
                           index=['a1', 'a2', 'a3'], columns=['x', 'y', 'z'])
        df2 = from_pandas(df1, chunk_size=2)

        # plain index
        expected = df1.iloc[1]
        df3 = df2.iloc[1]
        result = self.executor.execute_dataframe(df3, concat=True)[0]
        pd.testing.assert_series_equal(expected, result)

        # slice index
        expected = df1.iloc[:, 2:4]
        df4 = df2.iloc[:, 2:4]
        result = self.executor.execute_dataframe(df4, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # plain fancy index
        expected = df1.iloc[[0], [0, 1, 2]]
        df5 = df2.iloc[[0], [0, 1, 2]]
        result = self.executor.execute_dataframe(df5, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # fancy index
        expected = df1.iloc[[1, 2], [0, 1, 2]]
        df6 = df2.iloc[[1, 2], [0, 1, 2]]
        result = self.executor.execute_dataframe(df6, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # plain index
        expected = df1.iloc[1, 2]
        df7 = df2.iloc[1, 2]
        result = self.executor.execute_dataframe(df7, concat=True)[0]
        self.assertEqual(expected, result)

    def testILocSetItem(self):
        df1 = pd.DataFrame([[1,3,3], [4,2,6], [7, 8, 9]],
                            index=['a1', 'a2', 'a3'], columns=['x', 'y', 'z'])
        df2 = from_pandas(df1, chunk_size=2)

        # plain index
        expected = df1
        expected.iloc[1] = 100
        df2.iloc[1] = 100
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # slice index
        expected.iloc[:, 2:4] = 1111
        df2.iloc[:, 2:4] = 1111
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # plain fancy index
        expected.iloc[[0], [0, 1, 2]] = 2222
        df2.iloc[[0], [0, 1, 2]] = 2222
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # fancy index
        expected.iloc[[1, 2], [0, 1, 2]] = 3333
        df2.iloc[[1, 2], [0, 1, 2]] = 3333
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)

        # plain index
        expected.iloc[1, 2] = 4444
        df2.iloc[1, 2] = 4444
        result = self.executor.execute_dataframe(df2, concat=True)[0]
        pd.testing.assert_frame_equal(expected, result)
