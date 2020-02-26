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

from mars.tests.core import TestBase, parameterized, ExecutorForTest
from mars.dataframe.datasource.series import from_pandas as from_pandas_series
from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df

reduction_functions = dict(
    cummax=dict(func_name='cummax'),
    cummin=dict(func_name='cummin'),
    cumprod=dict(func_name='cumprod'),
    cumsum=dict(func_name='cumsum'),
)


@parameterized(**reduction_functions)
class Test(TestBase):
    def setUp(self):
        self.executor = ExecutorForTest()

    def compute(self, data, **kwargs):
        return getattr(data, self.func_name)(**kwargs)

    def testSeriesReduction(self):
        data = pd.Series(np.random.rand(20), index=[str(i) for i in range(20)], name='a')
        reduction_df1 = self.compute(from_pandas_series(data))
        pd.testing.assert_series_equal(
            self.compute(data), self.executor.execute_dataframe(reduction_df1, concat=True)[0])

        reduction_df2 = self.compute(from_pandas_series(data, chunk_size=6))
        pd.testing.assert_series_equal(
            self.compute(data), self.executor.execute_dataframe(reduction_df2, concat=True)[0])

        reduction_df3 = self.compute(from_pandas_series(data, chunk_size=3))
        pd.testing.assert_series_equal(
            self.compute(data), self.executor.execute_dataframe(reduction_df3, concat=True)[0])

        reduction_df4 = self.compute(from_pandas_series(data, chunk_size=4), axis='index')
        pd.testing.assert_series_equal(
            self.compute(data, axis='index'), self.executor.execute_dataframe(reduction_df4, concat=True)[0])

        data = pd.Series(np.random.rand(20), name='a')
        data[0] = 0.1  # make sure not all elements are NAN
        data[data > 0.5] = np.nan
        reduction_df1 = self.compute(from_pandas_series(data, chunk_size=3))
        pd.testing.assert_series_equal(
            self.compute(data), self.executor.execute_dataframe(reduction_df1, concat=True)[0])

        reduction_df2 = self.compute(from_pandas_series(data, chunk_size=3), skipna=False)
        pd.testing.assert_series_equal(
            self.compute(data, skipna=False), self.executor.execute_dataframe(reduction_df2, concat=True)[0])

    def testDataFrameReduction(self):
        data = pd.DataFrame(np.random.rand(20, 10))
        reduction_df1 = self.compute(from_pandas_df(data))
        pd.testing.assert_frame_equal(
            self.compute(data), self.executor.execute_dataframe(reduction_df1, concat=True)[0])

        reduction_df2 = self.compute(from_pandas_df(data, chunk_size=3))
        pd.testing.assert_frame_equal(
            self.compute(data), self.executor.execute_dataframe(reduction_df2, concat=True)[0])

        reduction_df4 = self.compute(from_pandas_df(data, chunk_size=3), axis=1)
        pd.testing.assert_frame_equal(
            self.compute(data, axis=1),
            self.executor.execute_dataframe(reduction_df4, concat=True)[0])

        # test null
        np_data = np.random.rand(20, 10)
        np_data[np_data > 0.6] = np.nan
        data = pd.DataFrame(np_data)

        reduction_df1 = self.compute(from_pandas_df(data, chunk_size=3))
        pd.testing.assert_frame_equal(
            self.compute(data), self.executor.execute_dataframe(reduction_df1, concat=True)[0])

        reduction_df2 = self.compute(from_pandas_df(data, chunk_size=3), skipna=False)
        pd.testing.assert_frame_equal(
            self.compute(data, skipna=False), self.executor.execute_dataframe(reduction_df2, concat=True)[0])

        reduction_df2 = self.compute(from_pandas_df(data, chunk_size=3), skipna=False)
        pd.testing.assert_frame_equal(
            self.compute(data, skipna=False), self.executor.execute_dataframe(reduction_df2, concat=True)[0])

        # test numeric_only
        data = pd.DataFrame(np.random.rand(10, 10), index=np.random.randint(-100, 100, size=(10,)),
                            columns=[np.random.bytes(10) for _ in range(10)])
        reduction_df1 = self.compute(from_pandas_df(data, chunk_size=2))
        pd.testing.assert_frame_equal(
            self.compute(data), self.executor.execute_dataframe(reduction_df1, concat=True)[0])

        reduction_df3 = self.compute(from_pandas_df(data, chunk_size=3), axis='columns')
        pd.testing.assert_frame_equal(
            self.compute(data, axis='columns'),
            self.executor.execute_dataframe(reduction_df3, concat=True)[0])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
