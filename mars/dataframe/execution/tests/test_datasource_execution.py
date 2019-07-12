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
from mars.dataframe.expressions.datasource.dataframe import from_pandas as from_pandas_df
from mars.dataframe.expressions.datasource.series import from_pandas as from_pandas_series
import mars.tensor as mt
from mars.dataframe.expressions.datasource.from_tensor import from_tensor


@unittest.skipIf(pd is None, 'pandas not installed')
class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.executor = Executor()

    def testFromPandasDataFrameExecution(self):
        pdf = pd.DataFrame(np.random.rand(20, 30), index=[np.arange(20), np.arange(20, 0, -1)])
        df = from_pandas_df(pdf, chunk_size=(13, 21))

        result = self.executor.execute_dataframe(df, concat=True)[0]
        pd.testing.assert_frame_equal(pdf, result)

    def testFromPandasSeriesExecution(self):
        ps = pd.Series(np.random.rand(20), index=[np.arange(20), np.arange(20, 0, -1)], name='a')
        series = from_pandas_series(ps, chunk_size=13)

        result = self.executor.execute_dataframe(series, concat=True)[0]
        pd.testing.assert_series_equal(ps, result)

    def testFromTensorExecution(self):
        tensor = mt.random.rand(10, 10, chunk_size=5)
        df = from_tensor(tensor)
        tensor_res = self.executor.execute_tensor(tensor)
        pdf_0 = pd.DataFrame(tensor_res[0])
        pdf_1 = pd.DataFrame(tensor_res[1])
        pdf_2 = pd.DataFrame(tensor_res[2])
        pdf_3 = pd.DataFrame(tensor_res[3])
        result = self.executor.execute_dataframe(df, concat=True)[0]

        pd.testing.assert_index_equal(result.index, pd.RangeIndex(0, 10))
        pd.testing.assert_index_equal(result.columns, pd.RangeIndex(0, 10))

        pd.testing.assert_frame_equal(pdf_0, result.iloc[0:5, 0:5])
        np.array_equal(pdf_1.values, result.iloc[0:5, 5:10].values)
        np.array_equal(pdf_2.values, result.iloc[5:10, 0:5].values)
        np.array_equal(pdf_3.values, result.iloc[5:10, 5:10].values)

        # test converted from 1-d tensor
        tensor2 = mt.array([1, 2, 3])
        df2 = from_tensor(tensor2)
        result2 = self.executor.execute_dataframe(df2, concat=True)[0]
        np.array_equal(np.array([1, 2, 3]), result2.values)
