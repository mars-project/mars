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

import numpy as np
import pandas as pd

import mars.tensor as mt
import mars.dataframe as md
from mars.executor import Executor
from mars.tests.core import TestBase
from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df
from mars.dataframe.datasource.series import from_pandas as from_pandas_series
from mars.dataframe.datasource.from_tensor import from_tensor
from mars.dataframe.datasource.from_records import from_records


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

    def testInitializerExecution(self):
        pdf = pd.DataFrame(np.random.rand(20, 30), index=[np.arange(20), np.arange(20, 0, -1)])
        df = md.DataFrame(pdf, chunk_size=(15, 10))
        result = self.executor.execute_dataframe(df, concat=True)[0]
        pd.testing.assert_frame_equal(pdf, result)

        ps = pd.Series(np.random.rand(20), index=[np.arange(20), np.arange(20, 0, -1)], name='a')
        series = md.Series(ps, chunk_size=7)
        result = self.executor.execute_dataframe(series, concat=True)[0]
        pd.testing.assert_series_equal(ps, result)

    def testFromTensorExecution(self):
        tensor = mt.random.rand(10, 10, chunk_size=5)
        df = from_tensor(tensor)
        tensor_res = self.executor.execute_tensor(tensor, concat=True)[0]
        pdf_expected = pd.DataFrame(tensor_res)
        df_result = self.executor.execute_dataframe(df, concat=True)[0]
        pd.testing.assert_index_equal(df_result.index, pd.RangeIndex(0, 10))
        pd.testing.assert_index_equal(df_result.columns, pd.RangeIndex(0, 10))
        pd.testing.assert_frame_equal(df_result, pdf_expected)

        # test converted with specified index_value and columns
        tensor2 = mt.random.rand(2, 2, chunk_size=1)
        df2 = from_tensor(tensor2, index=pd.Index(['a', 'b']), columns=pd.Index([3, 4]))
        df_result = self.executor.execute_dataframe(df2, concat=True)[0]
        pd.testing.assert_index_equal(df_result.index, pd.Index(['a', 'b']))
        pd.testing.assert_index_equal(df_result.columns, pd.Index([3, 4]))

        # test converted from 1-d tensor
        tensor3 = mt.array([1, 2, 3])
        df3 = from_tensor(tensor3)
        result3 = self.executor.execute_dataframe(df3, concat=True)[0]
        pdf_expected = pd.DataFrame(np.array([1, 2, 3]))
        pd.testing.assert_frame_equal(pdf_expected, result3)

        # test converted from identical chunks
        tensor4 = mt.ones((10, 10), chunk_size=3)
        df4 = from_tensor(tensor4)
        result4 = self.executor.execute_dataframe(df4, concat=True)[0]
        pdf_expected = pd.DataFrame(self.executor.execute_tensor(tensor4, concat=True)[0])
        pd.testing.assert_frame_equal(pdf_expected, result4)

        # from tensor with given index
        tensor5 = mt.ones((10, 10), chunk_size=3)
        df5 = from_tensor(tensor5, index=np.arange(0, 20, 2))
        result5 = self.executor.execute_dataframe(df5, concat=True)[0]
        pdf_expected = pd.DataFrame(self.executor.execute_tensor(tensor5, concat=True)[0],
                                    index=np.arange(0, 20, 2))
        pd.testing.assert_frame_equal(pdf_expected, result5)

        # from tensor with given columns
        tensor6 = mt.ones((10, 10), chunk_size=3)
        df6 = from_tensor(tensor6, columns=list('abcdefghij'))
        result6 = self.executor.execute_dataframe(df6, concat=True)[0]
        pdf_expected = pd.DataFrame(self.executor.execute_tensor(tensor6, concat=True)[0],
                                    columns=list('abcdefghij'))
        pd.testing.assert_frame_equal(pdf_expected, result6)

    def testFromRecordsExecution(self):
        dtype = np.dtype([('x', 'int'), ('y', 'double'), ('z', '<U16')])


        ndarr = np.ones((10,), dtype=dtype)
        pdf_expected = pd.DataFrame.from_records(ndarr, index=pd.RangeIndex(10))

        # from structured array of mars
        tensor = mt.ones((10,), dtype=dtype, chunk_size=3)
        df1 = from_records(tensor)
        df1_result = self.executor.execute_dataframe(df1, concat=True)[0]
        pd.testing.assert_frame_equal(df1_result, pdf_expected)

        # from structured array of numpy
        df2 = from_records(ndarr)
        df2_result = self.executor.execute_dataframe(df2, concat=True)[0]
        pd.testing.assert_frame_equal(df2_result, pdf_expected)
