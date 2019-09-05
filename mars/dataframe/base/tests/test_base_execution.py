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
try:
    import cudf
except ImportError:  # pragma: no cover
    cudf = None

from mars.tests.core import TestBase
from mars.executor import Executor
from mars.dataframe.base import to_gpu, to_cpu
from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df


class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.executor = Executor()

    @unittest.skipIf(cudf is None, 'cudf not installed')
    def testToGPUExecution(self):
        pdf = pd.DataFrame(np.random.rand(20, 30), index=np.arange(20, 0, -1))
        df = from_pandas_df(pdf, chunk_size=(13, 21))
        cdf = to_gpu(df)

        res = self.executor.execute_dataframe(cdf, concat=True)[0]
        self.assertIsInstance(res, cudf.DataFrame)
        pd.testing.assert_frame_equal(res.to_pandas(), pdf)

    @unittest.skipIf(cudf is None, 'cudf not installed')
    def testToCPUExecution(self):
        pdf = pd.DataFrame(np.random.rand(20, 30), index=np.arange(20, 0, -1))
        df = from_pandas_df(pdf, chunk_size=(13, 21))
        cdf = to_gpu(df)
        df2 = to_cpu(cdf)

        res = self.executor.execute_dataframe(df2, concat=True)[0]
        self.assertIsInstance(res, pd.DataFrame)
        pd.testing.assert_frame_equal(res, pdf)
