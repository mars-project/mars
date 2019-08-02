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
import numpy as np

from mars.tests.core import TestBase
from mars.dataframe.datasource.series import from_pandas as from_pandas_series
from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df


class Test(TestBase):
    def testSeriesSum(self):
        data = pd.Series(np.random.rand(20), index=[str(i) for i in range(20)], name='a')
        sum_df = from_pandas_series(data).sum()
        self.assertEqual(data.sum(), sum_df.execute())

        sum_df = from_pandas_series(data, chunk_size=6).sum()
        self.assertAlmostEqual(data.sum(), sum_df.execute())

        sum_df = from_pandas_series(data, chunk_size=3).sum()
        self.assertAlmostEqual(data.sum(), sum_df.execute())

        sum_df = from_pandas_series(data, chunk_size=4).sum(axis='index')
        self.assertAlmostEqual(data.sum(axis='index'), sum_df.execute())

        data = pd.Series(np.random.rand(20), name='a')
        data[data > 0.5] = np.nan
        sum_df = from_pandas_series(data, chunk_size=3).sum()
        self.assertAlmostEqual(data.sum(), sum_df.execute())

        sum_df = from_pandas_series(data, chunk_size=3).sum(skipna=False)
        self.assertTrue(np.isnan(sum_df.execute()))

    def testDataFrameSum(self):
        data = pd.DataFrame(np.random.rand(20, 10))
        sum_df = from_pandas_df(data, chunk_size=3).sum()
        pd.testing.assert_series_equal(data.sum(), sum_df.execute())
