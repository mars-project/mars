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

from mars.tests.core import TestBase
from mars.dataframe.core import SERIES_CHUNK_TYPE, Series, DataFrame, DATAFRAME_CHUNK_TYPE
from mars.dataframe.datasource.dataframe import from_pandas


class Test(TestBase):
    def testGetitem(self):
        data = pd.DataFrame(np.random.rand(10, 5), columns=['c1', 'c2', 'c3', 'c4', 'c5'])
        df = from_pandas(data, chunk_size=2)

        series = df['c3']
        self.assertIsInstance(series, Series)
        self.assertEqual(series.shape, (10,))
        self.assertEqual(series.name, 'c3')
        self.assertEqual(series.dtype, data['c3'].dtype)
        self.assertEqual(series.index_value, df.index_value)

        series.tiles()
        self.assertEqual(series.nsplits, ((2, 2, 2, 2, 2),))
        self.assertEqual(len(series.chunks), 5)
        for i, c in enumerate(series.chunks):
            self.assertIsInstance(c, SERIES_CHUNK_TYPE)
            self.assertEqual(c.index, (i,))
            self.assertEqual(c.shape, (2,))

        df1 = df[['c1', 'c2', 'c3']]
        self.assertIsInstance(df1, DataFrame)
        self.assertEqual(df1.shape, (10, 3))
        self.assertEqual(df1.index_value, df.index_value)
        pd.testing.assert_index_equal(df1.columns.to_pandas(), data[['c1', 'c2', 'c3']].columns)
        pd.testing.assert_series_equal(df1.dtypes, data[['c1', 'c2', 'c3']].dtypes)

        df1.tiles()
        self.assertEqual(df1.nsplits, ((2, 2, 2, 2, 2), (2, 1)))
        self.assertEqual(len(df1.chunks), 10)
        for i, c in enumerate(df1.chunks[slice(0, 10, 2)]):
            self.assertIsInstance(c, DATAFRAME_CHUNK_TYPE)
            self.assertEqual(c.index, (i, 0))
            self.assertEqual(c.shape, (2, 2))
        for i, c in enumerate(df1.chunks[slice(1, 10, 2)]):
            self.assertIsInstance(c, DATAFRAME_CHUNK_TYPE)
            self.assertEqual(c.index, (i, 1))
            self.assertEqual(c.shape, (2, 1))

