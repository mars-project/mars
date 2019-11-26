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

import numpy as np
import pandas as pd

from mars.tests.core import TestBase
from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df
from mars.dataframe.datasource.series import from_pandas as from_pandas_series
from mars.dataframe.base import to_gpu, to_cpu


class Test(TestBase):
    def testToGPU(self):
        # test dataframe
        data = pd.DataFrame(np.random.rand(10, 10), index=np.random.randint(-100, 100, size=(10,)),
                            columns=[np.random.bytes(10) for _ in range(10)])
        df = from_pandas_df(data)
        cdf = to_gpu(df)

        self.assertEqual(df.index_value, cdf.index_value)
        self.assertEqual(df.columns_value, cdf.columns_value)
        self.assertTrue(cdf.op.gpu)
        pd.testing.assert_series_equal(df.dtypes, cdf.dtypes)

        cdf.tiles()

        self.assertEqual(df.nsplits, cdf.nsplits)
        self.assertEqual(df.chunks[0].index_value, cdf.chunks[0].index_value)
        self.assertEqual(df.chunks[0].columns_value, cdf.chunks[0].columns_value)
        self.assertTrue(cdf.chunks[0].op.gpu)
        pd.testing.assert_series_equal(df.chunks[0].dtypes, cdf.chunks[0].dtypes)

        self.assertIs(cdf, to_gpu(cdf))

        # test series
        sdata = data.iloc[:, 0]
        series = from_pandas_series(sdata)
        cseries = to_gpu(series)

        self.assertEqual(series.index_value, cseries.index_value)
        self.assertTrue(cseries.op.gpu)

        cseries.tiles()

        self.assertEqual(series.nsplits, cseries.nsplits)
        self.assertEqual(series.chunks[0].index_value, cseries.chunks[0].index_value)
        self.assertTrue(cseries.chunks[0].op.gpu)

        self.assertIs(cseries, to_gpu(cseries))

    def testToCPU(self):
        data = pd.DataFrame(np.random.rand(10, 10), index=np.random.randint(-100, 100, size=(10,)),
                            columns=[np.random.bytes(10) for _ in range(10)])
        df = from_pandas_df(data)
        cdf = to_gpu(df)
        df2 = to_cpu(cdf)

        self.assertEqual(df.index_value, df2.index_value)
        self.assertEqual(df.columns_value, df2.columns_value)
        self.assertFalse(df2.op.gpu)
        pd.testing.assert_series_equal(df.dtypes, df2.dtypes)

        df2.tiles()

        self.assertEqual(df.nsplits, df2.nsplits)
        self.assertEqual(df.chunks[0].index_value, df2.chunks[0].index_value)
        self.assertEqual(df.chunks[0].columns_value, df2.chunks[0].columns_value)
        self.assertFalse(df2.chunks[0].op.gpu)
        pd.testing.assert_series_equal(df.chunks[0].dtypes, df2.chunks[0].dtypes)

        self.assertIs(df2, to_cpu(df2))

    def testRechunk(self):
        df = from_pandas_df(pd.DataFrame(np.random.rand(10, 10)), chunk_size=3)
        df2 = df.rechunk(4).tiles()

        self.assertEqual(df2.shape, (10, 10))
        self.assertEqual(len(df2.chunks), 9)

        self.assertEqual(df2.chunks[0].shape, (4, 4))
        pd.testing.assert_index_equal(df2.chunks[0].index_value.to_pandas(), pd.RangeIndex(4))
        pd.testing.assert_index_equal(df2.chunks[0].columns_value.to_pandas(), pd.RangeIndex(4))

        self.assertEqual(df2.chunks[2].shape, (4, 2))
        pd.testing.assert_index_equal(df2.chunks[2].index_value.to_pandas(), pd.RangeIndex(4))
        pd.testing.assert_index_equal(df2.chunks[2].columns_value.to_pandas(), pd.RangeIndex(8, 10))

        self.assertEqual(df2.chunks[-1].shape, (2, 2))
        pd.testing.assert_index_equal(df2.chunks[-1].index_value.to_pandas(), pd.RangeIndex(8, 10))
        pd.testing.assert_index_equal(df2.chunks[-1].columns_value.to_pandas(), pd.RangeIndex(8, 10))

        columns = [np.random.bytes(10) for _ in range(10)]
        index = np.random.randint(-100, 100, size=(4,))
        data = pd.DataFrame(np.random.rand(4, 10), index=index, columns=columns)
        df = from_pandas_df(data, chunk_size=3)
        df2 = df.rechunk(6).tiles()

        self.assertEqual(df2.shape, (4, 10))
        self.assertEqual(len(df2.chunks), 2)

        self.assertEqual(df2.chunks[0].shape, (4, 6))
        pd.testing.assert_index_equal(df2.chunks[0].index_value.to_pandas(), df.index_value.to_pandas())
        pd.testing.assert_index_equal(df2.chunks[0].columns_value.to_pandas(), pd.Index(columns[:6]))

        self.assertEqual(df2.chunks[1].shape, (4, 4))
        pd.testing.assert_index_equal(df2.chunks[1].index_value.to_pandas(), df.index_value.to_pandas())
        pd.testing.assert_index_equal(df2.chunks[1].columns_value.to_pandas(), pd.Index(columns[6:]))
