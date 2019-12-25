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

from mars.tests.core import TestBase, parameterized
from mars.dataframe.datasource.series import from_pandas as from_pandas_series
from mars.dataframe.datasource.dataframe import from_pandas as from_pandas_df


reduction_functions = dict(
    sum=dict(func_name='sum', has_min_count=True),
    prod=dict(func_name='prod', has_min_count=True),
    min=dict(func_name='min', has_min_count=False),
    max=dict(func_name='max', has_min_count=False)
)


@parameterized(**reduction_functions)
class Test(TestBase):
    def compute(self, data, **kwargs):
        return getattr(data, self.func_name)(**kwargs)

    def testSeriesReduction(self):
        data = pd.Series(np.random.rand(20), index=[str(i) for i in range(20)], name='a')
        reduction_df1 = self.compute(from_pandas_series(data))
        self.assertEqual(self.compute(data), reduction_df1.execute())

        reduction_df2 = self.compute(from_pandas_series(data, chunk_size=6))
        self.assertAlmostEqual(self.compute(data), reduction_df2.execute())

        reduction_df3 = self.compute(from_pandas_series(data, chunk_size=3))
        self.assertAlmostEqual(self.compute(data), reduction_df3.execute())

        reduction_df4 = self.compute(from_pandas_series(data, chunk_size=4), axis='index')
        self.assertAlmostEqual(self.compute(data, axis='index'), reduction_df4.execute())

        data = pd.Series(np.random.rand(20), name='a')
        data[0] = 0.1  # make sure not all elements are NAN
        data[data > 0.5] = np.nan
        reduction_df1 = self.compute(from_pandas_series(data, chunk_size=3))
        self.assertAlmostEqual(self.compute(data), reduction_df1.execute())

        reduction_df2 = self.compute(from_pandas_series(data, chunk_size=3), skipna=False)
        self.assertTrue(np.isnan(reduction_df2.execute()))

        if self.has_min_count:
            reduction_df3 = self.compute(from_pandas_series(data, chunk_size=3), skipna=False, min_count=2)
            self.assertTrue(np.isnan(reduction_df3.execute()))

            reduction_df4 = self.compute(from_pandas_series(data, chunk_size=3), min_count=1)
            self.assertAlmostEqual(self.compute(data, min_count=1), reduction_df4.execute())

            reduction_df5 = self.compute(from_pandas_series(data, chunk_size=3), min_count=21)
            self.assertTrue(np.isnan(reduction_df5.execute()))

    def testDataFrameReduction(self):
        data = pd.DataFrame(np.random.rand(20, 10))
        reduction_df1 = self.compute(from_pandas_df(data))
        pd.testing.assert_series_equal(self.compute(data), reduction_df1.execute())

        reduction_df2 = self.compute(from_pandas_df(data, chunk_size=3))
        pd.testing.assert_series_equal(self.compute(data), reduction_df2.execute())

        reduction_df3 = self.compute(from_pandas_df(data, chunk_size=6), axis='index', numeric_only=True)
        pd.testing.assert_series_equal(self.compute(data, axis='index', numeric_only=True), reduction_df3.execute())

        reduction_df4 = self.compute(from_pandas_df(data, chunk_size=3), axis=1)
        pd.testing.assert_series_equal(self.compute(data, axis=1), reduction_df4.execute())

        # test null
        np_data = np.random.rand(20, 10)
        np_data[np_data > 0.6] = np.nan
        data = pd.DataFrame(np_data)

        reduction_df1 = self.compute(from_pandas_df(data, chunk_size=3))
        pd.testing.assert_series_equal(self.compute(data), reduction_df1.execute())

        reduction_df2 = self.compute(from_pandas_df(data, chunk_size=3), skipna=False)
        pd.testing.assert_series_equal(self.compute(data, skipna=False), reduction_df2.execute())

        reduction_df2 = self.compute(from_pandas_df(data, chunk_size=3), skipna=False)
        pd.testing.assert_series_equal(self.compute(data, skipna=False), reduction_df2.execute())

        if self.has_min_count:
            reduction_df3 = self.compute(from_pandas_df(data, chunk_size=3), min_count=15)
            pd.testing.assert_series_equal(self.compute(data, min_count=15), reduction_df3.execute())

            reduction_df4 = self.compute(from_pandas_df(data, chunk_size=3), min_count=3)
            pd.testing.assert_series_equal(self.compute(data, min_count=3), reduction_df4.execute())

            reduction_df5 = self.compute(from_pandas_df(data, chunk_size=3), axis=1, min_count=3)
            pd.testing.assert_series_equal(self.compute(data, axis=1, min_count=3), reduction_df5.execute())

            reduction_df5 = self.compute(from_pandas_df(data, chunk_size=3), axis=1, min_count=8)
            pd.testing.assert_series_equal(self.compute(data, axis=1, min_count=8), reduction_df5.execute())

        # test numeric_only
        data = pd.DataFrame(np.random.rand(10, 10), index=np.random.randint(-100, 100, size=(10,)),
                            columns=[np.random.bytes(10) for _ in range(10)])
        reduction_df1 = self.compute(from_pandas_df(data, chunk_size=2))
        pd.testing.assert_series_equal(self.compute(data), reduction_df1.execute())

        reduction_df2 = self.compute(from_pandas_df(data, chunk_size=6), axis='index', numeric_only=True)
        pd.testing.assert_series_equal(self.compute(data, axis='index', numeric_only=True), reduction_df2.execute())

        reduction_df3 = self.compute(from_pandas_df(data, chunk_size=3), axis='columns')
        pd.testing.assert_series_equal(self.compute(data, axis='columns'), reduction_df3.execute())

        data_dict = dict((str(i), np.random.rand(10)) for i in range(10))
        data_dict['string'] = [str(i) for i in range(10)]
        data_dict['bool'] = np.random.choice([True, False], (10,))
        data = pd.DataFrame(data_dict)
        reduction_df = self.compute(from_pandas_df(data, chunk_size=3), axis='index', numeric_only=True)
        pd.testing.assert_series_equal(self.compute(data, axis='index', numeric_only=True), reduction_df.execute())


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
