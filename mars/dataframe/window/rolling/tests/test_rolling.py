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

from mars import dataframe as md
from mars.tests.core import TestBase


class Test(TestBase):
    def testRolling(self):
        df = pd.DataFrame(np.random.rand(4, 3), columns=list('abc'))
        df2 = md.DataFrame(df)

        r = df2.rolling(3, min_periods=1, center=True,
                        win_type='triang', closed='both')
        expected = df.rolling(3, min_periods=1, center=True,
                              win_type='triang', closed='both')
        self.assertEqual(repr(r), repr(expected))

        with self.assertRaises(KeyError):
            _ = r['d']

        with self.assertRaises(KeyError):
            _ = r['a', 'd']

    def testRollingAgg(self):
        df = pd.DataFrame(np.random.rand(4, 3), columns=list('abc'))
        df2 = md.DataFrame(df, chunk_size=3)

        r = df2.rolling(3).agg('max')
        expected = df.rolling(3).agg('max')

        self.assertEqual(r.shape, df.shape)
        self.assertIs(r.index_value, df2.index_value)
        pd.testing.assert_index_equal(r.columns_value.to_pandas(),
                                      expected.columns)
        pd.testing.assert_series_equal(r.dtypes, df2.dtypes)

        r = r.tiles()
        for c in r.chunks:
            self.assertEqual(c.shape, c.inputs[0].shape)
            self.assertIs(c.index_value, c.inputs[0].index_value)
            pd.testing.assert_index_equal(c.columns_value.to_pandas(),
                                          expected.columns)
            pd.testing.assert_series_equal(c.dtypes, expected.dtypes)
