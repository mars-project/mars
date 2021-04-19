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

import operator
import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from mars.core import enter_mode
from mars.dataframe.initializer import DataFrame


class Test(unittest.TestCase):
    def testComp(self):
        df1 = DataFrame(pd.DataFrame(np.random.rand(4, 3)))
        df2 = DataFrame(pd.DataFrame(np.random.rand(4, 3)))

        with enter_mode(build=True):
            self.assertFalse(df1.data == df2.data)
            self.assertTrue(df1.data == df1.data)

        for op in [operator.eq, operator.ne, operator.lt, operator.gt,
                   operator.le, operator.ge]:
            eq_df = op(df1, df2)
            pd.testing.assert_index_equal(eq_df.index_value.to_pandas(),
                                          df1.index_value.to_pandas())

            # index not identical
            df3 = DataFrame(pd.DataFrame(np.random.rand(4, 3),
                                         index=[1, 2, 3, 4]))
            with self.assertRaises(ValueError):
                op(df1, df3)

            # columns not identical
            df4 = DataFrame(pd.DataFrame(np.random.rand(4, 3),
                                         columns=['a', 'b', 'c']))
            with self.assertRaises(ValueError):
                op(df1, df4)

        # test datetime
        df = DataFrame(pd.DataFrame(pd.date_range('20130101', periods=6)))
        for op in [operator.eq, operator.ne, operator.lt, operator.gt,
                   operator.le, operator.ge]:
            r_df = op(df, datetime(2013, 1, 2))
            pd.testing.assert_index_equal(r_df.index_value.to_pandas(),
                                          df.index_value.to_pandas())
