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

import mars.dataframe as md
from mars.tests.core import TestBase


class Test(TestBase):
    def testSetIndex(self):
        df1 = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                            'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                            'c': list('aabaaddce')})
        mdf = md.DataFrame(df1, chunk_size=3)
        grouped = mdf.groupby('b')
        r = grouped.execute()
        expected = df1.groupby('b')
        for key, group in r:
            pd.testing.assert_frame_equal(group, expected.get_group(key))

        df2 = pd.DataFrame({'a': [3, 4, 5, 3, 5, 4, 1, 2, 3],
                            'b': [1, 3, 4, 5, 6, 5, 4, 4, 4],
                            'c': list('aabaaddce')},
                           index=['i' + str(i) for i in range(9)])
        mdf = md.DataFrame(df2, chunk_size=3)
        grouped = mdf.groupby('b')
        r = grouped.execute()
        expected = df2.groupby('b')
        for key, group in r:
            pd.testing.assert_frame_equal(group, expected.get_group(key))
