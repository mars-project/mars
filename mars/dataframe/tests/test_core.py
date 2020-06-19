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

from mars.dataframe.initializer import DataFrame, Series, Index
from mars.tests.core import TestBase


class Test(TestBase):
    def setUp(self):
        super().setUp()
        self.ctx, self.executor = self._create_test_context()

    def testDataFrameDir(self):
        df = DataFrame(pd.DataFrame(np.random.rand(4, 3), columns=list('ABC')))
        dir_result = set(dir(df))
        for c in df.dtypes.index:
            self.assertIn(c, dir_result)

    def testToFrameOrSeries(self):
        raw = pd.Series(np.random.rand(10), name='col')
        series = Series(raw)

        r = series.to_frame()
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_frame_equal(raw.to_frame(), result)

        r = series.to_frame(name='new_name')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_frame_equal(raw.to_frame(name='new_name'), result)

        raw = pd.Index(np.random.rand(10), name='col')
        index = Index(raw)

        r = index.to_frame()
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_frame_equal(raw.to_frame(), result)

        r = index.to_frame(index=False)
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_frame_equal(raw.to_frame(index=False), result)

        r = index.to_frame(name='new_name')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_frame_equal(raw.to_frame(name='new_name'), result)

        r = index.to_series()
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_series_equal(raw.to_series(), result)

        r = index.to_series(index=pd.RangeIndex(0, 10))
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_series_equal(raw.to_series(index=pd.RangeIndex(0, 10)), result)

        r = index.to_series(name='new_name')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_series_equal(raw.to_series(name='new_name'), result)

        raw = pd.MultiIndex.from_tuples([('A', 'E'), ('B', 'F'), ('C', 'G')])
        index = Index(raw, tupleize_cols=True)

        r = index.to_frame()
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_frame_equal(raw.to_frame(), result)

        with self.assertRaises(TypeError):
            index.to_frame(name='XY')

        with self.assertRaises(ValueError):
            index.to_frame(name=['X', 'Y', 'Z'])

        r = index.to_frame(name=['X', 'Y'])
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_frame_equal(raw.to_frame(name=['X', 'Y']), result)

        r = index.to_series(name='new_name')
        result = self.executor.execute_dataframe(r, concat=True)[0]
        pd.testing.assert_series_equal(raw.to_series(name='new_name'), result)
