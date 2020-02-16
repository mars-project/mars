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

import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from mars.dataframe import DataFrame
from mars.tests.core import TestBase, ExecutorForTest


class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.executor = ExecutorForTest()

    def testToCSVExecution(self):
        index = pd.RangeIndex(100, 0, -1, name='index')
        raw = pd.DataFrame({
            'col1': np.random.rand(100),
            'col2': np.random.choice(['a', 'b', 'c'], (100,)),
            'col3': np.arange(100)
        }, index=index)
        df = DataFrame(raw, chunk_size=33)

        base_path = tempfile.mkdtemp()
        try:
            # test one file
            path = os.path.join(base_path, 'out.csv')

            r = df.to_csv(path)
            self.executor.execute_dataframe(r)

            result = pd.read_csv(path, dtype=raw.dtypes.to_dict())
            result.set_index('index', inplace=True)
            pd.testing.assert_frame_equal(result, raw)

            # test multi files
            path = os.path.join(base_path, 'out-*.csv')
            r = df.to_csv(path)
            self.executor.execute_dataframe(r)

            dfs = [pd.read_csv(os.path.join(base_path, 'out-{}.csv'.format(i)),
                               dtype=raw.dtypes.to_dict())
                   for i in range(4)]
            result = pd.concat(dfs, axis=0)
            result.set_index('index', inplace=True)
            pd.testing.assert_frame_equal(result, raw)
            pd.testing.assert_frame_equal(dfs[1].set_index('index'), raw.iloc[33: 66])
        finally:
            shutil.rmtree(base_path)
