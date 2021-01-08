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
import tempfile

import numpy as np
import pandas as pd

import mars.dataframe as md
from mars.dataframe.indexing.iloc import DataFrameIlocGetItem
from mars.executor import register, Executor
from mars.tests.core import TestBase


class Test(TestBase):
    def setUp(self):
        self.ctx, self.executor = self._create_test_context()

    def testReadCSVHead(self):
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'test.csv')

            df = pd.DataFrame({'a': np.random.randint(10, size=100),
                               'b': np.random.rand(100),
                               'c': np.random.choice(list('abc'), size=100)})
            df.to_csv(file_path, index=False)

            size = os.stat(file_path).st_size / 2
            mdf = md.read_csv(file_path, chunk_bytes=size)

            def _execute_iloc(*_):  # pragma: no cover
                raise ValueError('cannot run iloc')

            with self.ctx:
                try:
                    register(DataFrameIlocGetItem, _execute_iloc)

                    hdf = mdf.head(5)
                    expected = df.head(5)
                    pd.testing.assert_frame_equal(hdf.execute().fetch(), expected)

                    with self.assertRaises(ValueError) as cm:
                        # need iloc
                        mdf.head(99).execute()

                    self.assertIn('cannot run iloc', str(cm.exception))
                finally:
                    del Executor._op_runners[DataFrameIlocGetItem]

                pd.testing.assert_frame_equal(
                    mdf.head(99).execute().fetch().reset_index(drop=True), df.head(99))
