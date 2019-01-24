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

import unittest

import numpy as np
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from mars.executor import Executor
from mars.tests.core import TestBase
from mars.dataframe.expressions.datasource.dataframe import from_pandas


@unittest.skipIf(pd is None, 'pandas not installed')
class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.executor = Executor()

    def testPandasExecution(self):
        pdf = pd.DataFrame(np.random.rand(20, 30), index=[np.arange(20), np.arange(20, 0, -1)])
        df = from_pandas(pdf, chunk_size = (13, 21))

        graph = df.build_graph(tiled=True)
        results = self.executor.execute_graph(graph, keys=[c.key for c in df.chunks])

        self.assertEqual(len(results), 4)
        pd.testing.assert_frame_equal(pdf.iloc[:13, :21], results[0])
        pd.testing.assert_frame_equal(pdf.iloc[:13, 21:], results[1])
        pd.testing.assert_frame_equal(pdf.iloc[13:, :21], results[2])
        pd.testing.assert_frame_equal(pdf.iloc[13:, 21:], results[3])
