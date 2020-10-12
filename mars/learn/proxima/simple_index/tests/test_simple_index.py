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
import unittest

import numpy as np
import pandas as pd
try:
    import pyproxima2
except ImportError:  # pragma: no cover
    pyproxima2 = None

import mars.dataframe as md
import mars.tensor as mt
from mars.learn.proxima.simple_index import build_index, search_index
from mars.session import new_session
from mars.tests.core import ExecutorForTest


@unittest.skipIf(pyproxima2 is None, 'pyproxima2 not installed')
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.session = new_session().as_default()
        self._old_executor = self.session._sess._executor
        self.executor = self.session._sess._executor = \
            ExecutorForTest('numpy', storage=self.session._sess._context)

    def tearDown(self) -> None:
        self.session._sess._executor = self._old_executor

    def testBuildAndSearchIndex(self):
        rs = np.random.RandomState(0)
        raw = pd.DataFrame(rs.rand(20, 10).astype(np.float32))
        df = md.DataFrame(raw, chunk_size=10)
        raw_t = rs.rand(15, 10).astype(np.float32)
        t = mt.tensor(raw_t, chunk_size=(5, 10))

        args = [
            (df, df.index),
            (raw.to_numpy(), range(20)),
        ]

        for arg in args:
            index = build_index(arg[0], arg[1], index_builder="HnswBuilder", session=self.session)
            paths = index.fetch()
            if not isinstance(paths, list):
                paths = [paths]

            try:
                for path in paths:
                    with open(path, 'rb') as f:
                        self.assertGreater(len(f.read()), 0)

                pk2, distance = search_index(t, range(15), index, 2, index_searcher="HnswSearcher",
                                             session=self.session)
                self.assertEqual(pk2.shape, (len(t), 2))
                self.assertEqual(distance.shape, (len(t), 2))
            finally:
                for path in paths:
                    os.remove(path)
