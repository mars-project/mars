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
from mars.learn.neighbors._proxima2 import build_proxima2_index
from mars.session import new_session


@unittest.skipIf(pyproxima2 is None, 'pyproxima2 not installed')
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.session = new_session().as_default()

    def testBuildIndex(self):
        raw = pd.DataFrame(np.random.rand(20, 10).astype(np.float32))
        df = md.DataFrame(raw, chunk_size=10)

        index = build_proxima2_index(df, df.index, session=self.session)
        paths = index.fetch()
        for path in paths:
            try:
                with open(path, 'rb') as f:
                    self.assertGreater(len(f.read()), 0)
            finally:
                os.remove(path)
