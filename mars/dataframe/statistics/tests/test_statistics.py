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

import numpy as np
import pandas as pd

from mars.tensor import Tensor
from mars.dataframe.core import Series
from mars.dataframe.datasource.series import from_pandas as series_from_pandas


class Test(unittest.TestCase):
    def testSeriesQuantile(self):
        raw = series_from_pandas(pd.Series(np.random.rand(10)), chunk_size=3)

        r = raw.quantile()
        self.assertIsInstance(r, Tensor)
        r.tiles()

        raw = series_from_pandas(pd.Series(np.random.rand(10)), chunk_size=3)

        r = raw.quantile([0.3, 0.7])
        self.assertIsInstance(r, Series)
        r.tiles()
