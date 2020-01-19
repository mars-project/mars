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
import scipy.sparse as sps

from mars.tests.core import ExecutorForTest
from mars.learn.utils.checks import check_non_negative_then_return_value
from mars import tensor as mt
from mars import dataframe as md


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = ExecutorForTest('numpy')

    def testCheckNonNegativeThenReturnValueExecution(self):
        raw = np.random.randint(10, size=(10, 5))
        c = mt.tensor(raw, chunk_size=(3, 2))

        r = check_non_negative_then_return_value(c, c, 'sth')
        result = self.executor.execute_tileable(r, concat=True)[0]
        np.testing.assert_array_equal(result, raw)

        raw = raw.copy()
        raw[1, 3] = -1
        c = mt.tensor(raw, chunk_size=(3, 2))

        r = check_non_negative_then_return_value(c, c, 'sth')
        with self.assertRaises(ValueError):
            _ = self.executor.execute_tileable(r, concat=True)[0]

        raw = sps.random(10, 5, density=.3, format='csr')
        c = mt.tensor(raw, chunk_size=(3, 2))

        r = check_non_negative_then_return_value(c, c, 'sth')
        result = self.executor.execute_tileable(r, concat=True)[0]
        np.testing.assert_array_equal(result.toarray(), raw.A)

        raw = raw.copy()
        raw[1, 3] = -1
        c = mt.tensor(raw, chunk_size=(3, 2))

        r = check_non_negative_then_return_value(c, c, 'sth')
        with self.assertRaises(ValueError):
            _ = self.executor.execute_tileable(r, concat=True)[0]

        raw = pd.DataFrame(np.random.rand(10, 4))
        c = md.DataFrame(raw, chunk_size=(3, 2))

        r = check_non_negative_then_return_value(c, c, 'sth')
        result = self.executor.execute_tileable(r, concat=True)[0]

        pd.testing.assert_frame_equal(result, raw)

        raw = raw.copy()
        raw.iloc[1, 3] = -1
        c = md.DataFrame(raw, chunk_size=(3, 2))

        r = check_non_negative_then_return_value(c, c, 'sth')
        with self.assertRaises(ValueError):
            _ = self.executor.execute_tileable(r, concat=True)[0]
