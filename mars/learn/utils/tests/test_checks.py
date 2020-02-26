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

from mars.config import option_context
from mars.tests.core import aio_case, ExecutorForTest
from mars.learn.utils.checks import check_non_negative_then_return_value, assert_all_finite
from mars import tensor as mt
from mars import dataframe as md


@aio_case
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
            _ = self.executor.execute_tileable(r, concat=True)[0]  # noqa: F841

    def testAssertAllFinite(self):
        raw = np.array([2.3, np.inf], dtype=np.float64)
        x = mt.tensor(raw)

        with self.assertRaises(ValueError):
            r = assert_all_finite(x)
            _ = self.executor.execute_tensor(r)

        raw = np.array([2.3, np.nan], dtype=np.float64)
        x = mt.tensor(raw)

        with self.assertRaises(ValueError):
            r = assert_all_finite(x, allow_nan=False)
            _ = self.executor.execute_tensor(r)

        max_float32 = np.finfo(np.float32).max
        raw = [max_float32] * 2
        self.assertFalse(np.isfinite(np.sum(raw)))
        x = mt.tensor(raw)

        r = assert_all_finite(x)
        result = self.executor.execute_tensor(r, concat=True)[0]
        self.assertTrue(result.item())

        raw = np.array([np.nan, 'a'], dtype=object)
        x = mt.tensor(raw)

        with self.assertRaises(ValueError):
            r = assert_all_finite(x)
            _ = self.executor.execute_tensor(r)

        raw = np.random.rand(10)
        x = mt.tensor(raw, chunk_size=2)

        r = assert_all_finite(x, check_only=False)
        result = self.executor.execute_tensor(r, concat=True)[0]
        np.testing.assert_array_equal(result, raw)

        r = assert_all_finite(x)
        result = self.executor.execute_tensor(r, concat=True)[0]
        self.assertTrue(result.item())

        with option_context() as options:
            options.learn.assume_finite = True

            self.assertIsNone(assert_all_finite(x))
            self.assertIs(assert_all_finite(x, check_only=False), x)

        # test sparse
        s = sps.random(10, 3, density=0.1, format='csr',
                       random_state=np.random.RandomState(0))
        s[0, 2] = np.nan

        with self.assertRaises(ValueError):
            r = assert_all_finite(s)
            _ = self.executor.execute_tensor(r)
