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

from mars.tensor import tensor
from mars.tests.core import ExecutorForTest

try:
    import scipy
    from scipy.stats import entropy as sp_entropy

    from mars.tensor.stats import entropy
except ImportError:
    scipy = None


@unittest.skipIf(scipy is None, 'scipy not installed')
class Test(unittest.TestCase):
    def setUp(self):
        self.executor = ExecutorForTest('numpy')

    def testEntropyExecution(self):
        rs = np.random.RandomState(0)
        a = rs.rand(10)

        t1 = tensor(a, chunk_size=4)
        r = entropy(t1)

        result = self.executor.execute_tensor(r, concat=True)[0]
        expected = sp_entropy(a)
        np.testing.assert_array_almost_equal(result, expected)

        b = rs.rand(10)
        base = 3.1

        t2 = tensor(b, chunk_size=4)
        r = entropy(t1, t2, base)

        result = self.executor.execute_tensor(r, concat=True)[0]
        expected = sp_entropy(a, b, base)
        np.testing.assert_array_almost_equal(result, expected)

        b = rs.rand(10)
        base = 3.1

        t2 = tensor(b, chunk_size=4)
        r = entropy(t1, t2, base)

        result = self.executor.execute_tensor(r, concat=True)[0]
        expected = sp_entropy(a, b, base)
        np.testing.assert_array_almost_equal(result, expected)

        r = entropy(t1, t2, t1.sum())

        result = self.executor.execute_tensor(r, concat=True)[0]
        expected = sp_entropy(a, b, a.sum())
        np.testing.assert_array_almost_equal(result, expected)

        with self.assertRaises(ValueError):
            entropy(t1, t2[:7])
