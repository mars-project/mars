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
    import scipy.sparse as sps
    import scipy.special as spspecial

    from mars.tensor import special as mt_special
except ImportError:
    scipy = None


@unittest.skipIf(scipy is None, 'scipy not installed')
class Test(unittest.TestCase):
    def setUp(self):
        self.executor = ExecutorForTest('numpy')

    def testUnaryExecution(self):
        funcs = [
            'gamma',
            'gammaln',
            'loggamma',
            'gammasgn',
            'psi',
            'rgamma',
            'digamma',
            'erf',
            'entr',
        ]

        for func in funcs:
            sp_func = getattr(spspecial, func)
            mt_func = getattr(mt_special, func)

            raw = np.random.rand(10, 8, 6)
            a = tensor(raw, chunk_size=3)

            r = mt_func(a)

            result = self.executor.execute_tensor(r, concat=True)[0]
            expected = sp_func(raw)

            np.testing.assert_array_equal(result, expected)

            # test sparse
            raw = sps.csr_matrix(np.array([0, 1.0, 1.01, np.nan]))
            a = tensor(raw, chunk_size=3)

            r = mt_func(a)

            result = self.executor.execute_tensor(r, concat=True)[0]

            data = sp_func(raw.data)
            expected = sps.csr_matrix((data, raw.indices, raw.indptr), raw.shape)

            np.testing.assert_array_equal(result.toarray(), expected.toarray())

    def testBinaryExecution(self):
        funcs = [
            'gammainc',
            'gammaincinv',
            'gammaincc',
            'gammainccinv',
            'beta',
            'betaln',
            'polygamma',
            'poch',
            'rel_entr',
            'kl_div',
            'xlogy',
        ]

        for func in funcs:
            sp_func = getattr(spspecial, func)
            mt_func = getattr(mt_special, func)

            raw1 = np.random.rand(4, 3, 2)
            raw2 = np.random.rand(4, 3, 2)
            a = tensor(raw1, chunk_size=3)
            b = tensor(raw2, chunk_size=3)

            r = mt_func(a, b)

            result = self.executor.execute_tensor(r, concat=True)[0]
            expected = sp_func(raw1, raw2)

            np.testing.assert_array_equal(result, expected)

            # test sparse
            raw1 = sps.csr_matrix(np.array([0, 1.0, 1.01, np.nan] * 3).reshape(4, 3))
            a = tensor(raw1, chunk_size=3)
            raw2 = np.random.rand(4, 3)
            b = tensor(raw2, chunk_size=3)

            r = mt_func(a, b)

            result = self.executor.execute_tensor(r, concat=True)[0]

            expected = sp_func(raw1.toarray(), raw2)
            np.testing.assert_array_equal(result.toarray(), expected)

    def testTripleExecution(self):
        funcs = [
            'betainc',
            'betaincinv',
        ]

        for func in funcs:
            sp_func = getattr(spspecial, func)
            mt_func = getattr(mt_special, func)

            raw1 = np.random.rand(4, 3, 2)
            raw2 = np.random.rand(4, 3, 2)
            raw3 = np.random.rand(4, 3, 2)
            a = tensor(raw1, chunk_size=3)
            b = tensor(raw2, chunk_size=3)
            c = tensor(raw3, chunk_size=3)

            r = mt_func(a, b, c)

            result = self.executor.execute_tensor(r, concat=True)[0]
            expected = sp_func(raw1, raw2, raw3)

            np.testing.assert_array_equal(result, expected)

            # test sparse
            raw1 = sps.csr_matrix(np.array([0, 1.0, 1.01, np.nan] * 3).reshape(4, 3))
            a = tensor(raw1, chunk_size=3)
            raw2 = np.random.rand(4, 3)
            b = tensor(raw2, chunk_size=3)
            raw3 = np.random.rand(4, 3)
            c = tensor(raw3, chunk_size=3)

            r = mt_func(a, b, c)

            result = self.executor.execute_tensor(r, concat=True)[0]

            expected = sp_func(raw1.toarray(), raw2, raw3)
            np.testing.assert_array_equal(result.toarray(), expected)
