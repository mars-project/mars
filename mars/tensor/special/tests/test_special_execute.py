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
    from scipy.special import gammaln as scipy_gammaln, \
        erf as scipy_erf, entr as scipy_entr, rel_entr as scipy_rel_entr

    from mars.tensor.special import gammaln, erf, entr, rel_entr
except ImportError:
    scipy = None


@unittest.skipIf(scipy is None, 'scipy not installed')
class Test(unittest.TestCase):
    def setUp(self):
        self.executor = ExecutorForTest('numpy')

    def testGammalnExecution(self):
        raw = np.random.rand(10, 8, 6)
        a = tensor(raw, chunk_size=3)

        r = gammaln(a)

        result = self.executor.execute_tensor(r, concat=True)[0]
        expected = scipy_gammaln(raw)

        np.testing.assert_array_equal(result, expected)

        # test sparse
        raw = sps.csr_matrix(np.array([0, 1.0, 1.01, np.nan]))
        a = tensor(raw, chunk_size=3)

        r = gammaln(a)

        result = self.executor.execute_tensor(r, concat=True)[0]

        data = scipy_gammaln(raw.data)
        expected = sps.csr_matrix((data, raw.indices, raw.indptr), raw.shape)

        np.testing.assert_array_equal(result.toarray(), expected.toarray())

    def testErfExecution(self):
        raw = np.random.rand(10, 8, 6)
        a = tensor(raw, chunk_size=3)

        r = erf(a)

        result = self.executor.execute_tensor(r, concat=True)[0]
        expected = scipy_erf(raw)

        np.testing.assert_array_equal(result, expected)

        # test sparse
        raw = sps.csr_matrix(np.array([0, 1.0, 1.01, np.nan]))
        a = tensor(raw, chunk_size=3)

        r = erf(a)

        result = self.executor.execute_tensor(r, concat=True)[0]

        data = scipy_erf(raw.data)
        expected = sps.csr_matrix((data, raw.indices, raw.indptr), raw.shape)

        np.testing.assert_array_equal(result.toarray(), expected.toarray())

    def testEntrExecution(self):
        raw = np.random.rand(10, 8, 6)
        a = tensor(raw, chunk_size=3)

        r = entr(a)

        result = self.executor.execute_tensor(r, concat=True)[0]
        expected = scipy_entr(raw)

        np.testing.assert_array_equal(result, expected)

        # test sparse
        raw = sps.csr_matrix(np.array([0, 1.0, 1.01, np.nan]))
        a = tensor(raw, chunk_size=3)

        r = entr(a)

        result = self.executor.execute_tensor(r, concat=True)[0]

        data = scipy_entr(raw.data)
        expected = sps.csr_matrix((data, raw.indices, raw.indptr), raw.shape)

        np.testing.assert_array_equal(result.toarray(), expected.toarray())

    def testRelEntrExecution(self):
        raw1 = np.random.rand(4, 3, 2)
        raw2 = np.random.rand(4, 3, 2)
        a = tensor(raw1, chunk_size=3)
        b = tensor(raw2, chunk_size=3)

        r = rel_entr(a, b)

        result = self.executor.execute_tensor(r, concat=True)[0]
        expected = scipy_rel_entr(raw1, raw2)

        np.testing.assert_array_equal(result, expected)

        # test sparse
        raw1 = sps.csr_matrix(np.array([0, 1.0, 1.01, np.nan] * 3).reshape(4, 3))
        a = tensor(raw1, chunk_size=3)
        raw2 = np.random.rand(4, 3)
        b = tensor(raw2, chunk_size=3)

        r = rel_entr(a, b)

        result = self.executor.execute_tensor(r, concat=True)[0]

        expected = scipy_rel_entr(raw1.toarray(), raw2)
        np.testing.assert_array_equal(result.toarray(), expected)
