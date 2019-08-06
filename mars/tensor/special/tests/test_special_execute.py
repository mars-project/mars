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

from mars.tensor import tensor
from mars.executor import Executor

try:
    import scipy
    import scipy.sparse as sps
    from scipy.special import gammaln as scipy_gammaln

    from mars.tensor.special import gammaln
except ImportError:
    scipy = None


@unittest.skipIf(scipy is None, 'scipy not installed')
class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')

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
