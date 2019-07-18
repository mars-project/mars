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


import numpy as np
import scipy.sparse as sps

from mars.executor import Executor
from mars.tests.core import TestBase
from mars.tensor.datasource import tensor, ones_like
from mars.lib.sparse import SparseNDArray


class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.executor = Executor()

    def testCreateSparseExecution(self):
        mat = sps.csr_matrix([[0, 0, 2], [2, 0, 0]])
        t = tensor(mat, dtype='f8', chunk_size=2)

        res = self.executor.execute_tensor(t)
        self.assertIsInstance(res[0], SparseNDArray)
        self.assertEqual(res[0].dtype, np.float64)
        np.testing.assert_array_equal(res[0].toarray(), mat[..., :2].toarray())
        np.testing.assert_array_equal(res[1].toarray(), mat[..., 2:].toarray())

        t2 = ones_like(t, dtype='f4')

        res = self.executor.execute_tensor(t2)
        expected = sps.csr_matrix([[0, 0, 1], [1, 0, 0]])
        self.assertIsInstance(res[0], SparseNDArray)
        self.assertEqual(res[0].dtype, np.float32)
        np.testing.assert_array_equal(res[0].toarray(), expected[..., :2].toarray())
        np.testing.assert_array_equal(res[1].toarray(), expected[..., 2:].toarray())

        t3 = tensor(np.array([[0, 0, 2], [2, 0, 0]]), chunk_size=2).tosparse()

        res = self.executor.execute_tensor(t3)
        self.assertIsInstance(res[0], SparseNDArray)
        self.assertEqual(res[0].dtype, np.int_)
        np.testing.assert_array_equal(res[0].toarray(), mat[..., :2].toarray())
        np.testing.assert_array_equal(res[1].toarray(), mat[..., 2:].toarray())
