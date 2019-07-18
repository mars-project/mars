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

from mars.tensor.datasource import tensor
from mars.tensor.datasource.array import CSRMatrixDataSource
from mars.tensor.core import SparseTensor
from mars.tests.core import TestBase


class Test(TestBase):
    def testFromSpmatrix(self):
        t = tensor(sps.csr_matrix([[0, 0, 1], [1, 0, 0]], dtype='f8'), chunk_size=2)

        self.assertIsInstance(t, SparseTensor)
        self.assertIsInstance(t.op, CSRMatrixDataSource)
        self.assertTrue(t.issparse())
        self.assertFalse(t.op.gpu)

        t.tiles()
        self.assertEqual(t.chunks[0].index, (0, 0))
        self.assertIsInstance(t.op, CSRMatrixDataSource)
        self.assertFalse(t.op.gpu)
        m = sps.csr_matrix([[0, 0], [1, 0]])
        self.assertTrue(np.array_equal(t.chunks[0].op.indices, m.indices))
        self.assertTrue(np.array_equal(t.chunks[0].op.indptr, m.indptr))
        self.assertTrue(np.array_equal(t.chunks[0].op.data, m.data))
        self.assertTrue(np.array_equal(t.chunks[0].op.shape, m.shape))
