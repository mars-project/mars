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

from mars.tests.core import TestBase, sps
from mars.tensor import tensor
from mars.tensor.expressions.arithmetic import angle, TensorAngle


class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.raw_data = np.random.rand(7, 8)
        self.raw_sparse_data = sps.random(7, 8)

    def testTensorSerialize(self):
        t = angle(tensor(self.raw_data, chunk_size=(3, 4)))

        serials = self._pb_serial(t)
        t2 = self._pb_deserial(serials)[t.data]

        self.assertIsInstance(t2.op, TensorAngle)
        self.assertEqual(t.issparse(), t2.issparse())

    def testChunkSerialize(self):
        t = angle(tensor(self.raw_data, chunk_size=(3, 4)))
        t.tiles()
        c = t.chunks[0]

        serials = self._pb_serial(c)
        c2 = self._pb_deserial(serials)[c.data]

        self.assertIsInstance(c2.op, TensorAngle)
        self.assertEqual(c.issparse(), c2.issparse())

    def testDenseExpr(self):
        t1 = tensor(self.raw_data, chunk_size=(3, 4))
        t = angle(t1, deg=True)
        t.tiles()

        self.assertEqual(t.shape, self.raw_data.shape)
        self.assertEqual(t.chunk_shape, (3, 2))
        self.assertEqual(t.dtype, self.raw_data.dtype)
        self.assertTrue(t.op.deg)
        self.assertFalse(t.issparse())

        # test each chunk
        for i, chunk in enumerate(t.chunks):
            # test chunk's inputs
            self.assertEqual(len(chunk.inputs), 1)
            self.assertIs(chunk.inputs[0], t1.chunks[i].data)
            # test chunk's index
            self.assertEqual(chunk.index, t1.chunks[i].index)
            # test chunk's dtype
            self.assertEqual(chunk.dtype, t.dtype)
            # test chunk's deg
            self.assertTrue(chunk.op.deg)
            # test chunk's sparse
            self.assertFalse(chunk.issparse())

    def testSparseExpr(self):
        t1 = tensor(self.raw_sparse_data, chunk_size=(3, 4))
        t = angle(t1, deg=True)
        t.tiles()

        self.assertEqual(t.shape, self.raw_sparse_data.shape)
        self.assertEqual(t.chunk_shape, (3, 2))
        self.assertEqual(t.dtype, self.raw_sparse_data.dtype)
        self.assertTrue(t.op.deg)
        self.assertTrue(t.issparse())

        # test each chunk
        for i, chunk in enumerate(t.chunks):
            # test chunk's inputs
            self.assertEqual(len(chunk.inputs), 1)
            self.assertIs(chunk.inputs[0], t1.chunks[i].data)
            # test chunk's index
            self.assertEqual(chunk.index, t1.chunks[i].index)
            # test chunk's dtype
            self.assertEqual(chunk.dtype, t.dtype)
            # test chunk's deg
            self.assertTrue(chunk.op.deg)
            # test chunk's sparse
            self.assertTrue(chunk.issparse())
