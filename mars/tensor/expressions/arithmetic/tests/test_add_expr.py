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
from mars.tensor.expressions.arithmetic import TensorAdd, TensorAddConstant


class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.raw_data1 = np.random.rand(7, 8)
        self.raw_data2 = np.random.random(8)
        self.raw_sparse_data1 = sps.random(7, 8)
        self.raw_sparse_data2 = sps.random(1, 8)

    def testTensorSerialize(self):
        # test add two dense tensors
        t = tensor(self.raw_data1, chunk_size=(3, 4)) + \
            tensor(self.raw_data2, chunk_size=4)

        serials = self._pb_serial(t)
        t2 = self._pb_deserial(serials)[t.data]

        self.assertIsInstance(t2.op, TensorAdd)
        self.assertEqual(t.issparse(), t2.issparse())

        # test add one dense with scalar
        t = tensor(self.raw_data1, chunk_size=(3, 4)) + 10

        serials = self._pb_serial(t)
        t2 = self._pb_deserial(serials)[t.data]

        self.assertIsInstance(t2.op, TensorAddConstant)
        self.assertEqual(t.issparse(), t2.issparse())

        # test add two sparse tensors
        t = tensor(self.raw_sparse_data1, chunk_size=(3, 4)) + \
            tensor(self.raw_sparse_data2, chunk_size=4)

        serials = self._pb_serial(t)
        t2 = self._pb_deserial(serials)[t.data]

        self.assertIsInstance(t2.op, TensorAdd)
        self.assertEqual(t.issparse(), t2.issparse())

        # test add one sparse with scalar
        t = tensor(self.raw_sparse_data1, chunk_size=(3, 4)) + 10

        serials = self._pb_serial(t)
        t2 = self._pb_deserial(serials)[t.data]

        self.assertIsInstance(t2.op, TensorAddConstant)
        self.assertEqual(t.issparse(), t2.issparse())

    def testChunkSerialize(self):
        # test add two dense tensors
        t = tensor(self.raw_data1, chunk_size=(3, 4)) + \
            tensor(self.raw_data2, chunk_size=4)
        t.tiles()
        c = t.chunks[0]

        serials = self._pb_serial(c)
        c2 = self._pb_deserial(serials)[c.data]

        self.assertIsInstance(c2.op, TensorAdd)
        self.assertEqual(c.issparse(), c2.issparse())

        # test add one dense with scalar
        t = tensor(self.raw_data1) + 10
        t.tiles()
        c = t.chunks[0]

        serials = self._pb_serial(c)
        c2 = self._pb_deserial(serials)[c.data]

        self.assertIsInstance(c2.op, TensorAddConstant)
        self.assertEqual(c.issparse(), c2.issparse())

        # test add two sparse tensors
        t = tensor(self.raw_sparse_data1) + tensor(self.raw_sparse_data2)
        t.tiles()
        c = t.chunks[0]

        serials = self._pb_serial(c)
        c2 = self._pb_deserial(serials)[c.data]

        self.assertIsInstance(c2.op, TensorAdd)
        self.assertEqual(c.issparse(), c2.issparse())

        # test add one sparse with scalar
        t = tensor(self.raw_sparse_data1) + 10
        t.tiles()
        c = t.chunks[0]

        serials = self._pb_serial(c)
        c2 = self._pb_deserial(serials)[c.data]

        self.assertIsInstance(c2.op, TensorAddConstant)
        self.assertEqual(c.issparse(), c2.issparse())

    def testDenseExpr(self):
        # test add two dense tensor
        t1 = tensor(self.raw_data1, chunk_size=(3, 4))
        t2 = tensor(self.raw_data2, chunk_size=4)
        t = t1 + t2

        self.assertEqual(t.shape, self.raw_data1.shape)
        self.assertEqual(t.dtype, self.raw_data1.dtype)
        self.assertFalse(t.issparse())

        t.tiles()

        self.assertEqual(t.shape, self.raw_data1.shape)
        self.assertEqual(t.chunk_shape, (3, 2))
        self.assertEqual(t.dtype, self.raw_data1.dtype)
        self.assertFalse(t.issparse())

        for i, chunk in enumerate(t.chunks):
            # test chunk's inputs
            self.assertEqual(len(chunk.inputs), 2)
            self.assertIs(chunk.inputs[0], t1.chunks[i].data)
            self.assertIs(chunk.inputs[1], t2.chunks[chunk.index[1]].data)
            # test chunk's index
            self.assertEqual(chunk.index, t1.chunks[i].index)
            # test chunk's dtype
            self.assertEqual(chunk.dtype, t.dtype)
            # test chunk's sparse
            self.assertFalse(chunk.issparse())

        # test add tensor with scalar
        t1 = tensor(self.raw_data1, chunk_size=(3, 4))
        t = t1 + 10

        self.assertEqual(t.shape, self.raw_data1.shape)
        self.assertEqual(t.dtype, self.raw_data1.dtype)
        self.assertFalse(t.issparse())

        t.tiles()

        self.assertEqual(t.shape, self.raw_data1.shape)
        self.assertEqual(t.chunk_shape, (3, 2))
        self.assertEqual(t.dtype, self.raw_data1.dtype)
        self.assertFalse(t.issparse())

        for i, chunk in enumerate(t.chunks):
            # test chunk's inputs
            self.assertEqual(len(chunk.inputs), 1)
            self.assertIs(chunk.inputs[0], t1.chunks[i].data)
            # test chunk's index
            self.assertEqual(chunk.index, t1.chunks[i].index)
            # test chunk's dtype
            self.assertEqual(chunk.dtype, t.dtype)
            # test chunk's sparse
            self.assertFalse(chunk.issparse())

        # test add scalar to scalar
        t1 = tensor(self.raw_data1, chunk_size=(3, 4))
        t = 10 + t1

        self.assertEqual(t.shape, self.raw_data1.shape)
        self.assertEqual(t.dtype, self.raw_data1.dtype)
        self.assertFalse(t.issparse())

        t.tiles()

        self.assertEqual(t.shape, self.raw_data1.shape)
        self.assertEqual(t.chunk_shape, (3, 2))
        self.assertEqual(t.dtype, self.raw_data1.dtype)
        self.assertFalse(t.issparse())

        for i, chunk in enumerate(t.chunks):
            # test chunk's inputs
            self.assertEqual(len(chunk.inputs), 1)
            self.assertIs(chunk.inputs[0], t1.chunks[i].data)
            # test chunk's index
            self.assertEqual(chunk.index, t1.chunks[i].index)
            # test chunk's dtype
            self.assertEqual(chunk.dtype, t.dtype)
            # test chunk's sparse
            self.assertFalse(chunk.issparse())

    def testSparseExpr(self):
        # test add two sparse tensor
        t1 = tensor(self.raw_sparse_data1, chunk_size=(3, 4))
        t2 = tensor(self.raw_sparse_data2, chunk_size=4)
        t = t1 + t2

        self.assertEqual(t.shape, self.raw_data1.shape)
        self.assertEqual(t.dtype, self.raw_data1.dtype)
        self.assertTrue(t.issparse())

        t.tiles()

        self.assertEqual(t.shape, self.raw_data1.shape)
        self.assertEqual(t.chunk_shape, (3, 2))
        self.assertEqual(t.dtype, self.raw_data1.dtype)
        self.assertTrue(t.issparse())

        for i, chunk in enumerate(t.chunks):
            # test chunk's inputs
            self.assertEqual(len(chunk.inputs), 2)
            self.assertIs(chunk.inputs[0], t1.chunks[i].data)
            self.assertIs(chunk.inputs[1], t2.chunks[chunk.index[1]].data)
            # test chunk's index
            self.assertEqual(chunk.index, t1.chunks[i].index)
            # test chunk's dtype
            self.assertEqual(chunk.dtype, t.dtype)
            # test chunk's sparse
            self.assertTrue(chunk.issparse())

        # test add dense and sparse tensor
        t1 = tensor(self.raw_sparse_data1, chunk_size=(3, 4))
        t2 = tensor(self.raw_data2, chunk_size=4)
        t = t1 + t2

        self.assertEqual(t.shape, self.raw_data1.shape)
        self.assertEqual(t.dtype, self.raw_data1.dtype)
        self.assertFalse(t.issparse())

        t.tiles()

        self.assertEqual(t.shape, self.raw_data1.shape)
        self.assertEqual(t.chunk_shape, (3, 2))
        self.assertEqual(t.dtype, self.raw_data1.dtype)
        self.assertFalse(t.issparse())

        for i, chunk in enumerate(t.chunks):
            # test chunk's inputs
            self.assertEqual(len(chunk.inputs), 2)
            self.assertIs(chunk.inputs[0], t1.chunks[i].data)
            self.assertIs(chunk.inputs[1], t2.chunks[chunk.index[1]].data)
            # test chunk's index
            self.assertEqual(chunk.index, t1.chunks[i].index)
            # test chunk's dtype
            self.assertEqual(chunk.dtype, t.dtype)
            # test chunk's sparse
            self.assertFalse(chunk.issparse())

        # test add sparse tensor with scalar != 0
        t1 = tensor(self.raw_sparse_data1, chunk_size=(3, 4))
        t = t1 + 10

        self.assertEqual(t.shape, self.raw_data1.shape)
        self.assertEqual(t.dtype, self.raw_data1.dtype)
        self.assertFalse(t.issparse())

        t.tiles()

        self.assertEqual(t.shape, self.raw_data1.shape)
        self.assertEqual(t.chunk_shape, (3, 2))
        self.assertEqual(t.dtype, self.raw_data1.dtype)
        self.assertFalse(t.issparse())

        for i, chunk in enumerate(t.chunks):
            # test chunk's inputs
            self.assertEqual(len(chunk.inputs), 1)
            self.assertIs(chunk.inputs[0], t1.chunks[i].data)
            # test chunk's index
            self.assertEqual(chunk.index, t1.chunks[i].index)
            # test chunk's dtype
            self.assertEqual(chunk.dtype, t.dtype)
            # test chunk's sparse
            self.assertFalse(chunk.issparse())

        # test add sparse tensor with scalar == 0
        t1 = tensor(self.raw_sparse_data1, chunk_size=(3, 4))
        t = t1 + 0

        self.assertEqual(t.shape, self.raw_data1.shape)
        self.assertEqual(t.dtype, self.raw_data1.dtype)
        self.assertTrue(t.issparse())

        t.tiles()

        self.assertEqual(t.shape, self.raw_data1.shape)
        self.assertEqual(t.chunk_shape, (3, 2))
        self.assertEqual(t.dtype, self.raw_data1.dtype)
        self.assertTrue(t.issparse())

        for i, chunk in enumerate(t.chunks):
            # test chunk's inputs
            self.assertEqual(len(chunk.inputs), 1)
            self.assertIs(chunk.inputs[0], t1.chunks[i].data)
            # test chunk's index
            self.assertEqual(chunk.index, t1.chunks[i].index)
            # test chunk's dtype
            self.assertEqual(chunk.dtype, t.dtype)
            # test chunk's sparse
            self.assertTrue(chunk.issparse())
