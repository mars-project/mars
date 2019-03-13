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
from mars.tensor.expressions.arithmetic import around, TensorAround


class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.raw_data = np.random.rand(7, 8)
        self.raw_sparse_data = sps.random(7, 8)

    def testTensorSerialize(self):
        t = around(tensor(self.raw_data, chunk_size=(3, 4)))

        serials = self._pb_serial(t)
        t2 = self._pb_deserial(serials)[t.data]

        self.assertIsInstance(t2.op, TensorAround)
        self.assertEqual(t.issparse(), t2.issparse())

    def testChunkSerialize(self):
        t = around(tensor(self.raw_data, chunk_size=(3, 4)))
        t.tiles()
        c = t.chunks[0]

        serials = self._pb_serial(c)
        c2 = self._pb_deserial(serials)[c.data]

        self.assertIsInstance(c2.op, TensorAround)
        self.assertEqual(c.issparse(), c2.issparse())

    def testDenseExpr(self):
        t1 = tensor(self.raw_data, chunk_size=(3, 4))
        t = around(t1, decimals=1)
        t.tiles()

        self.assertEqual(t.shape, self.raw_data.shape)
        self.assertEqual(t.chunk_shape, (3, 2))
        self.assertEqual(t.dtype, self.raw_data.dtype)
        self.assertEqual(t.op.decimals, 1)
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
            # test chunk's decimals
            self.assertEqual(chunk.op.decimals, 1)
            # test chunk's sparse
            self.assertFalse(chunk.issparse())

        t1 = tensor(self.raw_data, chunk_size=(3, 4))
        t = around(t1)

        # test around with out, wrong type of out
        with self.assertRaises(TypeError):
            around(t, out=self.raw_data)
        with self.assertRaises(TypeError):
            around(t, out=1)

        # test around with out, wrong shape of out
        with self.assertRaises(ValueError):
            around(t, out=t[:, 0])

        # test around with out, wrong dtype of out
        with self.assertRaises(TypeError):
            around(t, out=t.astype('i8'))

        # test out
        t1 = tensor(self.raw_data, chunk_size=(3, 4))
        t1_data = t1.data
        t2 = t[0, :].astype('f4')
        t3 = around(t2, out=t1)
        t3.tiles()

        self.assertEqual(t3.shape, t1.shape)
        self.assertEqual(t3.chunk_shape, (3, 2))
        self.assertEqual(t3.dtype, t1_data.dtype)
        self.assertFalse(t3.issparse())

        # test each chunk
        for i, chunk in enumerate(t3.chunks):
            # test chunk's inputs
            self.assertEqual(len(chunk.inputs), 2)
            self.assertIs(chunk.inputs[0], t2.chunks[chunk.index[1]].data)
            self.assertIs(chunk.inputs[1], t1_data.chunks[i].data)
            # test chunk's out
            self.assertIs(chunk.inputs[1], chunk.op.out)
            # test chunk's index
            self.assertEqual(chunk.index, t1_data.chunks[i].index)
            # test chunk's dtype
            self.assertEqual(chunk.dtype, t.dtype)
            # test chunk's sparse
            self.assertFalse(chunk.issparse())

        # test where which is not supported for around
        with self.assertRaises(TypeError):
            t1 = tensor(self.raw_data, chunk_size=(3, 4))
            t2 = t1[0] < 1
            around(t1, where=t2)

    def testSparseExpr(self):
        t1 = tensor(self.raw_sparse_data, chunk_size=(3, 4))
        t = around(t1, decimals=1)
        t.tiles()

        self.assertEqual(t.shape, self.raw_sparse_data.shape)
        self.assertEqual(t.chunk_shape, (3, 2))
        self.assertEqual(t.dtype, self.raw_sparse_data.dtype)
        self.assertEqual(t.op.decimals, 1)
        self.assertTrue(t.issparse())

        # test each chunk
        for i, chunk in enumerate(t.chunks):
            # test chunk's inZputs
            self.assertEqual(len(chunk.inputs), 1)
            self.assertIs(chunk.inputs[0], t1.chunks[i].data)
            # test chunk's index
            self.assertEqual(chunk.index, t1.chunks[i].index)
            # test chunk's dtype
            self.assertEqual(chunk.dtype, t.dtype)
            # test chunk's decimals
            self.assertEqual(chunk.op.decimals, 1)
            # test chunk's sparse
            self.assertTrue(chunk.issparse())

        t1 = tensor(self.raw_sparse_data, chunk_size=(3, 4))
        t = around(t1)

        # test around with out, wrong type of out
        with self.assertRaises(TypeError):
            around(t, out=self.raw_sparse_data)
        with self.assertRaises(TypeError):
            around(t, out=1)

        # test around with out, wrong shape of out
        with self.assertRaises(ValueError):
            around(t, out=t[:, 0])

        # test around with out, wrong dtype of out
        with self.assertRaises(TypeError):
            around(t, out=t.astype('i8'))

        # test around with out, sparse output to dense
        with self.assertRaises(ValueError):
            around(t, out=t.todense())

        # test out
        t1 = tensor(self.raw_sparse_data, chunk_size=(3, 4))
        t1_data = t1.data
        t2 = t[0, :].astype('f4')
        t3 = around(t2, out=t1)
        t3.tiles()

        self.assertEqual(t3.shape, t1.shape)
        self.assertEqual(t3.chunk_shape, (3, 2))
        self.assertEqual(t3.dtype, t1_data.dtype)
        self.assertTrue(t3.issparse())

        # test each chunk
        for i, chunk in enumerate(t3.chunks):
            # test chunk's inputs
            self.assertEqual(len(chunk.inputs), 2)
            self.assertIs(chunk.inputs[0], t2.chunks[chunk.index[1]].data)
            self.assertIs(chunk.inputs[1], t1_data.chunks[i].data)
            # test chunk's out
            self.assertIs(chunk.inputs[1], chunk.op.out)
            # test chunk's index
            self.assertEqual(chunk.index, t1_data.chunks[i].index)
            # test chunk's dtype
            self.assertEqual(chunk.dtype, t.dtype)
            # test chunk's sparse
            self.assertTrue(chunk.issparse())
