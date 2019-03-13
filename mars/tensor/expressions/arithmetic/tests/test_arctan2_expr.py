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
from mars.tensor.expressions.arithmetic import arctan2, TensorArctan2, TensorArct2Constant


class Test(TestBase):
    def setUp(self):
        super(Test, self).setUp()
        self.raw_data1 = np.random.rand(7, 8)
        self.raw_data2 = np.random.random(8)
        self.raw_sparse_data1 = sps.random(7, 8)
        self.raw_sparse_data2 = sps.random(1, 8)

    def testTensorSerialize(self):
        # test arctan2 two dense tensors
        t = arctan2(tensor(self.raw_data1, chunk_size=(3, 4)),
                    tensor(self.raw_data2, chunk_size=4))

        serials = self._pb_serial(t)
        t2 = self._pb_deserial(serials)[t.data]

        self.assertIsInstance(t2.op, TensorArctan2)
        self.assertEqual(t.issparse(), t2.issparse())

        # test arctan2 one dense with scalar
        t = arctan2(tensor(self.raw_data1, chunk_size=(3, 4)), 10)

        serials = self._pb_serial(t)
        t2 = self._pb_deserial(serials)[t.data]

        self.assertIsInstance(t2.op, TensorArct2Constant)
        self.assertEqual(t.issparse(), t2.issparse())

        # test arctan2 two sparse tensors
        t = arctan2(tensor(self.raw_sparse_data1, chunk_size=(3, 4)),
                    tensor(self.raw_sparse_data2, chunk_size=4))

        serials = self._pb_serial(t)
        t2 = self._pb_deserial(serials)[t.data]

        self.assertIsInstance(t2.op, TensorArctan2)
        self.assertEqual(t.issparse(), t2.issparse())

        # test arctan2 one sparse with scalar
        t = arctan2(tensor(self.raw_sparse_data1, chunk_size=(3, 4)), 10)

        serials = self._pb_serial(t)
        t2 = self._pb_deserial(serials)[t.data]

        self.assertIsInstance(t2.op, TensorArct2Constant)
        self.assertEqual(t.issparse(), t2.issparse())

        # test arctan2 with out
        t = tensor(self.raw_data1, chunk_size=(3, 4))
        t_op_type = type(t.op)
        t = arctan2(t, 1, out=t)

        serials = self._pb_serial(t)
        t2 = self._pb_deserial(serials)[t.data]

        self.assertIsInstance(t2.op, TensorArctan2)  # not constant
        self.assertIsInstance(t2.op.out.op, t_op_type)
        self.assertEqual(t.issparse(), t2.issparse())

        # test arctan2 with out and where
        t = tensor(self.raw_data1, chunk_size=(3, 4))
        w = tensor(self.raw_data1 < 0.5, chunk_size=(3, 4))
        t_op_type = type(t.op)
        w_op_type = type(w.op)
        t = arctan2(t, 1, out=t, where=w)

        serials = self._pb_serial(t)
        t2 = self._pb_deserial(serials)[t.data]

        self.assertIsInstance(t2.op, TensorArctan2)  # not constant
        self.assertIsInstance(t2.op.out.op, t_op_type)
        self.assertIsInstance(t2.op.where.op, w_op_type)
        self.assertEqual(t.issparse(), t2.issparse())

    def testChunkSerialize(self):
        # test arctan2 two dense tensors
        t = arctan2(tensor(self.raw_data1, chunk_size=(3, 4)),
                    tensor(self.raw_data2, chunk_size=4))
        t.tiles()
        c = t.chunks[0]

        serials = self._pb_serial(c)
        c2 = self._pb_deserial(serials)[c.data]

        self.assertIsInstance(c2.op, TensorArctan2)
        self.assertEqual(c.issparse(), c2.issparse())

        # test arctan2 one dense with scalar
        t = arctan2(tensor(self.raw_data1), 10)
        t.tiles()
        c = t.chunks[0]

        serials = self._pb_serial(c)
        c2 = self._pb_deserial(serials)[c.data]

        self.assertIsInstance(c2.op, TensorArct2Constant)
        self.assertEqual(c.issparse(), c2.issparse())

        # test arctan2 two sparse tensors
        t = arctan2(tensor(self.raw_sparse_data1), tensor(self.raw_sparse_data2))
        t.tiles()
        c = t.chunks[0]

        serials = self._pb_serial(c)
        c2 = self._pb_deserial(serials)[c.data]

        self.assertIsInstance(c2.op, TensorArctan2)
        self.assertEqual(c.issparse(), c2.issparse())

        # test arctan2 one sparse with scalar
        t = arctan2(tensor(self.raw_sparse_data1), 10)
        t.tiles()
        c = t.chunks[0]

        serials = self._pb_serial(c)
        c2 = self._pb_deserial(serials)[c.data]

        self.assertIsInstance(c2.op, TensorArct2Constant)
        self.assertEqual(c.issparse(), c2.issparse())

        # test arctan2 with out
        t = tensor(self.raw_data1, chunk_size=(3, 4))
        t_op_type = type(t.op)
        t = arctan2(t, 1, out=t)
        t.tiles()
        c = t.chunks[0]

        serials = self._pb_serial(c)
        c2 = self._pb_deserial(serials)[c.data]

        self.assertIsInstance(c2.op, TensorArctan2)  # not constant
        self.assertIsInstance(c2.op.out.op, t_op_type)
        self.assertEqual(c.issparse(), c2.issparse())

        # test arctan2 with out and where
        t = tensor(self.raw_data1, chunk_size=(3, 4))
        w = tensor(self.raw_data1 < 0.5, chunk_size=(3, 4))
        t_op_type = type(t.op)
        w_op_type = type(w.op)
        t = arctan2(t, 1, out=t, where=w)
        t.tiles()
        c = t.chunks[0]

        serials = self._pb_serial(c)
        c2 = self._pb_deserial(serials)[c.data]

        self.assertIsInstance(c2.op, TensorArctan2)  # not constant
        self.assertIsInstance(c2.op.out.op, t_op_type)
        self.assertIsInstance(c2.op.where.op, w_op_type)
        self.assertEqual(c.issparse(), c2.issparse())

    def testDenseExpr(self):
        # test arctan2 two dense tensor
        t1 = tensor(self.raw_data1, chunk_size=(3, 4))
        t2 = tensor(self.raw_data2, chunk_size=4)
        t = arctan2(t1, t2)

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

        # test arctan2 tensor with scalar
        t1 = tensor(self.raw_data1, chunk_size=(3, 4))
        t = arctan2(t1, 10)

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

        # test arctan2 scalar to scalar
        t1 = tensor(self.raw_data1, chunk_size=(3, 4))
        t = arctan2(10, t1)

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

        # test arctan2 with out
        t1 = tensor(self.raw_data1, chunk_size=(3, 4))
        t1_data = t1.data
        t = arctan2(t1, 10, out=t1)

        self.assertEqual(t.shape, self.raw_data1.shape)
        self.assertEqual(t.dtype, self.raw_data1.dtype)
        self.assertFalse(t.issparse())
        self.assertIs(t.op.out, t1_data)

        t.tiles()

        self.assertEqual(t.shape, self.raw_data1.shape)
        self.assertEqual(t.chunk_shape, (3, 2))
        self.assertEqual(t.dtype, self.raw_data1.dtype)
        self.assertFalse(t.issparse())

        for i, chunk in enumerate(t.chunks):
            # test chunk's inputs
            self.assertEqual(len(chunk.inputs), 3)
            self.assertIs(chunk.inputs[0], t1_data.chunks[i].data)
            self.assertEqual(chunk.inputs[1].op.data, 10)
            self.assertIs(chunk.inputs[0], chunk.inputs[2])  # out is input
            # test chunk's index
            self.assertEqual(chunk.index, t1.chunks[i].index)
            # test chunk's dtype
            self.assertEqual(chunk.dtype, t.dtype)
            # test chunk's sparse
            self.assertFalse(chunk.issparse())
            # test chunk's out
            self.assertIs(chunk.op.out, t1_data.chunks[i].data)

        # test arctan2 with out and where
        t1 = tensor(self.raw_data1, chunk_size=(3, 4))
        t1_data = t1.data
        w1 = tensor(self.raw_data1 < 0.5, chunk_size=(3, 4))
        w1_data = w1.data
        t = arctan2(t1, 10, out=t1, where=w1)

        self.assertEqual(t.shape, self.raw_data1.shape)
        self.assertEqual(t.dtype, self.raw_data1.dtype)
        self.assertFalse(t.issparse())
        self.assertIs(t.op.out, t1_data)
        self.assertIs(t.op.where, w1_data)

        t.tiles()

        self.assertEqual(t.shape, self.raw_data1.shape)
        self.assertEqual(t.chunk_shape, (3, 2))
        self.assertEqual(t.dtype, self.raw_data1.dtype)
        self.assertFalse(t.issparse())

        for i, chunk in enumerate(t.chunks):
            # test chunk's inputs
            self.assertEqual(len(chunk.inputs), 4)
            self.assertIs(chunk.inputs[0], t1_data.chunks[i].data)
            self.assertEqual(chunk.inputs[1].op.data, 10)
            self.assertIs(chunk.inputs[0], chunk.inputs[2])  # out is input
            self.assertIs(chunk.inputs[3], chunk.op.where)
            # test chunk's index
            self.assertEqual(chunk.index, t1.chunks[i].index)
            # test chunk's dtype
            self.assertEqual(chunk.dtype, t.dtype)
            # test chunk's sparse
            self.assertFalse(chunk.issparse())
            # test chunk's out
            self.assertIs(chunk.op.out, t1_data.chunks[i].data)
            # test chunk's where
            self.assertIs(chunk.op.where, w1_data.chunks[i].data)

    def testSparseExpr(self):
        # test arctan2 two sparse tensor
        t1 = tensor(self.raw_sparse_data1, chunk_size=(3, 4))
        t2 = tensor(self.raw_sparse_data2, chunk_size=4)
        t = arctan2(t1, t2)

        self.assertEqual(t.shape, self.raw_sparse_data1.shape)
        self.assertEqual(t.dtype, self.raw_sparse_data1.dtype)
        self.assertTrue(t.issparse())

        t.tiles()

        self.assertEqual(t.shape, self.raw_sparse_data1.shape)
        self.assertEqual(t.chunk_shape, (3, 2))
        self.assertEqual(t.dtype, self.raw_sparse_data1.dtype)
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

        # test arctan2 dense and sparse tensor
        t1 = tensor(self.raw_sparse_data1, chunk_size=(3, 4))
        t2 = tensor(self.raw_data2, chunk_size=4)
        t = arctan2(t1, t2)

        self.assertEqual(t.shape, self.raw_sparse_data1.shape)
        self.assertEqual(t.dtype, self.raw_sparse_data1.dtype)
        self.assertTrue(t.issparse())

        t.tiles()

        self.assertEqual(t.shape, self.raw_sparse_data1.shape)
        self.assertEqual(t.chunk_shape, (3, 2))
        self.assertEqual(t.dtype, self.raw_sparse_data1.dtype)
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

        # test arctan2 scalar != 0 with sparse tensor
        t1 = tensor(self.raw_sparse_data1, chunk_size=(3, 4))
        t = arctan2(10, t1)

        self.assertEqual(t.shape, self.raw_sparse_data1.shape)
        self.assertEqual(t.dtype, self.raw_sparse_data1.dtype)
        self.assertFalse(t.issparse())

        t.tiles()

        self.assertEqual(t.shape, self.raw_sparse_data1.shape)
        self.assertEqual(t.chunk_shape, (3, 2))
        self.assertEqual(t.dtype, self.raw_sparse_data1.dtype)
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

        # test arctan2 sparse tensor with scalar == 0
        t1 = tensor(self.raw_sparse_data1, chunk_size=(3, 4))
        t = arctan2(0, t1)

        self.assertEqual(t.shape, self.raw_sparse_data1.shape)
        self.assertEqual(t.dtype, self.raw_sparse_data1.dtype)
        self.assertTrue(t.issparse())

        t.tiles()

        self.assertEqual(t.shape, self.raw_sparse_data1.shape)
        self.assertEqual(t.chunk_shape, (3, 2))
        self.assertEqual(t.dtype, self.raw_sparse_data1.dtype)
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

        # test arctan2 with out
        t1 = tensor(self.raw_sparse_data1, chunk_size=(3, 4))
        t1_data = t1.data
        t = arctan2(t1, 10, out=t1)

        self.assertEqual(t.shape, self.raw_sparse_data1.shape)
        self.assertEqual(t.dtype, self.raw_sparse_data1.dtype)
        self.assertTrue(t.issparse())
        self.assertIs(t.op.out, t1_data)

        t.tiles()

        self.assertEqual(t.shape, self.raw_sparse_data1.shape)
        self.assertEqual(t.chunk_shape, (3, 2))
        self.assertEqual(t.dtype, self.raw_sparse_data1.dtype)
        self.assertTrue(t.issparse())

        for i, chunk in enumerate(t.chunks):
            # test chunk's inputs
            self.assertEqual(len(chunk.inputs), 3)
            self.assertIs(chunk.inputs[0], t1_data.chunks[i].data)
            self.assertEqual(chunk.inputs[1].op.data, 10)
            self.assertIs(chunk.inputs[0], chunk.inputs[2])  # out is input
            # test chunk's index
            self.assertEqual(chunk.index, t1.chunks[i].index)
            # test chunk's dtype
            self.assertEqual(chunk.dtype, t.dtype)
            # test chunk's sparse
            self.assertTrue(chunk.issparse())
            # test chunk's out
            self.assertIs(chunk.op.out, t1_data.chunks[i].data)

        # test arctan2 with out and where
        t1 = tensor(self.raw_sparse_data1, chunk_size=(3, 4))
        t1_data = t1.data
        w1 = tensor(self.raw_sparse_data1 < 0.5, chunk_size=(3, 4))
        w1_data = w1.data
        t = arctan2(t1, 10, out=t1, where=w1)

        self.assertEqual(t.shape, self.raw_sparse_data1.shape)
        self.assertEqual(t.dtype, self.raw_sparse_data1.dtype)
        self.assertTrue(t.issparse())
        self.assertIs(t.op.out, t1_data)
        self.assertIs(t.op.where, w1_data)

        t.tiles()

        self.assertEqual(t.shape, self.raw_sparse_data1.shape)
        self.assertEqual(t.chunk_shape, (3, 2))
        self.assertEqual(t.dtype, self.raw_sparse_data1.dtype)
        self.assertTrue(t.issparse())

        for i, chunk in enumerate(t.chunks):
            # test chunk's inputs
            self.assertEqual(len(chunk.inputs), 4)
            self.assertIs(chunk.inputs[0], t1_data.chunks[i].data)
            self.assertEqual(chunk.inputs[1].op.data, 10)
            self.assertIs(chunk.inputs[0], chunk.inputs[2])  # out is input
            self.assertIs(chunk.inputs[3], chunk.op.where)
            # test chunk's index
            self.assertEqual(chunk.index, t1.chunks[i].index)
            # test chunk's dtype
            self.assertEqual(chunk.dtype, t.dtype)
            # test chunk's sparse
            self.assertTrue(chunk.issparse())
            # test chunk's out
            self.assertIs(chunk.op.out, t1_data.chunks[i].data)
            # test chunk's where
            self.assertIs(chunk.op.where, w1_data.chunks[i].data)
