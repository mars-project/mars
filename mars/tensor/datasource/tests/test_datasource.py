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
import shutil
import tempfile
from weakref import ReferenceType
from copy import copy

import numpy as np
import scipy.sparse as sps
try:
    import tiledb
except (ImportError, OSError):  # pragma: no cover
    tiledb = None

import mars.dataframe as md
from mars.tensor import ones, zeros, tensor, full, arange, diag, linspace, triu, tril, ones_like, dot
from mars.tensor.datasource import array, fromtiledb, TensorTileDBDataSource, fromdense
from mars.tensor.datasource.tri import TensorTriu, TensorTril
from mars.tensor.datasource.zeros import TensorZeros
from mars.tensor.datasource.from_dense import DenseToSparse
from mars.tensor.datasource.array import CSRMatrixDataSource
from mars.tensor.datasource.ones import TensorOnes, TensorOnesLike
from mars.tensor.fuse.core import TensorFuseChunk
from mars.tensor.core import Tensor, SparseTensor, TensorChunk
from mars.tensor.datasource.from_dataframe import from_dataframe
from mars.graph import DAG
from mars.core import build_mode
from mars.serialize.protos.operand_pb2 import OperandDef
from mars.tests.core import TestBase


class Test(TestBase):
    def testChunkSerialize(self):
        t = ones((10, 3), chunk_size=(5, 2)).tiles()

        # pb
        chunk = t.chunks[0]
        serials = self._pb_serial(chunk)
        op, pb = serials[chunk.op, chunk.data]

        self.assertEqual(tuple(pb.index), chunk.index)
        self.assertEqual(pb.key, chunk.key)
        self.assertEqual(tuple(pb.shape), chunk.shape)
        self.assertEqual(int(op.type.split('.', 1)[1]), OperandDef.TENSOR_ONES)

        chunk2 = self._pb_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertEqual(chunk.op.dtype, chunk2.op.dtype)

        # json
        chunk = t.chunks[0]
        serials = self._json_serial(chunk)

        chunk2 = self._json_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertEqual(chunk.op.dtype, chunk2.op.dtype)

        t = tensor(np.random.random((10, 3)), chunk_size=(5, 2)).tiles()

        # pb
        chunk = t.chunks[0]
        serials = self._pb_serial(chunk)
        op, pb = serials[chunk.op, chunk.data]

        self.assertEqual(tuple(pb.index), chunk.index)
        self.assertEqual(pb.key, chunk.key)
        self.assertEqual(tuple(pb.shape), chunk.shape)
        self.assertEqual(int(op.type.split('.', 1)[1]), OperandDef.TENSOR_DATA_SOURCE)

        chunk2 = self._pb_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertTrue(np.array_equal(chunk.op.data, chunk2.op.data))

        # json
        chunk = t.chunks[0]
        serials = self._json_serial(chunk)

        chunk2 = self._json_deserial(serials)[chunk.data]

        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.key, chunk2.key)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertTrue(np.array_equal(chunk.op.data, chunk2.op.data))

        t = (tensor(np.random.random((10, 3)), chunk_size=(5, 2)) + 1).tiles()

        # pb
        chunk1 = t.chunks[0]
        chunk2 = t.chunks[1]
        fuse_op = TensorFuseChunk()
        composed_chunk = fuse_op.new_chunk(chunk1.inputs, shape=chunk2.shape,
                                           _key=chunk2.key, _composed=[chunk1.data, chunk2.data])
        serials = self._pb_serial(composed_chunk)
        op, pb = serials[composed_chunk.op, composed_chunk.data]

        self.assertEqual(pb.key, composed_chunk.key)
        self.assertEqual(int(op.type.split('.', 1)[1]), OperandDef.FUSE)
        self.assertEqual(len(pb.composed), 2)

        composed_chunk2 = self._pb_deserial(serials)[composed_chunk.data]

        self.assertEqual(composed_chunk.key, composed_chunk2.key)
        self.assertEqual(type(composed_chunk.op), type(composed_chunk2.op))
        self.assertEqual(composed_chunk.composed[0].inputs[0].key,
                         composed_chunk2.composed[0].inputs[0].key)
        self.assertEqual(composed_chunk.inputs[-1].key, composed_chunk2.inputs[-1].key)

        # json
        chunk1 = t.chunks[0]
        chunk2 = t.chunks[1]
        fuse_op = TensorFuseChunk()
        composed_chunk = fuse_op.new_chunk(chunk1.inputs, shape=chunk2.shape, _key=chunk2.key,
                                           _composed=[chunk1.data, chunk2.data])
        serials = self._json_serial(composed_chunk)

        composed_chunk2 = self._json_deserial(serials)[composed_chunk.data]

        self.assertEqual(composed_chunk.key, composed_chunk2.key)
        self.assertEqual(type(composed_chunk.op), type(composed_chunk2.op))
        self.assertEqual(composed_chunk.composed[0].inputs[0].key,
                         composed_chunk2.composed[0].inputs[0].key)
        self.assertEqual(composed_chunk.inputs[-1].key, composed_chunk2.inputs[-1].key)

        t1 = ones((10, 3), chunk_size=2)
        t2 = ones((3, 5), chunk_size=2)
        c = dot(t1, t2).tiles().chunks[0].inputs[0]

        # pb
        serials = self._pb_serial(c)
        c2 = self._pb_deserial(serials)[c]
        self.assertEqual(c.key, c2.key)

        # json
        serials = self._json_serial(c)
        c2 = self._json_deserial(serials)[c]
        self.assertEqual(c.key, c2.key)

    def testTensorSerialize(self):
        from mars.tensor import split

        t = ones((10, 10, 8), chunk_size=(3, 3, 5))

        serials = self._pb_serial(t)
        dt = self._pb_deserial(serials)[t.data]

        self.assertEqual(dt.extra_params.raw_chunk_size, (3, 3, 5))

        serials = self._json_serial(t)
        dt = self._json_deserial(serials)[t.data]

        self.assertEqual(dt.extra_params.raw_chunk_size, (3, 3, 5))

        t2, _ = split(t, 2)

        serials = self._pb_serial(t2)
        dt = self._pb_deserial(serials)[t2.data]
        self.assertEqual(dt.op.indices_or_sections, 2)

        t2, _, _ = split(t, ones(2, chunk_size=2))

        serials = self._pb_serial(t2)
        dt = self._pb_deserial(serials)[t2.data]
        with build_mode():
            self.assertIn(dt.op.indices_or_sections, dt.inputs)

    def testOnes(self):
        tensor = ones((10, 10, 8), chunk_size=(3, 3, 5))
        tensor.tiles()
        self.assertEqual(tensor.shape, (10, 10, 8))
        self.assertEqual(len(tensor.chunks), 32)

        tensor = ones((10, 3), chunk_size=(4, 2))
        tensor.tiles()
        self.assertEqual(tensor.shape, (10, 3))

        chunk = tensor.cix[1, 1]
        self.assertEqual(tensor.get_chunk_slices(chunk.index), (slice(4, 8), slice(2, 3)))

        tensor = ones((10, 5), chunk_size=(2, 3), gpu=True)
        tensor.tiles()

        self.assertTrue(tensor.op.gpu)
        self.assertTrue(tensor.chunks[0].op.gpu)

        tensor1 = ones((10, 10, 8), chunk_size=(3, 3, 5))
        tensor1.tiles()

        tensor2 = ones((10, 10, 8), chunk_size=(3, 3, 5))
        tensor2.tiles()

        self.assertEqual(tensor1.chunks[0].op.key, tensor2.chunks[0].op.key)
        self.assertEqual(tensor1.chunks[0].key, tensor2.chunks[0].key)
        self.assertNotEqual(tensor1.chunks[0].op.key, tensor1.chunks[1].op.key)
        self.assertNotEqual(tensor1.chunks[0].key, tensor1.chunks[1].key)

        tensor = ones((2, 3, 4))
        self.assertEqual(len(list(tensor)), 2)

        tensor2 = ones((2, 3, 4), chunk_size=1)
        # tensor's op key must be equal to tensor2
        self.assertEqual(tensor.op.key, tensor2.op.key)
        self.assertNotEqual(tensor.key, tensor2.key)

        tensor3 = ones((2, 3, 3))
        self.assertNotEqual(tensor.op.key, tensor3.op.key)
        self.assertNotEqual(tensor.key, tensor3.key)

        # test create chunk op of ones manually
        chunk_op1 = TensorOnes(dtype=tensor.dtype)
        chunk1 = chunk_op1.new_chunk(None, shape=(3, 3), index=(0, 0))
        chunk_op2 = TensorOnes(dtype=tensor.dtype)
        chunk2 = chunk_op2.new_chunk(None, shape=(3, 4), index=(0, 1))
        self.assertNotEqual(chunk1.op.key, chunk2.op.key)
        self.assertNotEqual(chunk1.key, chunk2.key)

        tensor = ones((100, 100), chunk_size=50)
        tensor.tiles()
        self.assertEqual(len({c.op.key for c in tensor.chunks}), 1)
        self.assertEqual(len({c.key for c in tensor.chunks}), 1)

    def testZeros(self):
        tensor = zeros((2, 3, 4))
        self.assertEqual(len(list(tensor)), 2)
        self.assertFalse(tensor.op.gpu)

        tensor2 = zeros((2, 3, 4), chunk_size=1)
        # tensor's op key must be equal to tensor2
        self.assertEqual(tensor.op.key, tensor2.op.key)
        self.assertNotEqual(tensor.key, tensor2.key)

        tensor3 = zeros((2, 3, 3))
        self.assertNotEqual(tensor.op.key, tensor3.op.key)
        self.assertNotEqual(tensor.key, tensor3.key)

        # test create chunk op of zeros manually
        chunk_op1 = TensorZeros(dtype=tensor.dtype)
        chunk1 = chunk_op1.new_chunk(None, shape=(3, 3), index=(0, 0))
        chunk_op2 = TensorZeros(dtype=tensor.dtype)
        chunk2 = chunk_op2.new_chunk(None, shape=(3, 4), index=(0, 1))
        self.assertNotEqual(chunk1.op.key, chunk2.op.key)
        self.assertNotEqual(chunk1.key, chunk2.key)

        tensor = zeros((100, 100), chunk_size=50)
        tensor.tiles()
        self.assertEqual(len({c.op.key for c in tensor.chunks}), 1)
        self.assertEqual(len({c.key for c in tensor.chunks}), 1)

    def testDataSource(self):
        from mars.tensor.base.broadcast_to import TensorBroadcastTo

        data = np.random.random((10, 3))
        t = tensor(data, chunk_size=2)
        self.assertFalse(t.op.gpu)
        t.tiles()
        self.assertTrue((t.chunks[0].op.data == data[:2, :2]).all())
        self.assertTrue((t.chunks[1].op.data == data[:2, 2:3]).all())
        self.assertTrue((t.chunks[2].op.data == data[2:4, :2]).all())
        self.assertTrue((t.chunks[3].op.data == data[2:4, 2:3]).all())

        self.assertEqual(t.key, tensor(data, chunk_size=2).tiles().key)
        self.assertNotEqual(t.key, tensor(data, chunk_size=3).tiles().key)
        self.assertNotEqual(t.key, tensor(np.random.random((10, 3)), chunk_size=2).tiles().key)

        t = tensor(data, chunk_size=2, gpu=True)
        t.tiles()

        self.assertTrue(t.op.gpu)
        self.assertTrue(t.chunks[0].op.gpu)

        t = full((2, 2), 2, dtype='f4')
        self.assertFalse(t.op.gpu)
        self.assertEqual(t.shape, (2, 2))
        self.assertEqual(t.dtype, np.float32)

        t = full((2, 2), [1.0, 2.0], dtype='f4')
        self.assertEqual(t.shape, (2, 2))
        self.assertEqual(t.dtype, np.float32)
        self.assertIsInstance(t.op, TensorBroadcastTo)

        with self.assertRaises(ValueError):
            full((2, 2), [1.0, 2.0, 3.0], dtype='f4')

    def testTensorGraphSerialize(self):
        t = ones((10, 3), chunk_size=(5, 2)) + tensor(np.random.random((10, 3)), chunk_size=(5, 2))
        graph = t.build_graph(tiled=False)

        pb = graph.to_pb()
        graph2 = DAG.from_pb(pb)
        self.assertEqual(len(graph), len(graph2))
        t = next(c for c in graph if c.inputs)
        t2 = next(c for c in graph2 if c.key == t.key)
        self.assertTrue(t2.op.outputs[0], ReferenceType)  # make sure outputs are all weak reference
        self.assertBaseEqual(t.op, t2.op)
        self.assertEqual(t.shape, t2.shape)
        self.assertEqual(sorted(i.key for i in t.inputs), sorted(i.key for i in t2.inputs))

        jsn = graph.to_json()
        graph2 = DAG.from_json(jsn)
        self.assertEqual(len(graph), len(graph2))
        t = next(c for c in graph if c.inputs)
        t2 = next(c for c in graph2 if c.key == t.key)
        self.assertTrue(t2.op.outputs[0], ReferenceType)  # make sure outputs are all weak reference
        self.assertBaseEqual(t.op, t2.op)
        self.assertEqual(t.shape, t2.shape)
        self.assertEqual(sorted(i.key for i in t.inputs), sorted(i.key for i in t2.inputs))

        # test graph with tiled tensor
        t2 = ones((10, 10), chunk_size=(5, 4)).tiles()
        graph = DAG()
        graph.add_node(t2)

        pb = graph.to_pb()
        graph2 = DAG.from_pb(pb)
        self.assertEqual(len(graph), len(graph2))
        chunks = next(iter(graph2)).chunks
        self.assertEqual(len(chunks), 6)
        self.assertIsInstance(chunks[0], TensorChunk)
        self.assertEqual(chunks[0].index, t2.chunks[0].index)
        self.assertBaseEqual(chunks[0].op, t2.chunks[0].op)

        jsn = graph.to_json()
        graph2 = DAG.from_json(jsn)
        self.assertEqual(len(graph), len(graph2))
        chunks = next(iter(graph2)).chunks
        self.assertEqual(len(chunks), 6)
        self.assertIsInstance(chunks[0], TensorChunk)
        self.assertEqual(chunks[0].index, t2.chunks[0].index)
        self.assertBaseEqual(chunks[0].op, t2.chunks[0].op)

    def testTensorGraphTiledSerialize(self):
        t = ones((10, 3), chunk_size=(5, 2)) + tensor(np.random.random((10, 3)), chunk_size=(5, 2))
        graph = t.build_graph(tiled=True)

        pb = graph.to_pb()
        graph2 = DAG.from_pb(pb)
        self.assertEqual(len(graph), len(graph2))
        chunk = next(c for c in graph if c.inputs)
        chunk2 = next(c for c in graph2 if c.key == chunk.key)
        self.assertBaseEqual(chunk.op, chunk2.op)
        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertEqual(sorted(i.key for i in chunk.inputs), sorted(i.key for i in chunk2.inputs))

        jsn = graph.to_json()
        graph2 = DAG.from_json(jsn)
        self.assertEqual(len(graph), len(graph2))
        chunk = next(c for c in graph if c.inputs)
        chunk2 = next(c for c in graph2 if c.key == chunk.key)
        self.assertBaseEqual(chunk.op, chunk2.op)
        self.assertEqual(chunk.index, chunk2.index)
        self.assertEqual(chunk.shape, chunk2.shape)
        self.assertEqual(sorted(i.key for i in chunk.inputs), sorted(i.key for i in chunk2.inputs))

        t = ones((10, 3), chunk_size=((3, 5, 2), 2)) + 2
        graph = t.build_graph(tiled=True)

        pb = graph.to_pb()
        graph2 = DAG.from_pb(pb)
        chunk = next(c for c in graph)
        chunk2 = next(c for c in graph2 if c.key == chunk.key)
        self.assertBaseEqual(chunk.op, chunk2.op)
        self.assertEqual(sorted(i.key for i in chunk.composed), sorted(i.key for i in chunk2.composed))

        jsn = graph.to_json()
        graph2 = DAG.from_json(jsn)
        chunk = next(c for c in graph)
        chunk2 = next(c for c in graph2 if c.key == chunk.key)
        self.assertBaseEqual(chunk.op, chunk2.op)
        self.assertEqual(sorted(i.key for i in chunk.composed), sorted(i.key for i in chunk2.composed))

    def testUfunc(self):
        t = ones((3, 10), chunk_size=2)

        x = np.add(t, [[1], [2], [3]])
        self.assertIsInstance(x, Tensor)

        y = np.sum(t, axis=1)
        self.assertIsInstance(y, Tensor)

    def testArange(self):
        t = arange(10, chunk_size=3)

        self.assertFalse(t.op.gpu)
        t.tiles()

        self.assertEqual(t.shape, (10,))
        self.assertEqual(t.nsplits, ((3, 3, 3, 1),))
        self.assertEqual(t.chunks[1].op.start, 3)
        self.assertEqual(t.chunks[1].op.stop, 6)

        t = arange(0, 10, 3, chunk_size=2)
        t.tiles()

        self.assertEqual(t.shape, (4,))
        self.assertEqual(t.nsplits, ((2, 2),))
        self.assertEqual(t.chunks[0].op.start, 0)
        self.assertEqual(t.chunks[0].op.stop, 6)
        self.assertEqual(t.chunks[0].op.step, 3)
        self.assertEqual(t.chunks[1].op.start, 6)
        self.assertEqual(t.chunks[1].op.stop, 12)
        self.assertEqual(t.chunks[1].op.step, 3)

        self.assertRaises(TypeError, lambda: arange(10, start=0))
        self.assertRaises(TypeError, lambda: arange(0, 10, stop=0))
        self.assertRaises(TypeError, lambda: arange())
        self.assertRaises(ValueError, lambda: arange('1066-10-13', dtype=np.datetime64, chunks=3))

    def testDiag(self):
        # test 2-d, shape[0] == shape[1], k == 0
        v = tensor(np.arange(16).reshape(4, 4), chunk_size=2)
        t = diag(v)

        self.assertEqual(t.shape, (4,))
        self.assertFalse(t.op.gpu)
        t.tiles()
        self.assertEqual(t.nsplits, ((2, 2),))

        v = tensor(np.arange(16).reshape(4, 4), chunk_size=(2, 3))
        t = diag(v)

        self.assertEqual(t.shape, (4,))
        t.tiles()
        self.assertEqual(t.nsplits, ((2, 1, 1),))

        # test 1-d, k == 0
        v = tensor(np.arange(3), chunk_size=2)
        t = diag(v, sparse=True)

        self.assertEqual(t.shape, (3, 3))
        t.tiles()
        self.assertEqual(t.nsplits, ((2, 1), (2, 1)))
        self.assertEqual(len([c for c in t.chunks
                              if c.op.__class__.__name__ == 'TensorDiag']), 2)
        self.assertTrue(t.chunks[0].op.sparse)

        # test 2-d, shape[0] != shape[1]
        v = tensor(np.arange(24).reshape(4, 6), chunk_size=2)
        t = diag(v)

        self.assertEqual(t.shape, np.diag(np.arange(24).reshape(4, 6)).shape)
        t.tiles()
        self.assertEqual(tuple(sum(s) for s in t.nsplits), t.shape)

        v = tensor(np.arange(24).reshape(4, 6), chunk_size=2)

        t = diag(v, k=1)
        self.assertEqual(t.shape, np.diag(np.arange(24).reshape(4, 6), k=1).shape)
        t.tiles()
        self.assertEqual(tuple(sum(s) for s in t.nsplits), t.shape)

        t = diag(v, k=2)
        self.assertEqual(t.shape, np.diag(np.arange(24).reshape(4, 6), k=2).shape)
        t.tiles()
        self.assertEqual(tuple(sum(s) for s in t.nsplits), t.shape)

        t = diag(v, k=-1)
        self.assertEqual(t.shape, np.diag(np.arange(24).reshape(4, 6), k=-1).shape)
        t.tiles()
        self.assertEqual(tuple(sum(s) for s in t.nsplits), t.shape)

        t = diag(v, k=-2)
        self.assertEqual(t.shape, np.diag(np.arange(24).reshape(4, 6), k=-2).shape)
        t.tiles()
        self.assertEqual(tuple(sum(s) for s in t.nsplits), t.shape)

        # test tiled zeros' keys
        a = arange(5, chunk_size=2)
        t = diag(a)
        t.tiles()
        # 1 and 2 of t.chunks is ones, they have different shapes
        self.assertNotEqual(t.chunks[1].op.key, t.chunks[2].op.key)

    def testLinspace(self):
        a = linspace(2.0, 3.0, num=5, chunk_size=2)

        self.assertEqual(a.shape, (5,))

        a.tiles()
        self.assertEqual(a.nsplits, ((2, 2, 1),))
        self.assertEqual(a.chunks[0].op.start, 2.)
        self.assertEqual(a.chunks[0].op.stop, 2.25)
        self.assertEqual(a.chunks[1].op.start, 2.5)
        self.assertEqual(a.chunks[1].op.stop, 2.75)
        self.assertEqual(a.chunks[2].op.start, 3.)
        self.assertEqual(a.chunks[2].op.stop, 3.)

        a = linspace(2.0, 3.0, num=5, endpoint=False, chunk_size=2)

        self.assertEqual(a.shape, (5,))

        a.tiles()
        self.assertEqual(a.nsplits, ((2, 2, 1),))
        self.assertEqual(a.chunks[0].op.start, 2.)
        self.assertEqual(a.chunks[0].op.stop, 2.2)
        self.assertEqual(a.chunks[1].op.start, 2.4)
        self.assertEqual(a.chunks[1].op.stop, 2.6)
        self.assertEqual(a.chunks[2].op.start, 2.8)
        self.assertEqual(a.chunks[2].op.stop, 2.8)

        _, step = linspace(2.0, 3.0, num=5, chunk_size=2, retstep=True)
        self.assertEqual(step, .25)

    def testTriuTril(self):
        a_data = np.arange(12).reshape(4, 3)
        a = tensor(a_data, chunk_size=2)

        t = triu(a)

        self.assertFalse(t.op.gpu)

        t.tiles()
        self.assertEqual(len(t.chunks), 4)
        self.assertIsInstance(t.chunks[0].op, TensorTriu)
        self.assertIsInstance(t.chunks[1].op, TensorTriu)
        self.assertIsInstance(t.chunks[2].op, TensorZeros)
        self.assertIsInstance(t.chunks[3].op, TensorTriu)

        t = triu(a, k=1)

        t.tiles()
        self.assertEqual(len(t.chunks), 4)
        self.assertIsInstance(t.chunks[0].op, TensorTriu)
        self.assertIsInstance(t.chunks[1].op, TensorTriu)
        self.assertIsInstance(t.chunks[2].op, TensorZeros)
        self.assertIsInstance(t.chunks[3].op, TensorZeros)

        t = triu(a, k=2)

        t.tiles()
        self.assertEqual(len(t.chunks), 4)
        self.assertIsInstance(t.chunks[0].op, TensorZeros)
        self.assertIsInstance(t.chunks[1].op, TensorTriu)
        self.assertIsInstance(t.chunks[2].op, TensorZeros)
        self.assertIsInstance(t.chunks[3].op, TensorZeros)

        t = triu(a, k=-1)

        t.tiles()
        self.assertEqual(len(t.chunks), 4)
        self.assertIsInstance(t.chunks[0].op, TensorTriu)
        self.assertIsInstance(t.chunks[1].op, TensorTriu)
        self.assertIsInstance(t.chunks[2].op, TensorTriu)
        self.assertIsInstance(t.chunks[3].op, TensorTriu)

        t = tril(a)

        self.assertFalse(t.op.gpu)

        t.tiles()
        self.assertEqual(len(t.chunks), 4)
        self.assertIsInstance(t.chunks[0].op, TensorTril)
        self.assertIsInstance(t.chunks[1].op, TensorZeros)
        self.assertIsInstance(t.chunks[2].op, TensorTril)
        self.assertIsInstance(t.chunks[3].op, TensorTril)

        t = tril(a, k=1)

        t.tiles()
        self.assertEqual(len(t.chunks), 4)
        self.assertIsInstance(t.chunks[0].op, TensorTril)
        self.assertIsInstance(t.chunks[1].op, TensorTril)
        self.assertIsInstance(t.chunks[2].op, TensorTril)
        self.assertIsInstance(t.chunks[3].op, TensorTril)

        t = tril(a, k=-1)

        t.tiles()
        self.assertEqual(len(t.chunks), 4)
        self.assertIsInstance(t.chunks[0].op, TensorTril)
        self.assertIsInstance(t.chunks[1].op, TensorZeros)
        self.assertIsInstance(t.chunks[2].op, TensorTril)
        self.assertIsInstance(t.chunks[3].op, TensorTril)

        t = tril(a, k=-2)

        t.tiles()
        self.assertEqual(len(t.chunks), 4)
        self.assertIsInstance(t.chunks[0].op, TensorZeros)
        self.assertIsInstance(t.chunks[1].op, TensorZeros)
        self.assertIsInstance(t.chunks[2].op, TensorTril)
        self.assertIsInstance(t.chunks[3].op, TensorZeros)

    def testSetTensorInputs(self):
        t1 = tensor([1, 2], chunk_size=2)
        t2 = tensor([2, 3], chunk_size=2)
        t3 = t1 + t2

        t1c = copy(t1)
        t2c = copy(t2)

        self.assertIsNot(t1c, t1)
        self.assertIsNot(t2c, t2)

        self.assertIs(t3.op.lhs, t1.data)
        self.assertIs(t3.op.rhs, t2.data)
        self.assertEqual(t3.op.inputs, [t1.data, t2.data])
        self.assertEqual(t3.inputs, [t1.data, t2.data])

        with self.assertRaises(StopIteration):
            t3.inputs = []

        t1 = tensor([1, 2], chunk_size=2)
        t2 = tensor([True, False], chunk_size=2)
        t3 = t1[t2]

        t1c = copy(t1)
        t2c = copy(t2)
        t3c = copy(t3)
        t3c.inputs = [t1c, t2c]

        with build_mode():
            self.assertIs(t3c.op.input, t1c.data)
            self.assertIs(t3c.op.indexes[0], t2c.data)

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

    def testFromDense(self):
        t = fromdense(tensor([[0, 0, 1], [1, 0, 0]], chunk_size=2))

        self.assertIsInstance(t, SparseTensor)
        self.assertIsInstance(t.op, DenseToSparse)
        self.assertTrue(t.issparse())

        t.tiles()
        self.assertEqual(t.chunks[0].index, (0, 0))
        self.assertIsInstance(t.op, DenseToSparse)

    def testOnesLike(self):
        t1 = tensor([[0, 0, 1], [1, 0, 0]], chunk_size=2).tosparse()
        t = ones_like(t1, dtype='f8')

        self.assertIsInstance(t, SparseTensor)
        self.assertIsInstance(t.op, TensorOnesLike)
        self.assertTrue(t.issparse())
        self.assertFalse(t.op.gpu)

        t.tiles()
        self.assertEqual(t.chunks[0].index, (0, 0))
        self.assertIsInstance(t.op, TensorOnesLike)
        self.assertTrue(t.chunks[0].issparse())

    def testFromArray(self):
        x = array([1, 2, 3])
        self.assertEqual(x.shape, (3,))

        y = array([x, x])
        self.assertEqual(y.shape, (2, 3))

        z = array((x, x, x))
        self.assertEqual(z.shape, (3, 3))

    @unittest.skipIf(tiledb is None, 'TileDB not installed')
    def testFromTileDB(self):
        ctx = tiledb.Ctx()

        for sparse in (True, False):
            dom = tiledb.Domain(
                tiledb.Dim(ctx=ctx, name="i", domain=(1, 30), tile=7, dtype=np.int32),
                tiledb.Dim(ctx=ctx, name="j", domain=(1, 20), tile=3, dtype=np.int32),
                tiledb.Dim(ctx=ctx, name="k", domain=(1, 10), tile=4, dtype=np.int32),
                ctx=ctx,
            )
            schema = tiledb.ArraySchema(ctx=ctx, domain=dom, sparse=sparse,
                                        attrs=[tiledb.Attr(ctx=ctx, name='a', dtype=np.float32)])

            tempdir = tempfile.mkdtemp()
            try:
                # create tiledb array
                array_type = tiledb.DenseArray if not sparse else tiledb.SparseArray
                array_type.create(tempdir, schema)

                tensor = fromtiledb(tempdir)
                self.assertIsInstance(tensor.op, TensorTileDBDataSource)
                self.assertEqual(tensor.op.issparse(), sparse)
                self.assertEqual(tensor.shape, (30, 20, 10))
                self.assertEqual(tensor.extra_params.raw_chunk_size, (7, 3, 4))
                self.assertIsNone(tensor.op.tiledb_config)
                self.assertEqual(tensor.op.tiledb_uri, tempdir)
                self.assertIsNone(tensor.op.tiledb_key)
                self.assertIsNone(tensor.op.tiledb_timestamp)

                tensor.tiles()

                self.assertEqual(len(tensor.chunks), 105)
                self.assertIsInstance(tensor.chunks[0].op, TensorTileDBDataSource)
                self.assertEqual(tensor.chunks[0].op.issparse(), sparse)
                self.assertEqual(tensor.chunks[0].shape, (7, 3, 4))
                self.assertIsNone(tensor.chunks[0].op.tiledb_config)
                self.assertEqual(tensor.chunks[0].op.tiledb_uri, tempdir)
                self.assertIsNone(tensor.chunks[0].op.tiledb_key)
                self.assertIsNone(tensor.chunks[0].op.tiledb_timestamp)
                self.assertEqual(tensor.chunks[0].op.tiledb_dim_starts, (1, 1, 1))

                # test axis_offsets of chunk op
                self.assertEqual(tensor.chunks[0].op.axis_offsets, (0, 0, 0))
                self.assertEqual(tensor.chunks[1].op.axis_offsets, (0, 0, 4))
                self.assertEqual(tensor.cix[0, 2, 2].op.axis_offsets, (0, 6, 8))
                self.assertEqual(tensor.cix[0, 6, 2].op.axis_offsets, (0, 18, 8))
                self.assertEqual(tensor.cix[4, 6, 2].op.axis_offsets, (28, 18, 8))

                tensor2 = fromtiledb(tempdir, ctx=ctx)
                self.assertEqual(tensor2.op.tiledb_config, ctx.config().dict())

                tensor2.tiles()

                self.assertEqual(tensor2.chunks[0].op.tiledb_config, ctx.config().dict())
            finally:
                shutil.rmtree(tempdir)

    @unittest.skipIf(tiledb is None, 'TileDB not installed')
    def testDimStartFloat(self):
        ctx = tiledb.Ctx()

        dom = tiledb.Domain(
            tiledb.Dim(ctx=ctx, name="i", domain=(0.0, 6.0), tile=6, dtype=np.float64),
            ctx=ctx,
        )
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, sparse=True,
                                    attrs=[tiledb.Attr(ctx=ctx, name='a', dtype=np.float32)])

        tempdir = tempfile.mkdtemp()
        try:
            # create tiledb array
            tiledb.SparseArray.create(tempdir, schema)

            with self.assertRaises(ValueError):
                fromtiledb(tempdir, ctx=ctx)
        finally:
            shutil.rmtree(tempdir)

    def testFromDataFrame(self):
        mdf = md.DataFrame({'a': [0, 1, 2], 'b': [3, 4, 5]}, index=['c', 'd', 'e'], chunk_size=2)
        tensor = from_dataframe(mdf)
        self.assertEqual(tensor.shape, (3, 2))
        self.assertEqual(mdf.dtypes[0], tensor.dtype)
