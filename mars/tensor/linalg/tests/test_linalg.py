#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import scipy.sparse as sps

import mars.tensor as mt
from mars.tensor import ones, tensor, dot, empty
from mars.graph import DirectedGraph
from mars.tensor.core import SparseTensor, Tensor
from mars.tensor.linalg import matmul


class Test(unittest.TestCase):
    def testQR(self):
        a = mt.random.rand(9, 6, chunk_size=(3, 6))
        q, r = mt.linalg.qr(a)

        self.assertEqual(q.shape, (9, 6))
        self.assertEqual(r.shape, (6, 6))

        q.tiles()

        self.assertEqual(len(q.chunks), 3)
        self.assertEqual(len(r.chunks), 1)
        self.assertEqual(q.nsplits, ((3, 3, 3), (6,)))
        self.assertEqual(r.nsplits, ((6,), (6,)))

        # for Short-and-Fat QR
        a = mt.random.rand(6, 18, chunk_size=(6, 6))
        q, r = mt.linalg.qr(a, method='sfqr')

        self.assertEqual(q.shape, (6, 6))
        self.assertEqual(r.shape, (6, 18))
        q.tiles()

        self.assertEqual(len(q.chunks), 1)
        self.assertEqual(len(r.chunks), 3)
        self.assertEqual(q.nsplits, ((6,), (6,)))
        self.assertEqual(r.nsplits, ((6,), (6, 6, 6)))

        # chunk width less than height
        a = mt.random.rand(6, 9, chunk_size=(6, 3))
        q, r = mt.linalg.qr(a, method='sfqr')

        self.assertEqual(q.shape, (6, 6))
        self.assertEqual(r.shape, (6, 9))

        q.tiles()

        self.assertEqual(len(q.chunks), 1)
        self.assertEqual(len(r.chunks), 2)
        self.assertEqual(q.nsplits, ((6,), (6,)))
        self.assertEqual(r.nsplits, ((6,), (6, 3)))

        a = mt.random.rand(9, 6, chunk_size=(9, 3))
        q, r = mt.linalg.qr(a, method='sfqr')

        self.assertEqual(q.shape, (9, 6))
        self.assertEqual(r.shape, (6, 6))

        q.tiles()

        self.assertEqual(len(q.chunks), 1)
        self.assertEqual(len(r.chunks), 1)
        self.assertEqual(q.nsplits, ((9,), (6,)))
        self.assertEqual(r.nsplits, ((6,), (6,)))

    def testNorm(self):
        data = np.random.rand(9, 6)

        a = mt.tensor(data, chunk_size=(2, 6))

        for ord in (None, 'nuc', np.inf, -np.inf, 0, 1, -1, 2, -2):
            for axis in (0, 1, (0, 1)):
                for keepdims in (True, False):
                    try:
                        res = mt.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims)
                        expect_shape = np.linalg.norm(data, ord=ord, axis=axis, keepdims=keepdims).shape
                        self.assertEqual(res.shape, expect_shape)
                    except ValueError:
                        continue

    def testSVD(self):
        a = mt.random.rand(9, 6, chunk_size=(3, 6))
        U, s, V = mt.linalg.svd(a)

        self.assertEqual(U.shape, (9, 6))
        self.assertEqual(s.shape, (6,))
        self.assertEqual(V.shape, (6, 6))

        U.tiles()
        self.assertEqual(len(U.chunks), 3)
        self.assertEqual(U.chunks[0].shape, (3, 6))
        self.assertEqual(len(s.chunks), 1)
        self.assertEqual(s.chunks[0].shape, (6,))
        self.assertEqual(len(V.chunks), 1)
        self.assertEqual(V.chunks[0].shape, (6, 6))

        self.assertEqual(s.ndim, 1)
        self.assertEqual(len(s.chunks[0].index), 1)

        a = mt.random.rand(9, 6, chunk_size=(9, 6))
        U, s, V = mt.linalg.svd(a)

        self.assertEqual(U.shape, (9, 6))
        self.assertEqual(s.shape, (6,))
        self.assertEqual(V.shape, (6, 6))

        U.tiles()
        self.assertEqual(len(U.chunks), 1)
        self.assertEqual(U.chunks[0].shape, (9, 6))
        self.assertEqual(len(s.chunks), 1)
        self.assertEqual(s.chunks[0].shape, (6,))
        self.assertEqual(len(V.chunks), 1)
        self.assertEqual(V.chunks[0].shape, (6, 6))

        self.assertEqual(s.ndim, 1)
        self.assertEqual(len(s.chunks[0].index), 1)

        a = mt.random.rand(6, 20, chunk_size=10)
        U, s, V = mt.linalg.svd(a)

        self.assertEqual(U.shape, (6, 6))
        self.assertEqual(s.shape, (6,))
        self.assertEqual(V.shape, (6, 20))

        U.tiles()
        self.assertEqual(len(U.chunks), 1)
        self.assertEqual(U.chunks[0].shape, (6, 6))
        self.assertEqual(len(s.chunks), 1)
        self.assertEqual(s.chunks[0].shape, (6,))
        self.assertEqual(len(V.chunks), 1)
        self.assertEqual(V.chunks[0].shape, (6, 20))

        a = mt.random.rand(6, 9, chunk_size=(6, 9))
        U, s, V = mt.linalg.svd(a)

        self.assertEqual(U.shape, (6, 6))
        self.assertEqual(s.shape, (6,))
        self.assertEqual(V.shape, (6, 9))

        rs = mt.random.RandomState(1)
        a = rs.rand(9, 6, chunk_size=(3, 6))
        U, s, V = mt.linalg.svd(a)

        # test tensor graph
        graph = DirectedGraph()
        U.build_graph(tiled=False, graph=graph)
        s.build_graph(tiled=False, graph=graph)
        new_graph = DirectedGraph.from_json(graph.to_json())
        self.assertEqual((len(new_graph)), 4)
        new_outputs = [n for n in new_graph if new_graph.count_predecessors(n) == 1]
        self.assertEqual(len(new_outputs), 3)
        self.assertEqual(len(set([o.op for o in new_outputs])), 1)

        # test tensor graph, do some caculation
        graph = DirectedGraph()
        (U + 1).build_graph(tiled=False, graph=graph)
        (s + 1).build_graph(tiled=False, graph=graph)
        new_graph = DirectedGraph.from_json(graph.to_json())
        self.assertEqual((len(new_graph)), 6)
        new_outputs = [n for n in new_graph if new_graph.count_predecessors(n) == 1]
        self.assertEqual(len(new_outputs), 5)
        self.assertEqual(len(set([o.op for o in new_outputs])), 3)

        a = rs.rand(20, 10, chunk_size=10)
        _, s, _ = mt.linalg.svd(a)
        del _
        graph = s.build_graph(tiled=False)
        self.assertEqual(len(graph), 4)

    def testLU(self):
        a = mt.random.randint(1, 10, (6, 6), chunk_size=3)
        p, l, u = mt.linalg.lu(a)
        l.tiles()

        self.assertEqual(l.shape, (6, 6))
        self.assertEqual(u.shape, (6, 6))
        self.assertEqual(p.shape, (6, 6))

        a = mt.random.randint(1, 10, (6, 6), chunk_size=(3, 2))
        p, l, u = mt.linalg.lu(a)
        l.tiles()

        self.assertEqual(l.shape, (6, 6))
        self.assertEqual(u.shape, (6, 6))
        self.assertEqual(p.shape, (6, 6))

        self.assertEqual(p.nsplits, ((3, 3), (3, 3)))
        self.assertEqual(l.nsplits, ((3, 3), (3, 3)))
        self.assertEqual(u.nsplits, ((3, 3), (3, 3)))

        a = mt.random.randint(1, 10, (7, 7), chunk_size=4)
        p, l, u = mt.linalg.lu(a)
        l.tiles()

        self.assertEqual(l.shape, (7, 7))
        self.assertEqual(u.shape, (7, 7))
        self.assertEqual(p.shape, (7, 7))

        self.assertEqual(p.nsplits, ((4, 3), (4, 3)))
        self.assertEqual(l.nsplits, ((4, 3), (4, 3)))
        self.assertEqual(u.nsplits, ((4, 3), (4, 3)))

        a = mt.random.randint(1, 10, (7, 5), chunk_size=4)
        p, l, u = mt.linalg.lu(a)
        l.tiles()

        self.assertEqual(l.shape, (7, 5))
        self.assertEqual(u.shape, (5, 5))
        self.assertEqual(p.shape, (7, 7))

        a = mt.random.randint(1, 10, (5, 7), chunk_size=4)
        p, l, u = mt.linalg.lu(a)
        l.tiles()

        self.assertEqual(l.shape, (5, 5))
        self.assertEqual(u.shape, (5, 7))
        self.assertEqual(p.shape, (5, 5))

        # test sparse
        data = sps.csr_matrix([[2, 0, 0, 0, 5, 2],
                               [0, 6, 1, 0, 0, 6],
                               [8, 0, 9, 0, 0, 2],
                               [0, 6, 0, 8, 7, 3],
                               [7, 0, 6, 1, 7, 0],
                               [0, 0, 0, 7, 0, 8]])
        t = mt.tensor(data, chunk_size=3)
        p, l, u = mt.linalg.lu(t)

        self.assertTrue(p.op.sparse)
        self.assertIsInstance(p, SparseTensor)
        self.assertTrue(l.op.sparse)
        self.assertIsInstance(l, SparseTensor)
        self.assertTrue(u.op.sparse)
        self.assertIsInstance(u, SparseTensor)

        p.tiles()
        self.assertTrue(all(c.is_sparse() for c in p.chunks))
        self.assertTrue(all(c.is_sparse() for c in l.chunks))
        self.assertTrue(all(c.is_sparse() for c in u.chunks))

    def testSolve(self):
        a = mt.random.randint(1, 10, (20, 20))
        b = mt.random.randint(1, 10, (20, ))
        x = mt.linalg.solve(a, b).tiles()

        self.assertEqual(x.shape, (20, ))

        a = mt.random.randint(1, 10, (20, 20), chunk_size=5)
        b = mt.random.randint(1, 10, (20, 3), chunk_size=5)
        x = mt.linalg.solve(a, b).tiles()

        self.assertEqual(x.shape, (20, 3))

        a = mt.random.randint(1, 10, (20, 20), chunk_size=12)
        b = mt.random.randint(1, 10, (20, 3))
        x = mt.linalg.solve(a, b).tiles()

        self.assertEqual(x.shape, (20, 3))
        self.assertEqual(x.nsplits, ((12, 8), (3, )))

        # test sparse
        a = sps.csr_matrix(np.random.randint(1, 10, (20, 20)))
        b = mt.random.randint(1, 10, (20, ), chunk_size=3)
        x = mt.linalg.solve(a, b).tiles()

        self.assertEqual(x.shape, (20, ))
        self.assertTrue(x.op.sparse)
        self.assertTrue(x.chunks[0].op.sparse)

        a = mt.tensor(a, chunk_size=7)
        b = mt.random.randint(1, 10, (20,))
        x = mt.linalg.solve(a, b).tiles()

        self.assertEqual(x.shape, (20,))
        self.assertEqual(x.nsplits, ((7, 7, 6),))

        x = mt.linalg.solve(a, b, sparse=False).tiles()
        self.assertFalse(x.op.sparse)
        self.assertFalse(x.chunks[0].op.sparse)

    def testInv(self):
        a = mt.random.randint(1, 10, (20, 20), chunk_size=4)
        a_inv = mt.linalg.inv(a).tiles()

        self.assertEqual(a_inv.shape, (20, 20))

        a = mt.random.randint(1, 10, (20, 20), chunk_size=11)
        a_inv = mt.linalg.inv(a).tiles()

        self.assertEqual(a_inv.shape, (20, 20))
        self.assertEqual(a_inv.nsplits, ((11, 9), (11, 9)))

        b = a.T.dot(a)
        b_inv = mt.linalg.inv(b).tiles()
        self.assertEqual(b_inv.shape, (20, 20))

        # test sparse
        data = sps.csr_matrix(np.random.randint(1, 10, (20, 20)))
        a = mt.tensor(data, chunk_size=5)
        a_inv = mt.linalg.inv(a).tiles()

        self.assertEqual(a_inv.shape, (20, 20))

        self.assertTrue(a_inv.op.sparse)
        self.assertIsInstance(a_inv, SparseTensor)
        self.assertTrue(all(c.is_sparse() for c in a_inv.chunks))

        b = a.T.dot(a)
        b_inv = mt.linalg.inv(b).tiles()
        self.assertEqual(b_inv.shape, (20, 20))

        self.assertTrue(b_inv.op.sparse)
        self.assertIsInstance(b_inv, SparseTensor)
        self.assertTrue(all(c.is_sparse() for c in b_inv.chunks))

        b_inv = mt.linalg.inv(b, sparse=False).tiles()
        self.assertFalse(b_inv.op.sparse)
        self.assertTrue(not all(c.is_sparse() for c in b_inv.chunks))

    def testTensordot(self):
        from mars.tensor.linalg import tensordot, dot, inner

        t1 = ones((3, 4, 6), chunk_size=2)
        t2 = ones((4, 3, 5), chunk_size=2)
        t3 = tensordot(t1, t2, axes=((0, 1), (1, 0)))

        self.assertEqual(t3.shape, (6, 5))

        t3.tiles()

        self.assertEqual(t3.shape, (6, 5))
        self.assertEqual(len(t3.chunks), 9)

        a = ones((10000, 20000), chunk_size=5000)
        b = ones((20000, 1000), chunk_size=5000)

        with self.assertRaises(ValueError):
            tensordot(a, b)

        a = ones(10, chunk_size=2)
        b = ones((10, 20), chunk_size=2)
        c = dot(a, b)
        self.assertEqual(c.shape, (20,))
        c.tiles()
        self.assertEqual(c.shape, tuple(sum(s) for s in c.nsplits))

        a = ones((10, 20), chunk_size=2)
        b = ones(20, chunk_size=2)
        c = dot(a, b)
        self.assertEqual(c.shape, (10,))
        c.tiles()
        self.assertEqual(c.shape, tuple(sum(s) for s in c.nsplits))

        v = ones((100, 100), chunk_size=10)
        tv = v.dot(v)
        self.assertEqual(tv.shape, (100, 100))
        tv.tiles()
        self.assertEqual(tv.shape, tuple(sum(s) for s in tv.nsplits))

        a = ones((10, 20), chunk_size=2)
        b = ones((30, 20), chunk_size=2)
        c = inner(a, b)
        self.assertEqual(c.shape, (10, 30))
        c.tiles()
        self.assertEqual(c.shape, tuple(sum(s) for s in c.nsplits))

    def testDot(self):
        t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()
        t2 = t1.T

        self.assertTrue(t1.dot(t2).issparse())
        self.assertIs(type(t1.dot(t2)), SparseTensor)
        self.assertFalse(t1.dot(t2, sparse=False).issparse())
        self.assertIs(type(t1.dot(t2, sparse=False)), Tensor)

        with self.assertRaises(TypeError):
            dot(t1, t2, out=1)

        with self.assertRaises(ValueError):
            dot(t1, t2, empty((3, 6)))

        with self.assertRaises(ValueError):
            dot(t1, t2, empty((3, 3), dtype='i4'))

        with self.assertRaises(ValueError):
            dot(t1, t2, empty((3, 3), order='F'))

        t1.dot(t2, out=empty((2, 2), dtype=t1.dtype))

    def testMatmul(self):
        t1 = tensor([[0, 1, 0], [1, 0, 0]], chunk_size=2).tosparse()
        t2 = t1.T

        t3 = matmul(t1, t2, out=empty((2, 2), dtype=t1.dtype, order='F'))
        self.assertEqual(t3.order.value, 'F')

        with self.assertRaises(TypeError):
            matmul(t1, t2, out=1)

        with self.assertRaises(TypeError):
            matmul(t1, t2, out=empty((2, 2), dtype='?'))

        with self.assertRaises(ValueError):
            matmul(t1, t2, out=empty((3, 2), dtype=t1.dtype))

        raw1 = np.asfortranarray(np.random.rand(3, 3))
        raw2 = np.asfortranarray(np.random.rand(3, 3))
        raw3 = np.random.rand(3, 3)

        self.assertEqual(matmul(tensor(raw1), tensor(raw2)).flags['C_CONTIGUOUS'],
                         np.matmul(raw1, raw2).flags['C_CONTIGUOUS'])
        self.assertEqual(matmul(tensor(raw1), tensor(raw2)).flags['F_CONTIGUOUS'],
                         np.matmul(raw1, raw2).flags['F_CONTIGUOUS'])

        self.assertEqual(matmul(tensor(raw1), tensor(raw2), order='A').flags['C_CONTIGUOUS'],
                         np.matmul(raw1, raw2, order='A').flags['C_CONTIGUOUS'])
        self.assertEqual(matmul(tensor(raw1), tensor(raw2), order='A').flags['F_CONTIGUOUS'],
                         np.matmul(raw1, raw2, order='A').flags['F_CONTIGUOUS'])

        self.assertEqual(matmul(tensor(raw1), tensor(raw3), order='A').flags['C_CONTIGUOUS'],
                         np.matmul(raw1, raw3, order='A').flags['C_CONTIGUOUS'])
        self.assertEqual(matmul(tensor(raw1), tensor(raw3), order='A').flags['F_CONTIGUOUS'],
                         np.matmul(raw1, raw3, order='A').flags['F_CONTIGUOUS'])
