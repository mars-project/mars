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

import scipy.sparse as sps

import mars.tensor as mt
from mars.tensor.expressions.fuse.core import TensorFuseChunk
from mars.tensor.expressions.datasource import CSRMatrixDataSource, SparseToDense
from mars.tensor.expressions.random import TensorRand
from mars.tensor.expressions.arithmetic import TensorAdd, TensorMultiply, TensorTrueDiv,\
    TensorDivide, TensorSubtract
from mars.tensor.expressions.merge import TensorConcatenate
from mars.tensor.expressions.reduction import TensorSum


class Test(unittest.TestCase):
    def testBuildFusedGraph(self):
        t = mt.random.rand(9, 9, chunk_size=3)
        t1 = (t + 1) * 2

        g = t1.build_graph(tiled=True, compose=True)
        graph_nodes = list(g)
        self.assertTrue(all(isinstance(n.op, TensorFuseChunk) for n in graph_nodes))
        self.assertTrue(all(n.shape == (3, 3) for n in graph_nodes))

        fuse_node = graph_nodes[0]
        self.assertEqual(fuse_node.shape, (3, 3))
        self.assertEqual(len(fuse_node.composed), 3)
        self.assertIsInstance(fuse_node.composed[0].op, TensorRand)
        self.assertIsInstance(fuse_node.composed[1].op, TensorAdd)
        self.assertIsInstance(fuse_node.composed[2].op, TensorMultiply)

        t2 = mt.sum((t / 2) - 1, axis=0)

        g = t2.build_graph(tiled=True, compose=True)
        graph_nodes = list(g.bfs())
        self.assertTrue(all(isinstance(n.op, TensorFuseChunk) for n in graph_nodes))

        reduction_node = graph_nodes[-1]
        self.assertEqual(reduction_node.shape, (3,))
        self.assertEqual(len(reduction_node.composed), 2)
        self.assertEqual(reduction_node.inputs, reduction_node.composed[0].inputs)
        self.assertIsInstance(reduction_node.composed[0].op, TensorConcatenate)
        self.assertIsInstance(reduction_node.composed[1].op, TensorSum)

        agg_node = graph_nodes[0]
        self.assertEqual(agg_node.shape, (1, 3))
        self.assertEqual(len(agg_node.composed), 4)
        self.assertIsInstance(agg_node.composed[0].op, TensorRand)
        self.assertIsInstance(agg_node.composed[1].op, (TensorTrueDiv, TensorDivide))
        self.assertIsInstance(agg_node.composed[2].op, TensorSubtract)
        self.assertIsInstance(agg_node.composed[3].op, TensorSum)

    def testSparse(self):
        data = sps.rand(9, 9, density=0.1)
        t = mt.tensor(data, chunk_size=3)

        t1 = t * 2 / 3
        g = t1.build_graph(tiled=True, compose=True)
        graph_nodes = list(g)
        self.assertTrue(all(isinstance(n.op, TensorFuseChunk) for n in graph_nodes))
        self.assertTrue(all(n.op.sparse for n in graph_nodes))
        self.assertTrue(all(n.shape == (3, 3) for n in graph_nodes))

        fuse_node = graph_nodes[0]
        self.assertEqual(fuse_node.shape, (3, 3))
        self.assertEqual(len(fuse_node.composed), 3)
        self.assertIsInstance(fuse_node.composed[0].op, CSRMatrixDataSource)
        self.assertIsInstance(fuse_node.composed[1].op, TensorMultiply)
        self.assertIsInstance(fuse_node.composed[2].op, (TensorTrueDiv, TensorDivide))
        self.assertTrue(all(c.op.sparse for c in fuse_node.composed))

        t2 = (t * 2).todense()
        g = t2.build_graph(tiled=True, compose=True)
        graph_nodes = list(g)
        self.assertTrue(all([isinstance(n.op, TensorFuseChunk) for n in graph_nodes]))
        self.assertTrue(all([not n.op.sparse for n in graph_nodes]))
        self.assertTrue(all(n.shape == (3, 3) for n in graph_nodes))

        fuse_node = graph_nodes[0]
        self.assertEqual(fuse_node.shape, (3, 3))
        self.assertEqual(len(fuse_node.composed), 3)
        self.assertIsInstance(fuse_node.composed[0].op, CSRMatrixDataSource)
        self.assertIsInstance(fuse_node.composed[1].op, TensorMultiply)
        self.assertTrue(fuse_node.composed[1].op.sparse)
        self.assertIsInstance(fuse_node.composed[2].op, SparseToDense)
        self.assertFalse(fuse_node.composed[2].op.sparse)
