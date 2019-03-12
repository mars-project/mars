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
from mars.tensor.expressions.datasource import CSRMatrixDataSource
from mars import operands


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
        self.assertIsInstance(fuse_node.composed[0].op, operands.Rand)
        self.assertIsInstance(fuse_node.composed[1].op, operands.AddConstant)
        self.assertIsInstance(fuse_node.composed[2].op, operands.MulConstant)

        t2 = mt.sum((t / 2) - 1, axis=0)

        g = t2.build_graph(tiled=True, compose=True)
        graph_nodes = list(g)
        self.assertTrue(all(isinstance(n.op, TensorFuseChunk) for n in graph_nodes))

        fuse_node = graph_nodes[0]
        self.assertEqual(fuse_node.shape, (1, 3))
        self.assertEqual(len(fuse_node.composed), 4)
        self.assertIsInstance(fuse_node.composed[0].op, operands.Rand)
        self.assertIsInstance(fuse_node.composed[1].op, operands.TDivConstant)
        self.assertIsInstance(fuse_node.composed[2].op, operands.SubConstant)
        self.assertIsInstance(fuse_node.composed[3].op, operands.Sum)

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
        self.assertIsInstance(fuse_node.composed[1].op, operands.MulConstant)
        self.assertIsInstance(fuse_node.composed[2].op, operands.TDivConstant)
        self.assertTrue(all(c.op.sparse for c in fuse_node.composed))

        # add constant will convert sparse matrix to dense matrix
        t2 = t * 2 + 3
        g = t2.build_graph(tiled=True, compose=True)
        graph_nodes = list(g)
        self.assertTrue(all([isinstance(n.op, TensorFuseChunk) for n in graph_nodes]))
        self.assertTrue(all([not n.op.sparse for n in graph_nodes]))
        self.assertTrue(all(n.shape == (3, 3) for n in graph_nodes))

        fuse_node = graph_nodes[0]
        self.assertEqual(fuse_node.shape, (3, 3))
        self.assertEqual(len(fuse_node.composed), 3)
        self.assertIsInstance(fuse_node.composed[0].op, CSRMatrixDataSource)
        self.assertIsInstance(fuse_node.composed[1].op, operands.MulConstant)
        self.assertTrue(fuse_node.composed[1].op.sparse)
        self.assertIsInstance(fuse_node.composed[2].op, operands.AddConstant)
        self.assertFalse(fuse_node.composed[2].op.sparse)
