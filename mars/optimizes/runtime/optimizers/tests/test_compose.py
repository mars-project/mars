#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

from mars.executor import Executor
from mars.tensor.arithmetic import TensorTreeAdd
from mars.tensor.indexing import TensorSlice
from mars.graph import DirectedGraph
from mars.optimizes.runtime.optimizers.ne import NeOptimizer


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')

    def testCompose(self):
        """
        test compose in build graph and optimize
        """

        r"""
        graph(@: node, #: composed_node):

        @ --> @ --> @   ========>    #
        """
        chunks = [TensorTreeAdd(_key=str(n)).new_chunk(None, None) for n in range(3)]
        graph = DirectedGraph()
        list(map(graph.add_node, chunks[:3]))
        graph.add_edge(chunks[0], chunks[1])
        graph.add_edge(chunks[1], chunks[2])

        composed_nodes = graph.compose()
        self.assertTrue(composed_nodes[0].composed == chunks[:3])

        # make the middle one as result chunk, thus the graph cannot be composed
        composed_nodes = graph.compose(keys=[chunks[1].key])
        self.assertEqual(len(composed_nodes), 0)

        r"""
        graph(@: node, #: composed_node):

        @             @              @       @
          \         /                  \   /
            @ --> @       ========>      #
          /         \                  /   \
        @             @              @       @
        """
        chunks = [TensorTreeAdd(_key=str(n)).new_chunk(None, None) for n in range(6)]
        graph = DirectedGraph()
        list(map(graph.add_node, chunks[:6]))

        chunks[2].op._inputs = [chunks[0], chunks[1]]
        chunks[3].op._inputs = [chunks[2]]
        chunks[4].op._inputs = [chunks[3]]
        chunks[5].op._inputs = [chunks[3]]

        graph.add_edge(chunks[0], chunks[2])
        graph.add_edge(chunks[1], chunks[2])
        graph.add_edge(chunks[2], chunks[3])
        graph.add_edge(chunks[3], chunks[4])
        graph.add_edge(chunks[3], chunks[5])

        composed_nodes = graph.compose()
        self.assertTrue(composed_nodes[0].composed == chunks[2:4])

        # to make sure the predecessors and successors of compose are right
        # 0 and 1's successors must be composed
        self.assertIn(composed_nodes[0], graph.successors(chunks[0]))
        self.assertIn(composed_nodes[0], graph.successors(chunks[1]))
        # check composed's inputs
        self.assertIn(chunks[0].key, [n.key for n in composed_nodes[0].inputs])
        self.assertIn(chunks[1].key, [n.key for n in composed_nodes[0].inputs])
        # check composed's predecessors
        self.assertIn(chunks[0], graph.predecessors(composed_nodes[0]))
        self.assertIn(chunks[1], graph.predecessors(composed_nodes[0]))
        # check 4 and 5's inputs
        self.assertIn(composed_nodes[0].key, [n.key for n in graph.successors(composed_nodes[0])[0].inputs])
        self.assertIn(composed_nodes[0].key, [n.key for n in graph.successors(composed_nodes[0])[0].inputs])
        # check 4 and 5's predecessors
        self.assertIn(composed_nodes[0], graph.predecessors(chunks[4]))
        self.assertIn(composed_nodes[0], graph.predecessors(chunks[5]))

        # test optimizer compose

        r"""
        graph(@: node, S: Slice Chunk, #: composed_node):

        @                   @              @             @
          \               /                  \         /
            @ --> @ --> S      ========>       # --> S
          /               \                  /         \
        @                   @              @             @

        compose stopped at S, because numexpr don't support Slice op
        """
        chunks = [TensorTreeAdd(_key=str(n)).new_chunk(None, None) for n in range(6)]
        chunk_slice = TensorSlice().new_chunk([None], None)
        graph = DirectedGraph()
        list(map(graph.add_node, chunks[:6]))
        graph.add_node(chunk_slice)
        graph.add_edge(chunks[0], chunks[2])
        graph.add_edge(chunks[1], chunks[2])
        graph.add_edge(chunks[2], chunks[3])
        graph.add_edge(chunks[3], chunk_slice)
        graph.add_edge(chunk_slice, chunks[4])
        graph.add_edge(chunk_slice, chunks[5])
        optimizer = NeOptimizer(graph)
        composed_nodes = optimizer.compose()
        self.assertTrue(composed_nodes[0].composed == chunks[2:4])

        r"""
            graph(@: node, S: Slice Chunk, #: composed_node):

            @ --> @ --> S --> @  ========>  # --> S --> @

        compose stopped at S, because numexpr don't support Slice op
        """
        chunks = [TensorTreeAdd(_key=str(n)).new_chunk(None, None) for n in range(4)]
        graph = DirectedGraph()
        list(map(graph.add_node, chunks[:3]))
        graph.add_node(chunk_slice)
        graph.add_edge(chunks[0], chunks[1])
        graph.add_edge(chunks[1], chunk_slice)
        graph.add_edge(chunk_slice, chunks[2])
        optimizer = NeOptimizer(graph)
        composed_nodes = optimizer.compose()
        self.assertTrue(composed_nodes[0].composed == chunks[:2])
        self.assertTrue(len(composed_nodes) == 1)

        r"""
            graph(@: node, S: Slice Chunk, #: composed_node):

            @ --> @ --> S --> @ --> @   ========>  # --> S --> #

        compose stopped at S, because numexpr don't support Slice op
        """
        chunks = [TensorTreeAdd(_key=str(n)).new_chunk(None, None) for n in range(4)]
        graph = DirectedGraph()
        list(map(graph.add_node, chunks[:4]))
        graph.add_node(chunk_slice)
        graph.add_edge(chunks[0], chunks[1])
        graph.add_edge(chunks[1], chunk_slice)
        graph.add_edge(chunk_slice, chunks[2])
        graph.add_edge(chunks[2], chunks[3])
        optimizer = NeOptimizer(graph)
        composed_nodes = optimizer.compose()
        self.assertTrue(composed_nodes[0].composed == chunks[:2])
        self.assertTrue(composed_nodes[1].composed == chunks[2:4])
