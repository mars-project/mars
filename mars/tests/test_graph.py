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

from mars.graph import DAG, GraphContainsCycleError


class Test(unittest.TestCase):
    def testDAG(self):
        r"""
        1 --- 4
        2 --- 6
          \  /
           5
         /
        3
        """

        dag = DAG()
        [dag.add_node(i) for i in range(1, 7)]
        dag.add_edge(1, 4)
        dag.add_edge(2, 6)
        dag.add_edge(2, 5)
        dag.add_edge(5, 6)
        dag.add_edge(3, 5)

        with self.assertRaises(KeyError):
            dag.add_edge(1, 10)
        with self.assertRaises(KeyError):
            dag.add_edge(10, 1)

        self.assertEqual(set(dag[2]), set([5, 6]))
        self.assertEqual(list(dag.topological_iter()), [3, 2, 5, 6, 1, 4])

        self.assertEqual(list(dag.dfs()), [3, 2, 5, 6, 1, 4])
        self.assertEqual(list(dag.bfs()), [1, 2, 3, 4, 5, 6])

        dag.add_edge(6, 1)
        dag.add_edge(1, 2)

        with self.assertRaises(KeyError):
            for _ in dag.iter_predecessors(-1):
                pass

        with self.assertRaises(KeyError):
            for _ in dag.iter_successors(-1):
                pass

        self.assertRaises(GraphContainsCycleError, lambda: list(dag.topological_iter()))

        dag.remove_edge(2, 5)
        self.assertFalse(dag.has_successor(2, 5))
        with self.assertRaises(KeyError):
            dag.remove_edge(2, 5)

        rev_dag = dag.build_reversed()
        for n in dag:
            self.assertIn(n, rev_dag)
            self.assertTrue(all(rev_dag.has_successor(n, pred)
                                for pred in dag.predecessors(n)))

        undigraph = dag.build_undirected()
        for n in dag:
            self.assertIn(n, undigraph)
            self.assertTrue(all(undigraph.has_predecessor(pred, n)
                                for pred in dag.predecessors(n)))
            self.assertTrue(all(undigraph.has_successor(n, pred)
                                for pred in dag.predecessors(n)))

    def testToDot(self):
        import mars.tensor as mt

        arr = mt.random.randint(10, size=(10, 8), chunk_size=4)
        arr_add = mt.random.randint(10, size=(10, 8), chunk_size=4)
        arr2 = arr + arr_add
        graph = arr2.build_graph(compose=False, tiled=True)

        dot = str(graph.to_dot(trunc_key=5))
        self.assertTrue(str(n.op.key)[5] in dot for n in graph)
