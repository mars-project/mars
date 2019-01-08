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

        self.assertEqual(set(dag[2]), set([5, 6]))
        self.assertEqual(list(dag.topological_iter()), [3, 2, 5, 6, 1, 4])

        self.assertEqual(list(dag.dfs()), [3, 2, 5, 6, 1, 4])
        self.assertEqual(list(dag.bfs()), [1, 2, 3, 4, 5, 6])

        dag.add_edge(6, 1)
        dag.add_edge(1, 2)

        self.assertRaises(GraphContainsCycleError, lambda: list(dag.topological_iter()))
