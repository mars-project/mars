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

from mars.tensor.execution.core import Executor
from mars import tensor as mt
from mars.session import LocalSession, Session


class Test(unittest.TestCase):
    def setUp(self):
        self.executor = Executor('numpy')
        local_session = LocalSession()
        local_session._executor = self.executor
        self.session = Session()
        self.session._sess = local_session

    def testDecref(self):
        a = mt.random.rand(10, 20, chunk_size=5)
        b = a + 1

        b.execute(session=self.session)

        self.assertEqual(len(self.executor.chunk_result), 8)

        del b
        # decref called
        self.assertEqual(len(self.executor.chunk_result), 0)

    def testMockExecuteSize(self):
        import mars.tensor as mt
        from mars.graph import DAG
        from mars.tensor.expressions.datasource import TensorFetchChunk
        from mars.tensor.expressions.arithmetic import TensorTreeAdd

        graph_add = DAG()
        input_chunks = []
        for _ in range(2):
            fetch_op = TensorFetchChunk(dtype=np.dtype('int64'))
            inp_chunk = fetch_op.new_chunk(None, shape=(100, 100)).data
            input_chunks.append(inp_chunk)

        add_op = TensorTreeAdd(dtype=np.dtype('int64'))
        add_chunk = add_op.new_chunk(input_chunks, shape=(100, 100), dtype=np.dtype('int64')).data
        graph_add.add_node(add_chunk)
        for inp_chunk in input_chunks:
            graph_add.add_node(inp_chunk)
            graph_add.add_edge(inp_chunk, add_chunk)

        executor = Executor()

        res = executor.execute_graph(graph_add, [add_chunk.key], compose=False, mock=True)[0]
        self.assertEqual(res, (80000, 80000))
        self.assertEqual(executor.mock_max_memory, 80000)

        for _ in range(3):
            new_add_op = TensorTreeAdd(dtype=np.dtype('int64'))
            new_add_chunk = new_add_op.new_chunk([add_chunk], shape=(100, 100), dtype=np.dtype('int64')).data
            graph_add.add_node(new_add_chunk)
            graph_add.add_edge(add_chunk, new_add_chunk)

            add_chunk = new_add_chunk

        executor = Executor()

        res = executor.execute_graph(graph_add, [add_chunk.key], compose=False, mock=True)[0]
        self.assertEqual(res, (80000, 80000))
        self.assertEqual(executor.mock_max_memory, 160000)

        a = mt.random.rand(10, 10, chunk_size=10)
        b = a[:, mt.newaxis, :] - a
        r = mt.triu(mt.sqrt(b ** 2).sum(axis=2))

        executor = Executor()

        res = executor.execute_tensor(r, concat=False, mock=True)
        # larger than maximal memory size in calc procedure
        self.assertGreaterEqual(res[0][0], 800)
        self.assertGreaterEqual(executor.mock_max_memory, 8000)

    def testArrayFunction(self):
        a = mt.ones((10, 20), chunk_size=5)

        # test sum
        self.assertEqual(np.sum(a).execute(), 200)

        # test qr
        q, r = np.linalg.qr(a)
        self.assertTrue(np.allclose(np.dot(q, r), a).execute())
