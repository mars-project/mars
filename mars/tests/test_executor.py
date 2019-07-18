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

import sys
import unittest

import numpy as np

import mars.tensor as mt
from mars.executor import Executor, register
from mars.serialize import Int64Field
from mars.tensor.operands import TensorOperand, TensorOperandMixin
from mars.graph import DirectedGraph
from mars.actors import create_actor_pool, Distributor, Actor


class FakeDistributor(Distributor):
    def distribute(self, uid):
        return int(uid.split(':')[0])


class RunActor(Actor):
    def on_receive(self, message):
        return message + 1


class ExecutorActor(Actor):
    def __init__(self):
        self._executor = Executor(sync_provider_type=Executor.SyncProviderType.GEVENT)

    def post_create(self):
        register(FakeOperand, fake_execution_maker(self.ctx))

    def on_receive(self, message):
        op = FakeOperand(_num=message)
        chunk = op.new_chunk(None, ())
        graph = DirectedGraph()
        graph.add_node(chunk.data)
        res = self._executor.execute_graph(graph, [chunk.key])
        assert res[0] == message + 1


class FakeOperand(TensorOperand, TensorOperandMixin):
    _num = Int64Field('num')

    @property
    def num(self):
        return self._num


def fake_execution_maker(actor_ctx):
    def run(ctx, chunk):
        actor = actor_ctx.create_actor(RunActor, uid='1-run')
        ctx[chunk.key] = actor.send(chunk.op.num)

    return run


class Test(unittest.TestCase):
    def testExecutorWithGeventProvider(self):
        executor = Executor(sync_provider_type=Executor.SyncProviderType.GEVENT)

        a = mt.ones((10, 10), chunk_size=2)
        res = executor.execute_tensor(a, concat=True)[0]
        np.testing.assert_array_equal(res, np.ones((10, 10)))

    @unittest.skipIf(sys.platform == 'win32', 'does not run in windows')
    def testActorInExecutor(self):
        with create_actor_pool(n_process=2) as pool:
            actor = pool.create_actor(ExecutorActor, uid='0-executor')
            self.assertIsNone(actor.send(1))

    def testMockExecuteSize(self):
        import mars.tensor as mt
        from mars.graph import DAG
        from mars.tensor.fetch import TensorFetch
        from mars.tensor.arithmetic import TensorTreeAdd

        graph_add = DAG()
        input_chunks = []
        for _ in range(2):
            fetch_op = TensorFetch(dtype=np.dtype('int64'))
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
