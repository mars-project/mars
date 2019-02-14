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

import mars.tensor as mt
from mars.executor import Executor, register
from mars.operands import Operand
from mars.serialize import Int64Field
from mars.tensor.expressions.core import TensorOperandMixin
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


class FakeOperand(Operand, TensorOperandMixin):
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

        a = mt.ones((2, 2), chunk_size=2)
        res = executor.execute_tensor(a)[0]
        np.testing.assert_array_equal(res, np.ones((2, 2)))

    def testActorInExecutor(self):
        with create_actor_pool(n_process=2) as pool:
            actor = pool.create_actor(ExecutorActor, uid='0-executor')
            self.assertIsNone(actor.send(1))
