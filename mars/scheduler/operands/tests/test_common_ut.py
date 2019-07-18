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

import contextlib
import unittest
import uuid
from collections import defaultdict

from mars import promise, tensor as mt
from mars.actors import create_actor_pool
from mars.graph import DAG
from mars.scheduler import OperandState, ResourceActor, ChunkMetaActor,\
    ChunkMetaClient, AssignerActor, GraphActor, OperandActor
from mars.scheduler.utils import SchedulerClusterInfoActor
from mars.tests.core import patch_method
from mars.utils import get_next_port, serialize_graph


class FakeExecutionActor(promise.PromiseActor):
    def __init__(self, exec_delay):
        super(FakeExecutionActor, self).__init__()
        self._exec_delay = exec_delay
        self._finished_keys = set()
        self._enqueue_callbacks = dict()
        self._finish_callbacks = defaultdict(list)
        self._undone_preds = dict()
        self._succs = dict()

    def mock_send_all_callbacks(self, graph_key):
        for cb in self._finish_callbacks[graph_key]:
            self.tell_promise(cb, {})
        self._finished_keys.add(graph_key)
        self._finish_callbacks[graph_key] = []
        try:
            for succ_key in self._succs[graph_key]:
                self._undone_preds[succ_key].difference_update([graph_key])
                if not self._undone_preds[succ_key]:
                    self.tell_promise(self._enqueue_callbacks[succ_key])
        except KeyError:
            pass

    def execute_graph(self, session_id, graph_key, graph_ser, io_meta, data_sizes,
                      send_addresses=None, callback=None):
        if callback:
            self._finish_callbacks[graph_key].append(callback)
        self.ref().mock_send_all_callbacks(graph_key, _tell=True, _delay=self._exec_delay)

    def add_finish_callback(self, session_id, graph_key, callback):
        if graph_key in self._finished_keys:
            self.tell_promise(callback)
        else:
            self._finish_callbacks[graph_key].append(callback)


@patch_method(ResourceActor._broadcast_sessions)
@patch_method(ResourceActor._broadcast_workers)
class Test(unittest.TestCase):
    @contextlib.contextmanager
    def _prepare_test_graph(self, session_id, graph_key, mock_workers):
        addr = '127.0.0.1:%d' % get_next_port()
        a1 = mt.random.random((100,))
        a2 = mt.random.random((100,))
        s = a1 + a2
        v1, v2 = mt.split(s, 2)

        graph = DAG()
        v1.build_graph(graph=graph, compose=False)
        v2.build_graph(graph=graph, compose=False)

        with create_actor_pool(n_process=1, backend='gevent', address=addr) as pool:
            pool.create_actor(SchedulerClusterInfoActor, [pool.cluster_info.address],
                              uid=SchedulerClusterInfoActor.default_uid())
            resource_ref = pool.create_actor(ResourceActor, uid=ResourceActor.default_uid())
            pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())
            pool.create_actor(AssignerActor, uid=AssignerActor.gen_uid(session_id))
            graph_ref = pool.create_actor(GraphActor, session_id, graph_key, serialize_graph(graph),
                                          uid=GraphActor.gen_uid(session_id, graph_key))

            for w in mock_workers:
                resource_ref.set_worker_meta(w, dict(hardware=dict(cpu=4, cpu_total=4, memory=1600)))

            graph_ref.prepare_graph()
            graph_ref.analyze_graph()
            graph_ref.create_operand_actors(_start=False)

            yield pool, graph_ref

    @staticmethod
    def _filter_graph_level_op_keys(graph_ref):
        from mars.tensor.expressions.random import TensorRandomSample
        from mars.tensor.expressions.indexing.getitem import TensorIndex
        from mars.tensor.arithmetic import TensorAdd

        graph = graph_ref.get_chunk_graph()

        return (
            [c.op.key for c in graph if isinstance(c.op, TensorRandomSample)],
            [c.op.key for c in graph if isinstance(c.op, TensorAdd)][0],
            [c.op.key for c in graph if isinstance(c.op, TensorIndex)],
        )

    @staticmethod
    def _filter_graph_level_chunk_keys(graph_ref):
        from mars.tensor.expressions.random import TensorRandomSample
        from mars.tensor.expressions.indexing.getitem import TensorIndex
        from mars.tensor.arithmetic import TensorAdd

        graph = graph_ref.get_chunk_graph()

        return (
            [c.key for c in graph if isinstance(c.op, TensorRandomSample)],
            [c.key for c in graph if isinstance(c.op, TensorAdd)][0],
            [c.key for c in graph if isinstance(c.op, TensorIndex)],
        )

    @patch_method(ResourceActor.allocate_resource, new=lambda *_, **__: True)
    @patch_method(ResourceActor.detach_dead_workers)
    @patch_method(ResourceActor.detect_dead_workers)
    def testReadyState(self, *_):
        session_id = str(uuid.uuid4())
        graph_key = str(uuid.uuid4())
        mock_workers = ['localhost:12345', 'localhost:23456']

        def _mock_get_workers_meta(*_, **__):
            return dict((w, dict(hardware=dict(cpu_total=1, memory=1024 ** 3))) for w in mock_workers)

        with patch_method(ResourceActor.get_workers_meta, new=_mock_get_workers_meta) as _, \
                self._prepare_test_graph(session_id, graph_key, mock_workers) as (pool, graph_ref):
            input_op_keys, mid_op_key, output_op_keys = self._filter_graph_level_op_keys(graph_ref)
            meta_client = ChunkMetaClient(pool, pool.actor_ref(SchedulerClusterInfoActor.default_uid()))
            op_ref = pool.actor_ref(OperandActor.gen_uid(session_id, mid_op_key))
            resource_ref = pool.actor_ref(ResourceActor.default_uid())

            input_refs = [pool.actor_ref(OperandActor.gen_uid(session_id, k)) for k in input_op_keys]

            def test_entering_state(target):
                for key in input_op_keys:
                    op_ref.remove_finished_predecessor(key)

                op_ref.start_operand(OperandState.UNSCHEDULED)
                for ref in input_refs:
                    ref.start_operand(OperandState.UNSCHEDULED)

                for ref in input_refs:
                    self.assertEqual(op_ref.get_state(), OperandState.UNSCHEDULED)
                    ref.start_operand(OperandState.FINISHED)
                pool.sleep(1)
                self.assertEqual(target, op_ref.get_state())
                for w in mock_workers:
                    resource_ref.deallocate_resource(session_id, mid_op_key, w)

            # test entering state with no input meta
            test_entering_state(OperandState.UNSCHEDULED)

            # fill meta
            input_chunk_keys, _, _ = self._filter_graph_level_chunk_keys(graph_ref)
            for ck in input_chunk_keys:
                meta_client.set_chunk_meta(session_id, ck, workers=('localhost:12345',), size=800)

            # test successful entering state
            test_entering_state(OperandState.READY)
