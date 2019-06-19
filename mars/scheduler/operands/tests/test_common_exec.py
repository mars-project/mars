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

import logging
import time
import unittest
import uuid
from collections import defaultdict

from mars import tensor as mt
from mars.actors import ActorAlreadyExist
from mars.config import options
from mars.errors import ExecutionInterrupted
from mars.scheduler import OperandActor, ResourceActor, GraphActor, AssignerActor, \
    ChunkMetaActor, GraphMetaActor
from mars.scheduler.utils import GraphState, SchedulerClusterInfoActor, SchedulerActor
from mars.worker.execution import GraphExecutionRecord
from mars.utils import serialize_graph, log_unhandled, build_exc_info
from mars.actors import create_actor_pool
from mars.tests.core import patch_method

logger = logging.getLogger(__name__)


class FakeExecutionActor(SchedulerActor):
    _retries = defaultdict(lambda: 0)

    def __init__(self, exec_delay=0.1, fail_count=0):
        super(FakeExecutionActor, self).__init__()

        self._fail_count = fail_count
        self._exec_delay = exec_delay

        self._results = dict()
        self._cancels = set()
        self._graph_records = dict()  # type: dict[tuple, GraphExecutionRecord]

    def post_create(self):
        super(FakeExecutionActor, self).post_create()
        self.set_cluster_info_ref()

    @classmethod
    def gen_uid(cls, addr):
        return 's:h1:%s$%s' % (cls.__name__, addr)

    @log_unhandled
    def actual_exec(self, session_id, graph_key):
        if graph_key in self._results:
            del self._results[graph_key]

        rec = self._graph_records[(session_id, graph_key)]
        if graph_key in self._cancels:
            exc = build_exc_info(ExecutionInterrupted)
            self._results[graph_key] = (exc, dict(_accept=False))
            for cb in rec.finish_callbacks:
                self.tell_promise(cb, *exc, **dict(_accept=False))
            rec.finish_callbacks = []
            return
        elif self._fail_count and self._retries[graph_key] < self._fail_count:
            exc = build_exc_info(ValueError)
            logger.debug('Key %r: %r', graph_key, self._retries.get(graph_key))
            self._retries[graph_key] += 1

            del self._graph_records[(session_id, graph_key)]
            self._results[graph_key] = (exc, dict(_accept=False))
            for cb in rec.finish_callbacks:
                self.tell_promise(cb, *exc, **dict(_accept=False))
            rec.finish_callbacks = []
            return

        chunk_graph = rec.graph
        key_to_chunks = defaultdict(list)
        for n in chunk_graph:
            key_to_chunks[n.key].append(n)
            self.chunk_meta.set_chunk_size(session_id, n.key, 0)

        for tk in rec.data_targets:
            for n in key_to_chunks[tk]:
                self.chunk_meta.add_worker(session_id, n.key, 'localhost:12345')
        self._results[graph_key] = ((dict(),), dict())
        for cb in rec.finish_callbacks:
            self.tell_promise(cb, {})
        rec.finish_callbacks = []

        for succ_key in (rec.succ_keys or ()):
            try:
                succ_rec = self._graph_records[(session_id, succ_key)]
            except KeyError:
                continue
            succ_rec.undone_pred_keys.difference_update([graph_key])
            if not succ_rec.undone_pred_keys:
                self.tell_promise(succ_rec.enqueue_callback)

    @log_unhandled
    def start_execution(self, session_id, graph_key, send_addresses=None, callback=None):
        rec = self._graph_records[(session_id, graph_key)]
        if callback:
            rec.finish_callbacks.append(callback)
        self.ref().actual_exec(session_id, graph_key, _tell=True, _delay=self._exec_delay)

    @log_unhandled
    def enqueue_graph(self, session_id, graph_key, graph_ser, io_meta, data_sizes,
                      priority_data=None, send_addresses=None, succ_keys=None,
                      pred_keys=None, callback=None):
        query_key = (session_id, graph_key)
        assert query_key not in self._graph_records

        pred_keys = pred_keys or ()
        actual_unfinished = []
        for k in pred_keys:
            if k in self._results and self._results[k][1].get('_accept', True):
                continue
            actual_unfinished.append(k)

        self._graph_records[query_key] = GraphExecutionRecord(
            graph_ser, None,
            data_targets=io_meta['chunks'],
            shared_input_chunks=set(io_meta.get('shared_input_chunks', [])),
            send_addresses=send_addresses,
            enqueue_callback=callback,
            succ_keys=succ_keys,
            undone_pred_keys=actual_unfinished,
        )
        if not actual_unfinished:
            self.tell_promise(callback)

    @log_unhandled
    def dequeue_graph(self, session_id, graph_key):
        try:
            del self._graph_records[(session_id, graph_key)]
        except KeyError:
            pass

    def update_priority(self, session_id, graph_key, priority_data):
        pass

    @log_unhandled
    def add_finish_callback(self, session_id, graph_key, callback):
        query_key = (session_id, graph_key)
        rec = self._graph_records[query_key]
        rec.finish_callbacks.append(callback)
        if query_key in self._results:
            args, kwargs = self._results[graph_key]
            for cb in rec.finish_callbacks:
                self.tell_promise(cb, *args, **kwargs)
            rec.finish_callbacks = []

    @log_unhandled
    def stop_execution(self, _, graph_key):
        self._cancels.add(graph_key)


@patch_method(ResourceActor._broadcast_sessions)
@patch_method(ResourceActor._broadcast_workers)
class Test(unittest.TestCase):
    @staticmethod
    def _run_operand_case(session_id, graph_key, tensor, execution_creator):
        graph = tensor.build_graph(compose=False)

        with create_actor_pool(n_process=1, backend='gevent') as pool:
            pool.create_actor(SchedulerClusterInfoActor, [pool.cluster_info.address],
                              uid=SchedulerClusterInfoActor.default_uid())
            resource_ref = pool.create_actor(ResourceActor, uid=ResourceActor.default_uid())
            pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())
            pool.create_actor(AssignerActor, uid=AssignerActor.default_uid())
            graph_ref = pool.create_actor(GraphActor, session_id, graph_key, serialize_graph(graph),
                                          uid=GraphActor.gen_uid(session_id, graph_key))

            def _build_mock_ref(uid=None, address=None):
                try:
                    return execution_creator(pool, FakeExecutionActor.gen_uid(address))
                except ActorAlreadyExist:
                    return pool.actor_ref(FakeExecutionActor.gen_uid(address))

            # handle mock objects
            OperandActor._get_raw_execution_ref.side_effect = _build_mock_ref

            mock_resource = dict(hardware=dict(cpu=4, cpu_total=4, memory=512))

            resource_ref.set_worker_meta('localhost:12345', mock_resource)
            resource_ref.set_worker_meta('localhost:23456', mock_resource)

            graph_ref.prepare_graph()
            fetched_graph = graph_ref.get_chunk_graph()

            graph_ref.analyze_graph()

            final_keys = set()
            for c in fetched_graph:
                if fetched_graph.count_successors(c) == 0:
                    final_keys.add(c.op.key)

            graph_ref.create_operand_actors()

            graph_meta_ref = pool.actor_ref(GraphMetaActor.gen_uid(session_id, graph_key))
            start_time = time.time()
            while True:
                pool.sleep(0.1)
                if time.time() - start_time > 30:
                    raise SystemError('Wait for execution finish timeout')
                if graph_meta_ref.get_state() in (GraphState.SUCCEEDED, GraphState.FAILED, GraphState.CANCELLED):
                    break

    @patch_method(OperandActor._get_raw_execution_ref)
    @patch_method(OperandActor._free_data_in_worker)
    def testOperandActor(self, *_):
        arr = mt.random.randint(10, size=(10, 8), chunk_size=4)
        arr_add = mt.random.randint(10, size=(10, 8), chunk_size=4)
        arr2 = arr + arr_add

        session_id = str(uuid.uuid4())
        graph_key = str(uuid.uuid4())
        self._run_operand_case(session_id, graph_key, arr2,
                               lambda pool, uid: pool.create_actor(FakeExecutionActor, uid=uid))

    @patch_method(OperandActor._get_raw_execution_ref)
    @patch_method(OperandActor._free_data_in_worker)
    def testOperandActorWithSameKey(self, *_):
        arr = mt.ones((5, 5), chunk_size=3)
        arr2 = mt.concatenate((arr, arr))

        session_id = str(uuid.uuid4())
        graph_key = str(uuid.uuid4())
        self._run_operand_case(session_id, graph_key, arr2,
                               lambda pool, uid: pool.create_actor(FakeExecutionActor, uid=uid))

    @patch_method(OperandActor._get_raw_execution_ref)
    @patch_method(OperandActor._free_data_in_worker)
    def testOperandActorWithRetry(self, *_):
        arr = mt.random.randint(10, size=(10, 8), chunk_size=4)
        arr_add = mt.random.randint(10, size=(10, 8), chunk_size=4)
        arr2 = arr + arr_add

        session_id = str(uuid.uuid4())
        graph_key = str(uuid.uuid4())
        try:
            options.scheduler.retry_delay = 0
            self._run_operand_case(session_id, graph_key, arr2,
                                   lambda pool, uid: pool.create_actor(FakeExecutionActor, fail_count=2, uid=uid))
        finally:
            options.scheduler.retry_delay = 60

    @patch_method(OperandActor._get_raw_execution_ref)
    @patch_method(OperandActor._free_data_in_worker)
    def testOperandActorWithRetryAndFail(self, *_):
        arr = mt.random.randint(10, size=(10, 8), chunk_size=4)
        arr_add = mt.random.randint(10, size=(10, 8), chunk_size=4)
        arr2 = arr + arr_add

        session_id = str(uuid.uuid4())
        graph_key = str(uuid.uuid4())
        try:
            options.scheduler.retry_delay = 0
            self._run_operand_case(session_id, graph_key, arr2,
                                   lambda pool, uid: pool.create_actor(FakeExecutionActor, fail_count=5, uid=uid))
        finally:
            options.scheduler.retry_delay = 60

    @patch_method(OperandActor._get_raw_execution_ref)
    @patch_method(OperandActor._free_data_in_worker)
    def testOperandActorWithCancel(self, *_):
        import logging
        logging.basicConfig(level=logging.DEBUG)

        arr = mt.random.randint(10, size=(10, 8), chunk_size=4)
        arr_add = mt.random.randint(10, size=(10, 8), chunk_size=4)
        arr2 = arr + arr_add

        session_id = str(uuid.uuid4())
        graph_key = str(uuid.uuid4())

        graph = arr2.build_graph(compose=False)

        with create_actor_pool(n_process=1, backend='gevent') as pool:
            pool.create_actor(SchedulerClusterInfoActor, [pool.cluster_info.address],
                              uid=SchedulerClusterInfoActor.default_uid())
            resource_ref = pool.create_actor(ResourceActor, uid=ResourceActor.default_uid())
            pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())
            pool.create_actor(AssignerActor, uid=AssignerActor.default_uid())
            graph_ref = pool.create_actor(GraphActor, session_id, graph_key, serialize_graph(graph),
                                          uid=GraphActor.gen_uid(session_id, graph_key))

            def _build_mock_ref(uid=None, address=None):
                try:
                    return pool.create_actor(
                        FakeExecutionActor, exec_delay=0.2, uid=FakeExecutionActor.gen_uid(address))
                except ActorAlreadyExist:
                    return pool.actor_ref(FakeExecutionActor.gen_uid(address))

            # handle mock objects
            OperandActor._get_raw_execution_ref.side_effect = _build_mock_ref

            mock_resource = dict(hardware=dict(cpu=4, cpu_total=4, memory=512))

            for idx in range(20):
                resource_ref.set_worker_meta('localhost:%d' % (idx + 12345), mock_resource)

            graph_ref.prepare_graph(compose=False)
            fetched_graph = graph_ref.get_chunk_graph()

            graph_ref.analyze_graph()

            final_keys = set()
            for c in fetched_graph:
                if fetched_graph.count_successors(c) == 0:
                    final_keys.add(c.op.key)

            graph_ref.create_operand_actors()
            graph_meta_ref = pool.actor_ref(GraphMetaActor.gen_uid(session_id, graph_key))
            start_time = time.time()
            cancel_called = False
            while True:
                pool.sleep(0.05)
                if not cancel_called and time.time() > start_time + 0.3:
                    cancel_called = True
                    graph_ref.stop_graph(_tell=True)
                if time.time() - start_time > 30:
                    raise SystemError('Wait for execution finish timeout')
                if graph_meta_ref.get_state() in (GraphState.SUCCEEDED, GraphState.FAILED, GraphState.CANCELLED):
                    break
            self.assertEqual(graph_meta_ref.get_state(), GraphState.CANCELLED)
