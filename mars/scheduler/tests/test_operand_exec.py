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
import sys
import time
import unittest
import uuid
from collections import defaultdict

import gevent

from mars import promise, tensor as mt
from mars.config import options
from mars.cluster_info import ClusterInfoActor
from mars.errors import ExecutionInterrupted
from mars.scheduler import OperandActor, ResourceActor, GraphActor, AssignerActor, \
    ChunkMetaActor, GraphMetaActor
from mars.scheduler.utils import GraphState
from mars.worker.execution import GraphExecutionRecord
from mars.utils import serialize_graph, log_unhandled
from mars.actors import create_actor_pool
from mars.tests.core import patch_method

logger = logging.getLogger(__name__)


class FakeExecutionActor(promise.PromiseActor):
    _retries = defaultdict(lambda: 0)

    def __init__(self, sleep=0.1, fail_count=0):
        super(FakeExecutionActor, self).__init__()

        self._chunk_meta_ref = None
        self._fail_count = fail_count
        self._sleep = sleep

        self._results = dict()
        self._cancels = set()
        self._graph_records = dict()  # type: dict[tuple, GraphExecutionRecord]

    def post_create(self):
        self._chunk_meta_ref = self.ctx.actor_ref(ChunkMetaActor.default_name())

    @log_unhandled
    def actual_exec(self, session_id, graph_key):
        if graph_key in self._results:
            del self._results[graph_key]

        rec = self._graph_records[(session_id, graph_key)]
        if graph_key in self._cancels:
            try:
                raise ExecutionInterrupted
            except:
                exc = sys.exc_info()

            self._results[graph_key] = (exc, dict(_accept=False))
            for cb in rec.finish_callbacks:
                self.tell_promise(cb, *exc, **dict(_accept=False))
            rec.finish_callbacks = []
            return
        elif self._fail_count and self._retries[graph_key] < self._fail_count:
            logger.debug('Key %r: %r', graph_key, self._retries.get(graph_key))
            self._retries[graph_key] += 1

            del self._graph_records[(session_id, graph_key)]
            self._results[graph_key] = ((), dict(_accept=False))
            for cb in rec.finish_callbacks:
                self.tell_promise(cb, _accept=False)
            rec.finish_callbacks = []
            return

        chunk_graph = rec.graph
        key_to_chunks = defaultdict(list)
        for n in chunk_graph:
            key_to_chunks[n.key].append(n)
            self._chunk_meta_ref.set_chunk_size(session_id, n.key, 0)

        for tk in rec.targets:
            for n in key_to_chunks[tk]:
                self._chunk_meta_ref.add_worker(session_id, n.key, 'localhost:12345')
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
        rec.finish_callbacks.append(callback)
        self.ref().actual_exec(session_id, graph_key, _tell=True, _delay=self._sleep)

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
            targets=io_meta['chunks'],
            chunks_use_once=set(io_meta.get('input_chunks', [])) - set(io_meta.get('shared_input_chunks', [])),
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


class Test(unittest.TestCase):
    @staticmethod
    def _run_operand_case(session_id, graph_key, tensor, execution_creator):
        graph = tensor.build_graph(compose=False)

        with create_actor_pool(n_process=1, backend='gevent') as pool:
            def execute_case():
                pool.create_actor(ClusterInfoActor, [pool.cluster_info.address],
                                  uid=ClusterInfoActor.default_name())
                resource_ref = pool.create_actor(ResourceActor, uid=ResourceActor.default_name())
                pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_name())
                pool.create_actor(AssignerActor, uid=AssignerActor.default_name())
                graph_ref = pool.create_actor(GraphActor, session_id, graph_key, serialize_graph(graph),
                                              uid=GraphActor.gen_name(session_id, graph_key))
                addr_dict = dict()

                def _build_mock_ref(uid=None, address=None):
                    if address in addr_dict:
                        return addr_dict[address]
                    else:
                        r = addr_dict[address] = execution_creator(pool)
                        return r

                # handle mock objects
                OperandActor._get_raw_execution_ref.side_effect = _build_mock_ref

                mock_resource = dict(hardware=dict(cpu=4, cpu_total=4, memory=512))

                def write_mock_meta():
                    resource_ref.set_worker_meta('localhost:12345', mock_resource)
                    resource_ref.set_worker_meta('localhost:23456', mock_resource)

                v = gevent.spawn(write_mock_meta)
                v.join()

                graph_ref.prepare_graph()
                fetched_graph = graph_ref.get_chunk_graph()

                graph_ref.analyze_graph()

                final_keys = set()
                for c in fetched_graph:
                    if fetched_graph.count_successors(c) == 0:
                        final_keys.add(c.op.key)

                graph_ref.create_operand_actors()

                graph_meta_ref = pool.actor_ref(GraphMetaActor.gen_name(session_id, graph_key))
                start_time = time.time()
                while True:
                    gevent.sleep(0.1)
                    if time.time() - start_time > 30:
                        raise SystemError('Wait for execution finish timeout')
                    if graph_meta_ref.get_state() in (GraphState.SUCCEEDED, GraphState.FAILED, GraphState.CANCELLED):
                        break

            v = gevent.spawn(execute_case)
            v.get()

    @patch_method(OperandActor._get_raw_execution_ref)
    @patch_method(OperandActor._free_worker_data)
    def testOperandActor(self, *_):
        arr = mt.random.randint(10, size=(10, 8), chunk_size=4)
        arr_add = mt.random.randint(10, size=(10, 8), chunk_size=4)
        arr2 = arr + arr_add

        session_id = str(uuid.uuid4())
        graph_key = str(uuid.uuid4())
        self._run_operand_case(session_id, graph_key, arr2,
                               lambda pool: pool.create_actor(FakeExecutionActor))

    @patch_method(OperandActor._get_raw_execution_ref)
    @patch_method(OperandActor._free_worker_data)
    def testOperandActorWithSameKey(self, *_):
        arr = mt.ones((5, 5), chunk_size=3)
        arr2 = mt.concatenate((arr, arr))

        session_id = str(uuid.uuid4())
        graph_key = str(uuid.uuid4())
        self._run_operand_case(session_id, graph_key, arr2,
                               lambda pool: pool.create_actor(FakeExecutionActor))

    @patch_method(OperandActor._get_raw_execution_ref)
    @patch_method(OperandActor._free_worker_data)
    def testOperandActorWithRetry(self, *_):
        arr = mt.random.randint(10, size=(10, 8), chunk_size=4)
        arr_add = mt.random.randint(10, size=(10, 8), chunk_size=4)
        arr2 = arr + arr_add

        session_id = str(uuid.uuid4())
        graph_key = str(uuid.uuid4())
        try:
            options.scheduler.retry_delay = 0
            self._run_operand_case(session_id, graph_key, arr2,
                                   lambda pool: pool.create_actor(FakeExecutionActor, fail_count=2))
        finally:
            options.scheduler.retry_delay = 60

    @patch_method(OperandActor._get_raw_execution_ref)
    @patch_method(OperandActor._free_worker_data)
    def testOperandActorWithRetryAndFail(self, *_):
        arr = mt.random.randint(10, size=(10, 8), chunk_size=4)
        arr_add = mt.random.randint(10, size=(10, 8), chunk_size=4)
        arr2 = arr + arr_add

        session_id = str(uuid.uuid4())
        graph_key = str(uuid.uuid4())
        try:
            options.scheduler.retry_delay = 0
            self._run_operand_case(session_id, graph_key, arr2,
                                   lambda pool: pool.create_actor(FakeExecutionActor, fail_count=5))
        finally:
            options.scheduler.retry_delay = 60

    @patch_method(OperandActor._get_raw_execution_ref)
    @patch_method(OperandActor._free_worker_data)
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
            def execute_case():
                pool.create_actor(ClusterInfoActor, [pool.cluster_info.address],
                                  uid=ClusterInfoActor.default_name())
                resource_ref = pool.create_actor(ResourceActor, uid=ResourceActor.default_name())
                pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_name())
                pool.create_actor(AssignerActor, uid=AssignerActor.default_name())
                graph_ref = pool.create_actor(GraphActor, session_id, graph_key, serialize_graph(graph),
                                              uid=GraphActor.gen_name(session_id, graph_key))
                addr_dict = dict()

                def _build_mock_ref(uid=None, address=None):
                    if address in addr_dict:
                        return addr_dict[address]
                    else:
                        r = addr_dict[address] = pool.create_actor(FakeExecutionActor, sleep=1)
                        return r

                # handle mock objects
                OperandActor._get_raw_execution_ref.side_effect = _build_mock_ref

                mock_resource = dict(hardware=dict(cpu=4, cpu_total=4, memory=512))

                def write_mock_meta():
                    resource_ref.set_worker_meta('localhost:12345', mock_resource)
                    resource_ref.set_worker_meta('localhost:23456', mock_resource)

                v = gevent.spawn(write_mock_meta)
                v.join()

                graph_ref.prepare_graph()
                fetched_graph = graph_ref.get_chunk_graph()

                graph_ref.analyze_graph()

                final_keys = set()
                for c in fetched_graph:
                    if fetched_graph.count_successors(c) == 0:
                        final_keys.add(c.op.key)

                graph_ref.create_operand_actors()
                graph_meta_ref = pool.actor_ref(GraphMetaActor.gen_name(session_id, graph_key))
                start_time = time.time()
                cancel_called = False
                while True:
                    gevent.sleep(0.1)
                    if not cancel_called and time.time() > start_time + 0.8:
                        cancel_called = True
                        graph_ref.stop_graph(_tell=True)
                    if time.time() - start_time > 30:
                        raise SystemError('Wait for execution finish timeout')
                    if graph_meta_ref.get_state() in (GraphState.SUCCEEDED, GraphState.FAILED, GraphState.CANCELLED):
                        break

            v = gevent.spawn(execute_case)
            v.get()
