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
import time
import uuid
from collections import defaultdict

import gevent

from mars import promise, tensor as mt
from mars.config import options
from mars.cluster_info import ClusterInfoActor
from mars.errors import ExecutionInterrupted
from mars.scheduler import OperandActor, ResourceActor, GraphActor, AssignerActor, KVStoreActor
from mars.utils import serialize_graph, deserialize_graph
from mars.actors import create_actor_pool
from mars.compat import unittest, mock


class FakeExecutionActor(promise.PromiseActor):
    def __init__(self, sleep=0, fail_count=0):
        self._kv_store_ref = None
        self._fail_count = fail_count
        self._sleep = sleep

        self._callbacks = defaultdict(list)
        self._cancels = set()
        self._retries = defaultdict(lambda: fail_count)

    def post_create(self):
        self._kv_store_ref = self.ctx.actor_ref(KVStoreActor.default_name())

    def actual_exec(self, session_id, graph_key, graph_ser, targets):
        if graph_key not in self._retries:
            self._retries[graph_key] = self._fail_count

        if graph_key in self._cancels:
            try:
                raise ExecutionInterrupted
            except:
                exc = sys.exc_info()

            for cb in self._callbacks[graph_key]:
                self.tell_promise(cb, *exc, **dict(_accept=False))
            del self._callbacks[graph_key]
            return
        elif self._fail_count and self._retries.get(graph_key):
            self._retries[graph_key] -= 1

            for cb in self._callbacks[graph_key]:
                self.tell_promise(cb, _accept=False)
            del self._callbacks[graph_key]
            return

        chunk_graph = deserialize_graph(graph_ser)
        key_to_chunks = defaultdict(list)
        for n in chunk_graph:
            key_to_chunks[n.key].append(n)
            self._kv_store_ref.write(
                '/sessions/%s/chunks/%s/data_size' % (session_id, n.key), 0)

        for tk in targets:
            for n in key_to_chunks[tk]:
                self._kv_store_ref.write('/sessions/%s/chunks/%s/workers/localhost:12345'
                                         % (session_id, n.key), '')
        for cb in self._callbacks[graph_key]:
            self.tell_promise(cb, {})
        del self._callbacks[graph_key]

    def execute_graph(self, session_id, graph_key, graph_ser, io_meta, data_sizes, send_targets=None, callback=None):
        self._callbacks[graph_key].append(callback)
        self.ref().actual_exec(session_id, graph_key, graph_ser, io_meta['chunks'],
                               _tell=True, _delay=self._sleep)

    def add_finish_callback(self, session_id, graph_key, callback):
        self._callbacks[graph_key].append(callback)

    def stop_execution(self, graph_key):
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
                kv_store_ref = pool.create_actor(KVStoreActor, uid=KVStoreActor.default_name())
                pool.create_actor(AssignerActor, uid=AssignerActor.gen_name(session_id))
                graph_ref = pool.create_actor(GraphActor, session_id, graph_key, serialize_graph(graph),
                                              uid=GraphActor.gen_name(session_id, graph_key))
                execution_ref = execution_creator(pool)

                # handle mock objects
                OperandActor._get_raw_execution_ref.side_effect = lambda: execution_ref

                mock_resource = dict(hardware=dict(cpu=4, cpu_total=4, memory=512))

                def write_mock_meta():
                    resource_ref.set_worker_meta('localhost:12345', mock_resource)
                    resource_ref.set_worker_meta('localhost:23456', mock_resource)

                v = gevent.spawn(write_mock_meta)
                v.join()

                graph_ref.prepare_graph()
                graph_data = kv_store_ref.read('/sessions/%s/graphs/%s/chunk_graph'
                                               % (session_id, graph_key)).value
                fetched_graph = deserialize_graph(graph_data)

                graph_ref.scan_node()
                graph_ref.place_initial_chunks()

                final_keys = set()
                for c in fetched_graph:
                    if fetched_graph.count_successors(c) == 0:
                        final_keys.add(c.op.key)

                graph_ref.create_operand_actors()
                start_time = time.time()
                while True:
                    gevent.sleep(0.1)
                    if time.time() - start_time > 30:
                        raise SystemError('Wait for execution finish timeout')
                    if kv_store_ref.read('/sessions/%s/graph/%s/state' % (session_id, graph_key)).value.lower() \
                            in ('succeeded', 'failed', 'cancelled'):
                        break

            v = gevent.spawn(execute_case)
            v.get()

    @mock.patch(OperandActor.__module__ + '.OperandActor._get_raw_execution_ref')
    @mock.patch(OperandActor.__module__ + '.OperandActor._free_worker_data')
    def testOperandActor(self, *_):
        arr = mt.random.randint(10, size=(10, 8), chunks=4)
        arr_add = mt.random.randint(10, size=(10, 8), chunks=4)
        arr2 = arr + arr_add

        session_id = str(uuid.uuid4())
        graph_key = str(uuid.uuid4())
        self._run_operand_case(session_id, graph_key, arr2,
                               lambda pool: pool.create_actor(FakeExecutionActor))

    @mock.patch(OperandActor.__module__ + '.OperandActor._get_raw_execution_ref')
    @mock.patch(OperandActor.__module__ + '.OperandActor._free_worker_data')
    def testOperandActorWithSameKey(self, *_):
        arr = mt.ones((5, 5), chunks=3)
        arr2 = mt.concatenate((arr, arr))

        session_id = str(uuid.uuid4())
        graph_key = str(uuid.uuid4())
        self._run_operand_case(session_id, graph_key, arr2,
                               lambda pool: pool.create_actor(FakeExecutionActor))

    @mock.patch(OperandActor.__module__ + '.OperandActor._get_raw_execution_ref')
    @mock.patch(OperandActor.__module__ + '.OperandActor._free_worker_data')
    def testOperandActorWithRetry(self, *_):
        arr = mt.random.randint(10, size=(10, 8), chunks=4)
        arr_add = mt.random.randint(10, size=(10, 8), chunks=4)
        arr2 = arr + arr_add

        session_id = str(uuid.uuid4())
        graph_key = str(uuid.uuid4())
        try:
            options.scheduler.retry_delay = 0
            self._run_operand_case(session_id, graph_key, arr2,
                                   lambda pool: pool.create_actor(FakeExecutionActor, fail_count=2))
        finally:
            options.scheduler.retry_delay = 60

    @mock.patch(OperandActor.__module__ + '.OperandActor._get_raw_execution_ref')
    @mock.patch(OperandActor.__module__ + '.OperandActor._free_worker_data')
    def testOperandActorWithRetryAndFail(self, *_):
        arr = mt.random.randint(10, size=(10, 8), chunks=4)
        arr_add = mt.random.randint(10, size=(10, 8), chunks=4)
        arr2 = arr + arr_add

        session_id = str(uuid.uuid4())
        graph_key = str(uuid.uuid4())
        try:
            options.scheduler.retry_delay = 0
            self._run_operand_case(session_id, graph_key, arr2,
                                   lambda pool: pool.create_actor(FakeExecutionActor, fail_count=4))
        finally:
            options.scheduler.retry_delay = 60

    @mock.patch(OperandActor.__module__ + '.OperandActor._get_raw_execution_ref')
    @mock.patch(OperandActor.__module__ + '.OperandActor._free_worker_data')
    def testOperandActorWithCancel(self, *_):
        import logging
        logging.basicConfig(level=logging.DEBUG)

        arr = mt.random.randint(10, size=(10, 8), chunks=4)
        arr_add = mt.random.randint(10, size=(10, 8), chunks=4)
        arr2 = arr + arr_add

        session_id = str(uuid.uuid4())
        graph_key = str(uuid.uuid4())

        graph = arr2.build_graph(compose=False)

        with create_actor_pool(n_process=1, backend='gevent') as pool:
            def execute_case():
                pool.create_actor(ClusterInfoActor, [pool.cluster_info.address],
                                  uid=ClusterInfoActor.default_name())
                resource_ref = pool.create_actor(ResourceActor, uid=ResourceActor.default_name())
                kv_store_ref = pool.create_actor(KVStoreActor, uid=KVStoreActor.default_name())
                pool.create_actor(AssignerActor, uid=AssignerActor.gen_name(session_id))
                graph_ref = pool.create_actor(GraphActor, session_id, graph_key, serialize_graph(graph),
                                              uid=GraphActor.gen_name(session_id, graph_key))
                execution_ref = pool.create_actor(FakeExecutionActor, sleep=1)

                # handle mock objects
                OperandActor._get_raw_execution_ref.side_effect = lambda: execution_ref

                mock_resource = dict(hardware=dict(cpu=4, cpu_total=4, memory=512))

                def write_mock_meta():
                    resource_ref.set_worker_meta('localhost:12345', mock_resource)
                    resource_ref.set_worker_meta('localhost:23456', mock_resource)

                v = gevent.spawn(write_mock_meta)
                v.join()

                graph_ref.prepare_graph()
                graph_data = kv_store_ref.read('/sessions/%s/graphs/%s/chunk_graph'
                                               % (session_id, graph_key)).value
                fetched_graph = deserialize_graph(graph_data)

                graph_ref.scan_node()
                graph_ref.place_initial_chunks()

                final_keys = set()
                for c in fetched_graph:
                    if fetched_graph.count_successors(c) == 0:
                        final_keys.add(c.op.key)

                graph_ref.create_operand_actors()
                start_time = time.time()
                cancel_called = False
                while True:
                    gevent.sleep(0.1)
                    if not cancel_called and time.time() > start_time + 0.8:
                        cancel_called = True
                        graph_ref.stop_graph(_tell=True)
                    if time.time() - start_time > 30:
                        raise SystemError('Wait for execution finish timeout')
                    if kv_store_ref.read('/sessions/%s/graph/%s/state' % (session_id, graph_key)).value.lower() \
                            in ('succeeded', 'failed', 'cancelled'):
                        break

            v = gevent.spawn(execute_case)
            v.get()
