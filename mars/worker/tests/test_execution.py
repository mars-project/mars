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
import tempfile
import threading
import time
import uuid
import weakref

import numpy as np
from numpy.testing import assert_array_equal

from mars import promise
from mars.actors import create_actor_pool
from mars.compat import six
from mars.config import options
from mars.errors import WorkerProcessStopped, ExecutionInterrupted, DependencyMissing
from mars.utils import get_next_port, serialize_graph
from mars.scheduler import ChunkMetaActor, ChunkMetaClient, ResourceActor
from mars.scheduler.utils import SchedulerClusterInfoActor
from mars.tests.core import patch_method
from mars.worker.tests.base import WorkerCase
from mars.worker import *
from mars.worker.chunkstore import PlasmaKeyMapActor
from mars.distributor import MarsDistributor
from mars.worker.prochelper import ProcessHelperActor
from mars.worker.utils import WorkerActor, WorkerClusterInfoActor


class MockInProcessCacheActor(WorkerActor):
    def __init__(self, session_id, mock_data):
        super(MockInProcessCacheActor, self).__init__()
        self._session_id = session_id
        self._mock_data = mock_data
        self._chunk_holder_ref = None

    def post_create(self):
        super(MockInProcessCacheActor, self).post_create()
        self._chunk_holder_ref = self.ctx.actor_ref(ChunkHolderActor.default_uid())

    def dump_cache(self, session_id, keys, callback):
        for k in keys:
            ref = self._chunk_store.put(self._session_id, k, self._mock_data)
            self._chunk_holder_ref.register_chunk(self._session_id, k)
            del ref
        self.tell_promise(callback, {})


class MockCpuCalcActor(WorkerActor):
    def __init__(self, session_id, mock_data, delay):
        super(MockCpuCalcActor, self).__init__()
        self._delay = delay
        self._session_id = session_id
        self._mock_data = mock_data
        self._inproc_ref = None
        self._dispatch_ref = None

    def post_create(self):
        uid_parts = self.uid.split(':')
        inproc_uid = 'w:' + uid_parts[1] + ':inproc-cache-' + str(uuid.uuid4())
        self._inproc_ref = self.ctx.create_actor(
            MockInProcessCacheActor, self._session_id, self._mock_data, uid=inproc_uid)
        daemon_ref = self.ctx.actor_ref(WorkerDaemonActor.default_uid())
        if self.ctx.has_actor(daemon_ref):
            daemon_ref.register_child_actor(self._inproc_ref, _tell=True)

        self._dispatch_ref = self.promise_ref(DispatchActor.default_uid())
        self._dispatch_ref.register_free_slot(self.uid, 'cpu')

    @promise.reject_on_exception
    def calc(self, session_id, ser_graph, targets, callback):
        self.ctx.sleep(self._delay)
        self.tell_promise(callback, self._inproc_ref.uid)
        self._dispatch_ref.register_free_slot(self.uid, 'cpu')


class MockSenderActor(WorkerActor):
    def __init__(self, mock_data, mode=None):
        super(MockSenderActor, self).__init__()
        self._mode = mode or 'in'
        self._mock_data = mock_data
        self._dispatch_ref = None

    def post_create(self):
        super(MockSenderActor, self).post_create()
        self._dispatch_ref = self.promise_ref(DispatchActor.default_uid())
        self._dispatch_ref.register_free_slot(self.uid, 'sender')

    @promise.reject_on_exception
    def send_data(self, session_id, chunk_key, target_endpoints, ensure_cached=True,
                  timeout=0, callback=None):
        if self._mode == 'in':
            self._chunk_store.put(session_id, chunk_key, self._mock_data)
        else:
            data = self._chunk_store.get(session_id, chunk_key)
            assert_array_equal(self._mock_data, data)
        self.tell_promise(callback, self._mock_data.nbytes)
        self._dispatch_ref.register_free_slot(self.uid, 'sender')


class ExecutionTestActor(WorkerActor):
    def __init__(self):
        super(ExecutionTestActor, self).__init__()
        self._results = []
        self._session_id = None
        self._graph_key = None
        self._array_key = None

    def run_simple_calc(self, session_id):
        self._session_id = session_id

        import mars.tensor as mt
        arr = mt.ones((4,), chunk_size=4) + 1
        graph = arr.build_graph(compose=False, tiled=True)

        self._array_key = arr.chunks[0].key

        graph_key = self._graph_key = str(uuid.uuid4())
        execution_ref = self.promise_ref(ExecutionActor.default_uid())
        execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                    dict(chunks=[arr.chunks[0].key]), None, _tell=True)

        execution_ref.add_finish_callback(session_id, graph_key, _promise=True) \
            .then(lambda *_: self._results.append((True,))) \
            .catch(lambda *exc: self._results.append((False, exc)))

    def get_graph_key(self):
        return self._graph_key

    def get_results(self):
        if not self._results:
            return None
        if self._results[0][0]:
            return self._chunk_store.get(self._session_id, self._array_key)
        else:
            six.reraise(*self._results[0][1])


class Test(WorkerCase):
    def tearDown(self):
        super(Test, self).tearDown()
        logger = logging.getLogger(ExecutionActor.__module__)
        logger.setLevel(logging.WARNING)
        self.rm_spill_dirs(options.worker.spill_directory)

    @classmethod
    def create_standard_actors(cls, pool, address, quota_size=None, with_daemon=True,
                               with_status=True, with_resource=False):
        quota_size = quota_size or (1024 * 1024)

        pool.create_actor(SchedulerClusterInfoActor, schedulers=[address],
                          uid=SchedulerClusterInfoActor.default_uid())
        pool.create_actor(WorkerClusterInfoActor, schedulers=[address],
                          uid=WorkerClusterInfoActor.default_uid())

        pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
        if with_resource:
            pool.create_actor(ResourceActor, uid=ResourceActor.default_uid())
        if with_daemon:
            pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
        if with_status:
            pool.create_actor(StatusActor, address, uid=StatusActor.default_uid())

        pool.create_actor(
            ChunkHolderActor, cls.plasma_storage_size, uid=ChunkHolderActor.default_uid())
        pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())
        pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())
        pool.create_actor(QuotaActor, quota_size, uid=MemQuotaActor.default_uid())
        pool.create_actor(ExecutionActor, uid=ExecutionActor.default_uid())

    @staticmethod
    def wait_for_result(pool, test_actor):
        check_time = time.time()
        while test_actor.get_results() is None:
            pool.sleep(1)
            if time.time() - check_time > 10:
                raise SystemError('Timed out')

    def testSimpleExecution(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        with create_actor_pool(n_process=1, backend='gevent', address=pool_address) as pool:
            self.create_standard_actors(pool, pool_address, with_daemon=False)
            pool.create_actor(CpuCalcActor, uid='w:1:calc-a')

            import mars.tensor as mt
            from mars.tensor.expressions.datasource import TensorOnes
            from mars.tensor.expressions.fetch import TensorFetch
            arr = mt.ones((10, 8), chunk_size=10)
            arr_add = mt.ones((10, 8), chunk_size=10)
            arr2 = arr + arr_add
            graph = arr2.build_graph(compose=False, tiled=True)

            for chunk in graph:
                if isinstance(chunk.op, TensorOnes):
                    chunk._op = TensorFetch(
                        dtype=chunk.dtype, _outputs=[weakref.ref(o) for o in chunk.op.outputs],
                        _key=chunk.op.key)

            with self.run_actor_test(pool) as test_actor:

                session_id = str(uuid.uuid4())
                chunk_holder_ref = test_actor.promise_ref(ChunkHolderActor.default_uid())

                refs = test_actor._chunk_store.put(session_id, arr.chunks[0].key,
                                                   np.ones((10, 8), dtype=np.int16))
                chunk_holder_ref.register_chunk(session_id, arr.chunks[0].key)
                del refs

                refs = test_actor._chunk_store.put(session_id, arr_add.chunks[0].key,
                                                   np.ones((10, 8), dtype=np.int16))
                chunk_holder_ref.register_chunk(session_id, arr_add.chunks[0].key)
                del refs

                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())

                def _validate(_):
                    data = test_actor._chunk_store.get(session_id, arr2.chunks[0].key)
                    assert_array_equal(data, 2 * np.ones((10, 8)))

                graph_key = str(uuid.uuid4())
                execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                            dict(chunks=[arr2.chunks[0].key]), None, _promise=True) \
                    .then(_validate) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            self.get_result()

            with self.run_actor_test(pool) as test_actor:
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())

                def _validate(_):
                    data = test_actor._chunk_store.get(session_id, arr2.chunks[0].key)
                    assert_array_equal(data, 2 * np.ones((10, 8)))

                execution_ref.add_finish_callback(session_id, graph_key, _promise=True) \
                    .then(_validate) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            self.get_result()

    @patch_method(ChunkHolderActor.pin_chunks)
    def testPrepareQuota(self, *_):
        pinned = [True]

        def _mock_pin(_graph_key, chunk_keys):
            from mars.errors import PinChunkFailed
            if pinned[0]:
                raise PinChunkFailed
            return chunk_keys

        ChunkHolderActor.pin_chunks.side_effect = _mock_pin

        pool_address = '127.0.0.1:%d' % get_next_port()
        session_id = str(uuid.uuid4())
        mock_data = np.array([1, 2, 3, 4])
        with create_actor_pool(n_process=1, backend='gevent', address=pool_address) as pool:
            self.create_standard_actors(pool, pool_address, with_daemon=False, with_status=False)
            pool.create_actor(MockSenderActor, mock_data, 'in', uid='w:mock_sender')
            pool.create_actor(CpuCalcActor)
            cluster_info_ref = pool.actor_ref(WorkerClusterInfoActor.default_uid())
            chunk_meta_client = ChunkMetaClient(pool, cluster_info_ref)

            import mars.tensor as mt
            from mars.tensor.expressions.fetch import TensorFetch
            arr = mt.ones((4,), chunk_size=4)
            arr_add = mt.array(mock_data)
            result_tensor = arr + arr_add
            graph = result_tensor.build_graph(compose=False, tiled=True)

            modified_chunk = arr_add.chunks[0]
            arr_add.chunks[0]._op = TensorFetch(
                dtype=modified_chunk.dtype, _outputs=[weakref.ref(o) for o in modified_chunk.op.outputs],
                _key=modified_chunk.op.key)
            chunk_meta_client.set_chunk_meta(session_id, modified_chunk.key, size=mock_data.nbytes,
                                             shape=mock_data.shape, workers=('0.0.0.0:1234', pool_address))
            with self.run_actor_test(pool) as test_actor:
                graph_key = str(uuid.uuid4())
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())

                start_time = time.time()

                execution_ref.execute_graph(
                    session_id, graph_key, serialize_graph(graph),
                    dict(chunks=[result_tensor.chunks[0].key]), None, _tell=True)

                execution_ref.add_finish_callback(session_id, graph_key, _promise=True) \
                    .then(lambda *_: test_actor.set_result(time.time())) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

                def _delay_fun():
                    time.sleep(1)
                    pinned[0] = False

                threading.Thread(target=_delay_fun).start()

            finish_time = self.get_result()
            self.assertGreaterEqual(finish_time, start_time + 1)

    def testPrepareSpilled(self):
        from mars.worker.spill import write_spill_file

        pool_address = '127.0.0.1:%d' % get_next_port()
        session_id = str(uuid.uuid4())
        mock_data = np.array([1, 2, 3, 4])

        options.worker.spill_directory = tempfile.mkdtemp(prefix='mars_worker_prep_spilled-')

        with create_actor_pool(n_process=1, backend='gevent', address=pool_address) as pool:
            self.create_standard_actors(pool, pool_address, with_daemon=False, with_status=False)
            pool.create_actor(SpillActor)
            pool.create_actor(CpuCalcActor)
            cluster_info_ref = pool.actor_ref(WorkerClusterInfoActor.default_uid())
            chunk_meta_client = ChunkMetaClient(pool, cluster_info_ref)
            pool.actor_ref(ChunkHolderActor.default_uid())

            import mars.tensor as mt
            from mars.tensor.expressions.fetch import TensorFetch
            arr = mt.ones((4,), chunk_size=4)
            arr_add = mt.array(mock_data)
            result_tensor = arr + arr_add
            graph = result_tensor.build_graph(compose=False, tiled=True)

            modified_chunk = arr_add.chunks[0]
            arr_add.chunks[0]._op = TensorFetch(
                dtype=modified_chunk.dtype, _outputs=[weakref.ref(o) for o in modified_chunk.op.outputs],
                _key=modified_chunk.op.key)

            # test meta missing
            with self.run_actor_test(pool) as test_actor:
                graph_key = str(uuid.uuid4())
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())
                execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                            dict(chunks=[result_tensor.chunks[0].key]), None, _promise=True) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            with self.assertRaises(DependencyMissing):
                self.get_result()

            chunk_meta_client.set_chunk_meta(session_id, modified_chunk.key, size=mock_data.nbytes,
                                             shape=mock_data.shape, workers=('0.0.0.0:1234', pool_address))
            write_spill_file(modified_chunk.key, mock_data)

            # test read from spilled file
            with self.run_actor_test(pool) as test_actor:
                def _validate(_):
                    data = test_actor._chunk_store.get(session_id, result_tensor.chunks[0].key)
                    assert_array_equal(data, mock_data + np.ones((4,)))

                graph_key = str(uuid.uuid4())
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())
                execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                            dict(chunks=[result_tensor.chunks[0].key]), None, _promise=True) \
                    .then(_validate) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            self.get_result()

    @patch_method(ResourceActor.allocate_resource, new=lambda *_, **__: True)
    def testEstimateGraphFinishTime(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        session_id = str(uuid.uuid4())
        with create_actor_pool(n_process=1, backend='gevent', address=pool_address) as pool:
            self.create_standard_actors(pool, pool_address, with_daemon=False)

            status_ref = pool.actor_ref(StatusActor.default_uid())
            execution_ref = pool.actor_ref(ExecutionActor.default_uid())
            pool.create_actor(CpuCalcActor)

            import mars.tensor as mt
            arr = mt.ones((10, 8), chunk_size=10)
            graph = arr.build_graph(compose=False, tiled=True)

            graph_key = str(uuid.uuid4())

            for _ in range(options.optimize.min_stats_count + 1):
                status_ref.update_mean_stats(
                    'calc_speed.' + type(arr.chunks[0].op).__name__, 10)
                status_ref.update_mean_stats('disk_read_speed', 10)
                status_ref.update_mean_stats('disk_write_speed', 10)
                status_ref.update_mean_stats('net_transfer_speed', 10)

            execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                        dict(chunks=[arr.chunks[0].key]), None)
            execution_ref.estimate_graph_finish_time(session_id, graph_key)

            stats_dict = status_ref.get_stats(['min_est_finish_time', 'max_est_finish_time'])
            self.assertIsNotNone(stats_dict.get('min_est_finish_time'))
            self.assertIsNotNone(stats_dict.get('max_est_finish_time'))

    def testCalcProcessFailure(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        session_id = str(uuid.uuid4())
        mock_data = np.array([1, 2, 3, 4])
        with create_actor_pool(n_process=2, backend='gevent',
                               address=pool_address, distributor=MarsDistributor(2, 'w:0:')) as pool:
            self.create_standard_actors(pool, pool_address, with_status=False)

            daemon_ref = pool.actor_ref(WorkerDaemonActor.default_uid())
            dispatch_ref = pool.actor_ref(DispatchActor.default_uid())
            calc_ref = daemon_ref.create_actor(
                MockCpuCalcActor, session_id, mock_data, 10, uid='w:1:cpu-calc-a')
            daemon_ref.create_actor(ProcessHelperActor, uid='w:1:proc-helper-a')

            test_actor = pool.create_actor(ExecutionTestActor, uid='w:0:test_actor')
            test_actor.run_simple_calc(session_id, _tell=True)

            pool.sleep(2)
            proc_id = pool.distributor.distribute(calc_ref.uid)
            daemon_ref.kill_actor_process(calc_ref)
            assert not daemon_ref.is_actor_process_alive(calc_ref)
            pool.restart_process(proc_id)
            daemon_ref.handle_process_down([proc_id])

            with self.assertRaises(WorkerProcessStopped):
                self.wait_for_result(pool, test_actor)
            self.assertEqual(len(dispatch_ref.get_slots('cpu')), 1)

    def testStopGraphCalc(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        session_id = str(uuid.uuid4())
        mock_data = np.array([1, 2, 3, 4])
        with create_actor_pool(n_process=2, backend='gevent',
                               address=pool_address, distributor=MarsDistributor(2, 'w:0:')) as pool:
            self.create_standard_actors(pool, pool_address, with_status=False)

            daemon_ref = pool.actor_ref(WorkerDaemonActor.default_uid())
            execution_ref = pool.actor_ref(ExecutionActor.default_uid())

            calc_ref = daemon_ref.create_actor(
                MockCpuCalcActor, session_id, mock_data, 10, uid='w:1:cpu-calc-a')
            daemon_ref.create_actor(ProcessHelperActor, uid='w:1:proc-helper-a')

            test_actor = pool.create_actor(ExecutionTestActor, uid='w:0:test_actor')
            test_actor.run_simple_calc(session_id, _tell=True)

            pool.sleep(2)
            proc_id = pool.distributor.distribute(calc_ref.uid)
            execution_ref.stop_execution(session_id, test_actor.get_graph_key(), _tell=True)
            while daemon_ref.is_actor_process_alive(calc_ref):
                pool.sleep(0.1)
            pool.restart_process(proc_id)
            daemon_ref.handle_process_down([proc_id])

            with self.assertRaises(ExecutionInterrupted):
                self.wait_for_result(pool, test_actor)

    def testFetchRemoteData(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        session_id = str(uuid.uuid4())
        mock_data = np.array([1, 2, 3, 4])
        with create_actor_pool(n_process=1, backend='gevent',
                               address=pool_address, distributor=MarsDistributor(2, 'w:0:')) as pool:
            self.create_standard_actors(pool, pool_address, with_daemon=False, with_status=False,
                                        with_resource=True)
            pool.create_actor(CpuCalcActor)
            pool.create_actor(MockSenderActor, mock_data, 'in', uid='w:mock_sender')
            cluster_info_ref = pool.actor_ref(WorkerClusterInfoActor.default_uid())
            chunk_meta_client = ChunkMetaClient(pool, cluster_info_ref)

            import mars.tensor as mt
            from mars.tensor.expressions.fetch import TensorFetch
            arr = mt.ones((4,), chunk_size=4)
            arr_add = mt.array(mock_data)
            result_tensor = arr + arr_add
            graph = result_tensor.build_graph(compose=False, tiled=True)

            modified_chunk = arr_add.chunks[0]
            arr_add.chunks[0]._op = TensorFetch(
                dtype=modified_chunk.dtype, _outputs=[weakref.ref(o) for o in modified_chunk.op.outputs],
                _key=modified_chunk.op.key)

            with self.run_actor_test(pool) as test_actor:
                graph_key = str(uuid.uuid4())
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())
                execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                            dict(chunks=[result_tensor.chunks[0].key]), None, _tell=True)

                execution_ref.add_finish_callback(session_id, graph_key, _promise=True) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            with self.assertRaises(DependencyMissing):
                self.get_result()

            chunk_meta_client.set_chunk_meta(session_id, modified_chunk.key, size=mock_data.nbytes,
                                             shape=mock_data.shape, workers=('0.0.0.0:1234',))
            with self.run_actor_test(pool) as test_actor:
                graph_key = str(uuid.uuid4())
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())
                execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                            dict(chunks=[result_tensor.chunks[0].key]), None, _tell=True)

                execution_ref.add_finish_callback(session_id, graph_key, _promise=True) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            with self.assertRaises(DependencyMissing):
                self.get_result()

            chunk_meta_client.set_chunk_meta(session_id, modified_chunk.key, size=mock_data.nbytes,
                                             shape=mock_data.shape, workers=('0.0.0.0:1234', pool_address))
            with self.run_actor_test(pool) as test_actor:
                def _validate(_):
                    data = test_actor._chunk_store.get(session_id, result_tensor.chunks[0].key)
                    assert_array_equal(data, mock_data + np.ones((4,)))

                graph_key = str(uuid.uuid4())
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())
                execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                            dict(chunks=[result_tensor.chunks[0].key]), None, _tell=True)

                execution_ref.add_finish_callback(session_id, graph_key, _promise=True) \
                    .then(_validate) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            self.get_result()

    def testSendTargets(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        session_id = str(uuid.uuid4())
        mock_data = np.array([1, 2, 3, 4])
        with create_actor_pool(n_process=1, backend='gevent',
                               address=pool_address, distributor=MarsDistributor(2, 'w:0:')) as pool:
            self.create_standard_actors(pool, pool_address, with_daemon=False, with_status=False)
            pool.create_actor(CpuCalcActor)

            import mars.tensor as mt
            arr = mt.ones((4,), chunk_size=4)
            arr_add = mt.array(mock_data)
            result_tensor = arr + arr_add
            graph = result_tensor.build_graph(compose=False, tiled=True)
            result_key = result_tensor.chunks[0].key

            pool.create_actor(MockSenderActor, mock_data + np.ones((4,)), 'out', uid='w:mock_sender')
            with self.run_actor_test(pool) as test_actor:
                def _validate(_):
                    data = test_actor._chunk_store.get(session_id, result_tensor.chunks[0].key)
                    assert_array_equal(data, mock_data + np.ones((4,)))

                graph_key = str(uuid.uuid4())
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())

                execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                            dict(chunks=[result_tensor.chunks[0].key]), None, _tell=True)
                execution_ref.send_data_to_workers(
                    session_id, graph_key, {result_key: (pool_address,)}, _tell=True)

                execution_ref.add_finish_callback(session_id, graph_key, _promise=True) \
                    .then(_validate) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            self.get_result()

    def testReExecuteExisting(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        session_id = str(uuid.uuid4())
        mock_data = np.array([1, 2, 3, 4])
        with create_actor_pool(n_process=1, backend='gevent',
                               address=pool_address, distributor=MarsDistributor(2, 'w:0:')) as pool:
            self.create_standard_actors(pool, pool_address, with_daemon=False, with_status=False)
            pool.create_actor(CpuCalcActor)

            import mars.tensor as mt
            arr = mt.ones((4,), chunk_size=4)
            arr_add = mt.array(mock_data)
            result_tensor = arr + arr_add
            graph = result_tensor.build_graph(compose=False, tiled=True)

            pool.create_actor(MockSenderActor, mock_data + np.ones((4,)), 'out', uid='w:mock_sender')

            def _validate(_):
                data = test_actor._chunk_store.get(session_id, result_tensor.chunks[0].key)
                assert_array_equal(data, mock_data + np.ones((4,)))

            with self.run_actor_test(pool) as test_actor:
                graph_key = str(uuid.uuid4())
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())
                execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                            dict(chunks=[result_tensor.chunks[0].key]), None, _promise=True) \
                    .then(_validate) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            self.get_result()

            with self.run_actor_test(pool) as test_actor:
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())
                execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                            dict(chunks=[result_tensor.chunks[0].key]), None, _promise=True) \
                    .then(_validate) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            self.get_result()
