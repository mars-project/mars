# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

import asyncio
import logging
import tempfile
import threading
import time
import uuid
import weakref

import numpy as np
from numpy.testing import assert_array_equal

from mars import promise
from mars.config import options
from mars.tiles import get_tiled
from mars.errors import WorkerProcessStopped, ExecutionInterrupted, DependencyMissing
from mars.utils import get_next_port, serialize_graph
from mars.scheduler import ChunkMetaActor, ResourceActor
from mars.scheduler.chunkmeta import WorkerMeta
from mars.scheduler.utils import SchedulerClusterInfoActor
from mars.tests.core import patch_method, create_actor_pool, aio_case
from mars.worker.tests.base import WorkerCase
from mars.worker import DispatchActor, ExecutionActor, CpuCalcActor, WorkerDaemonActor, \
    StorageManagerActor, StatusActor, QuotaActor, MemQuotaActor, IORunnerActor
from mars.worker.storage import PlasmaKeyMapActor, SharedHolderActor, InProcHolderActor, \
    DataStorageDevice
from mars.distributor import MarsDistributor
from mars.worker.prochelper import ProcessHelperActor
from mars.worker.utils import WorkerActor, WorkerClusterInfoActor


class MockCpuCalcActor(WorkerActor):
    def __init__(self, session_id, mock_data, delay):
        super().__init__()
        self._delay = delay
        self._session_id = session_id
        self._mock_data = mock_data
        self._dispatch_ref = None

    async def post_create(self):
        self._dispatch_ref = self.promise_ref(DispatchActor.default_uid())
        await self._dispatch_ref.register_free_slot(self.uid, 'cpu')

    @promise.reject_on_exception
    async def calc(self, session_id, graph_key, ser_graph, targets, callback):
        await asyncio.sleep(self._delay)
        await self.tell_promise(callback)
        await self._dispatch_ref.register_free_slot(self.uid, 'cpu')


class MockSenderActor(WorkerActor):
    def __init__(self, mock_data_list, mode=None):
        super().__init__()
        self._mode = mode or 'in'
        self._mock_data_list = mock_data_list
        self._dispatch_ref = None

    async def post_create(self):
        await super().post_create()
        self._dispatch_ref = self.promise_ref(DispatchActor.default_uid())
        await self._dispatch_ref.register_free_slot(self.uid, 'sender')

    @promise.reject_on_exception
    async def send_data(self, session_id, chunk_keys, target_endpoints, target_slots=None,
                  ensure_cached=True, compression=None, pin_token=None, timeout=None,
                  callback=None):
        if self._mode == 'in':
            self._dispatch_ref.register_free_slot(self.uid, 'sender')
            (await self.storage_client.put_objects(
                session_id, chunk_keys, self._mock_data_list, [DataStorageDevice.SHARED_MEMORY])) \
                .then(lambda *_: self.tell_promise(callback))
        else:
            for chunk_key, mock_data in zip(chunk_keys, self._mock_data_list):
                data = await self._shared_store.get(session_id, chunk_key)
                assert_array_equal(mock_data, data)
            await self.tell_promise(callback, [md.nbytes for md in self._mock_data_list])
            await self._dispatch_ref.register_free_slot(self.uid, 'sender')


class ExecutionTestActor(WorkerActor):
    def __init__(self):
        super().__init__()
        self._results = []
        self._session_id = None
        self._graph_key = None
        self._array_key = None

    async def run_simple_calc(self, session_id):
        self._session_id = session_id

        import mars.tensor as mt
        arr = mt.ones((4,), chunk_size=4) + 1
        graph = arr.build_graph(compose=False, tiled=True)

        arr = get_tiled(arr)
        self._array_key = arr.chunks[0].key

        graph_key = self._graph_key = str(uuid.uuid4())
        execution_ref = self.promise_ref(ExecutionActor.default_uid())
        await execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                          dict(chunks=[arr.chunks[0].key]), None, _tell=True)

        execution_ref.add_finish_callback(session_id, graph_key, _promise=True) \
            .then(lambda *_: self._results.append((True,))) \
            .catch(lambda *exc: self._results.append((False, exc)))

    def get_graph_key(self):
        return self._graph_key

    async def get_results(self):
        if not self._results:
            return None
        if self._results[0][0]:
            return await self._shared_store.get(self._session_id, self._array_key)
        else:
            exc_info = self._results[0][1]
            raise exc_info[1].with_traceback(exc_info[2])


@aio_case
class Test(WorkerCase):
    def tearDown(self):
        super().tearDown()
        logger = logging.getLogger(ExecutionActor.__module__)
        logger.setLevel(logging.WARNING)
        self.rm_spill_dirs()

    @classmethod
    async def create_standard_actors(cls, pool, address, quota_size=None, with_daemon=True,
                                     with_status=True, with_resource=False):
        quota_size = quota_size or (1024 * 1024)

        await pool.create_actor(SchedulerClusterInfoActor, [address],
                                uid=SchedulerClusterInfoActor.default_uid())
        await pool.create_actor(WorkerClusterInfoActor, [address],
                                uid=WorkerClusterInfoActor.default_uid())

        await pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
        await pool.create_actor(StorageManagerActor, uid=StorageManagerActor.default_uid())
        if with_resource:
            await pool.create_actor(ResourceActor, uid=ResourceActor.default_uid())
        if with_daemon:
            await pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
        if with_status:
            await pool.create_actor(StatusActor, address, uid=StatusActor.default_uid())

        await pool.create_actor(
            SharedHolderActor, cls.plasma_storage_size, uid=SharedHolderActor.default_uid())
        await pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())
        await pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())
        await pool.create_actor(QuotaActor, quota_size, uid=MemQuotaActor.default_uid())
        await pool.create_actor(ExecutionActor, uid=ExecutionActor.default_uid())

    @staticmethod
    async def wait_for_result(test_actor):
        check_time = time.time()
        while await test_actor.get_results() is None:
            await asyncio.sleep(1)
            if time.time() - check_time > 10:
                raise SystemError('Timed out')

    async def testSimpleExecution(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        async with create_actor_pool(n_process=1, address=pool_address) as pool:
            await self.create_standard_actors(pool, pool_address, with_daemon=False)
            await pool.create_actor(CpuCalcActor, uid='w:1:calc-a')
            await pool.create_actor(InProcHolderActor)

            import mars.tensor as mt
            from mars.tensor.datasource import TensorOnes
            from mars.tensor.fetch import TensorFetch
            arr = mt.ones((10, 8), chunk_size=10)
            arr_add = mt.ones((10, 8), chunk_size=10)
            arr2 = arr + arr_add
            graph = arr2.build_graph(compose=False, tiled=True)

            arr = get_tiled(arr)
            arr2 = get_tiled(arr2)

            metas = dict()
            for chunk in graph:
                if isinstance(chunk.op, TensorOnes):
                    chunk._op = TensorFetch(
                        dtype=chunk.dtype, _outputs=[weakref.ref(o) for o in chunk.op.outputs],
                        _key=chunk.op.key)
                    metas[chunk.key] = WorkerMeta(chunk.nbytes, chunk.shape, pool_address)

            async with self.run_actor_test(pool) as test_actor:
                session_id = str(uuid.uuid4())

                storage_client = test_actor.storage_client
                await self.waitp(
                    (await storage_client.put_objects(session_id, [arr.chunks[0].key], [np.ones((10, 8), dtype=np.int16)],
                                               [DataStorageDevice.SHARED_MEMORY])),
                )

                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())

                async def _validate(_):
                    data = await test_actor.shared_store.get(session_id, arr2.chunks[0].key)
                    assert_array_equal(data, 2 * np.ones((10, 8)))

                graph_key = str(uuid.uuid4())
                execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                            dict(chunks=[arr2.chunks[0].key]), metas, _promise=True) \
                    .then(_validate) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            await self.get_result()

            async with self.run_actor_test(pool) as test_actor:
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())

                async def _validate(_):
                    data = await test_actor.shared_store.get(session_id, arr2.chunks[0].key)
                    assert_array_equal(data, 2 * np.ones((10, 8)))

                execution_ref.add_finish_callback(session_id, graph_key, _promise=True) \
                    .then(_validate) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            await self.get_result()

    async def testPrepareQuota(self, *_):
        pinned = True

        orig_pin = SharedHolderActor.pin_data_keys

        def _mock_pin(self, session_id, chunk_keys, token):
            from mars.errors import PinDataKeyFailed
            if pinned:
                raise PinDataKeyFailed
            return orig_pin(self, session_id, chunk_keys, token)

        pool_address = '127.0.0.1:%d' % get_next_port()
        session_id = str(uuid.uuid4())
        mock_data = np.array([1, 2, 3, 4])
        async with patch_method(SharedHolderActor.pin_data_keys, new=_mock_pin, async_=True), \
                create_actor_pool(n_process=1, address=pool_address) as pool:
            await self.create_standard_actors(pool, pool_address, with_daemon=False, with_status=False)
            await pool.create_actor(MockSenderActor, [mock_data], 'in', uid='w:mock_sender')
            await pool.create_actor(CpuCalcActor)
            await pool.create_actor(InProcHolderActor)

            import mars.tensor as mt
            from mars.tensor.fetch import TensorFetch
            arr = mt.ones((4,), chunk_size=4)
            arr_add = mt.array(mock_data)
            result_tensor = arr + arr_add
            graph = result_tensor.build_graph(compose=False, tiled=True)

            arr_add = get_tiled(arr_add)
            result_tensor = get_tiled(result_tensor)

            modified_chunk = arr_add.chunks[0]
            arr_add.chunks[0]._op = TensorFetch(
                dtype=modified_chunk.dtype, _outputs=[weakref.ref(o) for o in modified_chunk.op.outputs],
                _key=modified_chunk.op.key)
            metas = {modified_chunk.key: WorkerMeta(
                mock_data.nbytes, mock_data.shape,
                ('0.0.0.0:1234', pool_address.replace('127.0.0.1', 'localhost')))}
            async with self.run_actor_test(pool) as test_actor:
                graph_key = str(uuid.uuid4())
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())

                start_time = time.time()

                await execution_ref.execute_graph(
                    session_id, graph_key, serialize_graph(graph),
                    dict(chunks=[result_tensor.chunks[0].key]), metas, _tell=True)

                execution_ref.add_finish_callback(session_id, graph_key, _promise=True) \
                    .then(lambda *_: test_actor.set_result(time.time())) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

                def _delay_fun():
                    nonlocal pinned
                    time.sleep(0.5)
                    pinned = False

                threading.Thread(target=_delay_fun).start()

            finish_time = await self.get_result()
            self.assertGreaterEqual(finish_time, start_time + 0.5)

    async def testPrepareSpilled(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        session_id = str(uuid.uuid4())
        mock_data = np.array([1, 2, 3, 4])

        options.worker.spill_directory = tempfile.mkdtemp(prefix='mars_worker_prep_spilled-')

        async with create_actor_pool(n_process=1, address=pool_address) as pool:
            await self.create_standard_actors(pool, pool_address, with_daemon=False, with_status=False)
            await pool.create_actor(IORunnerActor)
            await pool.create_actor(CpuCalcActor)
            await pool.create_actor(InProcHolderActor)

            import mars.tensor as mt
            from mars.tensor.fetch import TensorFetch
            arr = mt.ones((4,), chunk_size=4)
            arr_add = mt.array(mock_data)
            result_tensor = arr + arr_add
            graph = result_tensor.build_graph(compose=False, tiled=True)

            arr_add = get_tiled(arr_add)
            result_tensor = get_tiled(result_tensor)

            modified_chunk = arr_add.chunks[0]
            arr_add.chunks[0]._op = TensorFetch(
                dtype=modified_chunk.dtype, _outputs=[weakref.ref(o) for o in modified_chunk.op.outputs],
                _key=modified_chunk.op.key)

            # test meta missing
            async with self.run_actor_test(pool) as test_actor:
                graph_key = str(uuid.uuid4())
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())
                execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                            dict(chunks=[result_tensor.chunks[0].key]), None, _promise=True) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            with self.assertRaises(DependencyMissing):
                await self.get_result()

            metas = {modified_chunk.key: WorkerMeta(
                mock_data.nbytes, mock_data.shape, ('0.0.0.0:1234', pool_address))}

            # test read from spilled file
            async with self.run_actor_test(pool) as test_actor:
                await self.waitp(
                    (await test_actor.storage_client.put_objects(
                        session_id, [modified_chunk.key], [mock_data], [DataStorageDevice.PROC_MEMORY])) \
                        .then(lambda *_: test_actor.storage_client.copy_to(
                            session_id, [modified_chunk.key], [DataStorageDevice.DISK]))
                )
                await test_actor.storage_client.delete(session_id, [modified_chunk.key],
                                                       [DataStorageDevice.PROC_MEMORY])

                async def _validate(_):
                    data = await test_actor.shared_store.get(session_id, result_tensor.chunks[0].key)
                    assert_array_equal(data, mock_data + np.ones((4,)))

                graph_key = str(uuid.uuid4())
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())
                execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                            dict(chunks=[result_tensor.chunks[0].key]), metas, _promise=True) \
                    .then(_validate) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            await self.get_result()

    @patch_method(ResourceActor.allocate_resource, new=lambda *_, **__: True)
    async def testEstimateGraphFinishTime(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        session_id = str(uuid.uuid4())
        async with create_actor_pool(n_process=1, address=pool_address) as pool:
            await self.create_standard_actors(pool, pool_address, with_daemon=False)

            status_ref = pool.actor_ref(StatusActor.default_uid())
            execution_ref = pool.actor_ref(ExecutionActor.default_uid())
            await pool.create_actor(CpuCalcActor)

            import mars.tensor as mt
            arr = mt.ones((10, 8), chunk_size=10)
            graph = arr.build_graph(compose=False, tiled=True)

            arr = get_tiled(arr)

            graph_key = str(uuid.uuid4())

            for _ in range(options.optimize.min_stats_count + 1):
                await status_ref.update_mean_stats(
                    'calc_speed.' + type(arr.chunks[0].op).__name__, 10)
                await status_ref.update_mean_stats('disk_read_speed', 10)
                await status_ref.update_mean_stats('disk_write_speed', 10)
                await status_ref.update_mean_stats('net_transfer_speed', 10)

            await execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                              dict(chunks=[arr.chunks[0].key]), None)
            await execution_ref.estimate_graph_finish_time(session_id, graph_key)

            stats_dict = await status_ref.get_stats(['min_est_finish_time', 'max_est_finish_time'])
            self.assertIsNotNone(stats_dict.get('min_est_finish_time'))
            self.assertIsNotNone(stats_dict.get('max_est_finish_time'))

    async def testCalcProcessFailure(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        session_id = str(uuid.uuid4())
        mock_data = np.array([1, 2, 3, 4])
        async with create_actor_pool(
                n_process=2, address=pool_address, distributor=MarsDistributor(2, 'w:0:')) as pool:
            await self.create_standard_actors(pool, pool_address, with_status=False)

            daemon_ref = pool.actor_ref(WorkerDaemonActor.default_uid())
            dispatch_ref = pool.actor_ref(DispatchActor.default_uid())
            calc_ref = await daemon_ref.create_actor(
                MockCpuCalcActor, session_id, mock_data, 10, uid='w:1:cpu-calc-a')
            await daemon_ref.create_actor(ProcessHelperActor, uid='w:1:proc-helper-a')

            test_actor = await pool.create_actor(ExecutionTestActor, uid='w:0:test_actor')
            await test_actor.run_simple_calc(session_id, _tell=True)

            await asyncio.sleep(2)
            proc_id = pool.distributor.distribute(calc_ref.uid)
            await daemon_ref.kill_actor_process(calc_ref)
            assert not await daemon_ref.is_actor_process_alive(calc_ref)
            await pool.restart_process(proc_id)
            await daemon_ref.handle_process_down([proc_id])

            with self.assertRaises(WorkerProcessStopped):
                await self.wait_for_result(test_actor)
            self.assertEqual(len(await dispatch_ref.get_slots('cpu')), 1)

    async def testStopGraphCalc(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        session_id = str(uuid.uuid4())
        mock_data = np.array([1, 2, 3, 4])
        async with create_actor_pool(
                n_process=2, address=pool_address, distributor=MarsDistributor(2, 'w:0:')) as pool:
            await self.create_standard_actors(pool, pool_address, with_status=False)

            daemon_ref = pool.actor_ref(WorkerDaemonActor.default_uid())
            execution_ref = pool.actor_ref(ExecutionActor.default_uid())

            calc_ref = await daemon_ref.create_actor(
                MockCpuCalcActor, session_id, mock_data, 10, uid='w:1:cpu-calc-a')
            await daemon_ref.create_actor(ProcessHelperActor, uid='w:1:proc-helper-a')

            test_actor = await pool.create_actor(ExecutionTestActor, uid='w:0:test_actor')
            await test_actor.run_simple_calc(session_id, _tell=True)

            await asyncio.sleep(2)
            proc_id = pool.distributor.distribute(calc_ref.uid)
            await execution_ref.stop_execution(session_id, await test_actor.get_graph_key(), _tell=True)
            while await daemon_ref.is_actor_process_alive(calc_ref):
                await asyncio.sleep(0.1)
            await pool.restart_process(proc_id)
            await daemon_ref.handle_process_down([proc_id])

            with self.assertRaises(ExecutionInterrupted):
                await self.wait_for_result(test_actor)

    async def testFetchRemoteData(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        session_id = str(uuid.uuid4())
        mock_data = np.array([1, 2, 3, 4])
        async with create_actor_pool(
                n_process=1, address=pool_address, distributor=MarsDistributor(2, 'w:0:')) as pool:
            await self.create_standard_actors(pool, pool_address, with_daemon=False,
                                              with_status=False, with_resource=True)
            await pool.create_actor(CpuCalcActor)
            await pool.create_actor(InProcHolderActor)
            await pool.create_actor(MockSenderActor, [mock_data], 'in', uid='w:mock_sender')

            import mars.tensor as mt
            from mars.tensor.fetch import TensorFetch
            arr = mt.ones((4,), chunk_size=4)
            arr_add = mt.array(mock_data)
            result_tensor = arr + arr_add
            graph = result_tensor.build_graph(compose=False, tiled=True)

            arr_add = get_tiled(arr_add)
            result_tensor = get_tiled(result_tensor)

            modified_chunk = arr_add.chunks[0]
            arr_add.chunks[0]._op = TensorFetch(
                dtype=modified_chunk.dtype, _outputs=[weakref.ref(o) for o in modified_chunk.op.outputs],
                _key=modified_chunk.op.key)

            async with self.run_actor_test(pool) as test_actor:
                graph_key = str(uuid.uuid4())
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())
                await execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                                  dict(chunks=[result_tensor.chunks[0].key]), None, _tell=True)

                execution_ref.add_finish_callback(session_id, graph_key, _promise=True) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            with self.assertRaises(DependencyMissing):
                await self.get_result()

            metas = {modified_chunk.key: WorkerMeta(mock_data.nbytes, mock_data.shape, ('0.0.0.0:1234',))}
            async with self.run_actor_test(pool) as test_actor:
                graph_key = str(uuid.uuid4())
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())
                await execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                                  dict(chunks=[result_tensor.chunks[0].key]), metas, _tell=True)

                execution_ref.add_finish_callback(session_id, graph_key, _promise=True) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            with self.assertRaises(DependencyMissing):
                await self.get_result()

            metas[modified_chunk.key] = WorkerMeta(
                mock_data.nbytes, mock_data.shape,
                ('0.0.0.0:1234', pool_address.replace('127.0.0.1', 'localhost')))
            async with self.run_actor_test(pool) as test_actor:
                async def _validate(_):
                    data = await test_actor.shared_store.get(session_id, result_tensor.chunks[0].key)
                    assert_array_equal(data, mock_data + np.ones((4,)))

                graph_key = str(uuid.uuid4())
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())
                await execution_ref.execute_graph(
                    session_id, graph_key, serialize_graph(graph),
                    dict(chunks=[result_tensor.chunks[0].key]), metas, _tell=True)

                execution_ref.add_finish_callback(session_id, graph_key, _promise=True) \
                    .then(_validate) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            await self.get_result()

    async def testSendTargets(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        session_id = str(uuid.uuid4())
        mock_data = np.array([1, 2, 3, 4])
        async with create_actor_pool(
                n_process=1, address=pool_address, distributor=MarsDistributor(2, 'w:0:')) as pool:
            await self.create_standard_actors(pool, pool_address, with_daemon=False, with_status=False)
            await pool.create_actor(CpuCalcActor)
            await pool.create_actor(InProcHolderActor)

            import mars.tensor as mt
            arr = mt.ones((4,), chunk_size=4)
            arr_add = mt.array(mock_data)
            result_tensor = arr + arr_add
            graph = result_tensor.build_graph(compose=False, tiled=True)
            result_tensor = get_tiled(result_tensor)
            result_key = result_tensor.chunks[0].key

            await pool.create_actor(MockSenderActor, [mock_data + np.ones((4,))], 'out', uid='w:mock_sender')
            async with self.run_actor_test(pool) as test_actor:
                async def _validate(_):
                    data = await test_actor.shared_store.get(session_id, result_tensor.chunks[0].key)
                    assert_array_equal(data, mock_data + np.ones((4,)))

                graph_key = str(uuid.uuid4())
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())

                await execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                                  dict(chunks=[result_tensor.chunks[0].key]), None, _tell=True)
                await execution_ref.send_data_to_workers(
                    session_id, graph_key, {result_key: (pool_address,)}, _tell=True)

                execution_ref.add_finish_callback(session_id, graph_key, _promise=True) \
                    .then(_validate) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            await self.get_result()

    async def testReExecuteExisting(self):
        pool_address = '127.0.0.1:%d' % get_next_port()
        session_id = str(uuid.uuid4())
        mock_data = np.array([1, 2, 3, 4])
        async with create_actor_pool(
                n_process=1, address=pool_address, distributor=MarsDistributor(2, 'w:0:')) as pool:
            await self.create_standard_actors(pool, pool_address, with_daemon=False, with_status=False)
            await pool.create_actor(CpuCalcActor, uid='w:1:cpu-calc')
            await pool.create_actor(InProcHolderActor, uid='w:1:inproc-holder')

            import mars.tensor as mt
            arr = mt.ones((4,), chunk_size=4)
            arr_add = mt.array(mock_data)
            result_tensor = arr + arr_add
            graph = result_tensor.build_graph(compose=False, tiled=True)

            result_tensor = get_tiled(result_tensor)

            async def _validate(_):
                data = await test_actor.shared_store.get(session_id, result_tensor.chunks[0].key)
                assert_array_equal(data, mock_data + np.ones((4,)))

            async with self.run_actor_test(pool) as test_actor:
                graph_key = str(uuid.uuid4())
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())
                execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                            dict(chunks=[result_tensor.chunks[0].key]), None, _promise=True) \
                    .then(_validate) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            await self.get_result()

            async with self.run_actor_test(pool) as test_actor:
                execution_ref = test_actor.promise_ref(ExecutionActor.default_uid())
                execution_ref.execute_graph(session_id, graph_key, serialize_graph(graph),
                                            dict(chunks=[result_tensor.chunks[0].key]), None, _promise=True) \
                    .then(_validate) \
                    .then(lambda *_: test_actor.set_result(None)) \
                    .catch(lambda *exc: test_actor.set_result(exc, False))

            await self.get_result()
