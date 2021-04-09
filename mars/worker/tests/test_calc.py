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

import contextlib
import random
import uuid

import numpy as np

from mars.errors import StorageFull
from mars.core import ChunkGraph
from mars.utils import get_next_port, serialize_graph
from mars.scheduler import ChunkMetaActor
from mars.scheduler.utils import SchedulerClusterInfoActor
from mars.tests.core import patch_method
from mars.worker import WorkerDaemonActor, DispatchActor, StorageManagerActor, \
    CpuCalcActor, IORunnerActor, PlasmaKeyMapActor, SharedHolderActor, \
    InProcHolderActor, QuotaActor, MemQuotaActor, StatusActor
from mars.worker.storage import DataStorageDevice
from mars.worker.storage.sharedstore import PlasmaSharedStore
from mars.worker.tests.base import WorkerCase
from mars.worker.utils import build_quota_key, WorkerClusterInfoActor


class Test(WorkerCase):
    @contextlib.contextmanager
    def _start_calc_pool(self):
        mock_addr = f'127.0.0.1:{get_next_port()}'
        with self.create_pool(n_process=1, backend='gevent', address=mock_addr) as pool:
            pool.create_actor(SchedulerClusterInfoActor, [mock_addr],
                              uid=SchedulerClusterInfoActor.default_uid())
            pool.create_actor(WorkerClusterInfoActor, [mock_addr],
                              uid=WorkerClusterInfoActor.default_uid())

            pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())
            pool.create_actor(StatusActor, mock_addr, uid=StatusActor.default_uid())

            pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())
            pool.create_actor(StorageManagerActor, uid=StorageManagerActor.default_uid())
            pool.create_actor(IORunnerActor)
            pool.create_actor(QuotaActor, 1024 ** 2, uid=MemQuotaActor.default_uid())
            shared_holder_ref = pool.create_actor(
                SharedHolderActor, uid=SharedHolderActor.default_uid())
            pool.create_actor(InProcHolderActor)
            pool.create_actor(CpuCalcActor, uid=CpuCalcActor.default_uid())

            with self.run_actor_test(pool) as test_actor:
                try:
                    yield pool, test_actor
                finally:
                    shared_holder_ref.destroy()

    @staticmethod
    def _build_test_graph(data_list):
        from mars.tensor.fetch import TensorFetch
        from mars.tensor.arithmetic import TensorTreeAdd

        inputs = []
        for idx, d in enumerate(data_list):
            chunk_key = f'chunk-{random.randint(0, 999)}-{idx}'
            fetch_chunk = TensorFetch(source_key=chunk_key, dtype=d.dtype) \
                .new_chunk([], shape=d.shape, _key=chunk_key)
            inputs.append(fetch_chunk)
        add_chunk = TensorTreeAdd(args=inputs, dtype=data_list[0].dtype) \
            .new_chunk(inputs, shape=data_list[0].shape)

        exec_graph = ChunkGraph([add_chunk.data])
        exec_graph.add_node(add_chunk.data)
        for input_chunk in inputs:
            exec_graph.add_node(input_chunk.data)
            exec_graph.add_edge(input_chunk.data, add_chunk.data)
        return exec_graph, inputs, add_chunk

    def testCpuCalcSingleFetches(self):
        import gc
        with self._start_calc_pool() as (_pool, test_actor):
            quota_ref = test_actor.promise_ref(MemQuotaActor.default_uid())
            calc_ref = test_actor.promise_ref(CpuCalcActor.default_uid())

            session_id = str(uuid.uuid4())
            data_list = [np.random.random((10, 10)) for _ in range(3)]
            exec_graph, fetch_chunks, add_chunk = self._build_test_graph(data_list)

            storage_client = test_actor.storage_client

            for fetch_chunk, d in zip(fetch_chunks, data_list):
                self.waitp(
                    storage_client.put_objects(
                        session_id, [fetch_chunk.key], [d], [DataStorageDevice.SHARED_MEMORY]),
                )
            self.assertEqual(list(storage_client.get_data_locations(session_id, [fetch_chunks[0].key])[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])

            quota_batch = {
                build_quota_key(session_id, add_chunk.key, add_chunk.op.key): data_list[0].nbytes,
            }

            for idx in [1, 2]:
                quota_batch[build_quota_key(session_id, fetch_chunks[idx].key, add_chunk.op.key)] \
                    = data_list[idx].nbytes

                self.waitp(
                    storage_client.copy_to(session_id, [fetch_chunks[idx].key], [DataStorageDevice.DISK])
                        .then(lambda *_: storage_client.delete(
                            session_id, [fetch_chunks[idx].key], [DataStorageDevice.SHARED_MEMORY]))
                )
                self.assertEqual(
                    list(storage_client.get_data_locations(session_id, [fetch_chunks[idx].key])[0]),
                    [(0, DataStorageDevice.DISK)])

            self.waitp(
                quota_ref.request_batch_quota(quota_batch, _promise=True),
            )

            o_create = PlasmaSharedStore.create

            def _mock_plasma_create(store, session_id, data_key, size):
                if data_key == fetch_chunks[2].key:
                    raise StorageFull
                return o_create(store, session_id, data_key, size)

            id_type_set = set()

            def _extract_value_ref(*_):
                inproc_handler = storage_client.get_storage_handler((0, DataStorageDevice.PROC_MEMORY))
                obj = inproc_handler.get_objects(session_id, [add_chunk.key])[0]
                id_type_set.add((id(obj), type(obj)))
                del obj

            with patch_method(PlasmaSharedStore.create, _mock_plasma_create):
                self.waitp(
                    calc_ref.calc(session_id, add_chunk.op.key, serialize_graph(exec_graph),
                                  [add_chunk.key], _promise=True)
                        .then(_extract_value_ref)
                        .then(lambda *_: calc_ref.store_results(
                            session_id, add_chunk.op.key, [add_chunk.key], None, _promise=True))
                )

            self.assertTrue(all((id(obj), type(obj)) not in id_type_set
                                for obj in gc.get_objects()))

            self.assertEqual(sorted(storage_client.get_data_locations(session_id, [fetch_chunks[0].key])[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])
            self.assertEqual(sorted(storage_client.get_data_locations(session_id, [fetch_chunks[1].key])[0]),
                             [(0, DataStorageDevice.DISK)])
            self.assertEqual(sorted(storage_client.get_data_locations(session_id, [fetch_chunks[2].key])[0]),
                             [(0, DataStorageDevice.DISK)])
            self.assertEqual(sorted(storage_client.get_data_locations(session_id, [add_chunk.key])[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])

    def testCpuCalcErrorInRunning(self):
        with self._start_calc_pool() as (_pool, test_actor):
            calc_ref = test_actor.promise_ref(CpuCalcActor.default_uid())

            session_id = str(uuid.uuid4())
            data_list = [np.random.random((10, 10)) for _ in range(2)]
            exec_graph, fetch_chunks, add_chunk = self._build_test_graph(data_list)

            storage_client = test_actor.storage_client

            for fetch_chunk, d in zip(fetch_chunks, data_list):
                self.waitp(
                    storage_client.put_objects(
                        session_id, [fetch_chunk.key], [d], [DataStorageDevice.SHARED_MEMORY]),
                )

            def _mock_calc_results_error(*_, **__):
                raise ValueError

            with patch_method(CpuCalcActor._calc_results, _mock_calc_results_error), \
                    self.assertRaises(ValueError):
                self.waitp(
                    calc_ref.calc(session_id, add_chunk.op.key, serialize_graph(exec_graph),
                                  [add_chunk.key], _promise=True)
                        .then(lambda *_: calc_ref.store_results(
                            session_id, add_chunk.op.key, [add_chunk.key], None, _promise=True))
                )

    def testDestroyCalcActor(self):
        import gevent.event

        with self._start_calc_pool() as (_pool, test_actor):
            calc_ref = _pool.actor_ref(CpuCalcActor.default_uid())
            calc_ref.mark_destroy()
            gevent.sleep(0.8)
            self.assertFalse(_pool.has_actor(calc_ref))

        with self._start_calc_pool() as (_pool, test_actor):
            calc_ref = test_actor.promise_ref(CpuCalcActor.default_uid())

            session_id = str(uuid.uuid4())
            data_list = [np.random.random((10, 10)) for _ in range(2)]
            exec_graph, fetch_chunks, add_chunk = self._build_test_graph(data_list)
            exec_graph2, fetch_chunks2, add_chunk2 = self._build_test_graph(data_list[::-1])

            storage_client = test_actor.storage_client

            for fetch_chunk, d in zip(fetch_chunks, data_list):
                self.waitp(
                    storage_client.put_objects(
                        session_id, [fetch_chunk.key], [d], [DataStorageDevice.SHARED_MEMORY]),
                )
            for fetch_chunk2, d in zip(fetch_chunks2, data_list[::-1]):
                self.waitp(
                    storage_client.put_objects(
                        session_id, [fetch_chunk2.key], [d], [DataStorageDevice.SHARED_MEMORY]),
                )

            orig_calc_results = CpuCalcActor._calc_results

            start_event = gevent.event.Event()

            def _mock_calc_delayed(actor_obj, *args, **kwargs):
                start_event.set()
                gevent.sleep(1)
                return orig_calc_results(actor_obj, *args, **kwargs)

            with patch_method(CpuCalcActor._calc_results, _mock_calc_delayed):
                p = calc_ref.calc(session_id, add_chunk.op.key, serialize_graph(exec_graph),
                                  [add_chunk.key], _promise=True) \
                    .then(lambda *_: calc_ref.store_results(
                        session_id, add_chunk.op.key, [add_chunk.key], None, _promise=True))
                start_event.wait()
                calc_ref.mark_destroy()

                p2 = calc_ref.calc(session_id, add_chunk2.op.key, serialize_graph(exec_graph2),
                                   [add_chunk2.key], _promise=True) \
                    .then(lambda *_: calc_ref.store_results(
                        session_id, add_chunk2.op.key, [add_chunk2.key], None, _promise=True))

                self.assertTrue(_pool.has_actor(calc_ref._ref))
                self.waitp(p)
                self.waitp(p2)

            gevent.sleep(0.8)
            self.assertFalse(_pool.has_actor(calc_ref._ref))
