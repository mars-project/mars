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
import uuid
import weakref

import numpy as np

from mars.errors import StorageFull
from mars.graph import DAG
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
        mock_addr = '127.0.0.1:%d' % get_next_port()
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
            chunk_key = 'chunk-%d' % idx
            fetch_chunk = TensorFetch(to_fetch_key=chunk_key, dtype=d.dtype) \
                .new_chunk([], shape=d.shape, _key=chunk_key)
            inputs.append(fetch_chunk)
        add_chunk = TensorTreeAdd(data_list[0].dtype).new_chunk(inputs, shape=data_list[0].shape)

        exec_graph = DAG()
        exec_graph.add_node(add_chunk)
        for input_chunk in inputs:
            exec_graph.add_node(input_chunk)
            exec_graph.add_edge(input_chunk, add_chunk)
        return exec_graph, inputs, add_chunk

    def testCpuCalcSingleFetches(self):
        with self._start_calc_pool() as (_pool, test_actor):
            quota_ref = test_actor.promise_ref(MemQuotaActor.default_uid())
            calc_ref = test_actor.promise_ref(CpuCalcActor.default_uid())

            session_id = str(uuid.uuid4())
            data_list = [np.random.random((10, 10)) for _ in range(3)]
            exec_graph, fetch_chunks, add_chunk = self._build_test_graph(data_list)

            storage_client = test_actor.storage_client

            for fetch_chunk, d in zip(fetch_chunks, data_list):
                self.waitp(
                    storage_client.put_object(
                        session_id, fetch_chunk.key, d, [DataStorageDevice.SHARED_MEMORY]),
                )
            self.assertEqual(list(storage_client.get_data_locations(session_id, fetch_chunks[0].key)),
                             [(0, DataStorageDevice.SHARED_MEMORY)])

            quota_batch = {
                build_quota_key(session_id, add_chunk.key, add_chunk.op.key): data_list[0].nbytes,
            }

            for idx in [1, 2]:
                quota_batch[build_quota_key(session_id, fetch_chunks[idx].key, add_chunk.op.key)] \
                    = data_list[idx].nbytes

                self.waitp(
                    storage_client.copy_to(session_id, fetch_chunks[idx].key, [DataStorageDevice.DISK])
                        .then(lambda *_: storage_client.delete(
                            session_id, fetch_chunks[idx].key, [DataStorageDevice.SHARED_MEMORY]))
                )
                self.assertEqual(
                    list(storage_client.get_data_locations(session_id, fetch_chunks[idx].key)),
                    [(0, DataStorageDevice.DISK)])

            self.waitp(
                quota_ref.request_batch_quota(quota_batch, _promise=True),
            )

            o_create = PlasmaSharedStore.create

            def _mock_plasma_create(store, session_id, data_key, size):
                if data_key == fetch_chunks[2].key:
                    raise StorageFull
                return o_create(store, session_id, data_key, size)

            ref_store = []

            def _extract_value_ref(*_):
                inproc_handler = storage_client.get_storage_handler(DataStorageDevice.PROC_MEMORY)
                obj = inproc_handler.get_object(session_id, add_chunk.key)
                ref_store.append(weakref.ref(obj))
                del obj

            with patch_method(PlasmaSharedStore.create, _mock_plasma_create):
                self.waitp(
                    calc_ref.calc(session_id, add_chunk.op.key, serialize_graph(exec_graph),
                                  [add_chunk.key], _promise=True)
                        .then(_extract_value_ref)
                        .then(lambda *_: calc_ref.store_results(session_id, [add_chunk.key], _promise=True))
                )

            self.assertIsNone(ref_store[-1]())

            quota_dump = quota_ref.dump_data()
            self.assertEqual(len(quota_dump.allocations), 0)
            self.assertEqual(len(quota_dump.requests), 0)
            self.assertEqual(len(quota_dump.proc_sizes), 0)
            self.assertEqual(len(quota_dump.hold_sizes), 0)

            self.assertEqual(sorted(storage_client.get_data_locations(session_id, fetch_chunks[0].key)),
                             [(0, DataStorageDevice.SHARED_MEMORY)])
            self.assertEqual(sorted(storage_client.get_data_locations(session_id, fetch_chunks[1].key)),
                             [(0, DataStorageDevice.SHARED_MEMORY), (0, DataStorageDevice.DISK)])
            self.assertEqual(sorted(storage_client.get_data_locations(session_id, fetch_chunks[2].key)),
                             [(0, DataStorageDevice.DISK)])
            self.assertEqual(sorted(storage_client.get_data_locations(session_id, add_chunk.key)),
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
                    storage_client.put_object(
                        session_id, fetch_chunk.key, d, [DataStorageDevice.SHARED_MEMORY]),
                )

            def _mock_calc_results_error(*_, **__):
                raise ValueError

            with patch_method(CpuCalcActor._calc_results, _mock_calc_results_error), \
                    self.assertRaises(ValueError):
                self.waitp(
                    calc_ref.calc(session_id, add_chunk.op.key, serialize_graph(exec_graph),
                                  [add_chunk.key], _promise=True)
                        .then(lambda *_: calc_ref.store_results(session_id, [add_chunk.key], _promise=True))
                )
