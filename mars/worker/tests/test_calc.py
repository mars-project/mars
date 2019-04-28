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

import uuid

import numpy as np

from mars.graph import DAG
from mars.utils import get_next_port, serialize_graph
from mars.scheduler import ChunkMetaActor
from mars.worker import WorkerDaemonActor, DispatchActor, StorageManagerActor, \
    CpuCalcActor, IORunnerActor, PlasmaKeyMapActor, SharedHolderActor, \
    InProcHolderActor, QuotaActor, MemQuotaActor
from mars.worker.storage import DataStorageDevice
from mars.worker.tests.base import WorkerCase
from mars.worker.utils import WorkerClusterInfoActor


class Test(WorkerCase):
    def testCpuCalcSingleFetches(self):
        mock_scheduler_addr = '127.0.0.1:%d' % get_next_port()
        with self.create_pool(n_process=1, backend='gevent', address=mock_scheduler_addr) as pool:
            pool.create_actor(WorkerClusterInfoActor, schedulers=[mock_scheduler_addr],
                              uid=WorkerClusterInfoActor.default_uid())
            pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())

            pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())
            pool.create_actor(StorageManagerActor, uid=StorageManagerActor.default_uid())
            pool.create_actor(IORunnerActor)
            quota_ref = pool.create_actor(QuotaActor, 1024 ** 2, uid=MemQuotaActor.default_uid())
            pool.create_actor(SharedHolderActor, uid=SharedHolderActor.default_uid())
            pool.create_actor(InProcHolderActor)
            calc_ref = pool.create_actor(CpuCalcActor, uid=CpuCalcActor.default_uid())

            session_id = str(uuid.uuid4())

            data1 = np.random.random((10, 10))
            data2 = np.random.random((10, 10))

            from mars.tensor.expressions.fetch import TensorFetch
            from mars.tensor.expressions.arithmetic import TensorTreeAdd
            fetch_chunk1 = TensorFetch(to_fetch_key='chunk0', dtype=data1.dtype) \
                .new_chunk([], shape=data1.shape, _key='chunk0')
            fetch_chunk2 = TensorFetch(to_fetch_key='chunk1', dtype=data2.dtype) \
                .new_chunk([], shape=data1.shape, _key='chunk1')
            add_chunk = TensorTreeAdd(data1.dtype) \
                .new_chunk([fetch_chunk1, fetch_chunk2], shape=data1.shape)

            exec_graph = DAG()
            exec_graph.add_node(fetch_chunk1)
            exec_graph.add_node(fetch_chunk2)
            exec_graph.add_node(add_chunk)
            exec_graph.add_edge(fetch_chunk1, add_chunk)
            exec_graph.add_edge(fetch_chunk2, add_chunk)

            with self.run_actor_test(pool) as test_actor:
                storage_client = test_actor.storage_client
                quota_ref_p = test_actor.promise_ref(quota_ref)
                calc_ref_p = test_actor.promise_ref(calc_ref)

                self.waitp(
                    storage_client.put_object(
                        session_id, fetch_chunk1.key, data1, [DataStorageDevice.SHARED_MEMORY]),
                    storage_client.put_object(
                        session_id, fetch_chunk2.key, data2, [DataStorageDevice.SHARED_MEMORY]),
                )
                self.waitp(
                    storage_client.copy_to(session_id, fetch_chunk2.key, [DataStorageDevice.DISK])
                        .then(lambda *_: storage_client.delete(
                            session_id, fetch_chunk2.key, [DataStorageDevice.SHARED_MEMORY]))
                )
                self.assertEqual(list(storage_client.get_data_locations(session_id, fetch_chunk1.key)),
                                 [(0, DataStorageDevice.SHARED_MEMORY)])
                self.assertEqual(list(storage_client.get_data_locations(session_id, fetch_chunk2.key)),
                                 [(0, DataStorageDevice.DISK)])
                self.waitp(
                    quota_ref_p.request_quota((fetch_chunk1.key, session_id, add_chunk.op.key),
                                              data1.nbytes, _promise=True),
                    quota_ref_p.request_quota((fetch_chunk2.key, session_id, add_chunk.op.key),
                                              data2.nbytes, _promise=True),
                )
                self.waitp(
                    calc_ref_p.calc(session_id, add_chunk.op.key, serialize_graph(exec_graph),
                                    [add_chunk.key], _promise=True)
                        .then(lambda *_: calc_ref_p.store_results(session_id, [add_chunk.key], _promise=True))
                )

                quota_dump = quota_ref.dump_data()
                self.assertEqual(len(quota_dump.allocations), 0)
                self.assertEqual(len(quota_dump.requests), 0)
                self.assertEqual(len(quota_dump.proc_sizes), 0)
                self.assertEqual(len(quota_dump.hold_sizes), 0)

                self.assertEqual(sorted(storage_client.get_data_locations(session_id, fetch_chunk1.key)),
                                 [(0, DataStorageDevice.SHARED_MEMORY)])
                self.assertEqual(sorted(storage_client.get_data_locations(session_id, fetch_chunk2.key)),
                                 [(0, DataStorageDevice.SHARED_MEMORY), (0, DataStorageDevice.DISK)])
                self.assertEqual(sorted(storage_client.get_data_locations(session_id, add_chunk.key)),
                                 [(0, DataStorageDevice.SHARED_MEMORY)])
