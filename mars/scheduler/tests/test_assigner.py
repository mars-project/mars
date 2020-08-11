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

import unittest
import uuid

import gevent

from mars.actors import FunctionActor, create_actor_pool
from mars.config import options
from mars.scheduler import ResourceActor, AssignerActor, ChunkMetaClient, \
    ChunkMetaActor, OperandActor
from mars.scheduler.utils import SchedulerClusterInfoActor
from mars.tests.core import patch_method
from mars.worker.tests.base import WorkerCase
from mars.utils import get_next_port


class MockOperandActor(FunctionActor):
    def submit_to_worker(self, worker_ep, _data_sizes):
        self._worker_ep = worker_ep

    def get_worker_ep(self):
        return getattr(self, '_worker_ep', None)


@patch_method(ResourceActor._broadcast_sessions)
class Test(WorkerCase):
    def testAssignerActor(self, *_):
        mock_scheduler_addr = '127.0.0.1:%d' % get_next_port()
        with create_actor_pool(n_process=1, backend='gevent', address=mock_scheduler_addr) as pool:
            cluster_info_ref = pool.create_actor(SchedulerClusterInfoActor, [pool.cluster_info.address],
                                                 uid=SchedulerClusterInfoActor.default_uid())
            resource_ref = pool.create_actor(ResourceActor, uid=ResourceActor.default_uid())
            pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())

            endpoint1 = 'localhost:12345'
            endpoint2 = 'localhost:23456'
            res = dict(hardware=dict(cpu=4, mem_quota=4096))

            resource_ref.set_worker_meta(endpoint1, res)
            resource_ref.set_worker_meta(endpoint2, res)

            assigner_ref = pool.create_actor(AssignerActor, uid=AssignerActor.default_uid())

            session_id = str(uuid.uuid4())
            op_key = str(uuid.uuid4())
            chunk_key1 = str(uuid.uuid4())
            chunk_key2 = str(uuid.uuid4())
            chunk_key3 = str(uuid.uuid4())

            op_info = {
                'op_name': 'test_op',
                'io_meta': dict(input_chunks=[chunk_key1, chunk_key2, chunk_key3]),
                'retries': 0,
                'optimize': {
                    'depth': 0,
                    'demand_depths': (),
                    'successor_size': 1,
                    'descendant_size': 0
                }
            }

            chunk_meta_client = ChunkMetaClient(pool, cluster_info_ref)
            chunk_meta_client.set_chunk_meta(session_id, chunk_key1, size=512, workers=(endpoint1,))
            chunk_meta_client.set_chunk_meta(session_id, chunk_key2, size=512, workers=(endpoint1,))
            chunk_meta_client.set_chunk_meta(session_id, chunk_key3, size=512, workers=(endpoint2,))

            uid = OperandActor.gen_uid(session_id, op_key)
            reply_ref = pool.create_actor(MockOperandActor, uid=uid)
            assigner_ref.apply_for_resource(session_id, op_key, op_info)

            while not reply_ref.get_worker_ep():
                gevent.sleep(0.1)
            self.assertEqual(reply_ref.get_worker_ep(), endpoint1)

            with self.run_actor_test(pool) as test_actor, self.assertRaises(TimeoutError):
                assigner_p_ref = test_actor.promise_ref(assigner_ref)

                try:
                    options.scheduler.assign_timeout = 1
                    res = dict(hardware=dict(cpu=4, mem_quota=0))
                    resource_ref.set_worker_meta(endpoint1, res)
                    resource_ref.set_worker_meta(endpoint2, res)

                    self.waitp(
                        assigner_p_ref.apply_for_resource(session_id, op_key, op_info, _promise=True)
                    )
                finally:
                    options.scheduler.assign_timeout = 600
