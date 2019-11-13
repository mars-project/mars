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

import time
import unittest
import uuid

from mars.actors import FunctionActor
from mars.config import options
from mars.scheduler import ChunkMetaActor, ChunkMetaClient, GraphActor, \
    ResourceActor, SessionManagerActor
from mars.scheduler.utils import SchedulerClusterInfoActor
from mars.tests.core import mock, create_actor_pool
from mars.utils import get_next_port


class MockGraphActor(FunctionActor):
    def __init__(self, *_, **__):
        self._worker_change_args = None

    @staticmethod
    def gen_uid(session_id, graph_key):
        return GraphActor.gen_uid(session_id, graph_key)

    def execute_graph(self):
        pass

    def handle_worker_change(self, *args):
        self._worker_change_args = args

    def get_worker_change_args(self):
        return self._worker_change_args


class Test(unittest.TestCase):
    def tearDown(self):
        options.scheduler.worker_blacklist_time = 3600
        super(Test, self).tearDown()

    def testFailoverMessage(self):
        mock_session_id = str(uuid.uuid4())
        mock_graph_key = str(uuid.uuid4())
        mock_chunk_key = str(uuid.uuid4())
        addr = '127.0.0.1:%d' % get_next_port()
        mock_worker_addr = '127.0.0.1:54132'

        options.scheduler.worker_blacklist_time = 0.5

        with create_actor_pool(n_process=1, backend='gevent', address=addr) as pool:
            cluster_info_ref = pool.create_actor(SchedulerClusterInfoActor, [pool.cluster_info.address],
                                                 uid=SchedulerClusterInfoActor.default_uid())
            session_manager_ref = pool.create_actor(
                SessionManagerActor, uid=SessionManagerActor.default_uid())
            resource_ref = pool.create_actor(ResourceActor, uid=ResourceActor.default_uid())
            pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())

            session_ref = pool.actor_ref(session_manager_ref.create_session(mock_session_id))
            chunk_meta_client = ChunkMetaClient(pool, cluster_info_ref)
            chunk_meta_client.set_chunk_meta(mock_session_id, mock_chunk_key,
                                             size=80, shape=(10,), workers=(mock_worker_addr,))

            with mock.patch(GraphActor.__module__ + '.' + GraphActor.__name__, new=MockGraphActor):
                session_ref.submit_tileable_graph(None, mock_graph_key)
                graph_ref = pool.actor_ref(GraphActor.gen_uid(mock_session_id, mock_graph_key))

                expire_time = time.time() - options.scheduler.status_timeout - 1
                resource_ref.set_worker_meta(mock_worker_addr, dict(update_time=expire_time))

                resource_ref.detect_dead_workers(_tell=True)
                pool.sleep(0.2)

                _, removes, lost_chunks = graph_ref.get_worker_change_args()
                self.assertListEqual(removes, [mock_worker_addr])
                self.assertListEqual(lost_chunks, [mock_chunk_key])

                self.assertNotIn(mock_worker_addr, resource_ref.get_workers_meta())
                resource_ref.set_worker_meta(mock_worker_addr, dict(update_time=time.time()))
                self.assertNotIn(mock_worker_addr, resource_ref.get_workers_meta())

                pool.sleep(0.4)
                resource_ref.set_worker_meta(mock_worker_addr, dict(update_time=time.time()))
                self.assertIn(mock_worker_addr, resource_ref.get_workers_meta())
