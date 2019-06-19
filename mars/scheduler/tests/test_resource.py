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

import unittest
import uuid

import gevent

from mars.scheduler import ResourceActor
from mars.actors import create_actor_pool


class Test(unittest.TestCase):
    def testResourceActor(self):
        session_id = str(uuid.uuid4())
        with create_actor_pool(n_process=1, backend='gevent') as pool:
            resource_ref = pool.create_actor(ResourceActor, uid=ResourceActor.default_uid())
            mock_resource = dict(hardware=dict(cpu=4, memory=512))

            ep1 = 'localhost:12345'
            ep2 = 'localhost:23456'

            def write_mock_meta():
                resource_ref.set_worker_meta(ep1, mock_resource)
                resource_ref.set_worker_meta(ep2, mock_resource)
                return resource_ref.get_workers_meta()

            g = gevent.spawn(write_mock_meta)
            g.join()
            self.assertEqual({ep1: mock_resource, ep2: mock_resource}, g.value)

            key1 = str(uuid.uuid4())
            self.assertFalse(resource_ref.allocate_resource(session_id, key1, ep1, dict(cpu=5, memory=256)))
            key2 = str(uuid.uuid4())
            self.assertTrue(resource_ref.allocate_resource(session_id, key2, ep1, dict(cpu=2, memory=256)))
            key3 = str(uuid.uuid4())
            self.assertFalse(resource_ref.allocate_resource(session_id, key3, ep1, dict(cpu=2, memory=260)))
            key4 = str(uuid.uuid4())
            self.assertTrue(resource_ref.allocate_resource(session_id, key4, ep1, dict(cpu=2, memory=256)))
            key5 = str(uuid.uuid4())
            self.assertFalse(resource_ref.allocate_resource(session_id, key5, ep1, dict(cpu=2, memory=256)))
            resource_ref.deallocate_resource(session_id, key4, ep1)
            key6 = str(uuid.uuid4())
            self.assertTrue(resource_ref.allocate_resource(session_id, key6, ep1, dict(cpu=2, memory=256)))
            resource_ref.deallocate_resource(session_id, key6, ep1)
