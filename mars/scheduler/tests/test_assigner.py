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

from mars.cluster_info import ClusterInfoActor
from mars.scheduler import ResourceActor, AssignerActor, KVStoreActor
from mars.actors import FunctionActor, create_actor_pool


class PromiseReplyTestActor(FunctionActor):
    def __init__(self):
        super(PromiseReplyTestActor, self).__init__()
        self._replied = False
        self._value = None

    def reply(self, value):
        self._replied = True
        self._value = value

    def get_reply(self):
        if not self._replied:
            return None
        return self._replied, self._value


class Test(unittest.TestCase):

    def testAssignerActor(self):
        with create_actor_pool(backend='gevent') as pool:
            pool.create_actor(ClusterInfoActor, [pool.cluster_info.address],
                              uid=ClusterInfoActor.default_name())
            resource_ref = pool.create_actor(ResourceActor, uid=ResourceActor.default_name())
            kv_store_ref = pool.create_actor(KVStoreActor, uid=KVStoreActor.default_name())

            endpoint1 = 'localhost:12345'
            endpoint2 = 'localhost:23456'
            res = dict(hardware=dict(cpu=4, memory=4096))

            def write_mock_meta():
                resource_ref.set_worker_meta(endpoint1, res)
                resource_ref.set_worker_meta(endpoint2, res)

            g = gevent.spawn(write_mock_meta)
            g.join()

            assigner_ref = pool.create_actor(AssignerActor, uid='AssignerActor')

            session_id = str(uuid.uuid4())
            op_key = str(uuid.uuid4())
            chunk_key1 = str(uuid.uuid4())
            chunk_key2 = str(uuid.uuid4())
            chunk_key3 = str(uuid.uuid4())

            op_info = {
                'op_name': 'test_op',
                'io_meta': dict(input_chunks=[chunk_key1, chunk_key2, chunk_key3]),
                'output_size': 512,
                'retries': 0,
                'optimize': {
                    'depth': 0,
                    'demand_depths': (),
                    'successor_size': 1,
                    'descendant_size': 0
                }
            }

            kv_store_ref.write('/sessions/%s/chunks/%s/data_size' % (session_id, chunk_key1), 512)
            kv_store_ref.write('/sessions/%s/chunks/%s/workers/%s' % (session_id, chunk_key1, endpoint1), '')

            kv_store_ref.write('/sessions/%s/chunks/%s/data_size' % (session_id, chunk_key2), 512)
            kv_store_ref.write('/sessions/%s/chunks/%s/workers/%s'
                               % (session_id, chunk_key2, endpoint1), '')

            kv_store_ref.write('/sessions/%s/chunks/%s/data_size' % (session_id, chunk_key3), 512)
            kv_store_ref.write('/sessions/%s/chunks/%s/workers/%s'
                               % (session_id, chunk_key3, endpoint2), '')

            reply_ref = pool.create_actor(PromiseReplyTestActor)
            reply_callback = ((reply_ref.uid, reply_ref.address), 'reply')
            assigner_ref.apply_for_resource(session_id, op_key, op_info, callback=reply_callback)

            while not reply_ref.get_reply():
                gevent.sleep(0.1)
            _, ret_value = reply_ref.get_reply()
            self.assertEqual(ret_value, endpoint1)
