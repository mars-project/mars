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

import multiprocessing
import unittest
import uuid

import gevent

from mars.actors import create_actor_pool, new_client
from mars.cluster_info import ClusterInfoActor
from mars.scheduler import ChunkMetaActor
from mars.utils import get_next_port


def start_meta_worker(endpoints, idx):
    with create_actor_pool(n_process=1, backend='gevent', address=endpoints[idx]) as pool:
        pool.create_actor(ClusterInfoActor, endpoints, uid=ClusterInfoActor.default_name())
        pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_name())

        while True:
            gevent.sleep(10)


class Test(unittest.TestCase):
    def testChunkMeta(self):
        proc_count = 2
        endpoints = ['127.0.0.1:%d' % get_next_port() for _ in range(proc_count)]
        procs = []

        session_id = str(uuid.uuid4())
        try:
            procs = [
                multiprocessing.Process(target=start_meta_worker, args=(endpoints, idx))
                for idx in range(proc_count)
            ]
            [p.start() for p in procs]
            gevent.sleep(2)

            client = new_client()
            ref1 = client.actor_ref(ChunkMetaActor.default_name(), address=endpoints[0])
            ref2 = client.actor_ref(ChunkMetaActor.default_name(), address=endpoints[1])

            key1 = str(uuid.uuid4())
            ref1.set_chunk_size(session_id, key1, 512)
            key2 = str(uuid.uuid4())
            ref2.set_chunk_size(session_id, key2, 1024)

            self.assertEqual(ref1.get_chunk_size(session_id, key1), 512)
            self.assertEqual(ref2.get_chunk_size(session_id, key2), 1024)
            self.assertEqual(ref1.get_chunk_size(session_id, key2), 1024)
            self.assertEqual(ref2.get_chunk_size(session_id, key1), 512)

            self.assertListEqual(ref1.batch_get_chunk_size(session_id, [key1, key2]), [512, 1024])
            self.assertListEqual(ref2.batch_get_chunk_size(session_id, [key1, key2]), [512, 1024])

            ref1.add_worker(session_id, key1, 'abc')
            ref1.add_worker(session_id, key1, 'def')
            ref2.add_worker(session_id, key2, 'ghi')

            self.assertEqual(sorted(ref1.get_workers(session_id, key1)), sorted(('abc', 'def')))
            self.assertEqual(sorted(ref2.get_workers(session_id, key2)), sorted(('ghi',)))

            batch_result = ref1.batch_get_workers(session_id, [key1, key2])
            self.assertEqual(sorted(batch_result[0]), sorted(('abc', 'def')))
            self.assertEqual(sorted(batch_result[1]), sorted(('ghi',)))

            ref1.delete_meta(session_id, key1)
            self.assertIsNone(ref1.get_workers(session_id, key1))
            self.assertIsNone(ref1.batch_get_chunk_size(session_id, [key1, key2])[0])
            self.assertIsNone(ref1.batch_get_workers(session_id, [key1, key2])[0])

            ref2.batch_delete_meta(session_id, [key1, key2])
            self.assertIsNone(ref1.get_workers(session_id, key2))
            self.assertIsNone(ref1.batch_get_chunk_size(session_id, [key1, key2])[1])
            self.assertIsNone(ref1.batch_get_workers(session_id, [key1, key2])[1])
        finally:
            [p.terminate() for p in procs]
