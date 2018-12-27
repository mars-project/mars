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

import copy
import sys
import unittest
import uuid

from mars.actors import create_actor_pool, new_client
from mars.cluster_info import ClusterInfoActor
from mars.scheduler.chunkmeta import WorkerMeta, ChunkMetaStore, ChunkMetaCache, \
    ChunkMetaActor, LocalChunkMetaActor
from mars.tests.core import mock
from mars.utils import get_next_port


class Test(unittest.TestCase):
    def testChunkMetaStore(self):
        store = ChunkMetaStore()

        store['c0'] = WorkerMeta(0, ('w0',))
        self.assertIn('c0', store)
        self.assertEqual(store['c0'], WorkerMeta(0, ('w0',)))
        self.assertEqual(store.get('c0'), WorkerMeta(0, ('w0',)))
        self.assertIsNone(store.get('c1'))
        self.assertSetEqual(store.get_worker_chunk_keys('w0'), {'c0'})

        store['c0'] = WorkerMeta(0, ('w1',))
        self.assertEqual(store.get_worker_chunk_keys('w0'), set())
        self.assertSetEqual(store.get_worker_chunk_keys('w1'), {'c0'})

        del store['c0']
        self.assertNotIn('c0', store)

        store['c1'] = WorkerMeta(1, ('w0', 'w1'))
        store['c2'] = WorkerMeta(2, ('w1',))
        store['c3'] = WorkerMeta(3, ('w0',))
        store['c4'] = WorkerMeta(4, ('w0',))
        affected = store.remove_worker_keys('w0', lambda k: k[-1] < '4')
        self.assertListEqual(affected, ['c3'])
        self.assertEqual(store.get('c1'), WorkerMeta(1, ('w1',)))
        self.assertEqual(store.get('c2'), WorkerMeta(2, ('w1',)))
        self.assertSetEqual(store.get_worker_chunk_keys('w0'), {'c4'})
        self.assertSetEqual(store.get_worker_chunk_keys('w1'), {'c1', 'c2'})
        self.assertNotIn('c3', store)
        self.assertIn('c4', store)

        affected = store.remove_worker_keys('w0')
        self.assertListEqual(affected, ['c4'])
        self.assertNotIn('c4', store)
        self.assertIsNone(store.get_worker_chunk_keys('w0'))
        self.assertSetEqual(store.get_worker_chunk_keys('w1'), {'c1', 'c2'})

    def testChunkMetaCache(self):
        cache = ChunkMetaCache(9)

        for idx in range(10):
            cache['c%d' % idx] = WorkerMeta(idx, ('w0',))
        self.assertNotIn('c0', cache)
        self.assertTrue(all('c%d' % idx in cache for idx in range(1, 10)))
        self.assertListEqual(sorted(cache.get_worker_chunk_keys('w0')),
                             ['c%d' % idx for idx in range(1, 10)])

        dup_cache = copy.deepcopy(cache)
        dup_cache.get('c1')
        dup_cache['c10'] = WorkerMeta(10, ('w0',))
        self.assertIsNone(dup_cache.get('c0'))
        self.assertNotIn('c2', dup_cache)
        self.assertIn('c1', dup_cache)
        self.assertTrue(all('c%d' % idx in dup_cache for idx in range(3, 11)))

        dup_cache = copy.deepcopy(cache)
        _ = dup_cache['c1']
        dup_cache['c10'] = WorkerMeta(10, ('w0',))
        self.assertNotIn('c2', dup_cache)
        self.assertIn('c1', dup_cache)
        self.assertTrue(all('c%d' % idx in dup_cache for idx in range(3, 11)))

        dup_cache = copy.deepcopy(cache)
        dup_cache['c1'] = WorkerMeta(1, ('w0',))
        dup_cache['c10'] = WorkerMeta(10, ('w0',))
        self.assertNotIn('c2', dup_cache)
        self.assertIn('c1', dup_cache)
        self.assertTrue(all('c%d' % idx in dup_cache for idx in range(3, 11)))

    @unittest.skipIf(sys.platform == 'win32', 'Currently not support multiple pools under Windows')
    @mock.patch(ChunkMetaActor.__module__ + '.ChunkMetaActor.get_scheduler')
    def testChunkMetaActors(self, *_):
        proc_count = 2
        endpoints = ['127.0.0.1:%d' % get_next_port() for _ in range(proc_count)]
        keys = []

        def _mock_get_scheduler(key):
            return endpoints[keys.index(key[1]) % len(endpoints)]

        ChunkMetaActor.get_scheduler.side_effect = _mock_get_scheduler

        session1 = str(uuid.uuid4())
        session2 = str(uuid.uuid4())
        with create_actor_pool(n_process=1, backend='gevent', address=endpoints[0]) as pool1:
            pool1.create_actor(ClusterInfoActor, endpoints, uid=ClusterInfoActor.default_name())
            pool1.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_name())

            with create_actor_pool(n_process=1, backend='gevent', address=endpoints[1]) as pool2:
                pool2.create_actor(ClusterInfoActor, endpoints, uid=ClusterInfoActor.default_name())
                pool2.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_name())

                client = new_client()
                ref1 = client.actor_ref(ChunkMetaActor.default_name(), address=endpoints[0])
                ref2 = client.actor_ref(ChunkMetaActor.default_name(), address=endpoints[1])

                loc_ref1 = client.actor_ref(LocalChunkMetaActor.default_name(), address=endpoints[0])
                loc_ref2 = client.actor_ref(LocalChunkMetaActor.default_name(), address=endpoints[1])

                key1 = str(uuid.uuid4())
                key2 = str(uuid.uuid4())
                key3 = str(uuid.uuid4())
                keys = [key1, key2, key3]
                ref1.set_chunk_size(session1, key1, 512)
                ref2.set_chunk_size(session1, key2, 1024)
                ref2.set_chunk_size(session2, key3, 1024)

                self.assertEqual(ref1.get_chunk_size(session1, key1), 512)
                self.assertEqual(ref2.get_chunk_size(session1, key2), 1024)
                self.assertEqual(ref1.get_chunk_size(session1, key2), 1024)
                self.assertEqual(ref2.get_chunk_size(session1, key1), 512)

                self.assertListEqual(ref1.batch_get_chunk_size(session1, [key1, key2]), [512, 1024])
                self.assertListEqual(ref2.batch_get_chunk_size(session1, [key1, key2]), [512, 1024])

                ref1.add_worker(session1, key1, 'abc')
                ref1.add_worker(session1, key1, 'def')
                ref2.add_worker(session1, key2, 'ghi')

                ref1.add_worker(session2, key3, 'ghi')

                self.assertEqual(sorted(ref1.get_workers(session1, key1)), sorted(('abc', 'def')))
                self.assertEqual(sorted(ref2.get_workers(session1, key2)), sorted(('ghi',)))

                batch_result = ref1.batch_get_workers(session1, [key1, key2])
                self.assertEqual(sorted(batch_result[0]), sorted(('abc', 'def')))
                self.assertEqual(sorted(batch_result[1]), sorted(('ghi',)))

                affected = []
                for loc_ref in (loc_ref1, loc_ref2):
                    affected.extend(loc_ref.remove_workers_in_session(session2, ['ghi']))
                self.assertEqual(affected, [key3])
                self.assertEqual(sorted(ref1.get_workers(session1, key2)), sorted(('ghi',)))
                self.assertIsNone(ref1.get_workers(session2, key3))

                ref1.delete_meta(session1, key1)
                self.assertIsNone(ref1.get_workers(session1, key1))
                self.assertIsNone(ref1.batch_get_chunk_size(session1, [key1, key2])[0])
                self.assertIsNone(ref1.batch_get_workers(session1, [key1, key2])[0])

                ref2.batch_delete_meta(session1, [key1, key2])
                self.assertIsNone(ref1.get_workers(session1, key2))
                self.assertIsNone(ref1.batch_get_chunk_size(session1, [key1, key2])[1])
                self.assertIsNone(ref1.batch_get_workers(session1, [key1, key2])[1])

    @unittest.skipIf(sys.platform == 'win32', 'Currently not support multiple pools under Windows')
    @mock.patch(ChunkMetaActor.__module__ + '.ChunkMetaActor.get_scheduler')
    def testChunkBroadcast(self, *_):
        proc_count = 2
        endpoints = ['127.0.0.1:%d' % get_next_port() for _ in range(proc_count)]
        keys = []

        def _mock_get_scheduler(key):
            return endpoints[keys.index(key[1]) % len(endpoints)]

        ChunkMetaActor.get_scheduler.side_effect = _mock_get_scheduler

        session_id = str(uuid.uuid4())
        with create_actor_pool(n_process=1, backend='gevent', address=endpoints[0]) as pool1:
            pool1.create_actor(ClusterInfoActor, endpoints, uid=ClusterInfoActor.default_name())
            pool1.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_name())

            with create_actor_pool(n_process=1, backend='gevent', address=endpoints[1]) as pool2:
                pool2.create_actor(ClusterInfoActor, endpoints, uid=ClusterInfoActor.default_name())
                pool2.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_name())

                client = new_client()
                ref1 = client.actor_ref(ChunkMetaActor.default_name(), address=endpoints[0])
                local_ref1 = client.actor_ref(LocalChunkMetaActor.default_name(), address=endpoints[0])
                local_ref2 = client.actor_ref(LocalChunkMetaActor.default_name(), address=endpoints[1])

                key1 = str(uuid.uuid4())
                key2 = str(uuid.uuid4())
                keys = [key1, key2]

                ref1.set_chunk_broadcasts(session_id, key1, [endpoints[1]])
                ref1.set_chunk_size(session_id, key1, 512)
                pool2.sleep(0.1)

                self.assertEqual(local_ref1.get_chunk_meta(session_id, key1).chunk_size, 512)
                self.assertEqual(local_ref2.get_chunk_meta(session_id, key1).chunk_size, 512)

                ref1.delete_meta(session_id, key1)
                pool2.sleep(0.1)

                self.assertIsNone(local_ref1.get_chunk_meta(session_id, key1))
                self.assertIsNone(local_ref2.get_chunk_meta(session_id, key1))
