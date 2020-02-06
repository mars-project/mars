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

import numpy as np
from numpy.testing import assert_allclose

from mars.tests.core import aio_case, create_actor_pool
from mars.errors import StorageDataExists, StorageFull
from mars.utils import get_next_port, to_async_context_manager
from mars.worker.storage import PlasmaKeyMapActor
from mars.worker.storage.sharedstore import PlasmaSharedStore


@aio_case
class Test(unittest.TestCase):
    async def testPlasmaSharedStore(self):
        import pyarrow
        from pyarrow import plasma

        store_size = 10 * 1024 ** 2
        test_addr = '127.0.0.1:%d' % get_next_port()
        async with to_async_context_manager(plasma.start_plasma_store(store_size)) as (sckt, _), \
                create_actor_pool(n_process=1, address=test_addr) as pool:
            km_ref = await pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            try:
                plasma_client = plasma.connect(sckt)
            except TypeError:
                plasma_client = plasma.connect(sckt, '', 0)
            store = PlasmaSharedStore(plasma_client, km_ref)

            self.assertGreater(store.get_actual_capacity(store_size), store_size / 2)

            session_id = str(uuid.uuid4())
            data_list = [np.random.randint(0, 32767, (655360,), np.int16)
                         for _ in range(20)]
            key_list = [str(uuid.uuid4()) for _ in range(20)]

            self.assertFalse(await store.contains(session_id, str(uuid.uuid4())))
            with self.assertRaises(KeyError):
                await store.get(session_id, str(uuid.uuid4()))
            with self.assertRaises(KeyError):
                await store.get_actual_size(session_id, str(uuid.uuid4()))
            with self.assertRaises(KeyError):
                await store.seal(session_id, str(uuid.uuid4()))

            fake_data_key = str(uuid.uuid4())
            await km_ref.put(session_id, fake_data_key, plasma.ObjectID.from_random())
            self.assertFalse(await store.contains(session_id, fake_data_key))
            self.assertIsNone(await km_ref.get(session_id, fake_data_key))
            with self.assertRaises(KeyError):
                await km_ref.put(session_id, fake_data_key, plasma.ObjectID.from_random())
                await store.get(session_id, fake_data_key)
            self.assertIsNone(await km_ref.get(session_id, fake_data_key))
            with self.assertRaises(KeyError):
                await km_ref.put(session_id, fake_data_key, plasma.ObjectID.from_random())
                await store.seal(session_id, fake_data_key)
            self.assertIsNone(await km_ref.get(session_id, fake_data_key))
            with self.assertRaises(KeyError):
                await km_ref.put(session_id, fake_data_key, plasma.ObjectID.from_random())
                await store.get_actual_size(session_id, fake_data_key)
            self.assertIsNone(await km_ref.get(session_id, fake_data_key))
            with self.assertRaises(KeyError):
                await km_ref.put(session_id, fake_data_key, plasma.ObjectID.from_random())
                await store.get_buffer(session_id, fake_data_key)
            self.assertIsNone(await km_ref.get(session_id, fake_data_key))
            await store.delete(session_id, fake_data_key)

            with self.assertRaises(Exception):
                non_serial = type('non_serial', (object,), dict(nbytes=10))
                await store.put(session_id, fake_data_key, non_serial())
            self.assertIsNone(await km_ref.get(session_id, fake_data_key))
            with self.assertRaises(Exception):
                await store.create(session_id, fake_data_key, 'abcd')
            self.assertIsNone(await km_ref.get(session_id, fake_data_key))
            with self.assertRaises(StorageFull):
                await store.create(session_id, fake_data_key, store_size * 2)
            self.assertIsNone(await km_ref.get(session_id, fake_data_key))

            arrow_ser = pyarrow.serialize(data_list[0])
            buf = await store.create(session_id, key_list[0], arrow_ser.total_bytes)
            writer = pyarrow.FixedSizeBufferWriter(buf)
            arrow_ser.write_to(writer)
            writer.close()
            await store.seal(session_id, key_list[0])

            self.assertTrue(await store.contains(session_id, key_list[0]))
            self.assertEqual(await store.get_actual_size(session_id, key_list[0]),
                             arrow_ser.total_bytes)
            assert_allclose(await store.get(session_id, key_list[0]),
                            data_list[0])
            assert_allclose(pyarrow.deserialize(await store.get_buffer(session_id, key_list[0])),
                            data_list[0])

            with self.assertRaises(StorageDataExists):
                await store.create(session_id, key_list[0], arrow_ser.total_bytes)
            self.assertIsNotNone(await km_ref.get(session_id, key_list[0]))
            await store.delete(session_id, key_list[0])
            del buf

            bufs = []
            for key, data in zip(key_list, data_list):
                try:
                    bufs.append(await store.put(session_id, key, data))
                except StorageFull:
                    break
            del bufs
