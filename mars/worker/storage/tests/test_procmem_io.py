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

import uuid
import weakref

import numpy as np
from numpy.testing import assert_allclose

from mars.serialize import dataserializer
from mars.tests.core import aio_case
from mars.utils import get_next_port
from mars.worker import DispatchActor, WorkerDaemonActor, QuotaActor, MemQuotaActor
from mars.worker.tests.base import WorkerCase
from mars.worker.storage import StorageManagerActor, InProcHolderActor, IORunnerActor, \
    PlasmaKeyMapActor, SharedHolderActor, DataStorageDevice


@aio_case
class Test(WorkerCase):
    async def testProcMemPutAndGet(self):
        test_addr = '127.0.0.1:%d' % get_next_port()
        async with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
            await pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = await pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())
            await pool.create_actor(QuotaActor, 1024 ** 2, uid=MemQuotaActor.default_uid())
            await pool.create_actor(InProcHolderActor)

            data1 = np.random.random((10, 10))
            data2 = np.random.random((10, 10))
            ser_data2 = dataserializer.serialize(data2)
            bytes_data2 = ser_data2.to_buffer()

            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())
            data_key2 = str(uuid.uuid4())

            storage_client = test_actor.storage_client
            handler = await storage_client.get_storage_handler((0, DataStorageDevice.PROC_MEMORY))

            await handler.put_objects(session_id, [data_key1], [data1])
            self.assertEqual(sorted((await storage_manager_ref.get_data_locations(session_id, [data_key1]))[0]),
                             [(0, DataStorageDevice.PROC_MEMORY)])
            assert_allclose(data1, (await handler.get_objects(session_id, [data_key1]))[0])

            await handler.delete(session_id, [data_key1])
            self.assertEqual(list((await storage_manager_ref.get_data_locations(session_id, [data_key1]))[0]), [])
            with self.assertRaises(KeyError):
                await handler.get_objects(session_id, [data_key1])

            await handler.put_objects(session_id, [data_key2], [ser_data2], serialize=True)
            assert_allclose(data2, (await handler.get_objects(session_id, [data_key2]))[0])
            await handler.delete(session_id, [data_key2])

            await handler.put_objects(session_id, [data_key2], [bytes_data2], serialize=True)
            assert_allclose(data2, (await handler.get_objects(session_id, [data_key2]))[0])
            await handler.delete(session_id, [data_key2])

    async def testProcMemLoad(self):
        test_addr = '127.0.0.1:%d' % get_next_port()
        async with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
            await pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())
            await pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = await pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            await pool.create_actor(QuotaActor, 1024 ** 2, uid=MemQuotaActor.default_uid())
            await pool.create_actor(InProcHolderActor)
            await pool.create_actor(IORunnerActor)

            await pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            await pool.create_actor(SharedHolderActor, uid=SharedHolderActor.default_uid())

            data1 = np.random.random((10, 10))
            data2 = np.random.random((10, 10))
            ser_data1 = dataserializer.serialize(data1)

            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())
            data_key2 = str(uuid.uuid4())

            storage_client = test_actor.storage_client
            handler = await storage_client.get_storage_handler((0, DataStorageDevice.PROC_MEMORY))

            # load from bytes io
            disk_handler = await storage_client.get_storage_handler((0, DataStorageDevice.DISK))
            async with await disk_handler.create_bytes_writer(
                    session_id, data_key1, ser_data1.total_bytes) as writer:
                ser_data1.write_to(writer)

            await self.waitp((await handler.load_from_bytes_io(session_id, [data_key1], disk_handler))
                             .then(lambda: None))
            self.assertEqual(sorted((await storage_manager_ref.get_data_locations(session_id, [data_key1]))[0]),
                             [(0, DataStorageDevice.PROC_MEMORY), (0, DataStorageDevice.DISK)])

            await disk_handler.delete(session_id, [data_key1])

            data_load = (await handler.get_objects(session_id, [data_key1]))[0]
            ref_data = weakref.ref(data_load)
            del data_load
            await handler.delete(session_id, [data_key1])
            self.assertIsNone(ref_data())

            # load from object io
            shared_handler = await storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))
            await shared_handler.put_objects(session_id, [data_key2], [data2])

            await self.waitp((await handler.load_from_object_io(session_id, [data_key2], shared_handler))
                             .then(lambda *_: None))
            self.assertEqual(sorted((await storage_manager_ref.get_data_locations(session_id, [data_key2]))[0]),
                             [(0, DataStorageDevice.PROC_MEMORY), (0, DataStorageDevice.SHARED_MEMORY)])

            await shared_handler.delete(session_id, [data_key2])

            data_load = (await handler.get_objects(session_id, [data_key2]))[0]
            ref_data = weakref.ref(data_load)
            del data_load
            await handler.delete(session_id, [data_key2])
            self.assertIsNone(ref_data())
