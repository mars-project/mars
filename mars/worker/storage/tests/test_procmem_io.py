# Copyright 1999-2019 Alibaba Group Holding Ltd.
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
from numpy.testing import assert_allclose

from mars.serialize import dataserializer
from mars.utils import get_next_port
from mars.worker import WorkerDaemonActor, QuotaActor, MemQuotaActor
from mars.worker.tests.base import WorkerCase
from mars.worker.storage import *


class Test(WorkerCase):
    def testProcMemPutAndGet(self):
        test_addr = '127.0.0.1:%d' % get_next_port()
        with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
            pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())
            pool.create_actor(QuotaActor, 1024 ** 2, uid=MemQuotaActor.default_uid())
            pool.create_actor(InProcHolderActor)

            data1 = np.random.random((10, 10))
            data2 = np.random.random((10, 10))
            ser_data2 = dataserializer.serialize(data2)
            bytes_data2 = ser_data2.to_buffer()

            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())
            data_key2 = str(uuid.uuid4())

            storage_client = test_actor.storage_client
            handler = storage_client.get_storage_handler(DataStorageDevice.PROC_MEMORY)

            handler.put_object(session_id, data_key1, data1)
            self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, data_key1)),
                             [(0, DataStorageDevice.PROC_MEMORY)])
            assert_allclose(data1, handler.get_object(session_id, data_key1))

            handler.delete(session_id, data_key1)
            self.assertIsNone(storage_manager_ref.get_data_locations(session_id, data_key1))
            with self.assertRaises(KeyError):
                handler.get_object(session_id, data_key1)

            handler.put_object(session_id, data_key2, ser_data2, serialized=True)
            assert_allclose(data2, handler.get_object(session_id, data_key2))
            handler.delete(session_id, data_key2)

            handler.put_object(session_id, data_key2, bytes_data2, serialized=True)
            assert_allclose(data2, handler.get_object(session_id, data_key2))
            handler.delete(session_id, data_key2)

    def testProcMemLoad(self):
        test_addr = '127.0.0.1:%d' % get_next_port()
        with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor :
            pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            pool.create_actor(QuotaActor, 1024 ** 2, uid=MemQuotaActor.default_uid())
            pool.create_actor(InProcHolderActor)

            pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            pool.create_actor(SharedHolderActor, uid=SharedHolderActor.default_uid())

            data1 = np.random.random((10, 10))
            data2 = np.random.random((10, 10))
            ser_data1 = dataserializer.serialize(data1)

            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())
            data_key2 = str(uuid.uuid4())

            storage_client = test_actor.storage_client
            handler = storage_client.get_storage_handler(DataStorageDevice.PROC_MEMORY)

            # load from bytes io
            disk_handler = storage_client.get_storage_handler(DataStorageDevice.DISK)
            with disk_handler.create_bytes_writer(
                    session_id, data_key1, ser_data1.total_bytes) as writer:
                ser_data1.write_to(writer)

            handler.load_from_bytes_io(session_id, data_key1, disk_handler) \
                .then(lambda *_: test_actor.set_result(None),
                      lambda *exc: test_actor.set_result(exc, accept=False))
            self.get_result(5)
            self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, data_key1)),
                             [(0, DataStorageDevice.PROC_MEMORY), (0, DataStorageDevice.DISK)])

            disk_handler.delete(session_id, data_key1)
            handler.delete(session_id, data_key1)

            # load from object io
            shared_handler = storage_client.get_storage_handler(DataStorageDevice.SHARED_MEMORY)
            shared_handler.put_object(session_id, data_key2, data2)

            handler.load_from_object_io(session_id, data_key2, shared_handler) \
                .then(lambda *_: test_actor.set_result(None),
                      lambda *exc: test_actor.set_result(exc, accept=False))
            self.get_result(5)
            self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, data_key2)),
                             [(0, DataStorageDevice.PROC_MEMORY), (0, DataStorageDevice.SHARED_MEMORY)])

            shared_handler.delete(session_id, data_key2)
            handler.delete(session_id, data_key2)
