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

import functools
import os
import uuid
import weakref

import numpy as np
from numpy.testing import assert_allclose

from mars import promise
from mars.actors import create_actor_pool
from mars.config import options
from mars.errors import StorageFull
from mars.serialize import dataserializer
from mars.tests.core import patch_method
from mars.utils import get_next_port, build_exc_info
from mars.worker import WorkerDaemonActor, MemQuotaActor, QuotaActor, DispatchActor
from mars.worker.tests.base import WorkerCase
from mars.worker.storage import *


class Test(WorkerCase):
    def tearDown(self):
        options.worker.lock_free_fileio = False
        super(Test, self).tearDown()

    def testClientReadAndWrite(self):
        test_addr = '127.0.0.1:%d' % get_next_port()
        with create_actor_pool(n_process=1, address=test_addr) as pool:
            options.worker.lock_free_fileio = True
            pool.create_actor(
                WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            pool.create_actor(
                DispatchActor, uid=DispatchActor.default_uid())
            pool.create_actor(IORunnerActor)

            pool.create_actor(
                PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            pool.create_actor(
                SharedHolderActor, self.plasma_storage_size,
                uid=SharedHolderActor.default_uid())

            data1 = np.random.random((10, 10))
            ser_data1 = dataserializer.serialize(data1)

            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())

            with self.run_actor_test(pool) as test_actor:
                storage_client = test_actor.storage_client

                file_names = []

                def _write_data(ser, writer):
                    file_names.append(writer.filename)
                    self.assertEqual(writer.nbytes, ser_data1.total_bytes)
                    with writer:
                        ser.write_to(writer)

                # test creating writer and write
                storage_client.create_writer(
                        session_id, data_key1, ser_data1.total_bytes, (DataStorageDevice.DISK,)) \
                    .then(functools.partial(_write_data, ser_data1)) \
                    .then(lambda *_: test_actor.set_result(None),
                          lambda *exc: test_actor.set_result(exc, accept=False))
                self.get_result(5)
                self.assertTrue(os.path.exists(file_names[0]))
                self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, data_key1)),
                                 [(0, DataStorageDevice.DISK)])

                def _read_data(reader):
                    with reader:
                        return dataserializer.deserialize(reader.read())

                # test creating reader when data exist in location
                storage_client.create_reader(session_id, data_key1, (DataStorageDevice.DISK,)) \
                    .then(_read_data) \
                    .then(functools.partial(test_actor.set_result),
                          lambda *exc: test_actor.set_result(exc, accept=False))
                assert_allclose(self.get_result(5), data1)

                # test creating reader when no data in location (should raise)
                with self.assertRaises(IOError):
                    storage_client.create_reader(session_id, data_key1, (DataStorageDevice.SHARED_MEMORY,),
                                         _promise=False)

                # test creating reader when copy needed
                storage_client.create_reader(session_id, data_key1, (DataStorageDevice.SHARED_MEMORY,)) \
                    .then(_read_data) \
                    .then(functools.partial(test_actor.set_result),
                          lambda *exc: test_actor.set_result(exc, accept=False))
                self.get_result(5)
                self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, data_key1)),
                                 [(0, DataStorageDevice.SHARED_MEMORY), (0, DataStorageDevice.DISK)])

                storage_client.delete(session_id, data_key1)
                while os.path.exists(file_names[0]):
                    test_actor.ctx.sleep(0.05)
                self.assertFalse(os.path.exists(file_names[0]))

    def testClientSpill(self, *_):
        test_addr = '127.0.0.1:%d' % get_next_port()
        with self.create_pool(n_process=1, address=test_addr) as pool:
            pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())
            pool.create_actor(IORunnerActor)

            pool.create_actor(QuotaActor, 1024 ** 2, uid=MemQuotaActor.default_uid())
            pool.create_actor(InProcHolderActor)

            pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            pool.create_actor(SharedHolderActor, self.plasma_storage_size,
                              uid=SharedHolderActor.default_uid())

            session_id = str(uuid.uuid4())
            data_list = [np.random.randint(0, 32767, (655360,), np.int16)
                         for _ in range(20)]
            data_keys = [str(uuid.uuid4()) for _ in range(20)]

            with self.run_actor_test(pool) as test_actor:
                storage_client = test_actor.storage_client
                idx = 0

                shared_handler = storage_client.get_storage_handler(DataStorageDevice.SHARED_MEMORY)
                proc_handler = storage_client.get_storage_handler(DataStorageDevice.PROC_MEMORY)

                def _fill_data():
                    i = 0
                    for i, (key, data) in enumerate(zip(data_keys[idx:], data_list)):
                        try:
                            shared_handler.put_object(session_id, key, data)
                        except StorageFull:
                            break
                    return i + idx

                idx = _fill_data()

                # test copying non-existing keys
                storage_client.copy_to(session_id, 'non-exist-key', [DataStorageDevice.SHARED_MEMORY]) \
                    .then(lambda *_: test_actor.set_result(None),
                          lambda *exc: test_actor.set_result(exc, accept=False))
                with self.assertRaises(KeyError):
                    self.get_result(5)

                # test copying into containing locations
                storage_client.copy_to(session_id, data_keys[0], [DataStorageDevice.SHARED_MEMORY]) \
                    .then(lambda *_: test_actor.set_result(None),
                          lambda *exc: test_actor.set_result(exc, accept=False))
                self.get_result(5)

                self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, data_keys[0])),
                                 [(0, DataStorageDevice.SHARED_MEMORY)])

                # test unsuccessful copy when no data at target
                def _mock_load_from(*_, **__):
                    return promise.finished(*build_exc_info(SystemError), **dict(_accept=False))

                with patch_method(StorageHandler.load_from, _mock_load_from), \
                        self.assertRaises(SystemError):
                    storage_client.copy_to(session_id, data_keys[0], [DataStorageDevice.DISK]) \
                        .then(lambda *_: test_actor.set_result(None),
                              lambda *exc: test_actor.set_result(exc, accept=False))
                    self.get_result(5)

                # test successful copy
                ref_data = weakref.ref(data_list[idx])
                proc_handler.put_object(session_id, data_keys[idx], data_list[idx])
                data_list[idx] = None

                storage_client.copy_to(session_id, data_keys[idx],
                               [DataStorageDevice.SHARED_MEMORY, DataStorageDevice.DISK]) \
                    .then(lambda *_: test_actor.set_result(None),
                          lambda *exc: test_actor.set_result(exc, accept=False))
                self.get_result(5)

                proc_handler.delete(session_id, data_keys[idx])

                self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, data_keys[idx])),
                                 [(0, DataStorageDevice.DISK)])
                self.assertIsNone(ref_data())

                # test copy with spill
                idx += 1
                proc_handler.put_object(session_id, data_keys[idx], data_list[idx])

                storage_client.copy_to(session_id, data_keys[idx], [DataStorageDevice.SHARED_MEMORY]) \
                    .then(lambda *_: test_actor.set_result(None),
                          lambda *exc: test_actor.set_result(exc, accept=False))
                self.get_result(5)

                self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, data_keys[idx])),
                                 [(0, DataStorageDevice.PROC_MEMORY), (0, DataStorageDevice.SHARED_MEMORY)])
