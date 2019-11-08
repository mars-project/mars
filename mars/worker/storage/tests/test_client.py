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

import functools
import os
import time
import uuid
import weakref

import numpy as np
from numpy.testing import assert_allclose

from mars import promise
from mars.actors import create_actor_pool
from mars.compat import TimeoutError, six
from mars.config import options
from mars.distributor import MarsDistributor
from mars.errors import StorageFull
from mars.serialize import dataserializer
from mars.tests.core import patch_method
from mars.utils import get_next_port, build_exc_info
from mars.worker import WorkerDaemonActor, MemQuotaActor, QuotaActor, DispatchActor
from mars.worker.utils import WorkerActor
from mars.worker.tests.base import WorkerCase
from mars.worker.storage import *


class OtherProcessTestActor(WorkerActor):
    def __init__(self):
        super(OtherProcessTestActor, self).__init__()
        self._accept = None
        self._result = None
        self._manager_ref = None

    def post_create(self):
        super(OtherProcessTestActor, self).post_create()
        self._manager_ref = self.ctx.actor_ref(StorageManagerActor.default_uid())

    def set_result(self, result, accept=True):
        self._result, self._accept = result, accept

    def get_result(self):
        if self._accept is None:
            return None
        elif self._accept:
            return self._result
        else:
            six.reraise(*self._result)

    def run_copy_global_to_proc_test(self):
        self._accept, self._result = None, None

        session_id = str(uuid.uuid4())
        data1 = np.random.randint(0, 32767, (655360,), np.int16)
        key1 = str(uuid.uuid4())

        shared_handler = self.storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))
        proc_handler = self.storage_client.get_storage_handler((1, DataStorageDevice.PROC_MEMORY))

        def _verify_result(*_):
            result = proc_handler.get_object(session_id, key1)
            assert_allclose(result, data1)

            devices = self._manager_ref.get_data_locations(session_id, key1)
            if devices != {(0, DataStorageDevice.SHARED_MEMORY), (1, DataStorageDevice.PROC_MEMORY)}:
                raise AssertionError

        shared_handler.put_object(session_id, key1, data1)
        self.storage_client.copy_to(session_id, key1, [(1, DataStorageDevice.PROC_MEMORY)]) \
            .then(_verify_result) \
            .then(lambda *_: self.set_result(1),
                  lambda *exc: self.set_result(exc, accept=False))

    def run_copy_proc_to_global_test(self):
        self._accept, self._result = None, None

        session_id = str(uuid.uuid4())
        data1 = np.random.randint(0, 32767, (655360,), np.int16)
        key1 = str(uuid.uuid4())

        shared_handler = self.storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))
        proc_handler = self.storage_client.get_storage_handler((1, DataStorageDevice.PROC_MEMORY))

        def _verify_result(*_):
            result = shared_handler.get_object(session_id, key1)
            assert_allclose(result, data1)

            devices = self._manager_ref.get_data_locations(session_id, key1)
            if devices != {(0, DataStorageDevice.SHARED_MEMORY), (1, DataStorageDevice.PROC_MEMORY)}:
                raise AssertionError

        proc_handler.put_object(session_id, key1, data1)
        self.storage_client.copy_to(session_id, key1, [DataStorageDevice.SHARED_MEMORY]) \
            .then(_verify_result) \
            .then(lambda *_: self.set_result(1),
                  lambda *exc: self.set_result(exc, accept=False))


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

    def testLoadStoreInOtherProcess(self):
        test_addr = '127.0.0.1:%d' % get_next_port()
        with self.create_pool(n_process=2, address=test_addr, distributor=MarsDistributor(2)) as pool:
            pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())

            pool.create_actor(QuotaActor, 1024 ** 2, uid=MemQuotaActor.default_uid())

            pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            pool.create_actor(SharedHolderActor, self.plasma_storage_size,
                              uid=SharedHolderActor.default_uid())

            pool.create_actor(InProcHolderActor, uid='w:1:InProcHolderActor')
            pool.create_actor(IORunnerActor, lock_free=True, dispatched=False, uid=IORunnerActor.gen_uid(1))

            test_ref = pool.create_actor(OtherProcessTestActor, uid='w:0:OtherProcTest')

            test_ref.run_copy_global_to_proc_test(_tell=True)

            start_time = time.time()
            while test_ref.get_result() is None:
                pool.sleep(0.5)
                if time.time() - start_time > 10:
                    raise TimeoutError

            test_ref.run_copy_proc_to_global_test(_tell=True)

            start_time = time.time()
            while test_ref.get_result() is None:
                pool.sleep(0.5)
                if time.time() - start_time > 10:
                    raise TimeoutError

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

                shared_handler = storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))
                proc_handler = storage_client.get_storage_handler((0, DataStorageDevice.PROC_MEMORY))

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
