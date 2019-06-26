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

from mars.errors import StorageDataExists
from mars.serialize import dataserializer
from mars.tests.core import patch_method
from mars.utils import get_next_port
from mars.worker import WorkerDaemonActor, QuotaActor, MemQuotaActor, StatusActor
from mars.worker.storage import *
from mars.worker.tests.base import WorkerCase
from mars.worker.utils import WorkerClusterInfoActor


def mock_transfer_in_global_runner(self, session_id, data_key, src_handler, fallback=None):
    if fallback:
        return fallback()


@patch_method(StorageHandler.transfer_in_global_runner, new=mock_transfer_in_global_runner)
class Test(WorkerCase):
    @staticmethod
    def _get_compress_types():
        return {dataserializer.CompressType.NONE} \
            | dataserializer.get_supported_compressions()

    def testDiskReadAndWrite(self, *_):
        test_addr = '127.0.0.1:%d' % get_next_port()
        with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
            pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            data1 = np.random.random((10, 10))
            ser_data1 = dataserializer.serialize(data1)
            data2 = np.random.random((10, 10))
            ser_data2 = dataserializer.serialize(data2)

            session_id = str(uuid.uuid4())

            storage_client = test_actor.storage_client
            handler = storage_client.get_storage_handler(DataStorageDevice.DISK)

            for handler._compress in self._get_compress_types():
                data_key1 = str(uuid.uuid4())
                data_key2 = (str(uuid.uuid4()), 'subkey')

                storage_client.delete(session_id, data_key1)
                storage_client.delete(session_id, data_key2)
                self.rm_spill_dirs()

                def _write_data(ser, writer):
                    self.assertEqual(writer.nbytes, ser.total_bytes)
                    with writer:
                        ser.write_to(writer)
                    return writer.filename

                def _read_data(reader):
                    with reader:
                        return dataserializer.deserialize(reader.read())

                # test normal file write
                handler.create_bytes_writer(session_id, data_key1, ser_data1.total_bytes, _promise=True) \
                    .then(functools.partial(_write_data, ser_data1)) \
                    .then(test_actor.set_result,
                          lambda *exc: test_actor.set_result(exc, accept=False))
                file_name = self.get_result(5)
                self.assertTrue(os.path.exists(file_name))
                self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, data_key1)),
                                 [(0, DataStorageDevice.DISK)])

                # test write existing (this should produce an error)
                handler.create_bytes_writer(session_id, data_key1, ser_data1.total_bytes, _promise=True) \
                    .then(functools.partial(_write_data, ser_data1)) \
                    .then(test_actor.set_result,
                          lambda *exc: test_actor.set_result(exc, accept=False))
                with self.assertRaises(StorageDataExists):
                    self.get_result(5)

                # test writing with unreferenced file
                storage_manager_ref.unregister_data(session_id, data_key1, (0, DataStorageDevice.DISK))
                handler.create_bytes_writer(session_id, data_key1, ser_data1.total_bytes, _promise=True) \
                    .then(functools.partial(_write_data, ser_data1)) \
                    .then(test_actor.set_result,
                          lambda *exc: test_actor.set_result(exc, accept=False))
                file_name = self.get_result(5)
                self.assertTrue(os.path.exists(file_name))
                self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, data_key1)),
                                 [(0, DataStorageDevice.DISK)])

                # test reading and verifying written data
                handler.create_bytes_reader(session_id, data_key1, _promise=True) \
                    .then(_read_data) \
                    .then(test_actor.set_result,
                          lambda *exc: test_actor.set_result(exc, accept=False))
                assert_allclose(self.get_result(5), data1)

                # test unregistering data
                handler.delete(session_id, data_key1)
                while os.path.exists(file_name):
                    test_actor.ctx.sleep(0.05)
                self.assertFalse(os.path.exists(file_name))

                # test reading and writing with tuple keys
                handler.create_bytes_writer(session_id, data_key2, ser_data2.total_bytes, _promise=True) \
                    .then(functools.partial(_write_data, ser_data2)) \
                    .then(test_actor.set_result,
                          lambda *exc: test_actor.set_result(exc, accept=False))
                self.get_result(5)
                self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, data_key2)),
                                 [(0, DataStorageDevice.DISK)])

                handler.create_bytes_reader(session_id, data_key2, _promise=True) \
                    .then(_read_data) \
                    .then(functools.partial(test_actor.set_result),
                          lambda *exc: test_actor.set_result(exc, accept=False))
                assert_allclose(self.get_result(5), data2)

    def testDiskReadAndWritePacked(self, *_):
        test_addr = '127.0.0.1:%d' % get_next_port()
        with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
            pool.create_actor(WorkerClusterInfoActor, schedulers=[test_addr],
                              uid=WorkerClusterInfoActor.default_uid())
            pool.create_actor(StatusActor, test_addr, uid=StatusActor.default_uid())

            pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            session_id = str(uuid.uuid4())
            data1 = np.random.random((10, 10))
            ser_data1 = dataserializer.serialize(data1)

            storage_client = test_actor.storage_client
            handler = storage_client.get_storage_handler(DataStorageDevice.DISK)

            for handler._compress in self._get_compress_types():
                data_key1 = str(uuid.uuid4())

                storage_client.delete(session_id, data_key1)
                self.rm_spill_dirs()

                block_data1 = dataserializer.dumps(data1, handler._compress)

                def _write_data(ser, writer):
                    with writer:
                        writer.write(ser)
                    return writer.filename

                handler.create_bytes_writer(session_id, data_key1, ser_data1.total_bytes,
                                            packed=True, _promise=True) \
                    .then(functools.partial(_write_data, block_data1)) \
                    .then(test_actor.set_result,
                          lambda *exc: test_actor.set_result(exc, accept=False))
                file_name = self.get_result(5)
                self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, data_key1)),
                                 [(0, DataStorageDevice.DISK)])
                self.assertTrue(os.path.exists(file_name))

                def _read_data(reader):
                    with reader:
                        return dataserializer.loads(reader.read())

                handler.create_bytes_reader(session_id, data_key1, packed=True, _promise=True) \
                    .then(_read_data) \
                    .then(functools.partial(test_actor.set_result),
                          lambda *exc: test_actor.set_result(exc, accept=False))
                assert_allclose(self.get_result(5), data1)

    def testDiskLoad(self, *_):
        test_addr = '127.0.0.1:%d' % get_next_port()
        with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
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
            handler = storage_client.get_storage_handler(DataStorageDevice.DISK)

            # load from bytes io
            shared_handler = storage_client.get_storage_handler(DataStorageDevice.SHARED_MEMORY)
            with shared_handler.create_bytes_writer(
                    session_id, data_key1, ser_data1.total_bytes) as writer:
                ser_data1.write_to(writer)

            handler.load_from_bytes_io(session_id, data_key1, shared_handler) \
                .then(lambda *_: test_actor.set_result(None),
                      lambda *exc: test_actor.set_result(exc, accept=False))
            self.get_result(5)
            self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, data_key1)),
                             [(0, DataStorageDevice.SHARED_MEMORY), (0, DataStorageDevice.DISK)])

            shared_handler.delete(session_id, data_key1)
            handler.delete(session_id, data_key1)

            # load from object io
            ref_data2 = weakref.ref(data2)
            proc_handler = storage_client.get_storage_handler(DataStorageDevice.PROC_MEMORY)
            proc_handler.put_object(session_id, data_key2, data2)
            del data2

            handler.load_from_object_io(session_id, data_key2, proc_handler) \
                .then(lambda *_: test_actor.set_result(None),
                      lambda *exc: test_actor.set_result(exc, accept=False))
            self.get_result(5)
            self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, data_key2)),
                             [(0, DataStorageDevice.PROC_MEMORY), (0, DataStorageDevice.DISK)])

            proc_handler.delete(session_id, data_key2)
            self.assertIsNone(ref_data2())
            handler.delete(session_id, data_key2)
