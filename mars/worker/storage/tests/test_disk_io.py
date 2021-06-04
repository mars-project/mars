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
import random
import uuid
import weakref

import numpy as np
from numpy.testing import assert_allclose

from mars import promise
from mars.config import options
from mars.errors import StorageDataExists
from mars.serialize import dataserializer
from mars.tests.core import patch_method
from mars.utils import get_next_port
from mars.worker import WorkerDaemonActor, QuotaActor, MemQuotaActor, StatusActor, EventsActor
from mars.worker.storage import StorageHandler, StorageManagerActor, InProcHolderActor, \
    PlasmaKeyMapActor, SharedHolderActor, DiskFileMergerActor, DataStorageDevice
from mars.worker.tests.base import WorkerCase
from mars.worker.utils import WorkerClusterInfoActor


def mock_transfer_in_global_runner(self, session_id, data_key, src_handler, fallback=None):
    if fallback:
        return fallback()


@patch_method(StorageHandler.transfer_in_runner, new=mock_transfer_in_global_runner)
class Test(WorkerCase):
    def tearDown(self):
        options.worker.filemerger.max_file_size = 10 * 1024 ** 2
        options.worker.filemerger.concurrency = 128
        super().tearDown()

    @staticmethod
    def _get_compress_types():
        return {dataserializer.CompressType.NONE} \
            | dataserializer.get_supported_compressions()

    def testDiskReadAndWrite(self, *_):
        test_addr = f'127.0.0.1:{get_next_port()}'
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
            handler = storage_client.get_storage_handler((0, DataStorageDevice.DISK))

            for handler._compress in self._get_compress_types():
                data_key1 = str(uuid.uuid4())
                data_key2 = (str(uuid.uuid4()), 'subkey')

                storage_client.delete(session_id, [data_key1])
                storage_client.delete(session_id, [data_key2])
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
                self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, [data_key1])[0]),
                                 [(0, DataStorageDevice.DISK)])

                # test write existing (this should produce an error)
                handler.create_bytes_writer(session_id, data_key1, ser_data1.total_bytes, _promise=True) \
                    .then(functools.partial(_write_data, ser_data1)) \
                    .then(test_actor.set_result,
                          lambda *exc: test_actor.set_result(exc, accept=False))
                with self.assertRaises(StorageDataExists):
                    self.get_result(5)

                # test writing with unreferenced file
                storage_manager_ref.unregister_data(session_id, [data_key1], (0, DataStorageDevice.DISK))
                handler.create_bytes_writer(session_id, data_key1, ser_data1.total_bytes, _promise=True) \
                    .then(functools.partial(_write_data, ser_data1)) \
                    .then(test_actor.set_result,
                          lambda *exc: test_actor.set_result(exc, accept=False))
                file_name = self.get_result(5)
                self.assertTrue(os.path.exists(file_name))
                self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, [data_key1])[0]),
                                 [(0, DataStorageDevice.DISK)])

                # test reading and verifying written data
                handler.create_bytes_reader(session_id, data_key1, _promise=True) \
                    .then(_read_data) \
                    .then(test_actor.set_result,
                          lambda *exc: test_actor.set_result(exc, accept=False))
                assert_allclose(self.get_result(5), data1)

                # test unregistering data
                handler.delete(session_id, [data_key1])
                while os.path.exists(file_name):
                    test_actor.ctx.sleep(0.05)
                self.assertFalse(os.path.exists(file_name))

                # test reading and writing with tuple keys
                handler.create_bytes_writer(session_id, data_key2, ser_data2.total_bytes, _promise=True) \
                    .then(functools.partial(_write_data, ser_data2)) \
                    .then(test_actor.set_result,
                          lambda *exc: test_actor.set_result(exc, accept=False))
                self.get_result(5)
                self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, [data_key2])[0]),
                                 [(0, DataStorageDevice.DISK)])

                handler.create_bytes_reader(session_id, data_key2, _promise=True) \
                    .then(_read_data) \
                    .then(functools.partial(test_actor.set_result),
                          lambda *exc: test_actor.set_result(exc, accept=False))
                assert_allclose(self.get_result(5), data2)

    def testDiskReadAndWritePacked(self, *_):
        test_addr = f'127.0.0.1:{get_next_port()}'
        with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
            pool.create_actor(WorkerClusterInfoActor, [test_addr],
                              uid=WorkerClusterInfoActor.default_uid())
            pool.create_actor(StatusActor, test_addr, uid=StatusActor.default_uid())
            pool.create_actor(EventsActor, uid=EventsActor.default_uid())

            pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            session_id = str(uuid.uuid4())
            data1 = np.random.random((10, 10))
            ser_data1 = dataserializer.serialize(data1)

            storage_client = test_actor.storage_client
            handler = storage_client.get_storage_handler((0, DataStorageDevice.DISK))

            for handler._compress in self._get_compress_types():
                data_key1 = str(uuid.uuid4())

                storage_client.delete(session_id, [data_key1])
                self.rm_spill_dirs()

                block_data1 = dataserializer.dumps(data1, compress=handler._compress)

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
                self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, [data_key1])[0]),
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

    def testDiskReadAndWriteMerger(self):
        test_addr = f'127.0.0.1:{get_next_port()}'
        options.worker.filemerger.max_file_size = 2400
        options.worker.filemerger.concurrency = 16

        with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
            pool.create_actor(WorkerClusterInfoActor, [test_addr],
                              uid=WorkerClusterInfoActor.default_uid())
            pool.create_actor(StatusActor, test_addr, uid=StatusActor.default_uid())
            pool.create_actor(EventsActor, uid=EventsActor.default_uid())

            disk_file_merger_ref = pool.create_actor(
                DiskFileMergerActor, uid=DiskFileMergerActor.default_uid())

            pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            session_id = str(uuid.uuid4())
            data_count = 30
            data = [np.random.rand(random.randint(10, 30), random.randint(10, 30))
                    for _ in range(data_count)]
            ser_data = [dataserializer.serialize(d) for d in data]

            storage_client = test_actor.storage_client
            handler = storage_client.get_storage_handler((0, DataStorageDevice.DISK))

            for handler._compress in self._get_compress_types():
                data_keys = [str(uuid.uuid4()) for _ in range(data_count)]

                promises = []
                for idx in range(data_count):
                    block_data = dataserializer.dumps(data[idx], compress=handler._compress)

                    def _write_data(ser, writer):
                        with writer:
                            writer.write(ser)
                        return writer.filename

                    promises.append(
                        handler.create_bytes_writer(session_id, data_keys[idx], ser_data[idx].total_bytes,
                                                    packed=True, with_merger_lock=True, _promise=True)
                            .then(functools.partial(_write_data, block_data))
                    )
                promise.all_(promises).then(lambda *_: test_actor.set_result(0),
                                            lambda *exc: test_actor.set_result(exc, accept=False))
                self.get_result(50)

                for key in data_keys:
                    self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, [key])[0]),
                                     [(0, DataStorageDevice.DISK)])

                dump_result = disk_file_merger_ref.dump_info()
                written_files = list(dump_result[2])
                for fn in written_files:
                    self.assertTrue(os.path.exists(fn))

                data_store = [None] * len(data)
                promises = []
                for idx in range(data_count):
                    def _read_data(reader, idx):
                        with reader:
                            data_store[idx] = dataserializer.loads(reader.read())

                    promises.append(
                        handler.create_bytes_reader(session_id, data_keys[idx],
                                                    with_merger_lock=True, packed=True, _promise=True)
                            .then(functools.partial(_read_data, idx=idx))
                    )
                promise.all_(promises).then(lambda *_: test_actor.set_result(0),
                                            lambda *exc: test_actor.set_result(exc, accept=False))
                self.get_result(50)
                for true_data, read_data in zip(data, data_store):
                    assert_allclose(true_data, read_data)

                data_store = [None] * len(data)
                promises = []
                for idx in range(data_count):
                    def _read_data(reader, idx):
                        with reader:
                            data_store[idx] = dataserializer.deserialize(reader.read())

                    promises.append(
                        handler.create_bytes_reader(session_id, data_keys[idx], _promise=True)
                            .then(functools.partial(_read_data, idx=idx))
                    )
                promise.all_(promises).then(lambda *_: test_actor.set_result(0),
                                            lambda *exc: test_actor.set_result(exc, accept=False))
                self.get_result(50)
                for true_data, read_data in zip(data, data_store):
                    assert_allclose(true_data, read_data)

                storage_client.delete(session_id, data_keys)
                pool.sleep(0.1)
                for fn in written_files:
                    self.assertFalse(os.path.exists(fn))

    def testDiskLoad(self, *_):
        test_addr = f'127.0.0.1:{get_next_port()}'
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
            handler = storage_client.get_storage_handler((0, DataStorageDevice.DISK))

            # load from bytes io
            shared_handler = storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))
            with shared_handler.create_bytes_writer(
                    session_id, data_key1, ser_data1.total_bytes) as writer:
                ser_data1.write_to(writer)

            handler.load_from_bytes_io(session_id, [data_key1], shared_handler) \
                .then(lambda *_: test_actor.set_result(None),
                      lambda *exc: test_actor.set_result(exc, accept=False))
            self.get_result(5)
            self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, [data_key1])[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY), (0, DataStorageDevice.DISK)])

            shared_handler.delete(session_id, [data_key1])
            handler.delete(session_id, [data_key1])

            # load from object io
            ref_data2 = weakref.ref(data2)
            proc_handler = storage_client.get_storage_handler((0, DataStorageDevice.PROC_MEMORY))
            proc_handler.put_objects(session_id, [data_key2], [data2])
            del data2

            handler.load_from_object_io(session_id, [data_key2], proc_handler) \
                .then(lambda *_: test_actor.set_result(None),
                      lambda *exc: test_actor.set_result(exc, accept=False))
            self.get_result(5)
            self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, [data_key2])[0]),
                             [(0, DataStorageDevice.PROC_MEMORY), (0, DataStorageDevice.DISK)])

            proc_handler.delete(session_id, [data_key2])
            self.assertIsNone(ref_data2())
            handler.delete(session_id, [data_key2])
