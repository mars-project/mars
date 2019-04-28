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

import numpy as np
from numpy.testing import assert_allclose

from mars.actors import create_actor_pool
from mars.serialize import dataserializer
from mars.tests.core import patch_method
from mars.utils import get_next_port
from mars.worker import WorkerDaemonActor, QuotaActor, MemQuotaActor, StatusActor
from mars.worker.tests.base import WorkerCase
from mars.worker.utils import WorkerClusterInfoActor
from mars.worker.storage import *


@patch_method(StorageHandler.transfer_in_global_runner, new=lambda *_, **__: None)
class Test(WorkerCase):
    def testDiskReadAndWrite(self, *_):
        test_addr = '127.0.0.1:%d' % get_next_port()
        with create_actor_pool(n_process=1, address=test_addr) as pool:
            pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            data1 = np.random.random((10, 10))
            ser_data1 = dataserializer.serialize(data1)
            data2 = np.random.random((10, 10))
            ser_data2 = dataserializer.serialize(data2)

            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())
            data_key2 = (str(uuid.uuid4()), 'subkey')

            with self.run_actor_test(pool) as test_actor:
                storage_client = test_actor.storage_client
                handler = storage_client.get_storage_handler(DataStorageDevice.DISK)

                compress_types = {dataserializer.CompressType.NONE} \
                    | dataserializer.get_supported_compressions()
                for handler._compress in compress_types:
                    storage_client.delete(session_id, data_key1)
                    storage_client.delete(session_id, data_key2)
                    self.rm_spill_dirs()

                    file_names = []

                    def _write_data(ser, writer):
                        file_names.append(writer.filename)
                        self.assertEqual(writer.nbytes, ser_data1.total_bytes)
                        with writer:
                            ser.write_to(writer)

                    handler.create_bytes_writer(session_id, data_key1, ser_data1.total_bytes, _promise=True) \
                        .then(functools.partial(_write_data, ser_data1)) \
                        .then(lambda *_: test_actor.set_result(None),
                              lambda *exc: test_actor.set_result(exc, accept=False))
                    self.get_result(5)
                    self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, data_key1)),
                                     [(0, DataStorageDevice.DISK)])
                    self.assertTrue(os.path.exists(file_names[-1]))

                    def _read_data(reader):
                        with reader:
                            return dataserializer.deserialize(reader.read())

                    handler.create_bytes_reader(session_id, data_key1, _promise=True) \
                        .then(_read_data) \
                        .then(functools.partial(test_actor.set_result),
                              lambda *exc: test_actor.set_result(exc, accept=False))
                    assert_allclose(self.get_result(5), data1)

                    handler.delete(session_id, data_key1)
                    while os.path.exists(file_names[-1]):
                        test_actor.ctx.sleep(0.05)
                    self.assertFalse(os.path.exists(file_names[-1]))

                    def _write_data(ser, writer):
                        self.assertEqual(writer.nbytes, ser_data2.total_bytes)
                        with writer:
                            ser.write_to(writer)

                    handler.create_bytes_writer(session_id, data_key2, ser_data2.total_bytes, _promise=True) \
                        .then(functools.partial(_write_data, ser_data2)) \
                        .then(lambda *_: test_actor.set_result(None),
                              lambda *exc: test_actor.set_result(exc, accept=False))
                    self.get_result(5)
                    self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, data_key2)),
                                     [(0, DataStorageDevice.DISK)])

                    def _read_data(reader):
                        with reader:
                            return dataserializer.deserialize(reader.read())

                    handler.create_bytes_reader(session_id, data_key2, _promise=True) \
                        .then(_read_data) \
                        .then(functools.partial(test_actor.set_result),
                              lambda *exc: test_actor.set_result(exc, accept=False))
                    assert_allclose(self.get_result(5), data2)

    def testDiskReadAndWritePacked(self, *_):
        test_addr = '127.0.0.1:%d' % get_next_port()
        with create_actor_pool(n_process=1, address=test_addr) as pool:
            pool.create_actor(WorkerClusterInfoActor, schedulers=[test_addr],
                              uid=WorkerClusterInfoActor.default_uid())
            pool.create_actor(StatusActor, test_addr, uid=StatusActor.default_uid())

            pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            data1 = np.random.random((10, 10))
            ser_data1 = dataserializer.serialize(data1)
            block_data1 = dataserializer.dumps(data1, dataserializer.CompressType.LZ4)

            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())

            with self.run_actor_test(pool) as test_actor:
                storage_client = test_actor.storage_client
                handler = storage_client.get_storage_handler(DataStorageDevice.DISK)

                compress_types = [dataserializer.CompressType.NONE, dataserializer.CompressType.LZ4]
                for handler._compress in compress_types:
                    storage_client.delete(session_id, data_key1)
                    self.rm_spill_dirs()

                    file_names = []

                    def _write_data(ser, writer):
                        file_names.append(writer.filename)
                        with writer:
                            writer.write(ser)

                    handler.create_bytes_writer(session_id, data_key1, ser_data1.total_bytes,
                                                packed=True, _promise=True) \
                        .then(functools.partial(_write_data, block_data1)) \
                        .then(lambda *_: test_actor.set_result(None),
                              lambda *exc: test_actor.set_result(exc, accept=False))
                    self.get_result(5)
                    self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, data_key1)),
                                     [(0, DataStorageDevice.DISK)])
                    self.assertTrue(os.path.exists(file_names[-1]))

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
        with self.create_pool(n_process=1, address=test_addr) as pool:
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

            with self.run_actor_test(pool) as test_actor:
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
                proc_handler = storage_client.get_storage_handler(DataStorageDevice.PROC_MEMORY)
                proc_handler.put_object(session_id, data_key2, data2)

                handler.load_from_object_io(session_id, data_key2, proc_handler) \
                    .then(lambda *_: test_actor.set_result(None),
                          lambda *exc: test_actor.set_result(exc, accept=False))
                self.get_result(5)
                self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, data_key2)),
                                 [(0, DataStorageDevice.PROC_MEMORY), (0, DataStorageDevice.DISK)])

                proc_handler.delete(session_id, data_key2)
                handler.delete(session_id, data_key2)
