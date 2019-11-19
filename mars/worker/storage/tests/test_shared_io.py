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
import uuid
import weakref

import numpy as np
from numpy.testing import assert_allclose

from mars.compat import BytesIO
from mars.errors import StorageFull
from mars.serialize import dataserializer
from mars.tests.core import patch_method
from mars.utils import get_next_port
from mars.worker import WorkerDaemonActor, QuotaActor, MemQuotaActor
from mars.worker.tests.base import WorkerCase
from mars.worker.storage import *


def mock_transfer_in_global_runner(self, session_id, data_key, src_handler, fallback=None):
    if fallback:
        return fallback()


@patch_method(StorageHandler.transfer_in_runner, new=mock_transfer_in_global_runner)
class Test(WorkerCase):
    plasma_storage_size = 1024 * 1024 * 10

    def testSharedReadAndWrite(self, *_):
        test_addr = '127.0.0.1:%d' % get_next_port()
        io_size = dataserializer.HEADER_LENGTH * 2
        with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
            pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            pool.create_actor(SharedHolderActor, uid=SharedHolderActor.default_uid())

            data1 = np.random.random((100, 100))
            ser_data1 = dataserializer.serialize(data1)

            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())

            storage_client = test_actor.storage_client
            handler = storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))

            def _write_data(ser, writer):
                self.assertEqual(writer.nbytes, ser_data1.total_bytes)
                with writer:
                    ser.write_to(writer)

            handler.create_bytes_writer(session_id, data_key1, ser_data1.total_bytes, _promise=True) \
                .then(functools.partial(_write_data, ser_data1)) \
                .then(lambda *_: test_actor.set_result(None),
                      lambda *exc: test_actor.set_result(exc, accept=False))
            self.get_result(5)
            self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, [data_key1])[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])
            handler.delete(session_id, [data_key1])

            def _write_data(ser, writer):
                with writer:
                    for start in range(0, len(ser), io_size):
                        writer.write(ser[start:start + io_size])

            handler.create_bytes_writer(session_id, data_key1, ser_data1.total_bytes, _promise=True) \
                .then(functools.partial(_write_data, ser_data1.to_buffer())) \
                .then(lambda *_: test_actor.set_result(None),
                      lambda *exc: test_actor.set_result(exc, accept=False))
            self.get_result(5)
            self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, [data_key1])[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])

            def _read_data_all(reader):
                with reader:
                    return dataserializer.deserialize(reader.read())

            handler.create_bytes_reader(session_id, data_key1, _promise=True) \
                .then(_read_data_all) \
                .then(functools.partial(test_actor.set_result),
                      lambda *exc: test_actor.set_result(exc, accept=False))
            assert_allclose(self.get_result(5), data1)

            def _read_data_batch(reader):
                bio = BytesIO()
                with reader:
                    while True:
                        buf = reader.read(io_size)
                        if buf:
                            bio.write(buf)
                        else:
                            break
                return dataserializer.deserialize(bio.getvalue())

            handler.create_bytes_reader(session_id, data_key1, _promise=True) \
                .then(_read_data_batch) \
                .then(functools.partial(test_actor.set_result),
                      lambda *exc: test_actor.set_result(exc, accept=False))
            assert_allclose(self.get_result(5), data1)
            handler.delete(session_id, [data_key1])

    def testSharedReadAndWritePacked(self, *_):
        test_addr = '127.0.0.1:%d' % get_next_port()
        io_size = dataserializer.HEADER_LENGTH * 2
        with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
            pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            pool.create_actor(SharedHolderActor, uid=SharedHolderActor.default_uid())

            data1 = np.random.random((100, 100))
            ser_data1 = dataserializer.serialize(data1)
            block_data1 = dataserializer.dumps(data1, dataserializer.CompressType.NONE)

            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())

            storage_client = test_actor.storage_client
            handler = storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))

            def _write_data(ser, writer):
                with writer:
                    writer.write(ser)

            handler.create_bytes_writer(session_id, data_key1, ser_data1.total_bytes,
                                        packed=True, _promise=True) \
                .then(functools.partial(_write_data, block_data1)) \
                .then(lambda *_: test_actor.set_result(None),
                      lambda *exc: test_actor.set_result(exc, accept=False))
            self.get_result(5)
            self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, [data_key1])[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])
            handler.delete(session_id, [data_key1])

            def _write_data(ser, writer):
                with writer:
                    with self.assertRaises(IOError):
                        writer.write(ser[:1])

                    for start in range(0, len(ser), io_size):
                        writer.write(ser[start:start + io_size])

            handler.create_bytes_writer(session_id, data_key1, ser_data1.total_bytes,
                                        packed=True, _promise=True) \
                .then(functools.partial(_write_data, block_data1)) \
                .then(lambda *_: test_actor.set_result(None),
                      lambda *exc: test_actor.set_result(exc, accept=False))
            self.get_result(5)
            self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, [data_key1])[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])

            def _read_data_all(reader):
                with reader:
                    return dataserializer.loads(reader.read())

            handler.create_bytes_reader(session_id, data_key1, packed=True, _promise=True) \
                .then(_read_data_all) \
                .then(functools.partial(test_actor.set_result),
                      lambda *exc: test_actor.set_result(exc, accept=False))
            assert_allclose(self.get_result(5), data1)

            def _read_data_batch(reader):
                bio = BytesIO()
                with reader:
                    while True:
                        buf = reader.read(io_size)
                        if buf:
                            bio.write(buf)
                        else:
                            break
                return dataserializer.loads(bio.getvalue())

            handler.create_bytes_reader(session_id, data_key1, packed=True, _promise=True) \
                .then(_read_data_batch) \
                .then(functools.partial(test_actor.set_result),
                      lambda *exc: test_actor.set_result(exc, accept=False))
            assert_allclose(self.get_result(5), data1)
            handler.delete(session_id, [data_key1])

    def testSharedPutAndGet(self, *_):
        test_addr = '127.0.0.1:%d' % get_next_port()
        with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
            pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            pool.create_actor(SharedHolderActor, uid=SharedHolderActor.default_uid())

            data1 = np.random.random((10, 10))
            data2 = np.random.random((10, 10))
            ser_data2 = dataserializer.serialize(data2)
            bytes_data2 = ser_data2.to_buffer()

            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())
            data_key2 = str(uuid.uuid4())

            storage_client = test_actor.storage_client
            handler = storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))

            handler.put_objects(session_id, [data_key1], [data1])
            self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, [data_key1])[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])
            assert_allclose(data1, handler.get_objects(session_id, [data_key1])[0])

            handler.delete(session_id, [data_key1])
            self.assertEqual(list(storage_manager_ref.get_data_locations(session_id, [data_key1])[0]), [])
            with self.assertRaises(KeyError):
                handler.get_objects(session_id, [data_key1])

            handler.put_objects(session_id, [data_key2], [ser_data2], serialized=True)
            assert_allclose(data2, handler.get_objects(session_id, [data_key2])[0])
            handler.delete(session_id, [data_key2])

            handler.put_objects(session_id, [data_key2], [bytes_data2], serialized=True)
            assert_allclose(data2, handler.get_objects(session_id, [data_key2])[0])
            handler.delete(session_id, [data_key2])

    def testSharedLoadFromBytes(self, *_):
        import logging
        logging.basicConfig(level=logging.DEBUG)
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
            ser_data1 = dataserializer.serialize(data1)

            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())

            storage_client = test_actor.storage_client
            handler = storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))

            # load from bytes io
            disk_handler = storage_client.get_storage_handler((0, DataStorageDevice.DISK))
            with disk_handler.create_bytes_writer(
                    session_id, data_key1, ser_data1.total_bytes) as writer:
                ser_data1.write_to(writer)

            handler.load_from_bytes_io(session_id, [data_key1], disk_handler) \
                .then(lambda *_: test_actor.set_result(None),
                      lambda *exc: test_actor.set_result(exc, accept=False))
            self.get_result(5)
            self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, [data_key1])[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY), (0, DataStorageDevice.DISK)])

            disk_handler.delete(session_id, [data_key1])
            handler.delete(session_id, [data_key1])

            # load from bytes io till no capacity
            data_list = [np.random.randint(0, 32767, (655360,), np.int16)
                         for _ in range(20)]
            data_keys = [str(uuid.uuid4()) for _ in range(20)]
            for key, data in zip(data_keys, data_list):
                ser_data = dataserializer.serialize(data)
                with disk_handler.create_bytes_writer(
                        session_id, key, ser_data.total_bytes) as writer:
                    ser_data.write_to(writer)

            handler.load_from_bytes_io(session_id, data_keys, disk_handler) \
                .then(lambda *_: test_actor.set_result(None),
                      lambda *exc: test_actor.set_result(exc, accept=False))

            affected_keys = set()
            try:
                self.get_result(5)
            except StorageFull as ex:
                affected_keys.update(ex.affected_keys)

            storage_client.delete(session_id, data_keys, [DataStorageDevice.DISK])

            self.assertLess(len(affected_keys), len(data_keys))
            self.assertGreater(len(affected_keys), 1)
            for k, size in zip(data_keys, storage_client.get_data_sizes(session_id, data_keys)):
                if k in affected_keys:
                    self.assertIsNone(size)
                else:
                    self.assertIsNotNone(size)

    def testSharedLoadFromObjects(self, *_):
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

            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())

            storage_client = test_actor.storage_client
            handler = storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))

            # load from object io
            ref_data1 = weakref.ref(data1)

            proc_handler = storage_client.get_storage_handler((0, DataStorageDevice.PROC_MEMORY))
            proc_handler.put_objects(session_id, [data_key1], [data1])
            del data1

            handler.load_from_object_io(session_id, [data_key1], proc_handler) \
                .then(lambda *_: test_actor.set_result(None),
                      lambda *exc: test_actor.set_result(exc, accept=False))
            self.get_result(5)
            self.assertEqual(sorted(storage_manager_ref.get_data_locations(session_id, [data_key1])[0]),
                             [(0, DataStorageDevice.PROC_MEMORY), (0, DataStorageDevice.SHARED_MEMORY)])

            proc_handler.delete(session_id, [data_key1])
            self.assertIsNone(ref_data1())
            handler.delete(session_id, [data_key1])

    def testSharedSpill(self, *_):
        test_addr = '127.0.0.1:%d' % get_next_port()
        with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
            pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            holder_ref = pool.create_actor(
                SharedHolderActor, self.plasma_storage_size,
                uid=SharedHolderActor.default_uid())

            session_id = str(uuid.uuid4())
            data_list = [np.random.randint(0, 32767, (655360,), np.int16)
                         for _ in range(20)]
            data_keys = [str(uuid.uuid4()) for _ in range(20)]

            storage_client = test_actor.storage_client
            handler = storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))
            idx = 0

            def _fill_data():
                i = 0
                for i, (key, data) in enumerate(zip(data_keys[idx:], data_list)):
                    try:
                        handler.put_objects(session_id, [key], [data])
                    except StorageFull:
                        break
                return i + idx

            def _do_spill():
                data_size = storage_manager_ref.get_data_sizes(session_id, [data_keys[0]])[0]
                handler.spill_size(2 * data_size) \
                    .then(lambda *_: test_actor.set_result(None),
                          lambda *exc: test_actor.set_result(exc, accept=False))
                self.get_result(5)

            # test lift data key
            idx = _fill_data()
            handler.lift_data_keys(session_id, [data_keys[0]])
            _do_spill()

            self.assertEqual(list(storage_manager_ref.get_data_locations(session_id, [data_keys[0]])[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])
            self.assertEqual(list(storage_manager_ref.get_data_locations(session_id, [data_keys[1]])[0]),
                             [(0, DataStorageDevice.DISK)])

            handler.put_objects(session_id, [data_keys[idx]], [data_list[idx]])
            self.assertEqual(list(storage_manager_ref.get_data_locations(session_id, [data_keys[idx]])[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])
            idx += 1

            # test pin data key
            idx = _fill_data()
            holder_ref.lift_data_keys(session_id, [data_keys[0]], last=False)
            pin_token = str(uuid.uuid4())
            pinned_keys = handler.pin_data_keys(session_id, (data_keys[0],), pin_token)
            self.assertIn(data_keys[0], pinned_keys)
            _do_spill()

            self.assertEqual(list(storage_manager_ref.get_data_locations(session_id, [data_keys[0]])[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])
            self.assertEqual(list(storage_manager_ref.get_data_locations(session_id, [data_keys[1]])[0]),
                             [(0, DataStorageDevice.DISK)])

            handler.put_objects(session_id, [data_keys[idx]], [data_list[idx]])
            self.assertEqual(list(storage_manager_ref.get_data_locations(session_id, [data_keys[idx]])[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])
            idx += 1

            # test unpin data key
            idx = _fill_data()
            handler.unpin_data_keys(session_id, (data_keys[0],), pin_token)
            _do_spill()

            self.assertEqual(list(storage_manager_ref.get_data_locations(session_id, [data_keys[0]])[0]),
                             [(0, DataStorageDevice.DISK)])
