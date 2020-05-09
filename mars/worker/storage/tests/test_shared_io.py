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
from io import BytesIO

import numpy as np
from numpy.testing import assert_allclose

from mars.errors import StorageFull
from mars.serialize import dataserializer
from mars.tests.core import aio_case, patch_method
from mars.utils import get_next_port
from mars.worker import WorkerDaemonActor, QuotaActor, MemQuotaActor
from mars.worker.tests.base import WorkerCase
from mars.worker.storage import StorageManagerActor, PlasmaKeyMapActor, SharedHolderActor, \
    InProcHolderActor, StorageHandler, DataStorageDevice


def mock_transfer_in_global_runner(self, session_id, data_key, src_handler, fallback=None):
    if fallback:
        return fallback()


@patch_method(StorageHandler.transfer_in_runner, new=mock_transfer_in_global_runner)
@aio_case
class Test(WorkerCase):
    plasma_storage_size = 1024 * 1024 * 10

    async def testSharedReadAndWrite(self, *_):
        test_addr = '127.0.0.1:%d' % get_next_port()
        io_size = dataserializer.HEADER_LENGTH * 2
        async with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
            await pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = await pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            await pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            await pool.create_actor(SharedHolderActor, uid=SharedHolderActor.default_uid())

            data1 = np.random.random((100, 100))
            ser_data1 = dataserializer.serialize(data1)

            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())

            storage_client = test_actor.storage_client
            handler = await storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))

            async def _write_data(ser, writer):
                self.assertEqual(writer.nbytes, ser_data1.total_bytes)
                async with writer:
                    ser.write_to(writer)

            await self.waitp(
                (await handler.create_bytes_writer(session_id, data_key1, ser_data1.total_bytes, _promise=True))
                    .then(functools.partial(_write_data, ser_data1))
            )
            self.assertEqual(sorted((await storage_manager_ref.get_data_locations(session_id, [data_key1]))[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])
            await handler.delete(session_id, [data_key1])

            async def _write_data(ser, writer):
                async with writer:
                    for start in range(0, len(ser), io_size):
                        writer.write(ser[start:start + io_size])

            await self.waitp(
                (await handler.create_bytes_writer(session_id, data_key1, ser_data1.total_bytes, _promise=True))
                    .then(functools.partial(_write_data, ser_data1.to_buffer()))
            )
            self.assertEqual(sorted((await storage_manager_ref.get_data_locations(session_id, [data_key1]))[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])

            async def _read_data_all(reader):
                async with reader:
                    return dataserializer.deserialize(reader.read())

            result = await self.waitp(
                (await handler.create_bytes_reader(session_id, data_key1, _promise=True)).then(_read_data_all))
            assert_allclose(result, data1)

            async def _read_data_batch(reader):
                bio = BytesIO()
                async with reader:
                    while True:
                        buf = reader.read(io_size)
                        if buf:
                            bio.write(buf)
                        else:
                            break
                return dataserializer.deserialize(bio.getvalue())

            result = await self.waitp(
                (await handler.create_bytes_reader(session_id, data_key1, _promise=True)).then(_read_data_batch))
            assert_allclose(result, data1)
            await handler.delete(session_id, [data_key1])

    async def testSharedReadAndWritePacked(self, *_):
        test_addr = '127.0.0.1:%d' % get_next_port()
        io_size = dataserializer.HEADER_LENGTH * 2
        async with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
            await pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = await pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            await pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            await pool.create_actor(SharedHolderActor, uid=SharedHolderActor.default_uid())

            data1 = np.random.random((100, 100))
            ser_data1 = dataserializer.serialize(data1)
            block_data1 = dataserializer.dumps(data1, dataserializer.CompressType.NONE)

            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())

            storage_client = test_actor.storage_client
            handler = await storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))

            async def _write_data(ser, writer):
                async with writer:
                    writer.write(ser)

            await self.waitp(
                (await handler.create_bytes_writer(session_id, data_key1, ser_data1.total_bytes,
                                                   packed=True, _promise=True))
                    .then(functools.partial(_write_data, block_data1)))
            self.assertEqual(sorted((await storage_manager_ref.get_data_locations(session_id, [data_key1]))[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])
            await handler.delete(session_id, [data_key1])

            async def _write_data(ser, writer):
                async with writer:
                    with self.assertRaises(IOError):
                        writer.write(ser[:1])

                    for start in range(0, len(ser), io_size):
                        writer.write(ser[start:start + io_size])

            await self.waitp(
                (await handler.create_bytes_writer(session_id, data_key1, ser_data1.total_bytes,
                                                   packed=True, _promise=True))
                    .then(functools.partial(_write_data, block_data1)))
            self.assertEqual(sorted((await storage_manager_ref.get_data_locations(session_id, [data_key1]))[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])

            async def _read_data_all(reader):
                async with reader:
                    return dataserializer.loads(reader.read())

            result = await self.waitp(
                (await handler.create_bytes_reader(session_id, data_key1, packed=True, _promise=True)) \
                    .then(_read_data_all))
            assert_allclose(result, data1)

            async def _read_data_batch(reader):
                bio = BytesIO()
                async with reader:
                    while True:
                        buf = reader.read(io_size)
                        if buf:
                            bio.write(buf)
                        else:
                            break
                return dataserializer.loads(bio.getvalue())

            result = await self.waitp(
                (await handler.create_bytes_reader(session_id, data_key1, packed=True, _promise=True))
                    .then(_read_data_batch))
            assert_allclose(result, data1)
            await handler.delete(session_id, [data_key1])

    async def testSharedPutAndGet(self, *_):
        test_addr = '127.0.0.1:%d' % get_next_port()
        async with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
            await pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = await pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            await pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            await pool.create_actor(SharedHolderActor, uid=SharedHolderActor.default_uid())

            data1 = np.random.random((10, 10))
            data2 = np.random.random((10, 10))
            ser_data2 = dataserializer.serialize(data2)
            bytes_data2 = ser_data2.to_buffer()

            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())
            data_key2 = str(uuid.uuid4())

            storage_client = test_actor.storage_client
            handler = await storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))

            await handler.put_objects(session_id, [data_key1], [data1])
            self.assertEqual(sorted((await storage_manager_ref.get_data_locations(session_id, [data_key1]))[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])
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

    async def testSharedLoadFromBytes(self, *_):
        test_addr = '127.0.0.1:%d' % get_next_port()
        async with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
            await pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = await pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            await pool.create_actor(QuotaActor, 1024 ** 2, uid=MemQuotaActor.default_uid())
            await pool.create_actor(InProcHolderActor)

            await pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            await pool.create_actor(SharedHolderActor, uid=SharedHolderActor.default_uid())

            data1 = np.random.random((10, 10))
            ser_data1 = dataserializer.serialize(data1)

            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())

            storage_client = test_actor.storage_client
            handler = await storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))

            # load from bytes io
            disk_handler = await storage_client.get_storage_handler((0, DataStorageDevice.DISK))
            async with await disk_handler.create_bytes_writer(
                    session_id, data_key1, ser_data1.total_bytes) as writer:
                ser_data1.write_to(writer)

            await self.waitp((await handler.load_from_bytes_io(session_id, [data_key1], disk_handler))
                             .then(lambda *_: None))
            self.assertEqual(sorted((await storage_manager_ref.get_data_locations(session_id, [data_key1]))[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY), (0, DataStorageDevice.DISK)])

            await disk_handler.delete(session_id, [data_key1])
            await handler.delete(session_id, [data_key1])

            # load from bytes io till no capacity
            data_list = [np.random.randint(0, 32767, (655360,), np.int16)
                         for _ in range(20)]
            data_keys = [str(uuid.uuid4()) for _ in range(20)]
            for key, data in zip(data_keys, data_list):
                ser_data = dataserializer.serialize(data)
                async with await disk_handler.create_bytes_writer(
                        session_id, key, ser_data.total_bytes) as writer:
                    ser_data.write_to(writer)

            affected_keys = set()
            try:
                await self.waitp(
                    (await handler.load_from_bytes_io(session_id, data_keys, disk_handler))
                    .then(lambda *_: None))
            except StorageFull as ex:
                affected_keys.update(ex.affected_keys)

            await storage_client.delete(session_id, data_keys, [DataStorageDevice.DISK])

            self.assertLess(len(affected_keys), len(data_keys))
            self.assertGreater(len(affected_keys), 1)
            for k, size in zip(data_keys, await storage_client.get_data_sizes(session_id, data_keys)):
                if k in affected_keys:
                    self.assertIsNone(size)
                else:
                    self.assertIsNotNone(size)

    async def testSharedLoadFromObjects(self, *_):
        test_addr = '127.0.0.1:%d' % get_next_port()
        async with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
            await pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = await pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            await pool.create_actor(QuotaActor, 1024 ** 2, uid=MemQuotaActor.default_uid())
            await pool.create_actor(InProcHolderActor)

            await pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            await pool.create_actor(SharedHolderActor, uid=SharedHolderActor.default_uid())

            data1 = np.random.random((10, 10))

            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())

            storage_client = test_actor.storage_client
            handler = await storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))

            # load from object io
            ref_data1 = weakref.ref(data1)

            proc_handler = await storage_client.get_storage_handler((0, DataStorageDevice.PROC_MEMORY))
            await proc_handler.put_objects(session_id, [data_key1], [data1])
            del data1

            await self.waitp((await handler.load_from_object_io(session_id, [data_key1], proc_handler))
                             .then(lambda *_: None))
            self.assertEqual(sorted((await storage_manager_ref.get_data_locations(session_id, [data_key1]))[0]),
                             [(0, DataStorageDevice.PROC_MEMORY), (0, DataStorageDevice.SHARED_MEMORY)])

            await proc_handler.delete(session_id, [data_key1])
            self.assertIsNone(ref_data1())
            await handler.delete(session_id, [data_key1])

    async def testSharedSpill(self, *_):
        test_addr = '127.0.0.1:%d' % get_next_port()
        async with self.create_pool(n_process=1, address=test_addr) as pool, \
                self.run_actor_test(pool) as test_actor:
            await pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = await pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            await pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            holder_ref = await pool.create_actor(
                SharedHolderActor, self.plasma_storage_size, uid=SharedHolderActor.default_uid())

            session_id = str(uuid.uuid4())
            data_list = [np.random.randint(0, 32767, (655360,), np.int16)
                         for _ in range(20)]
            data_keys = [str(uuid.uuid4()) for _ in range(20)]

            storage_client = test_actor.storage_client
            handler = await storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))
            idx = 0

            async def _fill_data():
                i = 0
                for i, (key, data) in enumerate(zip(data_keys[idx:], data_list)):
                    try:
                        await handler.put_objects(session_id, [key], [data])
                    except StorageFull:
                        break
                return i + idx

            async def _do_spill():
                data_size = (await storage_manager_ref.get_data_sizes(session_id, [data_keys[0]]))[0]
                await self.waitp(
                    (await handler.spill_size(2 * data_size)).then(lambda *_: None)
                )

            # test lift data key
            idx = await _fill_data()
            await handler.lift_data_keys(session_id, [data_keys[0]], _tell=False)
            await _do_spill()

            self.assertEqual(list((await storage_manager_ref.get_data_locations(session_id, [data_keys[0]]))[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])
            self.assertEqual(list((await storage_manager_ref.get_data_locations(session_id, [data_keys[1]]))[0]),
                             [(0, DataStorageDevice.DISK)])

            await handler.put_objects(session_id, [data_keys[idx]], [data_list[idx]])
            self.assertEqual(list((await storage_manager_ref.get_data_locations(session_id, [data_keys[idx]]))[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])
            idx += 1

            # test pin data key
            idx = await _fill_data()
            await holder_ref.lift_data_keys(session_id, [data_keys[0]], last=False)
            pin_token = str(uuid.uuid4())
            pinned_keys = await handler.pin_data_keys(session_id, (data_keys[0],), pin_token)
            self.assertIn(data_keys[0], pinned_keys)
            await _do_spill()

            self.assertEqual(list((await storage_manager_ref.get_data_locations(session_id, [data_keys[0]]))[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])
            self.assertEqual(list((await storage_manager_ref.get_data_locations(session_id, [data_keys[1]]))[0]),
                             [(0, DataStorageDevice.DISK)])

            await handler.put_objects(session_id, [data_keys[idx]], [data_list[idx]])
            self.assertEqual(list((await storage_manager_ref.get_data_locations(session_id, [data_keys[idx]]))[0]),
                             [(0, DataStorageDevice.SHARED_MEMORY)])
            idx += 1

            # test unpin data key
            idx = await _fill_data()
            await handler.unpin_data_keys(session_id, (data_keys[0],), pin_token)
            await _do_spill()

            self.assertEqual(list((await storage_manager_ref.get_data_locations(session_id, [data_keys[0]]))[0]),
                             [(0, DataStorageDevice.DISK)])
