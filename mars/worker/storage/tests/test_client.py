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

import asyncio
import functools
import os
import time
import uuid
import weakref
from collections import defaultdict

import numpy as np
from numpy.testing import assert_allclose

from mars import promise
from mars.config import options
from mars.distributor import MarsDistributor
from mars.errors import StorageFull
from mars.serialize import dataserializer
from mars.tests.core import aio_case, patch_method
from mars.utils import get_next_port, build_exc_info
from mars.worker import WorkerDaemonActor, MemQuotaActor, QuotaActor, DispatchActor
from mars.worker.utils import WorkerActor
from mars.worker.tests.base import WorkerCase
from mars.worker.storage import StorageManagerActor, IORunnerActor, SharedHolderActor, \
    PlasmaKeyMapActor, InProcHolderActor, StorageHandler, DataStorageDevice


class OtherProcessTestActor(WorkerActor):
    def __init__(self):
        super().__init__()
        self._accept = None
        self._result = None
        self._manager_ref = None

    async def post_create(self):
        await super().post_create()
        self._manager_ref = self.ctx.actor_ref(StorageManagerActor.default_uid())

    def set_result(self, result, accept=True):
        self._result, self._accept = result, accept

    def get_result(self):
        if self._accept is None:
            return None
        elif self._accept:
            return self._result
        else:
            raise self._result[1].with_traceback(self._result[2])

    async def run_copy_test(self, src_location, dest_location):
        self._accept, self._result = None, None

        session_id = str(uuid.uuid4())
        data1 = np.random.randint(0, 32767, (655360,), np.int16)
        key1 = str(uuid.uuid4())

        src_handler = await self.storage_client.get_storage_handler(src_location)
        proc_handler = await self.storage_client.get_storage_handler(dest_location)

        async def _verify_result(*_):
            result = (await proc_handler.get_objects(session_id, [key1]))[0]
            assert_allclose(result, data1)

            devices = (await self._manager_ref.get_data_locations(session_id, [key1]))[0]
            await self.storage_client.delete(session_id, [key1])
            if devices != {src_location, dest_location}:
                raise AssertionError

        await src_handler.put_objects(session_id, [key1], [data1])
        (await self.storage_client.copy_to(session_id, [key1], [dest_location])) \
            .then(_verify_result) \
            .then(lambda *_: self.set_result(1),
                  lambda *exc: self.set_result(exc, accept=False))


@aio_case
class Test(WorkerCase):
    plasma_storage_size = 1024 * 1024 * 10

    def tearDown(self):
        options.worker.lock_free_fileio = False
        super().tearDown()

    async def testClientReadAndWrite(self):
        test_addr = '127.0.0.1:%d' % get_next_port()
        async with self.create_pool(n_process=1, address=test_addr) as pool:
            options.worker.lock_free_fileio = True
            await pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            await pool.create_actor(StorageManagerActor, uid=StorageManagerActor.default_uid())

            await pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())
            await pool.create_actor(IORunnerActor)

            await pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            await pool.create_actor(
                SharedHolderActor, self.plasma_storage_size, uid=SharedHolderActor.default_uid())

            data1 = np.random.random((10, 10))
            ser_data1 = dataserializer.serialize(data1)

            session_id = str(uuid.uuid4())
            data_key1 = str(uuid.uuid4())
            data_key2 = str(uuid.uuid4())

            async with self.run_actor_test(pool) as test_actor:
                storage_client = test_actor.storage_client

                file_names = []

                async def _write_data(ser, writer):
                    file_names.append(writer.filename)
                    self.assertEqual(writer.nbytes, ser_data1.total_bytes)
                    ser.write_to(writer)

                # test creating non-promised writer and write
                async with await storage_client.create_writer(
                        session_id, data_key1, ser_data1.total_bytes, (DataStorageDevice.DISK,),
                        _promise=False) as writer:
                    await _write_data(ser_data1, writer)
                self.assertTrue(os.path.exists(file_names[0]))
                self.assertEqual(sorted((await storage_client.get_data_locations(session_id, [data_key1]))[0]),
                                 [(0, DataStorageDevice.DISK)])

                await storage_client.delete(session_id, [data_key1])

                # test creating promised writer and write
                file_names[:] = []
                await self.waitp((await storage_client.create_writer(
                        session_id, data_key2, ser_data1.total_bytes, (DataStorageDevice.DISK,)))
                    .then(functools.partial(_write_data, ser_data1)))
                self.assertTrue(os.path.exists(file_names[0]))
                self.assertEqual(sorted((await storage_client.get_data_locations(session_id, [data_key2]))[0]),
                                 [(0, DataStorageDevice.DISK)])

                async def _read_data(reader):
                    async with reader:
                        return dataserializer.deserialize(reader.read())

                # test creating reader when data exist in location
                result = (await self.waitp(
                    (await storage_client.create_reader(session_id, data_key2, (DataStorageDevice.DISK,)))
                    .then(_read_data)))[0]
                assert_allclose(result, data1)

                # test creating reader when no data in location (should raise)
                with self.assertRaises(IOError):
                    await storage_client.create_reader(
                        session_id, data_key2, (DataStorageDevice.SHARED_MEMORY,), _promise=False)

                # test creating reader when copy needed
                await (self.waitp(
                    (await storage_client.create_reader(session_id, data_key2, (DataStorageDevice.SHARED_MEMORY,)))
                    .then(_read_data)))
                self.assertEqual(sorted((await storage_client.get_data_locations(session_id, [data_key2]))[0]),
                                 [(0, DataStorageDevice.SHARED_MEMORY), (0, DataStorageDevice.DISK)])

                await storage_client.delete(session_id, [data_key2])
                while os.path.exists(file_names[0]):
                    await asyncio.sleep(0.05)
                self.assertFalse(os.path.exists(file_names[0]))

    async def testClientPutAndGet(self):
        import gc
        test_addr = '127.0.0.1:%d' % get_next_port()
        async with self.create_pool(n_process=1, address=test_addr) as pool:
            await pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            await pool.create_actor(StorageManagerActor, uid=StorageManagerActor.default_uid())

            await pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())
            await pool.create_actor(IORunnerActor)

            await pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            await pool.create_actor(
                SharedHolderActor, self.plasma_storage_size, uid=SharedHolderActor.default_uid())
            await pool.create_actor(InProcHolderActor, uid='w:1:InProcHolderActor')

            session_id = str(uuid.uuid4())
            data_list = [np.random.randint(0, 32767, (655360,), np.int16)
                         for _ in range(20)]
            data_keys = [str(uuid.uuid4()) for _ in range(20)]
            data_dict = dict(zip(data_keys, data_list))

            async with self.run_actor_test(pool) as test_actor:
                storage_client = test_actor.storage_client

                # check batch object put with size exceeds
                (await storage_client.put_objects(
                    session_id, data_keys, data_list,
                    [DataStorageDevice.SHARED_MEMORY, DataStorageDevice.PROC_MEMORY])) \
                    .then(functools.partial(test_actor.set_result),
                          lambda *exc: test_actor.set_result(exc, accept=False))
                await self.get_result(5)
                locations = await storage_client.get_data_locations(session_id, data_keys)
                loc_to_keys = defaultdict(list)
                for key, location in zip(data_keys, locations):
                    self.assertEqual(len(location), 1)
                    loc_to_keys[list(location)[0][-1]].append(key)
                self.assertGreater(len(loc_to_keys[DataStorageDevice.PROC_MEMORY]), 1)
                self.assertGreater(len(loc_to_keys[DataStorageDevice.SHARED_MEMORY]), 1)

                # check get object with all cases
                with self.assertRaises(IOError):
                    first_shared_key = loc_to_keys[DataStorageDevice.SHARED_MEMORY][0]
                    await storage_client.get_object(session_id, first_shared_key,
                                                    [DataStorageDevice.PROC_MEMORY], _promise=False)

                shared_objs = await storage_client.get_objects(
                    session_id, [first_shared_key], [DataStorageDevice.SHARED_MEMORY], _promise=False)
                self.assertEqual(len(shared_objs), 1)
                assert_allclose(shared_objs[0], data_dict[first_shared_key])

                (await storage_client.get_object(session_id, first_shared_key,
                                                 [DataStorageDevice.PROC_MEMORY], _promise=True)) \
                    .then(functools.partial(test_actor.set_result),
                          lambda *exc: test_actor.set_result(exc, accept=False))
                assert_allclose(await self.get_result(5), data_dict[first_shared_key])

                await storage_client.delete(session_id, data_keys)
                ref = weakref.ref(data_dict[data_keys[0]])
                await storage_client.put_objects(
                    session_id, data_keys[:1], [ref()], [DataStorageDevice.SHARED_MEMORY])
                await asyncio.sleep(0)
                data_list[:] = []
                data_dict.clear()
                self.assertIsNone(ref())

    async def testLoadStoreInOtherProcess(self):
        test_addr = '127.0.0.1:%d' % get_next_port()
        async with self.create_pool(n_process=3, address=test_addr, distributor=MarsDistributor(3)) as pool:
            await pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            await pool.create_actor(StorageManagerActor, uid=StorageManagerActor.default_uid())

            await pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())

            await pool.create_actor(QuotaActor, 1024 ** 2, uid=MemQuotaActor.default_uid())

            await pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            await pool.create_actor(SharedHolderActor, self.plasma_storage_size,
                                    uid=SharedHolderActor.default_uid())

            await pool.create_actor(InProcHolderActor, uid='w:1:InProcHolderActor1')
            await pool.create_actor(InProcHolderActor, uid='w:2:InProcHolderActor2')
            await pool.create_actor(IORunnerActor, lock_free=True, dispatched=False, uid=IORunnerActor.gen_uid(1))

            test_ref = await pool.create_actor(OtherProcessTestActor, uid='w:0:OtherProcTest')

            async def _get_result():
                start_time = time.time()
                while await test_ref.get_result() is None:
                    await asyncio.sleep(0.5)
                    if time.time() - start_time > 10:
                        raise TimeoutError

            await test_ref.run_copy_test((0, DataStorageDevice.SHARED_MEMORY),
                                         (1, DataStorageDevice.PROC_MEMORY), _tell=True)
            await _get_result()

            await test_ref.run_copy_test((1, DataStorageDevice.PROC_MEMORY),
                                         (0, DataStorageDevice.SHARED_MEMORY), _tell=True)
            await _get_result()

            await test_ref.run_copy_test((1, DataStorageDevice.PROC_MEMORY),
                                         (2, DataStorageDevice.PROC_MEMORY), _tell=True)
            await _get_result()

    async def testClientSpill(self, *_):
        test_addr = '127.0.0.1:%d' % get_next_port()
        async with self.create_pool(n_process=1, address=test_addr) as pool:
            await pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
            storage_manager_ref = await pool.create_actor(
                StorageManagerActor, uid=StorageManagerActor.default_uid())

            await pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())
            await pool.create_actor(IORunnerActor)

            await pool.create_actor(QuotaActor, 1024 ** 2, uid=MemQuotaActor.default_uid())
            await pool.create_actor(InProcHolderActor)

            await pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            await pool.create_actor(SharedHolderActor, self.plasma_storage_size,
                                    uid=SharedHolderActor.default_uid())

            session_id = str(uuid.uuid4())
            data_list = [np.random.randint(0, 32767, (655360,), np.int16)
                         for _ in range(20)]
            data_keys = [str(uuid.uuid4()) for _ in range(20)]

            async with self.run_actor_test(pool) as test_actor:
                storage_client = test_actor.storage_client
                idx = 0

                shared_handler = await storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))
                proc_handler = await storage_client.get_storage_handler((0, DataStorageDevice.PROC_MEMORY))

                async def _fill_data():
                    i = 0
                    for i, (key, data) in enumerate(zip(data_keys[idx:], data_list)):
                        try:
                            await shared_handler.put_objects(session_id, [key], [data])
                        except StorageFull:
                            break
                    return i + idx

                idx = await _fill_data()

                # test copying non-existing keys
                (await storage_client.copy_to(session_id, ['non-exist-key'], [DataStorageDevice.SHARED_MEMORY])) \
                    .then(lambda *_: test_actor.set_result(None),
                          lambda *exc: test_actor.set_result(exc, accept=False))
                with self.assertRaises(KeyError):
                    await self.get_result(5)

                # test copying into containing locations
                (await storage_client.copy_to(session_id, [data_keys[0]], [DataStorageDevice.SHARED_MEMORY])) \
                    .then(lambda *_: test_actor.set_result(None),
                          lambda *exc: test_actor.set_result(exc, accept=False))
                await self.get_result(5)

                self.assertEqual(sorted((await storage_manager_ref.get_data_locations(session_id, [data_keys[0]]))[0]),
                                 [(0, DataStorageDevice.SHARED_MEMORY)])

                # test unsuccessful copy when no data at target
                def _mock_load_from(*_, **__):
                    return promise.finished(*build_exc_info(SystemError), _accept=False)

                with patch_method(StorageHandler.load_from, _mock_load_from), \
                        self.assertRaises(SystemError):
                    (await storage_client.copy_to(session_id, [data_keys[0]], [DataStorageDevice.DISK])) \
                        .then(lambda *_: test_actor.set_result(None),
                              lambda *exc: test_actor.set_result(exc, accept=False))
                    await self.get_result(5)

                # test successful copy for multiple objects
                await storage_client.delete(session_id, [data_keys[idx - 1]])
                ref_data = weakref.ref(data_list[idx])
                ref_data2 = weakref.ref(data_list[idx + 1])
                await proc_handler.put_objects(session_id, data_keys[idx:idx + 2], data_list[idx:idx + 2])
                data_list[idx:idx + 2] = [None, None]

                (await storage_client.copy_to(session_id, data_keys[idx:idx + 2],
                                              [DataStorageDevice.SHARED_MEMORY, DataStorageDevice.DISK])) \
                    .then(lambda *_: test_actor.set_result(None),
                          lambda *exc: test_actor.set_result(exc, accept=False))
                await self.get_result(5)

                await proc_handler.delete(session_id, data_keys[idx:idx + 2])

                self.assertEqual(await storage_manager_ref.get_data_locations(session_id, data_keys[idx:idx + 2]),
                                 [{(0, DataStorageDevice.SHARED_MEMORY)}, {(0, DataStorageDevice.DISK)}])
                self.assertIsNone(ref_data())
                self.assertIsNone(ref_data2())

                # test copy with spill
                idx += 2
                await proc_handler.put_objects(session_id, [data_keys[idx]], [data_list[idx]])

                (await storage_client.copy_to(session_id, [data_keys[idx]], [DataStorageDevice.SHARED_MEMORY])) \
                    .then(lambda *_: test_actor.set_result(None),
                          lambda *exc: test_actor.set_result(exc, accept=False))
                await self.get_result(5)

                self.assertEqual(sorted((await storage_manager_ref.get_data_locations(session_id, [data_keys[idx]]))[0]),
                                 [(0, DataStorageDevice.PROC_MEMORY), (0, DataStorageDevice.SHARED_MEMORY)])
