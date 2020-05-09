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
import uuid

import numpy as np
from numpy.testing import assert_allclose

from mars.config import options
from mars.errors import StorageFull, SpillSizeExceeded, PinDataKeyFailed, NoDataToSpill
from mars.tests.core import aio_case
from mars.utils import get_next_port
from mars.worker import WorkerDaemonActor, StatusActor, DispatchActor
from mars.worker.utils import WorkerActor, WorkerClusterInfoActor, build_exc_info
from mars.worker.tests.base import WorkerCase
from mars.worker.storage import StorageManagerActor, PlasmaKeyMapActor, SharedHolderActor, \
    DataStorageDevice


class MockIORunnerActor(WorkerActor):
    """
    Actor handling spill read and write in single disk partition
    """
    _io_runner = True

    def __init__(self):
        super().__init__()
        self._work_items = dict()
        self._submissions = dict()

    async def post_create(self):
        await super().post_create()

        dispatch_ref = self.ctx.actor_ref(DispatchActor.default_uid())
        await dispatch_ref.register_free_slot(self.uid, 'iorunner')

    async def load_from(self, dest_device, session_id, data_keys, src_device, callback):
        session_data_key = (session_id, data_keys[0])
        self._work_items[session_data_key] = (dest_device, src_device, callback)
        if session_data_key in self._submissions:
            exc_info = self._submissions[session_data_key]
            await self.submit_item(session_id, data_keys[0], exc_info)

    async def submit_item(self, session_id, data_key, exc_info=None):
        try:
            dest_device, src_device, cb = self._work_items.pop((session_id, data_key))
        except KeyError:
            self._submissions[(session_id, data_key)] = exc_info
            return

        if exc_info is not None:
            await self.tell_promise(cb, *exc_info, _accept=False)
        else:
            src_handler = await self.storage_client.get_storage_handler(src_device)
            dest_handler = await self.storage_client.get_storage_handler(dest_device)
            (await dest_handler.load_from(session_id, [data_key], src_handler)) \
                .then(lambda *_: self.tell_promise(cb),
                      lambda *exc: self.tell_promise(cb, *exc, _accept=False))

    def get_request_keys(self):
        return [tp[1] for tp in self._work_items.keys()]

    def clear_submissions(self):
        self._submissions.clear()


@aio_case
class Test(WorkerCase):
    def setUp(self):
        super().setUp()
        self._old_min_spill_size = options.worker.min_spill_size
        options.worker.min_spill_size = 0

    def tearDown(self):
        super().tearDown()
        options.worker.min_spill_size = self._old_min_spill_size

    def _start_shared_holder_pool(self):
        this = self
        test_addr = '127.0.0.1:%d' % get_next_port()
        pool, test_actor_ctx = None, None

        class _AsyncContextManager:
            async def __aenter__(self):
                nonlocal pool, test_actor_ctx
                pool = await this.create_pool(n_process=1, address=test_addr).__aenter__()
                test_actor_ctx = this.run_actor_test(pool)
                test_actor = await test_actor_ctx.__aenter__()

                await pool.create_actor(WorkerClusterInfoActor, [test_addr],
                                        uid=WorkerClusterInfoActor.default_uid())
                await pool.create_actor(StatusActor, test_addr, uid=StatusActor.default_uid())

                await pool.create_actor(WorkerDaemonActor, uid=WorkerDaemonActor.default_uid())
                await pool.create_actor(StorageManagerActor, uid=StorageManagerActor.default_uid())
                await pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
                await pool.create_actor(SharedHolderActor, this.plasma_storage_size,
                                        uid=SharedHolderActor.default_uid())
                return pool, test_actor

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await pool.__aexit__(exc_type, exc_val, exc_tb)
                await test_actor_ctx.__aexit__(exc_type, exc_val, exc_tb)

        return _AsyncContextManager()

    async def _fill_shared_storage(self, session_id, key_list, data_list, idx=0):
        storage_client = self._test_actor.storage_client
        shared_handler = await storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))
        i = 0
        for i, (key, data) in enumerate(zip(key_list[idx:], data_list)):
            try:
                await shared_handler.put_objects(session_id, [key], [data])
            except StorageFull:
                break
        return i + idx

    async def testSharedHolderPutAndGet(self):
        async with self._start_shared_holder_pool() as (_pool, test_actor):
            storage_client = test_actor.storage_client

            session_id = str(uuid.uuid4())
            data_list = [np.random.randint(0, 32767, (655360,), np.int16)
                         for _ in range(20)]
            key_list = [str(uuid.uuid4()) for _ in range(20)]

            shared_handler = await storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))
            last_idx = await self._fill_shared_storage(session_id, key_list, data_list)

            pin_token1 = str(uuid.uuid4())
            pinned = await shared_handler.pin_data_keys(session_id, key_list, pin_token1)
            self.assertEqual(sorted(key_list[:last_idx]), sorted(pinned))

            await shared_handler.delete(session_id, [key_list[0]])
            await shared_handler.delete(session_id, [key_list[1]])
            await shared_handler.put_objects(session_id, [key_list[last_idx]], [data_list[last_idx]])
            assert_allclose(data_list[last_idx],
                            (await shared_handler.get_objects(session_id, [key_list[last_idx]]))[0])

            pin_token2 = str(uuid.uuid4())
            pinned = await shared_handler.pin_data_keys(session_id, key_list, pin_token2)
            self.assertEqual(sorted(key_list[2:last_idx + 1]), sorted(pinned))

            await shared_handler.put_objects(session_id, [key_list[last_idx]], [data_list[last_idx]])
            assert_allclose(data_list[last_idx],
                            (await shared_handler.get_objects(session_id, [key_list[last_idx]]))[0])

            unpinned = await shared_handler.unpin_data_keys(session_id, key_list, pin_token1)
            self.assertEqual(sorted(key_list[2:last_idx]), sorted(unpinned))

            unpinned = await shared_handler.unpin_data_keys(session_id, key_list, pin_token2)
            self.assertEqual(sorted(key_list[2:last_idx + 1]), sorted(unpinned))

    async def testSharedHolderSpill(self):
        async with self._start_shared_holder_pool() as (pool, test_actor):
            await pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())
            await pool.create_actor(MockIORunnerActor, uid=MockIORunnerActor.default_uid())

            manager_ref = pool.actor_ref(StorageManagerActor.default_uid())
            shared_holder_ref = pool.actor_ref(SharedHolderActor.default_uid())
            mock_runner_ref = pool.actor_ref(MockIORunnerActor.default_uid())

            storage_client = test_actor.storage_client
            shared_handler = await storage_client.get_storage_handler((0, DataStorageDevice.SHARED_MEMORY))

            session_id = str(uuid.uuid4())
            data_list = [np.random.randint(0, 32767, (655360,), np.int16)
                         for _ in range(20)]
            key_list = [str(uuid.uuid4()) for _ in range(20)]

            await self._fill_shared_storage(session_id, key_list, data_list)
            data_size = (await manager_ref.get_data_sizes(session_id, [key_list[0]]))[0]

            # spill huge sizes
            with self.assertRaises(SpillSizeExceeded):
                await self.waitp(
                    await shared_handler.spill_size(self.plasma_storage_size * 2),
                )

            # spill size of two data chunks
            keys_before = [tp[1] for tp in await shared_holder_ref.dump_keys()]
            pin_token = str(uuid.uuid4())
            await shared_holder_ref.pin_data_keys(session_id, key_list[1:2], pin_token)

            expect_spills = key_list[2:4]

            await shared_holder_ref.lift_data_keys(session_id, [key_list[0]])
            future = asyncio.ensure_future(self.waitp(
                (await shared_handler.spill_size(data_size * 1.5)).then(lambda: None)))

            await asyncio.sleep(0.5)
            # when the key is in spill (here we trigger it manually in mock),
            # it cannot be spilled
            with self.assertRaises(PinDataKeyFailed):
                await shared_holder_ref.pin_data_keys(session_id, key_list[2:3], str(uuid.uuid4()))

            for k in key_list[2:6]:
                await mock_runner_ref.submit_item(session_id, k)
            await future

            await shared_holder_ref.unpin_data_keys(session_id, key_list[1:2], pin_token)
            keys_after = [tp[1] for tp in await shared_holder_ref.dump_keys()]
            self.assertSetEqual(set(keys_before) - set(keys_after), set(expect_spills))

            # spill size of a single chunk, should return immediately
            keys_before = [tp[1] for tp in await shared_holder_ref.dump_keys()]

            await self.waitp((await shared_handler.spill_size(data_size)).then(lambda: None))

            keys_after = [tp[1] for tp in await shared_holder_ref.dump_keys()]
            self.assertSetEqual(set(keys_before), set(keys_after))

            # when all pinned, nothing can be spilled
            # and spill_size() should raises an error
            pin_token = str(uuid.uuid4())
            await shared_holder_ref.pin_data_keys(session_id, key_list, pin_token)

            with self.assertRaises(NoDataToSpill):
                await self.waitp((await shared_handler.spill_size(data_size * 3)).then(lambda: None))

            await shared_holder_ref.unpin_data_keys(session_id, key_list, pin_token)

            # when some errors raise when spilling,
            # spill_size() should report it

            await mock_runner_ref.clear_submissions()
            future = asyncio.ensure_future(self.waitp(
                (await shared_handler.spill_size(data_size * 3)).then(lambda: None)))

            await asyncio.sleep(0.5)
            spill_keys = await mock_runner_ref.get_request_keys()
            await mock_runner_ref.submit_item(
                session_id, spill_keys[0], build_exc_info(SystemError))
            for k in spill_keys[1:]:
                await mock_runner_ref.submit_item(session_id, k)

            with self.assertRaises(SystemError):
                await future
