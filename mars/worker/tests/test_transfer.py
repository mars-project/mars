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
import multiprocessing
import os
import signal
import tempfile
import time
import uuid
from collections import defaultdict
from io import BytesIO
from queue import Empty

import numpy as np
from numpy.testing import assert_array_equal
from pyarrow import plasma

from mars import promise
from mars.config import options
from mars.errors import DependencyMissing, ExecutionInterrupted, WorkerDead
from mars.scheduler import ChunkMetaActor
from mars.scheduler.utils import SchedulerClusterInfoActor
from mars.serialize import dataserializer
from mars.tests.core import aio_case, patch_method, create_actor_pool
from mars.utils import get_next_port, build_exc_info, wait_results, aio_run
from mars.worker import SenderActor, ReceiverManagerActor, ReceiverWorkerActor, \
    DispatchActor, QuotaActor, MemQuotaActor, StorageManagerActor, IORunnerActor, \
    StatusActor, SharedHolderActor, InProcHolderActor
from mars.worker.storage import DataStorageDevice, StorageClient
from mars.worker.storage.sharedstore import PlasmaKeyMapActor
from mars.worker.tests.base import WorkerCase, StorageClientActor
from mars.worker.transfer import ReceiveStatus, ReceiverDataMeta
from mars.worker.utils import WorkerActor, WorkerClusterInfoActor


class MockReceiverWorkerActor(WorkerActor):
    """
    Actor handling receiving data from a SenderActor
    """
    def __init__(self):
        super().__init__()
        self._dispatch_ref = None
        self._receiver_manager_ref = None

        self._data_metas = dict()
        self._data_writers = dict()
        self._callbacks = defaultdict(list)

        self._receive_delays = dict()
        self._receive_errors = set()

    async def post_create(self):
        await super().post_create()
        self._dispatch_ref = self.ctx.actor_ref(DispatchActor.default_uid())
        await self._dispatch_ref.register_free_slot(self.uid, 'receiver')
        self._receiver_manager_ref = self.ctx.actor_ref(ReceiverManagerActor.default_uid())
        if not await self.ctx.has_actor(self._receiver_manager_ref):
            self._receiver_manager_ref = None

    def set_status(self, session_id, chunk_key, status):
        query_key = (session_id, chunk_key)
        try:
            self._data_metas[query_key].status = status
        except KeyError:
            self._data_metas[query_key] = ReceiverDataMeta(status=status)

    def get_result_data(self, session_id, chunk_key):
        buf = self._data_writers[(session_id, chunk_key)].getvalue()
        return dataserializer.loads(buf)

    def set_receive_delay_key(self, session_id, chunk_key, delay):
        self._receive_delays[(session_id, chunk_key)] = delay

    def set_receive_error_key(self, session_id, chunk_key):
        self._receive_errors.add((session_id, chunk_key))

    def check_statuses(self, session_id, chunk_keys):
        try:
            return [self._data_metas[(session_id, k)].status for k in chunk_keys]
        except KeyError:
            return ReceiveStatus.NOT_STARTED

    async def create_data_writers(self, session_id, chunk_keys, data_sizes, sender_ref,
                                  ensure_cached=True, pin_token=None, timeout=0,
                                  use_promise=True, callback=None):
        for chunk_key, data_size in zip(chunk_keys, data_sizes):
            query_key = (session_id, chunk_key)
            if query_key in self._data_metas and \
                    self._data_metas[query_key].status in (ReceiveStatus.RECEIVED, ReceiveStatus.RECEIVING):
                await self.tell_promise(callback, self.address, self._data_metas[query_key].status)
                return
            self._data_metas[query_key] = ReceiverDataMeta(chunk_size=data_size, status=ReceiveStatus.RECEIVING)
            self._data_writers[query_key] = BytesIO()
        if callback:
            await self.tell_promise(callback)

    async def receive_data_part(self, session_id, chunk_keys, end_marks, *data_parts):
        finished_keys = []
        for chunk_key, data_part, end_mark in zip(chunk_keys, data_parts, end_marks):
            query_key = (session_id, chunk_key)
            if query_key in self._receive_delays:
                await asyncio.sleep(self._receive_delays[query_key])
            if query_key in self._receive_errors:
                raise ValueError
            meta = self._data_metas[query_key]  # type: ReceiverDataMeta
            self._data_writers[query_key].write(data_part)
            if end_mark:
                finished_keys.append(chunk_key)
                meta.status = ReceiveStatus.RECEIVED
        if self._receiver_manager_ref:
            self._receiver_manager_ref.notify_keys_finish(session_id, finished_keys, _tell=True, _wait=False)

    async def cancel_receive(self, session_id, chunk_keys, exc_info=None):
        if exc_info is None:
            exc_info = build_exc_info(ExecutionInterrupted)
        for k in chunk_keys:
            self._data_metas[(session_id, k)].status = ReceiveStatus.ERROR
        if self._receiver_manager_ref:
            await self._receiver_manager_ref.notify_keys_finish(
                session_id, chunk_keys, *exc_info, _accept=False, _tell=True)


def start_transfer_test_pool(**kwargs):
    address = kwargs.pop('address')
    plasma_size = kwargs.pop('plasma_size')

    class _AsyncContextManager:
        def __init__(self):
            self._pool = None

        async def __aenter__(self):
            pool = self._pool = await create_actor_pool(
                n_process=1, address=address, **kwargs).__aenter__()
            await pool.create_actor(SchedulerClusterInfoActor, [address],
                                    uid=SchedulerClusterInfoActor.default_uid())
            await pool.create_actor(WorkerClusterInfoActor, [address],
                                    uid=WorkerClusterInfoActor.default_uid())

            await pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
            await pool.create_actor(StorageManagerActor, uid=StorageManagerActor.default_uid())
            await pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())
            await pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())
            await pool.create_actor(QuotaActor, 1024 * 1024 * 20, uid=MemQuotaActor.default_uid())
            await pool.create_actor(SharedHolderActor, plasma_size,
                                    uid=SharedHolderActor.default_uid())
            await pool.create_actor(StatusActor, address, uid=StatusActor.default_uid())
            await pool.create_actor(IORunnerActor)
            await pool.create_actor(StorageClientActor, uid=StorageClientActor.default_uid())
            await pool.create_actor(InProcHolderActor)
            await pool.create_actor(ReceiverManagerActor, uid=ReceiverManagerActor.default_uid())
            return pool

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self._pool.actor_ref(SharedHolderActor.default_uid()).destroy()
            await self._pool.__aexit__(exc_type, exc_val, exc_tb)

    return _AsyncContextManager()


def run_transfer_worker(pool_address, session_id, chunk_keys, spill_dir, msg_queue):
    async def _run_transfer_worker():
        options.worker.spill_directory = spill_dir
        plasma_size = 1024 * 1024 * 10

        # don't use multiple with-statement as we need the options be forked
        with plasma.start_plasma_store(plasma_size) as store_args:
            options.worker.plasma_socket = plasma_socket = store_args[0]

            async with start_transfer_test_pool(address=pool_address, plasma_size=plasma_size) as pool:
                storage_client_ref = await pool.create_actor(StorageClientActor)

                for _ in range(2):
                    await pool.create_actor(SenderActor, uid='%s' % str(uuid.uuid4()))
                    await pool.create_actor(ReceiverWorkerActor, uid='%s' % str(uuid.uuid4()))

                for idx in range(0, len(chunk_keys) - 7):
                    data = np.ones((640 * 1024,), dtype=np.int16) * idx
                    await storage_client_ref.put_objects(
                        session_id, [chunk_keys[idx]], [data], [DataStorageDevice.PROC_MEMORY])
                for idx in range(len(chunk_keys) - 7, len(chunk_keys)):
                    data = np.ones((640 * 1024,), dtype=np.int16) * idx
                    await storage_client_ref.put_objects(
                        session_id, [chunk_keys[idx]], [data], [DataStorageDevice.SHARED_MEMORY])

                locations = await storage_client_ref.get_data_locations(session_id, chunk_keys)
                while not all(locations):
                    await asyncio.sleep(0.1)

                for idx in range(0, len(chunk_keys) - 7):
                    await storage_client_ref.copy_to(
                        session_id, [chunk_keys[idx]], [DataStorageDevice.DISK])

                data_locations = await storage_client_ref.get_data_locations(session_id, chunk_keys[:-7])
                while not all((0, DataStorageDevice.DISK) in locations
                              for locations in data_locations):
                    data_locations = await storage_client_ref.get_data_locations(session_id, chunk_keys[:-7])
                    await asyncio.sleep(0.1)

                msg_queue.put(plasma_socket)
                t = time.time()
                while True:
                    try:
                        msg_queue.get_nowait()
                    except Empty:
                        if time.time() > t + 60:
                            raise SystemError('Transfer finish timed out.')
                        await asyncio.sleep(0.1)

    aio_run(_run_transfer_worker())


@aio_case
class Test(WorkerCase):
    def setUp(self):
        super().setUp()
        self._old_block_size = options.worker.transfer_block_size
        options.worker.transfer_block_size = 4 * 1024

    def tearDown(self):
        super().tearDown()
        options.worker.transfer_block_size = self._old_block_size
        self.rm_spill_dirs(options.worker.spill_directory)

    async def testSender(self):
        send_pool_addr = '127.0.0.1:%d' % get_next_port()
        recv_pool_addr = '127.0.0.1:%d' % get_next_port()
        recv_pool_addr2 = '127.0.0.1:%d' % get_next_port()

        options.worker.spill_directory = tempfile.mkdtemp(prefix='mars_test_sender_')
        session_id = str(uuid.uuid4())

        mock_data = np.array([1, 2, 3, 4])
        chunk_key1 = str(uuid.uuid4())
        chunk_key2 = str(uuid.uuid4())
        chunk_key3 = str(uuid.uuid4())
        chunk_key4 = str(uuid.uuid4())
        chunk_key5 = str(uuid.uuid4())
        chunk_key6 = str(uuid.uuid4())

        def start_send_recv_pool():
            this = self
            sp, rp = None, None

            class _AsyncContextManager:
                async def __aenter__(self):
                    nonlocal sp, rp
                    sp = await start_transfer_test_pool(
                        address=send_pool_addr, plasma_size=this.plasma_storage_size).__aenter__()
                    rp = await start_transfer_test_pool(
                        address=recv_pool_addr, plasma_size=this.plasma_storage_size).__aenter__()
                    await sp.create_actor(SenderActor, uid=SenderActor.default_uid())
                    await rp.create_actor(MockReceiverWorkerActor, uid=ReceiverWorkerActor.default_uid())
                    return sp, rp

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    await rp.__aexit__(exc_type, exc_val, exc_tb)
                    await sp.__aexit__(exc_type, exc_val, exc_tb)

            return _AsyncContextManager()

        async with start_send_recv_pool() as (send_pool, recv_pool), \
                self.run_actor_test(send_pool) as test_actor:
            sender_ref = send_pool.actor_ref(SenderActor.default_uid())

            storage_client = test_actor.storage_client
            sender_ref_p = test_actor.promise_ref(sender_ref)

            # SCENARIO 1: send when data missing
            with self.assertRaises(DependencyMissing):
                await self.waitp(sender_ref_p.send_data(
                    session_id, ['non_exist'], [recv_pool_addr], _promise=True))

            # SCENARIO 2: send data to non-exist endpoint which causes error
            await self.waitp(await storage_client.put_objects(
                session_id, [chunk_key1], [mock_data], [DataStorageDevice.SHARED_MEMORY]))
            with self.assertRaises((BrokenPipeError, ConnectionRefusedError)):
                await self.waitp(sender_ref_p.send_data(
                    session_id, [chunk_key1], [recv_pool_addr2], _promise=True))

            async with start_transfer_test_pool(
                    address=recv_pool_addr2, plasma_size=self.plasma_storage_size) as rp2:
                mock_recv_ref2 = await rp2.create_actor(
                    MockReceiverWorkerActor, uid=ReceiverWorkerActor.default_uid())

                # SCENARIO 3: send data to multiple targets
                await self.waitp(await storage_client.put_objects(
                    session_id, [chunk_key2], [mock_data], [DataStorageDevice.SHARED_MEMORY]))
                await self.waitp(sender_ref_p.send_data(
                    session_id, [chunk_key2], [recv_pool_addr, recv_pool_addr2],
                    block_size=128, _promise=True))
                # send data to already transferred / transferring
                await self.waitp(sender_ref_p.send_data(
                    session_id, [chunk_key2], [recv_pool_addr, recv_pool_addr2], _promise=True))
                assert_array_equal(mock_data, await mock_recv_ref2.get_result_data(session_id, chunk_key2))

                # SCENARIO 4: send multiple data at one time
                await self.waitp(await storage_client.put_objects(
                    session_id, [chunk_key3, chunk_key4], [mock_data] * 2,
                    [DataStorageDevice.SHARED_MEMORY]))
                await self.waitp(sender_ref_p.send_data(
                    session_id, [chunk_key3, chunk_key4], [recv_pool_addr2], _promise=True))
                assert_array_equal(mock_data, await mock_recv_ref2.get_result_data(session_id, chunk_key3))
                assert_array_equal(mock_data, await mock_recv_ref2.get_result_data(session_id, chunk_key4))

                # SCENARIO 5: send chunks already under transfer
                await self.waitp(await storage_client.put_objects(
                    session_id, [chunk_key5], [mock_data], [DataStorageDevice.SHARED_MEMORY]))
                mock_recv_ref2.set_receive_delay_key(session_id, chunk_key5, 1)
                sender_ref_p.send_data(
                    session_id, [chunk_key2, chunk_key5], [recv_pool_addr2], _promise=True)
                await self.waitp(sender_ref_p.send_data(
                    session_id, [chunk_key5], [recv_pool_addr2], _promise=True))
                assert_array_equal(mock_data, await mock_recv_ref2.get_result_data(session_id, chunk_key5))

                # SCENARIO 6: send chunks already under transfer
                await self.waitp(await storage_client.put_objects(
                    session_id, [chunk_key6], [mock_data], [DataStorageDevice.SHARED_MEMORY]))
                await mock_recv_ref2.set_receive_error_key(session_id, chunk_key6)
                with self.assertRaises(ValueError):
                    await self.waitp(sender_ref_p.send_data(
                        session_id, [chunk_key6], [recv_pool_addr2], _promise=True))

    async def testReceiverManager(self):
        pool_addr = 'localhost:%d' % get_next_port()
        session_id = str(uuid.uuid4())

        mock_data = np.array([1, 2, 3, 4])
        serialized_arrow_data = dataserializer.serialize(mock_data)
        data_size = serialized_arrow_data.total_bytes
        serialized_mock_data = serialized_arrow_data.to_buffer()

        chunk_key1 = str(uuid.uuid4())
        chunk_key2 = str(uuid.uuid4())
        chunk_key3 = str(uuid.uuid4())
        chunk_key4 = str(uuid.uuid4())
        chunk_key5 = str(uuid.uuid4())
        chunk_key6 = str(uuid.uuid4())
        chunk_key7 = str(uuid.uuid4())

        async with start_transfer_test_pool(
                address=pool_addr, plasma_size=self.plasma_storage_size) as pool, \
                self.run_actor_test(pool) as test_actor:
            mock_receiver_ref = await pool.create_actor(MockReceiverWorkerActor, uid=str(uuid.uuid4()))
            storage_client = test_actor.storage_client
            receiver_manager_ref = test_actor.promise_ref(ReceiverManagerActor.default_uid())

            # SCENARIO 1: test transferring existing keys
            await self.waitp(
                (await storage_client.create_writer(session_id, chunk_key1,
                                                    serialized_arrow_data.total_bytes,
                                                    [DataStorageDevice.DISK]))
                    .then(lambda writer: promise.finished().then(lambda *_: writer.write(serialized_arrow_data))
                          .then(lambda *_: writer.close()))
            )
            result = await self.waitp(receiver_manager_ref.create_data_writers(
                    session_id, [chunk_key1], [data_size], test_actor, _promise=True))
            self.assertEqual(result[0].uid, mock_receiver_ref.uid)
            self.assertEqual(result[1][0], ReceiveStatus.RECEIVED)

            # test adding callback for transferred key (should return immediately)
            result = await self.waitp(receiver_manager_ref.add_keys_callback(
                session_id, [chunk_key1], _promise=True))
            self.assertTupleEqual(result, ())

            await receiver_manager_ref.register_pending_keys(session_id, [chunk_key1, chunk_key2])
            self.assertEqual(await receiver_manager_ref.filter_receiving_keys(
                session_id, [chunk_key1, chunk_key2, 'non_exist']), [chunk_key2])

            # SCENARIO 2: test transferring new keys and wait on listeners
            result = await self.waitp(receiver_manager_ref.create_data_writers(
                session_id, [chunk_key2, chunk_key3], [data_size] * 2, test_actor, _promise=True))
            self.assertEqual(result[0].uid, mock_receiver_ref.uid)
            self.assertIsNone(result[1][0])

            # transfer with transferring keys will report RECEIVING
            result = await self.waitp(receiver_manager_ref.create_data_writers(
                session_id, [chunk_key2], [data_size], test_actor, _promise=True))
            self.assertEqual(result[1][0], ReceiveStatus.RECEIVING)

            # add listener and finish transfer
            receiver_manager_ref.add_keys_callback(session_id, [chunk_key1, chunk_key2], _promise=True) \
                .then(lambda *s: test_actor.set_result(s))
            await mock_receiver_ref.receive_data_part(
                session_id, [chunk_key2], [True], serialized_mock_data)
            await mock_receiver_ref.receive_data_part(
                session_id, [chunk_key3], [True], serialized_mock_data)
            await self.get_result(5)

            # SCENARIO 3: test listening on multiple transfers
            receiver_manager_ref.create_data_writers(
                session_id, [chunk_key4, chunk_key5], [data_size] * 2, test_actor, _promise=True) \
                .then(lambda *s: test_actor.set_result(s))
            await self.get_result(5)
            # add listener
            receiver_manager_ref.add_keys_callback(session_id, [chunk_key4, chunk_key5], _promise=True) \
                .then(lambda *s: test_actor.set_result(s))
            await mock_receiver_ref.receive_data_part(
                session_id, [chunk_key4], [True], serialized_mock_data)
            # when some chunks are not transferred, promise will not return
            with self.assertRaises(TimeoutError):
                await self.get_result(0.5)
            await mock_receiver_ref.receive_data_part(
                session_id, [chunk_key5], [True], serialized_mock_data)
            await self.get_result(5)

            # SCENARIO 4: test listening on transfer with errors
            await self.waitp(receiver_manager_ref.create_data_writers(
                session_id, [chunk_key6], [data_size], test_actor, _promise=True))
            receiver_manager_ref.add_keys_callback(session_id, [chunk_key6], _promise=True) \
                .then(lambda *s: test_actor.set_result(s)) \
                .catch(lambda *exc: test_actor.set_result(exc, accept=False))
            await mock_receiver_ref.cancel_receive(session_id, [chunk_key6])
            with self.assertRaises(ExecutionInterrupted):
                await self.get_result(5)

            # SCENARIO 5: test creating writers without promise
            ref, statuses = await receiver_manager_ref.create_data_writers(
                session_id, [chunk_key7], [data_size], test_actor, use_promise=False)
            self.assertIsNone(statuses[0])
            self.assertEqual(ref.uid, mock_receiver_ref.uid)

    async def testReceiverWorker(self):
        pool_addr = 'localhost:%d' % get_next_port()
        options.worker.spill_directory = tempfile.mkdtemp(prefix='mars_test_receiver_')
        session_id = str(uuid.uuid4())

        mock_data = np.array([1, 2, 3, 4])
        serialized_arrow_data = dataserializer.serialize(mock_data)
        data_size = serialized_arrow_data.total_bytes
        dumped_mock_data = dataserializer.dumps(mock_data)

        chunk_key1 = str(uuid.uuid4())
        chunk_key2 = str(uuid.uuid4())
        chunk_key3 = str(uuid.uuid4())
        chunk_key4 = str(uuid.uuid4())
        chunk_key5 = str(uuid.uuid4())
        chunk_key6 = str(uuid.uuid4())
        chunk_key7 = str(uuid.uuid4())
        chunk_key8 = str(uuid.uuid4())
        chunk_key9 = str(uuid.uuid4())

        async with start_transfer_test_pool(
                address=pool_addr, plasma_size=self.plasma_storage_size) as pool, \
                self.run_actor_test(pool) as test_actor:
            storage_client = test_actor.storage_client
            receiver_ref = test_actor.promise_ref(
                await pool.create_actor(ReceiverWorkerActor, uid=str(uuid.uuid4())))
            receiver_manager_ref = test_actor.promise_ref(ReceiverManagerActor.default_uid())

            # SCENARIO 1: create two writers and write with chunks
            await self.waitp(receiver_ref.create_data_writers(
                session_id, [chunk_key1, chunk_key2], [data_size] * 2, test_actor, _promise=True))
            await receiver_ref.receive_data_part(
                session_id, [chunk_key1, chunk_key2], [True, False],
                dumped_mock_data, dumped_mock_data[:len(dumped_mock_data) // 2])
            self.assertEqual(await receiver_ref.check_status(session_id, chunk_key1),
                             ReceiveStatus.RECEIVED)
            self.assertEqual(await receiver_ref.check_status(session_id, chunk_key2),
                             ReceiveStatus.RECEIVING)
            await receiver_ref.receive_data_part(
                session_id, [chunk_key2], [True], dumped_mock_data[len(dumped_mock_data) // 2:])
            self.assertEqual(await receiver_ref.check_status(session_id, chunk_key2),
                             ReceiveStatus.RECEIVED)
            assert_array_equal(await storage_client.get_object(
                session_id, chunk_key1, [DataStorageDevice.SHARED_MEMORY], _promise=False), mock_data)
            assert_array_equal(await storage_client.get_object(
                session_id, chunk_key2, [DataStorageDevice.SHARED_MEMORY], _promise=False), mock_data)

            # SCENARIO 2: one of the writers failed to create,
            # will test both existing and non-existing keys
            old_create_writer = StorageClient.create_writer

            async def _create_writer_with_fail(self, session_id, chunk_key, *args, **kwargs):
                if chunk_key == fail_key:
                    if kwargs.get('_promise', True):
                        return promise.finished(*build_exc_info(ValueError), **dict(_accept=False))
                    else:
                        raise ValueError
                return await old_create_writer(self, session_id, chunk_key, *args, **kwargs)

            with patch_method(StorageClient.create_writer, new=_create_writer_with_fail), \
                    self.assertRaises(ValueError):
                fail_key = chunk_key4
                await self.waitp(receiver_ref.create_data_writers(
                    session_id, [chunk_key3, chunk_key4, chunk_key5],
                    [data_size] * 3, test_actor, ensure_cached=False, _promise=True))
            self.assertEqual(await receiver_ref.check_status(session_id, chunk_key3),
                             ReceiveStatus.NOT_STARTED)
            self.assertEqual(await receiver_ref.check_status(session_id, chunk_key4),
                             ReceiveStatus.NOT_STARTED)
            self.assertEqual(await receiver_ref.check_status(session_id, chunk_key5),
                             ReceiveStatus.NOT_STARTED)

            with patch_method(StorageClient.create_writer, new=_create_writer_with_fail):
                fail_key = chunk_key2
                await self.waitp(receiver_ref.create_data_writers(
                    session_id, [chunk_key2, chunk_key3], [data_size] * 2, test_actor,
                    ensure_cached=False, _promise=True))

            # SCENARIO 3: transfer timeout
            await receiver_manager_ref.register_pending_keys(session_id, [chunk_key6])
            await self.waitp(receiver_ref.create_data_writers(
                session_id, [chunk_key6], [data_size], test_actor, timeout=1, _promise=True))
            with self.assertRaises(TimeoutError):
                await self.waitp(receiver_manager_ref.add_keys_callback(
                    session_id, [chunk_key6], _promise=True))

            # SCENARIO 4: cancelled transfer (both before and during transfer)
            await receiver_manager_ref.register_pending_keys(session_id, [chunk_key7])
            await self.waitp(receiver_ref.create_data_writers(
                session_id, [chunk_key7], [data_size], test_actor, timeout=1, _promise=True))
            await receiver_ref.cancel_receive(session_id, [chunk_key2, chunk_key7])
            with self.assertRaises(ExecutionInterrupted):
                await receiver_ref.receive_data_part(session_id, [chunk_key7], [False],
                                                     dumped_mock_data[:len(dumped_mock_data) // 2])
            with self.assertRaises(ExecutionInterrupted):
                await self.waitp(receiver_manager_ref.add_keys_callback(
                    session_id, [chunk_key7], _promise=True))

            # SCENARIO 5: sender halt and receiver is notified (reusing previous unsuccessful key)
            await receiver_manager_ref.register_pending_keys(session_id, [chunk_key7])
            mock_ref = pool.actor_ref(test_actor.uid, address='MOCK_ADDR')
            await self.waitp(receiver_ref.create_data_writers(
                session_id, [chunk_key7], [data_size], mock_ref, timeout=1, _promise=True))
            receiver_ref.notify_dead_senders(['MOCK_ADDR'], _tell=True, _wait=False)
            with self.assertRaises(WorkerDead):
                await self.waitp(receiver_manager_ref.add_keys_callback(
                    session_id, [chunk_key7], _promise=True))

            # SCENARIO 6: successful transfer without promise
            await receiver_ref.create_data_writers(session_id, [chunk_key8], [data_size], mock_ref,
                                                   use_promise=False)
            await receiver_ref.receive_data_part(session_id, [chunk_key8], [True], dumped_mock_data)
            self.assertEqual(await receiver_ref.check_status(session_id, chunk_key8),
                             ReceiveStatus.RECEIVED)
            assert_array_equal(await storage_client.get_object(
                session_id, chunk_key8, [DataStorageDevice.SHARED_MEMORY], _promise=False), mock_data)

            # SCENARIO 7: failed transfer without promise
            with patch_method(StorageClient.create_writer, new=_create_writer_with_fail), \
                    self.assertRaises(ValueError):
                fail_key = chunk_key9
                await receiver_ref.create_data_writers(session_id, [chunk_key9], [data_size],
                                                       mock_ref, use_promise=False)

    async def testSimpleTransfer(self):
        session_id = str(uuid.uuid4())

        local_pool_addr = 'localhost:%d' % get_next_port()
        remote_pool_addr = 'localhost:%d' % get_next_port()
        remote_chunk_keys = [str(uuid.uuid4()) for _ in range(9)]
        msg_queue = multiprocessing.Queue()

        remote_spill_dir = tempfile.mkdtemp(prefix='mars_test_simple_transfer_')

        proc = multiprocessing.Process(
            target=run_transfer_worker,
            args=(remote_pool_addr, session_id, remote_chunk_keys, remote_spill_dir, msg_queue)
        )
        proc.start()
        try:
            remote_plasma_socket = msg_queue.get(timeout=30)
        except Empty:
            if proc.is_alive():
                proc.terminate()
            raise

        async with start_transfer_test_pool(
                address=local_pool_addr, plasma_size=self.plasma_storage_size) as pool:
            sender_refs, receiver_refs = [], []
            for _ in range(2):
                sender_refs.append(await pool.create_actor(SenderActor, uid=str(uuid.uuid4())))
                receiver_refs.append(await pool.create_actor(ReceiverWorkerActor, uid=str(uuid.uuid4())))

            try:
                for data_id in (-1, 0):
                    chunk_key = remote_chunk_keys[data_id]

                    async with self.run_actor_test(pool) as test_actor:
                        remote_dispatch_ref = test_actor.promise_ref(
                            DispatchActor.default_uid(), address=remote_pool_addr)

                        def _call_send_data(sender_uid):
                            sender_ref = test_actor.promise_ref(sender_uid, address=remote_pool_addr)
                            return sender_ref.send_data(
                                session_id, [chunk_key], [local_pool_addr], _promise=True)

                        async def _test_data_exist(*_):
                            local_client_ref = test_actor.promise_ref(StorageClientActor.default_uid())
                            remote_client_ref = test_actor.promise_ref(StorageClientActor.default_uid(),
                                                                       address=remote_pool_addr)

                            def _get(local_data):
                                return remote_client_ref.get_object(
                                    session_id, chunk_key, targets, _promise=True) \
                                        .then(lambda remote_data: assert_array_equal(local_data, remote_data))

                            targets = [DataStorageDevice.PROC_MEMORY]
                            return local_client_ref.get_object(session_id, chunk_key, targets, _promise=True) \
                                .then(_get)

                        remote_dispatch_ref.get_free_slot('sender', _promise=True) \
                            .then(_call_send_data) \
                            .then(_test_data_exist) \
                            .then(
                            lambda *_: test_actor.set_result(chunk_key),
                            lambda *exc: test_actor.set_result(exc, False),
                        )
                        self.assertEqual(await self.get_result(60), chunk_key)

                msg_queue.put(1)
            finally:
                await wait_results(pool.destroy_actor(ref) for ref in sender_refs + receiver_refs)

                os.unlink(remote_plasma_socket)
                os.kill(proc.pid, signal.SIGINT)

                t = time.time()
                while proc.is_alive() and time.time() < t + 2:
                    time.sleep(1)
                if proc.is_alive():
                    proc.terminate()

                self.rm_spill_dirs(remote_spill_dir)
