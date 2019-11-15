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

import contextlib
import multiprocessing
import os
import signal
import tempfile
import time
import uuid
import zlib
from collections import defaultdict

import numpy as np
from numpy.testing import assert_array_equal
from pyarrow import plasma

from mars import promise
from mars.compat import Empty, BrokenPipeError, TimeoutError
from mars.config import options
from mars.errors import ChecksumMismatch, DependencyMissing, StorageFull,\
    SpillNotConfigured, ExecutionInterrupted, WorkerDead
from mars.scheduler import ChunkMetaActor
from mars.scheduler.utils import SchedulerClusterInfoActor
from mars.serialize import dataserializer
from mars.tests.core import patch_method, create_actor_pool
from mars.utils import get_next_port
from mars.worker import SenderActor, ReceiverActor, DispatchActor, QuotaActor, \
    MemQuotaActor, StorageManagerActor, IORunnerActor, StatusActor, \
    SharedHolderActor, InProcHolderActor
from mars.worker.storage import DataStorageDevice
from mars.worker.storage.sharedstore import PlasmaSharedStore, PlasmaKeyMapActor
from mars.worker.tests.base import WorkerCase, StorageClientActor
from mars.worker.transfer import ReceiveStatus, ReceiverDataMeta
from mars.worker.utils import WorkerActor, WorkerClusterInfoActor


class MockReceiverActor(WorkerActor):
    """
    Actor handling receiving data from a SenderActor
    """
    def __init__(self):
        super(MockReceiverActor, self).__init__()
        self._dispatch_ref = None

        self._data_metas = dict()
        self._data_writers = dict()
        self._callbacks = defaultdict(list)

    def post_create(self):
        super(MockReceiverActor, self).post_create()
        self._dispatch_ref = self.ctx.actor_ref(DispatchActor.default_uid())
        self._dispatch_ref.register_free_slot(self.uid, 'receiver')

    def set_status(self, session_id, chunk_key, status):
        query_key = (session_id, chunk_key)
        try:
            self._data_metas[query_key].status = status
        except KeyError:
            self._data_metas[query_key] = ReceiverDataMeta(status=status)

    def get_result_data(self, session_id, chunk_key):
        buf = self._data_writers[(session_id, chunk_key)].getvalue()
        return dataserializer.loads(buf)

    def check_status(self, session_id, chunk_key):
        try:
            return self._data_metas[(session_id, chunk_key)].status
        except KeyError:
            return ReceiveStatus.NOT_STARTED

    def register_finish_callback(self, session_id, chunk_key, callback):
        query_key = (session_id, chunk_key)
        try:
            meta = self._data_metas[query_key]
            if meta.status in (ReceiveStatus.RECEIVED, ReceiveStatus.ERROR):
                self.tell_promise(callback, *meta.callback_args, **meta.callback_kwargs)
            else:
                raise KeyError
        except KeyError:
            self._callbacks[query_key].append(callback)

    def create_data_writer(self, session_id, chunk_key, data_size, sender_ref,
                           ensure_cached=True, timeout=0, callback=None):
        from mars.compat import BytesIO
        query_key = (session_id, chunk_key)
        if query_key in self._data_metas and \
                self._data_metas[query_key].status in (ReceiveStatus.RECEIVED, ReceiveStatus.RECEIVING):
            self.tell_promise(callback, self.address, self._data_metas[query_key].status)
            return
        self._data_metas[query_key] = ReceiverDataMeta(chunk_size=data_size, status=ReceiveStatus.RECEIVING)
        self._data_writers[query_key] = BytesIO()
        self.tell_promise(callback, self.address, None)

    def receive_data_part(self, session_id, chunk_key, data_part, checksum, is_last=False):
        query_key = (session_id, chunk_key)
        meta = self._data_metas[query_key]  # type: ReceiverDataMeta
        if data_part:
            new_checksum = zlib.crc32(data_part, meta.checksum)
            if new_checksum != checksum:
                raise ChecksumMismatch
            meta.checksum = checksum
            self._data_writers[query_key].write(data_part)
        if is_last:
            meta.status = ReceiveStatus.RECEIVED
        for cb in self._callbacks[query_key]:
            self.tell_promise(cb)

    def cancel_receive(self, session_id, chunk_key):
        pass


@contextlib.contextmanager
def start_transfer_test_pool(**kwargs):
    address = kwargs.pop('address')
    plasma_size = kwargs.pop('plasma_size')
    with create_actor_pool(n_process=1, backend='gevent', address=address, **kwargs) as pool:
        pool.create_actor(SchedulerClusterInfoActor, [address],
                          uid=SchedulerClusterInfoActor.default_uid())
        pool.create_actor(WorkerClusterInfoActor, [address],
                          uid=WorkerClusterInfoActor.default_uid())

        pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
        pool.create_actor(StorageManagerActor, uid=StorageManagerActor.default_uid())
        pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())
        pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())
        pool.create_actor(QuotaActor, 1024 * 1024 * 20, uid=MemQuotaActor.default_uid())
        shared_holder_ref = pool.create_actor(SharedHolderActor,
                                              plasma_size, uid=SharedHolderActor.default_uid())
        pool.create_actor(StatusActor, address, uid=StatusActor.default_uid())
        pool.create_actor(IORunnerActor)
        pool.create_actor(StorageClientActor, uid=StorageClientActor.default_uid())
        pool.create_actor(InProcHolderActor)

        yield pool

        shared_holder_ref.destroy()


def run_transfer_worker(pool_address, session_id, chunk_keys, spill_dir, msg_queue):
    options.worker.spill_directory = spill_dir
    plasma_size = 1024 * 1024 * 10

    # don't use multiple with-statement as we need the options be forked
    with plasma.start_plasma_store(plasma_size) as store_args:
        options.worker.plasma_socket = plasma_socket = store_args[0]

        with start_transfer_test_pool(address=pool_address, plasma_size=plasma_size) as pool:
            storage_client_ref = pool.create_actor(StorageClientActor)

            for _ in range(2):
                pool.create_actor(SenderActor, uid='%s' % str(uuid.uuid4()))
                pool.create_actor(ReceiverActor, uid='%s' % str(uuid.uuid4()))

            for idx in range(0, len(chunk_keys) - 7):
                data = np.ones((640 * 1024,), dtype=np.int16) * idx
                storage_client_ref.put_objects(
                    session_id, [chunk_keys[idx]], [data], [DataStorageDevice.PROC_MEMORY])
            for idx in range(len(chunk_keys) - 7, len(chunk_keys)):
                data = np.ones((640 * 1024,), dtype=np.int16) * idx
                storage_client_ref.put_objects(
                    session_id, [chunk_keys[idx]], [data], [DataStorageDevice.SHARED_MEMORY])

            while not all(storage_client_ref.get_data_locations(session_id, chunk_keys)):
                pool.sleep(0.1)

            for idx in range(0, len(chunk_keys) - 7):
                storage_client_ref.copy_to(session_id, [chunk_keys[idx]], [DataStorageDevice.DISK])

            while not all((0, DataStorageDevice.DISK) in locations
                          for locations in storage_client_ref.get_data_locations(session_id, chunk_keys[:-7])):
                pool.sleep(0.1)

            msg_queue.put(plasma_socket)
            t = time.time()
            while True:
                try:
                    msg_queue.get_nowait()
                except Empty:
                    if time.time() > t + 60:
                        raise SystemError('Transfer finish timed out.')
                    pool.sleep(0.1)


class Test(WorkerCase):
    def setUp(self):
        super(Test, self).setUp()
        self._old_block_size = options.worker.transfer_block_size
        options.worker.transfer_block_size = 4 * 1024

    def tearDown(self):
        super(Test, self).tearDown()
        options.worker.transfer_block_size = self._old_block_size
        self.rm_spill_dirs(options.worker.spill_directory)

    def testSender(self):
        send_pool_addr = 'localhost:%d' % get_next_port()
        recv_pool_addr = 'localhost:%d' % get_next_port()
        recv_pool_addr2 = 'localhost:%d' % get_next_port()

        options.worker.spill_directory = tempfile.mkdtemp(prefix='mars_test_sender_')
        session_id = str(uuid.uuid4())

        mock_data = np.array([1, 2, 3, 4])
        chunk_key1 = str(uuid.uuid4())
        chunk_key2 = str(uuid.uuid4())

        @contextlib.contextmanager
        def start_send_recv_pool():
            with start_transfer_test_pool(
                    address=send_pool_addr, plasma_size=self.plasma_storage_size) as sp:
                sp.create_actor(SenderActor, uid=SenderActor.default_uid())
                with start_transfer_test_pool(
                        address=recv_pool_addr, plasma_size=self.plasma_storage_size) as rp:
                    rp.create_actor(MockReceiverActor, uid=ReceiverActor.default_uid())
                    yield sp, rp

        with start_send_recv_pool() as (send_pool, recv_pool):
            sender_ref = send_pool.actor_ref(SenderActor.default_uid())
            receiver_ref = recv_pool.actor_ref(ReceiverActor.default_uid())

            with self.run_actor_test(send_pool) as test_actor:
                storage_client = test_actor.storage_client

                # send when data missing
                sender_ref_p = test_actor.promise_ref(sender_ref)
                sender_ref_p.send_data(session_id, str(uuid.uuid4()), recv_pool_addr, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s)) \
                    .catch(lambda *exc: test_actor.set_result(exc, accept=False))
                with self.assertRaises(DependencyMissing):
                    self.get_result(5)

                # send data in spill
                serialized = dataserializer.serialize(mock_data)
                self.waitp(
                    storage_client.create_writer(session_id, chunk_key1, serialized.total_bytes,
                                                 [DataStorageDevice.DISK])
                        .then(lambda writer: promise.finished().then(lambda *_: writer.write(serialized))
                              .then(lambda *_: writer.close()))
                )

                sender_ref_p.send_data(session_id, chunk_key1, recv_pool_addr, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s)) \
                    .catch(lambda *exc: test_actor.set_result(exc, accept=False))
                self.get_result(5)
                assert_array_equal(mock_data, receiver_ref.get_result_data(session_id, chunk_key1))
                storage_client.delete(session_id, [chunk_key1])

                # send data in plasma store
                self.waitp(
                    storage_client.put_objects(
                        session_id, [chunk_key1], [mock_data], [DataStorageDevice.SHARED_MEMORY])
                )

                sender_ref_p.send_data(session_id, chunk_key1, recv_pool_addr, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s)) \
                    .catch(lambda *exc: test_actor.set_result(exc, accept=False))
                self.get_result(5)
                assert_array_equal(mock_data, receiver_ref.get_result_data(session_id, chunk_key1))

                # send data to multiple targets
                with start_transfer_test_pool(
                        address=recv_pool_addr2, plasma_size=self.plasma_storage_size) as rp2:
                    recv_ref2 = rp2.create_actor(MockReceiverActor, uid=ReceiverActor.default_uid())

                    self.waitp(
                        sender_ref_p.send_data(session_id, chunk_key1,
                                               [recv_pool_addr, recv_pool_addr2], _promise=True)
                    )
                    # send data to already transferred / transferring
                    sender_ref_p.send_data(session_id, chunk_key1,
                                           [recv_pool_addr, recv_pool_addr2], _promise=True) \
                        .then(lambda *s: test_actor.set_result(s)) \
                        .catch(lambda *exc: test_actor.set_result(exc, accept=False))
                    self.get_result(5)
                    assert_array_equal(mock_data, recv_ref2.get_result_data(session_id, chunk_key1))

                # send data to non-exist endpoint which causes error
                self.waitp(
                    storage_client.put_objects(
                        session_id, [chunk_key2], [mock_data], [DataStorageDevice.SHARED_MEMORY])
                )

                sender_ref_p.send_data(session_id, chunk_key2, recv_pool_addr2, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s)) \
                    .catch(lambda *exc: test_actor.set_result(exc, accept=False))
                with self.assertRaises(BrokenPipeError):
                    self.get_result(5)

                def mocked_receive_data_part(*_, **__):
                    raise ChecksumMismatch

                with patch_method(MockReceiverActor.receive_data_part, new=mocked_receive_data_part):
                    sender_ref_p.send_data(session_id, chunk_key2, recv_pool_addr, _promise=True) \
                        .then(lambda *s: test_actor.set_result(s)) \
                        .catch(lambda *exc: test_actor.set_result(exc, accept=False))
                    with self.assertRaises(ChecksumMismatch):
                        self.get_result(5)

    def testReceiver(self):
        pool_addr = 'localhost:%d' % get_next_port()
        options.worker.spill_directory = tempfile.mkdtemp(prefix='mars_test_receiver_')
        session_id = str(uuid.uuid4())

        mock_data = np.array([1, 2, 3, 4])
        serialized_arrow_data = dataserializer.serialize(mock_data)
        data_size = serialized_arrow_data.total_bytes
        serialized_mock_data = serialized_arrow_data.to_buffer()
        serialized_crc32 = zlib.crc32(serialized_arrow_data.to_buffer())

        chunk_key1 = str(uuid.uuid4())
        chunk_key2 = str(uuid.uuid4())
        chunk_key3 = str(uuid.uuid4())
        chunk_key4 = str(uuid.uuid4())
        chunk_key5 = str(uuid.uuid4())
        chunk_key6 = str(uuid.uuid4())
        chunk_key7 = str(uuid.uuid4())

        with start_transfer_test_pool(address=pool_addr, plasma_size=self.plasma_storage_size) as pool:
            receiver_ref = pool.create_actor(ReceiverActor, uid=str(uuid.uuid4()))

            with self.run_actor_test(pool) as test_actor:
                storage_client = test_actor.storage_client

                # check_status on receiving and received
                self.assertEqual(receiver_ref.check_status(session_id, chunk_key1),
                                 ReceiveStatus.NOT_STARTED)

                self.waitp(
                    storage_client.create_writer(session_id, chunk_key1, serialized_arrow_data.total_bytes,
                                                 [DataStorageDevice.DISK])
                        .then(lambda writer: promise.finished().then(lambda *_: writer.write(serialized_arrow_data))
                              .then(lambda *_: writer.close()))
                )
                self.assertEqual(receiver_ref.check_status(session_id, chunk_key1),
                                 ReceiveStatus.RECEIVED)
                storage_client.delete(session_id, [chunk_key1])

                self.waitp(
                    storage_client.put_objects(
                        session_id, [chunk_key1], [mock_data], [DataStorageDevice.SHARED_MEMORY])
                )

                self.assertEqual(receiver_ref.check_status(session_id, chunk_key1),
                                 ReceiveStatus.RECEIVED)

                receiver_ref_p = test_actor.promise_ref(receiver_ref)

                # cancel on an un-run / missing result will result in nothing
                receiver_ref_p.cancel_receive(session_id, chunk_key2)

                # start creating writer
                receiver_ref_p.create_data_writer(session_id, chunk_key1, data_size, test_actor, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s))
                self.assertTupleEqual(self.get_result(5), (receiver_ref.address, ReceiveStatus.RECEIVED))

                result = receiver_ref_p.create_data_writer(session_id, chunk_key1, data_size, test_actor,
                                                           use_promise=False)
                self.assertTupleEqual(result, (receiver_ref.address, ReceiveStatus.RECEIVED))

                receiver_ref_p.create_data_writer(session_id, chunk_key2, data_size, test_actor, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s))
                self.assertTupleEqual(self.get_result(5), (receiver_ref.address, None))

                result = receiver_ref_p.create_data_writer(session_id, chunk_key2, data_size, test_actor,
                                                           use_promise=False)
                self.assertTupleEqual(result, (receiver_ref.address, ReceiveStatus.RECEIVING))

                receiver_ref_p.create_data_writer(session_id, chunk_key2, data_size, test_actor, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s))
                self.assertTupleEqual(self.get_result(5), (receiver_ref.address, ReceiveStatus.RECEIVING))

                receiver_ref_p.cancel_receive(session_id, chunk_key2)
                self.assertEqual(receiver_ref.check_status(session_id, chunk_key2),
                                 ReceiveStatus.NOT_STARTED)

                # test checksum error on receive_data_part
                receiver_ref_p.create_data_writer(session_id, chunk_key2, data_size, test_actor, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s))
                self.get_result(5)

                receiver_ref_p.register_finish_callback(session_id, chunk_key2, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s)) \
                    .catch(lambda *exc: test_actor.set_result(exc, accept=False))

                receiver_ref_p.receive_data_part(session_id, chunk_key2, serialized_mock_data, 0)

                with self.assertRaises(ChecksumMismatch):
                    self.get_result(5)

                receiver_ref_p.cancel_receive(session_id, chunk_key2)

                # test intermediate cancellation
                receiver_ref_p.create_data_writer(session_id, chunk_key2, data_size, test_actor, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s))
                self.assertTupleEqual(self.get_result(5), (receiver_ref.address, None))

                receiver_ref_p.register_finish_callback(session_id, chunk_key2, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s)) \
                    .catch(lambda *exc: test_actor.set_result(exc, accept=False))

                receiver_ref_p.receive_data_part(session_id, chunk_key2, serialized_mock_data[:64],
                                                 zlib.crc32(serialized_mock_data[:64]))
                receiver_ref_p.cancel_receive(session_id, chunk_key2)
                receiver_ref_p.receive_data_part(session_id, chunk_key2, serialized_mock_data[64:],
                                                 serialized_crc32)
                with self.assertRaises(ExecutionInterrupted):
                    self.get_result(5)

                # test transfer in memory
                receiver_ref_p.register_finish_callback(session_id, chunk_key3, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s)) \
                    .catch(lambda *exc: test_actor.set_result(exc, accept=False))

                receiver_ref_p.create_data_writer(session_id, chunk_key3, data_size, test_actor, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s))
                self.assertTupleEqual(self.get_result(5), (receiver_ref.address, None))

                receiver_ref_p.receive_data_part(session_id, chunk_key3, serialized_mock_data[:64],
                                                 zlib.crc32(serialized_mock_data[:64]))
                receiver_ref_p.receive_data_part(
                    session_id, chunk_key3, serialized_mock_data[64:], serialized_crc32, is_last=True)

                self.assertTupleEqual((), self.get_result(5))

                receiver_ref_p.create_data_writer(session_id, chunk_key3, data_size, test_actor, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s))
                self.assertTupleEqual(self.get_result(5), (receiver_ref.address, ReceiveStatus.RECEIVED))

                # test transfer in spill file
                def mocked_store_create(*_):
                    raise StorageFull

                with patch_method(PlasmaSharedStore.create, new=mocked_store_create):
                    with self.assertRaises(StorageFull):
                        receiver_ref_p.create_data_writer(session_id, chunk_key4, data_size, test_actor,
                                                          ensure_cached=True, use_promise=False)
                    # test receive aborted
                    receiver_ref_p.create_data_writer(
                        session_id, chunk_key4, data_size, test_actor, ensure_cached=False, _promise=True) \
                        .then(lambda *s: test_actor.set_result(s))
                    self.assertTupleEqual(self.get_result(5), (receiver_ref.address, None))

                    receiver_ref_p.register_finish_callback(session_id, chunk_key4, _promise=True) \
                        .then(lambda *s: test_actor.set_result(s)) \
                        .catch(lambda *exc: test_actor.set_result(exc, accept=False))

                    receiver_ref_p.receive_data_part(session_id, chunk_key4, serialized_mock_data[:64],
                                                     zlib.crc32(serialized_mock_data[:64]))
                    receiver_ref_p.cancel_receive(session_id, chunk_key4)
                    with self.assertRaises(ExecutionInterrupted):
                        self.get_result(5)

                    # test receive into spill
                    receiver_ref_p.create_data_writer(
                        session_id, chunk_key4, data_size, test_actor, ensure_cached=False, _promise=True) \
                        .then(lambda *s: test_actor.set_result(s))
                    self.assertTupleEqual(self.get_result(5), (receiver_ref.address, None))

                    receiver_ref_p.register_finish_callback(session_id, chunk_key4, _promise=True) \
                        .then(lambda *s: test_actor.set_result(s)) \
                        .catch(lambda *exc: test_actor.set_result(exc, accept=False))

                    receiver_ref_p.receive_data_part(
                        session_id, chunk_key4, serialized_mock_data, serialized_crc32, is_last=True)
                    self.assertTupleEqual((), self.get_result(5))

                # test intermediate error
                def mocked_store_create(*_):
                    raise SpillNotConfigured

                with patch_method(PlasmaSharedStore.create, new=mocked_store_create):
                    receiver_ref_p.create_data_writer(
                        session_id, chunk_key5, data_size, test_actor, ensure_cached=False, _promise=True) \
                        .then(lambda *s: test_actor.set_result(s),
                              lambda *s: test_actor.set_result(s, accept=False))

                    with self.assertRaises(SpillNotConfigured):
                        self.get_result(5)

                # test receive timeout
                receiver_ref_p.register_finish_callback(session_id, chunk_key6, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s)) \
                    .catch(lambda *exc: test_actor.set_result(exc, accept=False))

                receiver_ref_p.create_data_writer(session_id, chunk_key6, data_size, test_actor,
                                                  timeout=2, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s))
                self.assertTupleEqual(self.get_result(5), (receiver_ref.address, None))
                receiver_ref_p.receive_data_part(session_id, chunk_key6, serialized_mock_data[:64],
                                                 zlib.crc32(serialized_mock_data[:64]))

                with self.assertRaises(TimeoutError):
                    self.get_result(5)

                # test sender halt
                receiver_ref_p.register_finish_callback(session_id, chunk_key7, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s)) \
                    .catch(lambda *exc: test_actor.set_result(exc, accept=False))

                mock_ref = pool.actor_ref(test_actor.uid, address='MOCK_ADDR')
                receiver_ref_p.create_data_writer(
                    session_id, chunk_key7, data_size, mock_ref, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s))
                self.assertTupleEqual(self.get_result(5), (receiver_ref.address, None))
                receiver_ref_p.receive_data_part(session_id, chunk_key7, serialized_mock_data[:64],
                                                 zlib.crc32(serialized_mock_data[:64]))
                receiver_ref_p.notify_dead_senders(['MOCK_ADDR'])

                with self.assertRaises(WorkerDead):
                    self.get_result(5)

    def testSimpleTransfer(self):
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

        with start_transfer_test_pool(address=local_pool_addr, plasma_size=self.plasma_storage_size) as pool:
            sender_refs, receiver_refs = [], []
            for _ in range(2):
                sender_refs.append(pool.create_actor(SenderActor, uid=str(uuid.uuid4())))
                receiver_refs.append(pool.create_actor(ReceiverActor, uid=str(uuid.uuid4())))

            try:
                for data_id in (-1, 0):
                    chunk_key = remote_chunk_keys[data_id]

                    with self.run_actor_test(pool) as test_actor:
                        remote_dispatch_ref = test_actor.promise_ref(
                            DispatchActor.default_uid(), address=remote_pool_addr)

                        def _call_send_data(sender_uid):
                            sender_ref = test_actor.promise_ref(sender_uid, address=remote_pool_addr)
                            return sender_ref.send_data(session_id, chunk_key, local_pool_addr, _promise=True)

                        def _test_data_exist(*_):
                            local_client_ref = test_actor.promise_ref(StorageClientActor.default_uid())
                            remote_client_ref = test_actor.promise_ref(StorageClientActor.default_uid(),
                                                                       address=remote_pool_addr)

                            targets = [DataStorageDevice.PROC_MEMORY]
                            return local_client_ref.get_object(session_id, chunk_key, targets, _promise=True) \
                                .then(lambda local_data: remote_client_ref.get_object(
                                    session_id, chunk_key, targets, _promise=True)
                                      .then(lambda remote_data: assert_array_equal(local_data, remote_data))) \

                        remote_dispatch_ref.get_free_slot('sender', _promise=True) \
                            .then(_call_send_data) \
                            .then(_test_data_exist) \
                            .then(
                            lambda *_: test_actor.set_result(chunk_key),
                            lambda *exc: test_actor.set_result(exc, False),
                        )
                        self.assertEqual(self.get_result(60), chunk_key)

                msg_queue.put(1)
            finally:
                [pool.destroy_actor(ref) for ref in sender_refs + receiver_refs]

                os.unlink(remote_plasma_socket)
                os.kill(proc.pid, signal.SIGINT)

                t = time.time()
                while proc.is_alive() and time.time() < t + 2:
                    time.sleep(1)
                if proc.is_alive():
                    proc.terminate()

                self.rm_spill_dirs(remote_spill_dir)
