# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

from mars.actors import create_actor_pool
from mars.compat import Empty, BrokenPipeError, TimeoutError
from mars.config import options
from mars.errors import ChecksumMismatch, DependencyMissing, StoreFull,\
    SpillNotConfigured, ExecutionInterrupted, WorkerDead
from mars.scheduler import ChunkMetaActor
from mars.scheduler.utils import SchedulerClusterInfoActor
from mars.serialize import dataserializer
from mars.tests.core import patch_method
from mars.utils import get_next_port
from mars.worker import SenderActor, ReceiverActor, DispatchActor, QuotaActor, \
    MemQuotaActor, ChunkHolderActor, SpillActor, StatusActor
from mars.worker.chunkstore import PlasmaChunkStore, PlasmaKeyMapActor
from mars.worker.spill import build_spill_file_name, write_spill_file
from mars.worker.tests.base import WorkerCase
from mars.worker.transfer import ReceiveStatus, ReceiverDataMeta
from mars.worker.utils import WorkerActor, WorkerClusterInfoActor
from pyarrow import plasma


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

    def receive_data_part(self, session_id, chunk_key, data_part, checksum):
        query_key = (session_id, chunk_key)
        meta = self._data_metas[query_key]  # type: ReceiverDataMeta
        new_checksum = zlib.crc32(data_part, meta.checksum)
        if new_checksum != checksum:
            raise ChecksumMismatch
        meta.checksum = checksum
        self._data_writers[query_key].write(data_part)

    def finish_receive(self, session_id, chunk_key, checksum):
        query_key = (session_id, chunk_key)
        meta = self._data_metas[query_key]  # type: ReceiverDataMeta
        if meta.checksum != checksum:
            raise ChecksumMismatch
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
        pool.create_actor(SchedulerClusterInfoActor, schedulers=[address],
                          uid=SchedulerClusterInfoActor.default_uid())
        pool.create_actor(WorkerClusterInfoActor, schedulers=[address],
                          uid=WorkerClusterInfoActor.default_uid())

        pool.create_actor(PlasmaKeyMapActor, uid=PlasmaKeyMapActor.default_uid())
        pool.create_actor(ChunkMetaActor, uid=ChunkMetaActor.default_uid())
        pool.create_actor(DispatchActor, uid=DispatchActor.default_uid())
        pool.create_actor(QuotaActor, 1024 * 1024 * 20, uid=MemQuotaActor.default_uid())
        chunk_holder_ref = pool.create_actor(ChunkHolderActor,
                                             plasma_size, uid=ChunkHolderActor.default_uid())
        pool.create_actor(SpillActor)
        pool.create_actor(StatusActor, address, uid=StatusActor.default_uid())

        yield pool

        chunk_holder_ref.destroy()


def run_transfer_worker(pool_address, session_id, chunk_keys, spill_dir, msg_queue):
    options.worker.spill_directory = spill_dir
    plasma_size = 1024 * 1024 * 10

    # don't use multiple with-statement as we need the options be forked
    with plasma.start_plasma_store(plasma_size) as store_args:
        options.worker.plasma_socket = plasma_socket = store_args[0]
        try:
            plasma_client = plasma.connect(plasma_socket)
        except TypeError:
            plasma_client = plasma.connect(options.worker.plasma_socket, '', 0)

        with start_transfer_test_pool(address=pool_address, plasma_size=plasma_size) as pool:
            chunk_holder_ref = pool.actor_ref(ChunkHolderActor.default_uid())
            mapper_ref = pool.actor_ref(PlasmaKeyMapActor.default_uid())
            plasma_store = PlasmaChunkStore(plasma_client, mapper_ref)

            for _ in range(2):
                pool.create_actor(SenderActor, uid='%s' % str(uuid.uuid4()))
                pool.create_actor(ReceiverActor, uid='%s' % str(uuid.uuid4()))

            for idx in range(0, len(chunk_keys) - 7):
                data = np.ones((640 * 1024,), dtype=np.int16) * idx
                write_spill_file(chunk_keys[idx], data)
            for idx in range(len(chunk_keys) - 7, len(chunk_keys)):
                data = np.ones((640 * 1024,), dtype=np.int16) * idx
                plasma_store.put(session_id, chunk_keys[idx], data)
                chunk_holder_ref.register_chunk(session_id, chunk_keys[idx])

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
            chunk_holder_ref = send_pool.actor_ref(ChunkHolderActor.default_uid())
            sender_ref = send_pool.actor_ref(SenderActor.default_uid())
            receiver_ref = recv_pool.actor_ref(ReceiverActor.default_uid())

            sender_mapper_ref = send_pool.actor_ref(PlasmaKeyMapActor.default_uid())
            store = PlasmaChunkStore(self._plasma_client, sender_mapper_ref)

            with self.run_actor_test(send_pool) as test_actor:
                # send when data missing
                sender_ref_p = test_actor.promise_ref(sender_ref)
                sender_ref_p.send_data(session_id, str(uuid.uuid4()), recv_pool_addr, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s)) \
                    .catch(lambda *exc: test_actor.set_result(exc, accept=False))
                with self.assertRaises(DependencyMissing):
                    self.get_result(5)

                # send data in spill
                write_spill_file(chunk_key1, mock_data)

                sender_ref_p.send_data(session_id, chunk_key1, recv_pool_addr, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s)) \
                    .catch(lambda *exc: test_actor.set_result(exc, accept=False))
                self.get_result(5)
                assert_array_equal(mock_data, receiver_ref.get_result_data(session_id, chunk_key1))
                os.unlink(build_spill_file_name(chunk_key1))

                # send data in plasma store
                store.put(session_id, chunk_key1, mock_data)
                chunk_holder_ref.register_chunk(session_id, chunk_key1)

                sender_ref_p.send_data(session_id, chunk_key1, recv_pool_addr, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s)) \
                    .catch(lambda *exc: test_actor.set_result(exc, accept=False))
                self.get_result(5)
                assert_array_equal(mock_data, receiver_ref.get_result_data(session_id, chunk_key1))

                # send data to multiple targets
                with start_transfer_test_pool(
                        address=recv_pool_addr2, plasma_size=self.plasma_storage_size) as rp2:
                    recv_ref2 = rp2.create_actor(MockReceiverActor, uid=ReceiverActor.default_uid())

                    sender_ref_p.send_data(session_id, chunk_key1,
                                           [recv_pool_addr, recv_pool_addr2], _promise=True) \
                        .then(lambda *s: test_actor.set_result(s)) \
                        .catch(lambda *exc: test_actor.set_result(exc, accept=False))
                    self.get_result(5)

                    # send data to already transferred / transferring
                    sender_ref_p.send_data(session_id, chunk_key1,
                                           [recv_pool_addr, recv_pool_addr2], _promise=True) \
                        .then(lambda *s: test_actor.set_result(s)) \
                        .catch(lambda *exc: test_actor.set_result(exc, accept=False))
                    self.get_result(5)
                    assert_array_equal(mock_data, recv_ref2.get_result_data(session_id, chunk_key1))

                # send data to non-exist endpoint which causes error
                store.put(session_id, chunk_key2, mock_data)
                chunk_holder_ref.register_chunk(session_id, chunk_key2)

                sender_ref_p.send_data(session_id, chunk_key2, recv_pool_addr2, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s)) \
                    .catch(lambda *exc: test_actor.set_result(exc, accept=False))
                with self.assertRaises(BrokenPipeError):
                    self.get_result(5)

                def mocked_receive_data_part(*_):
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
        serialized_mock_data = dataserializer.dumps(mock_data)
        serialized_crc32 = zlib.crc32(serialized_mock_data)

        chunk_key1 = str(uuid.uuid4())
        chunk_key2 = str(uuid.uuid4())
        chunk_key3 = str(uuid.uuid4())
        chunk_key4 = str(uuid.uuid4())
        chunk_key5 = str(uuid.uuid4())
        chunk_key6 = str(uuid.uuid4())
        chunk_key7 = str(uuid.uuid4())
        chunk_key8 = str(uuid.uuid4())

        with start_transfer_test_pool(address=pool_addr, plasma_size=self.plasma_storage_size) as pool:
            chunk_holder_ref = pool.actor_ref(ChunkHolderActor.default_uid())
            mapper_ref = pool.actor_ref(PlasmaKeyMapActor.default_uid())
            receiver_ref = pool.create_actor(ReceiverActor, uid=str(uuid.uuid4()))

            store = PlasmaChunkStore(self._plasma_client, mapper_ref)

            # check_status on receiving and received
            self.assertEqual(receiver_ref.check_status(session_id, chunk_key1),
                             ReceiveStatus.NOT_STARTED)

            write_spill_file(chunk_key1, mock_data)
            self.assertEqual(receiver_ref.check_status(session_id, chunk_key1),
                             ReceiveStatus.RECEIVED)
            os.unlink(build_spill_file_name(chunk_key1))

            ref = store.put(session_id, chunk_key1, mock_data)
            data_size = store.get_actual_size(session_id, chunk_key1)
            chunk_holder_ref.register_chunk(session_id, chunk_key1)
            del ref
            self.assertEqual(receiver_ref.check_status(session_id, chunk_key1),
                             ReceiveStatus.RECEIVED)

            with self.run_actor_test(pool) as test_actor:
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

                # test checksum error on finish_receive
                receiver_ref_p.create_data_writer(session_id, chunk_key2, data_size, test_actor, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s))
                self.assertTupleEqual(self.get_result(5), (receiver_ref.address, None))

                receiver_ref_p.receive_data_part(session_id, chunk_key2, serialized_mock_data, serialized_crc32)
                receiver_ref_p.finish_receive(session_id, chunk_key2, 0)

                receiver_ref_p.register_finish_callback(session_id, chunk_key2, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s)) \
                    .catch(lambda *exc: test_actor.set_result(exc, accept=False))

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
                receiver_ref_p.receive_data_part(session_id, chunk_key3, serialized_mock_data[64:], serialized_crc32)
                receiver_ref_p.finish_receive(session_id, chunk_key3, serialized_crc32)

                self.assertTupleEqual((), self.get_result(5))

                receiver_ref_p.create_data_writer(session_id, chunk_key3, data_size, test_actor, _promise=True) \
                    .then(lambda *s: test_actor.set_result(s))
                self.assertTupleEqual(self.get_result(5), (receiver_ref.address, ReceiveStatus.RECEIVED))

                # test transfer in spill file
                def mocked_store_create(*_):
                    raise StoreFull

                with patch_method(PlasmaChunkStore.create, new=mocked_store_create):
                    with self.assertRaises(StoreFull):
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

                    receiver_ref_p.receive_data_part(session_id, chunk_key4, serialized_mock_data, serialized_crc32)
                    receiver_ref_p.finish_receive(session_id, chunk_key4, serialized_crc32)

                    self.assertTupleEqual((), self.get_result(5))

                # test intermediate error
                def mocked_store_create(*_):
                    raise SpillNotConfigured

                with patch_method(PlasmaChunkStore.create, new=mocked_store_create):
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

                # test checksum error on finish_receive
                result = receiver_ref_p.create_data_writer(session_id, chunk_key8, data_size, test_actor,
                                                           use_promise=False)
                self.assertTupleEqual(result, (receiver_ref.address, None))

                receiver_ref_p.receive_data_part(session_id, chunk_key8, serialized_mock_data, serialized_crc32)
                receiver_ref_p.finish_receive(session_id, chunk_key8, 0)

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
                        remote_mapper_ref = pool.actor_ref(
                            PlasmaKeyMapActor.default_uid(), address=remote_pool_addr)
                        try:
                            remote_plasma_client = plasma.connect(remote_plasma_socket)
                        except TypeError:
                            remote_plasma_client = plasma.connect(remote_plasma_socket, '', 0)
                        remote_store = PlasmaChunkStore(remote_plasma_client, remote_mapper_ref)

                        def _call_send_data(sender_uid):
                            sender_ref = test_actor.promise_ref(sender_uid, address=remote_pool_addr)
                            return sender_ref.send_data(session_id, chunk_key, local_pool_addr, _promise=True)

                        def _test_data_exist(*_):
                            try:
                                local_data = test_actor._chunk_store.get(session_id, chunk_key)
                            except KeyError:
                                with open(build_spill_file_name(chunk_key), 'rb') as spill_file:
                                    local_data = dataserializer.load(spill_file)

                            try:
                                remote_data = remote_store.get(session_id, chunk_key)
                            except KeyError:
                                with open(build_spill_file_name(chunk_key, remote_spill_dir), 'rb') as spill_file:
                                    remote_data = dataserializer.load(spill_file)
                            assert_array_equal(local_data, remote_data)

                            del local_data, remote_data

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
