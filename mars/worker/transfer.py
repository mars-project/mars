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
import operator
import logging
import sys
import time
from collections import defaultdict
from enum import Enum

from .. import promise
from ..config import options
from ..errors import DependencyMissing, ExecutionInterrupted, WorkerDead
from ..serialize import dataserializer
from ..scheduler.chunkmeta import WorkerMeta
from ..utils import log_unhandled, build_exc_info
from .events import EventContext, EventCategory, EventLevel, ProcedureEventType
from .storage import DataStorageDevice
from .utils import WorkerActor

logger = logging.getLogger(__name__)


class ReceiveStatus(Enum):
    NOT_STARTED = 0
    PENDING = 1
    RECEIVING = 2
    RECEIVED = 3
    ERROR = 4


class EndpointTransferState(object):
    """
    Structure providing transfer status in an endpoint
    """
    __slots__ = 'parts', 'total_size', 'keys', 'end_marks', 'send_future'

    def __init__(self):
        self.reset()
        self.send_future = None

    def reset(self):
        self.parts = []
        self.total_size = 0
        self.keys = []
        self.end_marks = []


class SenderActor(WorkerActor):
    """
    Actor handling sending data to ReceiverActors in other workers
    """
    def __init__(self):
        super().__init__()
        self._dispatch_ref = None
        self._events_ref = None

    def post_create(self):
        from .dispatcher import DispatchActor
        from .events import EventsActor

        super().post_create()

        self._events_ref = self.ctx.actor_ref(EventsActor.default_uid())
        if not self.ctx.has_actor(self._events_ref):
            self._events_ref = None

        self._dispatch_ref = self.promise_ref(DispatchActor.default_uid())
        self._dispatch_ref.register_free_slot(self.uid, 'sender')

    @promise.reject_on_exception
    @log_unhandled
    def send_data(self, session_id, chunk_keys, target_endpoints, ensure_cached=True,
                  compression=None, block_size=None, pin_token=None, timeout=None, callback=None):
        """
        Send data to other workers
        :param session_id: session id
        :param chunk_keys: chunks to be sent
        :param target_endpoints: endpoints to receive this chunk
        :param ensure_cached: if True, make sure the data is in the shared storage of the target worker
        :param compression: compression type when transfer in network
        :param block_size: size of data block
        :param pin_token: token to pin the data
        :param timeout: timeout of data sending
        :param callback: promise callback
        """
        chunk_keys = list(chunk_keys)
        target_endpoints = list(target_endpoints)
        block_size = block_size or options.worker.transfer_block_size

        data_sizes = self.storage_client.get_data_sizes(session_id, chunk_keys)
        if any(s is None for s in data_sizes):
            raise DependencyMissing('Dependencies %r not met when sending.'
                                    % [k for k, s in zip(chunk_keys, data_sizes) if s is None])
        compression = compression or dataserializer.CompressType(options.worker.transfer_compression)

        wait_refs = []
        addrs_to_chunks = dict()
        keys_to_readers = dict()
        receiver_refs = []
        receiver_manager_ref_dict = dict(
            (ep, self.promise_ref(ReceiverManagerActor.default_uid(), address=ep))
            for ep in target_endpoints)

        @log_unhandled
        def _create_local_readers():
            promises = []
            for k in chunk_keys:
                promises.append(self.storage_client.create_reader(
                    session_id, k, source_devices, packed=True, packed_compression=compression)
                    .then(functools.partial(operator.setitem, keys_to_readers, k)))
            return promise.all_(promises)

        @log_unhandled
        def _create_remote_writers():
            nonlocal data_sizes
            create_write_promises = []
            data_sizes = self.storage_client.get_data_sizes(session_id, chunk_keys)
            for ref in receiver_manager_ref_dict.values():
                # register transfer actions
                create_write_promises.append(
                    ref.create_data_writers(
                        session_id, chunk_keys, data_sizes, self.ref(), ensure_cached=ensure_cached,
                        timeout=timeout, pin_token=pin_token, _timeout=timeout, _promise=True
                    ).then(_handle_created)
                )
            return promise.all_(create_write_promises).then(lambda *_: keys_to_readers)

        @log_unhandled
        def _handle_created(ref, statuses):
            manager_ref = receiver_manager_ref_dict[ref.address]
            # filter out endpoints already transferred or already started transfer
            keys_receiving, keys_to_receive = [], []
            for k, status in zip(chunk_keys, statuses):
                if status == ReceiveStatus.RECEIVING:
                    keys_receiving.append(k)
                elif status is None:
                    keys_to_receive.append(k)

            if keys_to_receive:
                addrs_to_chunks[ref.address] = keys_to_receive
                receiver_refs.append(self.promise_ref(ref))
            if keys_receiving:
                wait_refs.append(manager_ref.add_keys_callback(
                    session_id, keys_receiving, _timeout=timeout, _promise=True))

        @log_unhandled
        def _finalize(*_):
            self._dispatch_ref.register_free_slot(self.uid, 'sender', _tell=True)
            self.tell_promise(callback, data_sizes)

        @log_unhandled
        def _handle_rejection(*exc):
            logger.exception('Transfer chunks %r to %r failed', chunk_keys, target_endpoints, exc_info=exc)
            self._dispatch_ref.register_free_slot(self.uid, 'sender', _tell=True)

            for reader in keys_to_readers.values():
                reader.close()
            keys_to_readers.clear()

            for ref in receiver_refs:
                ref.cancel_receive(session_id, addrs_to_chunks[ref.address], _tell=True, _wait=False)
            self.tell_promise(callback, *exc, _accept=False)

        try:
            if options.vineyard.socket:
                source_devices = [DataStorageDevice.VINEYARD, DataStorageDevice.DISK]  # pragma: no cover
            else:
                source_devices = [DataStorageDevice.SHARED_MEMORY, DataStorageDevice.DISK]
            _create_local_readers().then(_create_remote_writers) \
                .then(lambda *_: self._compress_and_send(
                    session_id, addrs_to_chunks, receiver_refs, keys_to_readers,
                    block_size=block_size, timeout=timeout,
                )) \
                .then(lambda *_: promise.all_(wait_refs)) \
                .then(_finalize, _handle_rejection)
        except:  # noqa: E722
            _handle_rejection(*sys.exc_info())
            return

    @log_unhandled
    def _compress_and_send(self, session_id, addrs_to_chunks, receiver_refs, keys_to_readers,
                           block_size, timeout=None):
        """
        Compress and send data to receivers in chunked manner
        :param session_id: session id
        :param addrs_to_chunks: dict mapping endpoints to chunks to send
        :param receiver_refs: refs to send data to
        """
        # collect data targets
        chunks_to_addrs = defaultdict(set)
        for addr, chunks in addrs_to_chunks.items():
            for key in chunks:
                chunks_to_addrs[key].add(addr)
        all_chunk_keys = sorted(chunks_to_addrs.keys(), key=lambda k: len(chunks_to_addrs[k]))
        addr_statuses = dict((k, EndpointTransferState()) for k in addrs_to_chunks.keys())
        addr_to_refs = dict((ref.address, ref) for ref in receiver_refs)

        # start compress and send data into targets
        logger.debug('Data writer for chunks %r allocated at targets, start transmission', all_chunk_keys)

        # filter out endpoints we need to send to
        try:
            if not receiver_refs:
                self._dispatch_ref.register_free_slot(self.uid, 'sender', _tell=True, _wait=False)
                return
            cur_key_id = 0
            cur_key = all_chunk_keys[cur_key_id]
            cur_reader = keys_to_readers[cur_key]
            with EventContext(self._events_ref, EventCategory.PROCEDURE, EventLevel.NORMAL,
                              ProcedureEventType.NETWORK, self.uid):
                while cur_key_id < len(all_chunk_keys):
                    # read a data part from reader we defined above
                    pool = cur_reader.get_io_pool()
                    next_part = pool.submit(cur_reader.read, block_size).result()
                    file_eof = len(next_part) < block_size

                    for addr in chunks_to_addrs[cur_key]:
                        addr_status = addr_statuses[addr]
                        addr_status.parts.append(next_part)
                        addr_status.keys.append(cur_key)
                        addr_status.total_size += len(next_part)
                        addr_status.end_marks.append(file_eof)

                        if addr_status.total_size >= block_size:
                            if addr_status.send_future:
                                addr_status.send_future.result(timeout=timeout)
                            addr_status.send_future = addr_to_refs[addr].receive_data_part(
                                session_id, addr_status.keys, addr_status.end_marks,
                                *addr_status.parts, _wait=False)
                            addr_status.reset()

                    # when some part goes to end, move to the next chunk
                    if file_eof:
                        cur_reader.close()
                        cur_key_id += 1
                        if cur_key_id < len(all_chunk_keys):
                            # still some chunks left unhandled
                            cur_key = all_chunk_keys[cur_key_id]
                            cur_reader = keys_to_readers[cur_key]
                        else:
                            # all chunks handled
                            for addr, addr_status in addr_statuses.items():
                                if addr_status.send_future:
                                    addr_status.send_future.result(timeout=timeout)
                                if addr_status.parts:
                                    # send remaining chunks
                                    addr_status.end_marks[-1] = True
                                    addr_status.send_future = addr_to_refs[addr].receive_data_part(
                                        session_id, addr_status.keys, addr_status.end_marks,
                                        *addr_status.parts, _wait=False)
                                addr_status.reset()
                            for addr_status in addr_statuses.values():
                                if addr_status.send_future:
                                    addr_status.send_future.result(timeout=timeout)
                                addr_status.reset()

        except:  # noqa: E722
            for ref in receiver_refs:
                ref.cancel_receive(session_id, addrs_to_chunks[ref.address], _tell=True, _wait=False)
            raise
        finally:
            for reader in keys_to_readers.values():
                reader.close()


class ReceiverDataMeta(object):
    __slots__ = 'start_time', 'chunk_size', 'source_address',\
                'status', 'transfer_event_id', 'receiver_worker_uid', \
                'callback_ids', 'callback_args', 'callback_kwargs'

    def __init__(self, start_time=None, chunk_size=None, source_address=None,
                 transfer_event_id=None, receiver_worker_uid=None, status=None,
                 callback_ids=None, callback_args=None, callback_kwargs=None):
        self.start_time = start_time
        self.chunk_size = chunk_size
        self.source_address = source_address
        self.status = status
        self.transfer_event_id = transfer_event_id
        self.receiver_worker_uid = receiver_worker_uid
        self.callback_ids = callback_ids or []
        self.callback_args = callback_args or ()
        self.callback_kwargs = callback_kwargs or {}

    def update(self, **kwargs):
        kwargs['callback_ids'] = list(set(self.callback_ids) | set(kwargs.get('callback_ids') or ()))
        for k, v in kwargs.items():
            setattr(self, k, v)


class ReceiverManagerActor(WorkerActor):
    def __init__(self):
        super().__init__()
        self._data_metas = dict()
        self._max_callback_id = 0
        self._callback_id_to_callbacks = dict()
        self._callback_id_to_keys = dict()

        self._dispatch_ref = None

    def post_create(self):
        super().post_create()

        from .dispatcher import DispatchActor
        self._dispatch_ref = self.promise_ref(DispatchActor.default_uid())

    def _update_data_meta(self, session_id, data_key, **kwargs):
        try:
            self._data_metas[(session_id, data_key)].update(**kwargs)
        except KeyError:
            self._data_metas[(session_id, data_key)] = ReceiverDataMeta(**kwargs)

    @promise.reject_on_exception
    @log_unhandled
    def create_data_writers(self, session_id, data_keys, data_sizes, sender_ref,
                            ensure_cached=True, pin_token=None, timeout=0,
                            use_promise=True, callback=None):
        sender_address = None if sender_ref is None else sender_ref.address
        logger.debug('Begin creating transmission data writer for chunks %r from %s',
                     data_keys, sender_address)
        data_locations = dict(zip(
            data_keys, self.storage_client.get_data_locations(session_id, data_keys)))
        keys_to_fetch = []
        sizes_to_fetch = []
        statuses = []

        slot_ref = self.promise_ref(self._dispatch_ref.get_hash_slot('receiver', repr(data_keys)))
        for chunk_key, data_size in zip(data_keys, data_sizes):
            session_chunk_key = (session_id, chunk_key)

            try:
                data_meta = self._data_metas[session_chunk_key]
            except KeyError:
                data_meta = self._data_metas[session_chunk_key] = \
                    ReceiverDataMeta(chunk_size=data_size, source_address=sender_address,
                                     status=ReceiveStatus.NOT_STARTED)

            if data_locations.get(chunk_key):
                data_meta.status = ReceiveStatus.RECEIVED
                statuses.append(ReceiveStatus.RECEIVED)
                self._update_data_meta(session_id, chunk_key, status=ReceiveStatus.RECEIVED)
                continue
            elif data_meta.status == ReceiveStatus.RECEIVING:
                # data transfer already started
                logger.debug('Chunk %s already started transmission', chunk_key)
                statuses.append(ReceiveStatus.RECEIVING)
                continue
            elif data_meta.status == ReceiveStatus.RECEIVED:
                data_meta = self._data_metas[session_chunk_key] = \
                    ReceiverDataMeta(chunk_size=data_size, source_address=sender_address,
                                     status=ReceiveStatus.NOT_STARTED)

            data_meta.start_time = time.time()
            data_meta.receiver_worker_uid = slot_ref.uid
            data_meta.source_address = sender_address
            data_meta.status = ReceiveStatus.RECEIVING

            self._update_data_meta(session_id, chunk_key, chunk_size=data_size,
                                   source_address=sender_address, status=ReceiveStatus.RECEIVING)
            keys_to_fetch.append(chunk_key)
            sizes_to_fetch.append(data_size)
            statuses.append(None)  # this notifies the sender to transmit data

        if use_promise:
            if keys_to_fetch:
                slot_ref.create_data_writers(
                    session_id, keys_to_fetch, sizes_to_fetch, sender_ref, ensure_cached=ensure_cached,
                    timeout=timeout, pin_token=pin_token, use_promise=use_promise, _promise=True) \
                    .then(lambda *_: self.tell_promise(callback, slot_ref, statuses))
            else:
                self.tell_promise(callback, slot_ref, statuses)
        else:
            slot_ref.create_data_writers(
                session_id, keys_to_fetch, sizes_to_fetch, sender_ref, ensure_cached=ensure_cached,
                timeout=timeout, pin_token=pin_token, use_promise=use_promise)
            return slot_ref, statuses

    def register_pending_keys(self, session_id, data_keys):
        for key in data_keys:
            session_data_key = (session_id, key)
            if session_data_key not in self._data_metas \
                    or self._data_metas[session_data_key].status == ReceiveStatus.ERROR:
                self._update_data_meta(session_id, key, status=ReceiveStatus.PENDING,
                                       callback_args=(), callback_kwargs={})

    def filter_receiving_keys(self, session_id, data_keys):
        keys = []
        receiving_status = (ReceiveStatus.PENDING, ReceiveStatus.RECEIVING)
        for k in data_keys:
            try:
                if self._data_metas[(session_id, k)].status in receiving_status:
                    keys.append(k)
            except KeyError:
                pass
        return keys

    def add_keys_callback(self, session_id, data_keys, callback):
        cb_id = self._max_callback_id
        self._max_callback_id += 1

        receiving_status = (ReceiveStatus.PENDING, ReceiveStatus.RECEIVING)
        registered_session_keys = []
        args, kwargs = (), {}
        for k in data_keys:
            session_data_key = (session_id, k)
            data_meta = self._data_metas[session_data_key]  # type: ReceiverDataMeta
            if data_meta.status in receiving_status:
                registered_session_keys.append(session_data_key)
                data_meta.callback_ids.append(cb_id)
            else:
                args, kwargs = data_meta.callback_args, data_meta.callback_kwargs

        if registered_session_keys:
            self._callback_id_to_callbacks[cb_id] = callback
            self._callback_id_to_keys[cb_id] = set(registered_session_keys)
            logger.debug('Callback for transferring %r registered', registered_session_keys)
        else:
            self._max_callback_id = cb_id
            self.tell_promise(callback, *args, **kwargs)

    def notify_keys_finish(self, session_id, data_keys, *args, **kwargs):
        keys_to_clear = []
        for data_key in data_keys:
            session_data_key = (session_id, data_key)
            try:
                data_meta = self._data_metas[session_data_key]  # type: ReceiverDataMeta
            except KeyError:
                logger.debug('Record of %s not found.', data_key)
                continue

            try:
                data_meta.callback_args = args
                data_meta.callback_kwargs = kwargs
                if kwargs.get('_accept', True):
                    data_meta.status = ReceiveStatus.RECEIVED
                else:
                    data_meta.status = ReceiveStatus.ERROR

                cb_ids = data_meta.callback_ids
                data_meta.callback_ids = []
                if not cb_ids:
                    continue

                kwargs['_wait'] = False
                notified = 0
                for cb_id in cb_ids:
                    cb_keys = self._callback_id_to_keys[cb_id]
                    cb_keys.remove(session_data_key)
                    if not cb_keys:
                        del self._callback_id_to_keys[cb_id]
                        cb = self._callback_id_to_callbacks.pop(cb_id)
                        notified += 1
                        self.tell_promise(cb, *args, **kwargs)
                logger.debug('%d transfer listeners of %s notified.', notified, data_key)
            finally:
                if data_meta.status == ReceiveStatus.RECEIVED:
                    keys_to_clear.append(data_key)
        self.ref().clear_keys(session_id, keys_to_clear, _tell=True, _delay=5)

    def clear_keys(self, session_id, keys):
        for k in keys:
            self._data_metas.pop((session_id, k), None)


class ReceiverWorkerActor(WorkerActor):
    """
    Actor handling receiving data from a SenderActor
    """
    def __init__(self):
        super().__init__()
        self._chunk_holder_ref = None
        self._dispatch_ref = None
        self._receiver_manager_ref = None
        self._events_ref = None
        self._status_ref = None

        self._data_writers = dict()
        self._writing_futures = dict()
        self._data_metas = dict()

    def post_create(self):
        from .events import EventsActor
        from .status import StatusActor
        from .dispatcher import DispatchActor

        super().post_create()

        self._events_ref = self.ctx.actor_ref(EventsActor.default_uid())
        if not self.ctx.has_actor(self._events_ref):
            self._events_ref = None

        self._status_ref = self.ctx.actor_ref(StatusActor.default_uid())
        if not self.ctx.has_actor(self._status_ref):
            self._status_ref = None

        self._receiver_manager_ref = self.ctx.actor_ref(ReceiverManagerActor.default_uid())
        if not self.ctx.has_actor(self._receiver_manager_ref):
            self._receiver_manager_ref = None

        self._dispatch_ref = self.promise_ref(DispatchActor.default_uid())
        self._dispatch_ref.register_free_slot(self.uid, 'receiver')

    @log_unhandled
    def check_status(self, session_id, chunk_key):
        """
        Check if data exists or is being transferred in the target worker
        :param session_id: session id
        :param chunk_key: chunk key
        """
        session_chunk_key = (session_id, chunk_key)

        if self.storage_client.get_data_locations(session_id, [chunk_key])[0]:
            return ReceiveStatus.RECEIVED
        if session_chunk_key in self._data_writers:
            # data still being transferred
            return ReceiveStatus.RECEIVING
        return ReceiveStatus.NOT_STARTED

    @promise.reject_on_exception
    @log_unhandled
    def create_data_writers(self, session_id, chunk_keys, data_sizes, sender_ref,
                            ensure_cached=True, timeout=0, pin_token=None,
                            use_promise=True, callback=None):
        """
        Create a data writer for subsequent data transfer. The writer can either work on
        shared storage or spill.

        :param session_id: session id
        :param chunk_keys: chunk keys
        :param data_sizes: uncompressed data sizes
        :param sender_ref: ActorRef of SenderActor
        :param ensure_cached: if True, the data should be stored in shared memory,
                              otherwise spill is acceptable
        :param timeout: timeout if the chunk receiver does not close
        :param pin_token: token to pin the data
        :param use_promise: if True, we use promise callback to notify accomplishment
                            of writer creation, otherwise the function returns directly
                            and when sill is needed, a StorageFull will be raised instead.
        :param callback: promise callback
        """
        promises = []
        failed = False
        if options.vineyard.socket:
            device_order = [DataStorageDevice.VINEYARD]  # pragma: no cover
        else:
            device_order = [DataStorageDevice.SHARED_MEMORY]
        source_address = sender_ref.address if sender_ref is not None else None
        if not ensure_cached:
            device_order += [DataStorageDevice.DISK]

        def _handle_accept_key(key, writer):
            if failed:
                writer.close(finished=False)
            else:
                self._data_writers[(session_id, key)] = writer

        @log_unhandled
        def _handle_reject_key(key, *exc):
            nonlocal failed
            if self.check_status(session_id, key) == ReceiveStatus.RECEIVED:
                logger.debug('Chunk %s already received', key)
            else:
                logger.debug('Rejecting %s from putting into plasma.', key)
                failed = True
                self._stop_transfer_with_exc(session_id, chunk_keys, exc)
                if callback is not None:
                    self.tell_promise(callback, *exc, _accept=False)

        # configure timeout callback
        if timeout:
            self.ref().handle_receive_timeout(session_id, chunk_keys, _delay=timeout, _tell=True)

        for chunk_key, data_size in zip(chunk_keys, data_sizes):
            self._data_metas[(session_id, chunk_key)] = ReceiverDataMeta(
                start_time=time.time(), chunk_size=data_size, source_address=source_address)
            if use_promise:
                promises.append(self.storage_client.create_writer(
                    session_id, chunk_key, data_size, device_order, packed=True,
                    pin_token=pin_token, _promise=True)
                    .then(functools.partial(_handle_accept_key, chunk_key),
                          functools.partial(_handle_reject_key, chunk_key)))
            else:
                try:
                    _writer = self.storage_client.create_writer(
                        session_id, chunk_key, data_size, device_order, packed=True,
                        pin_token=pin_token, _promise=False)
                    _handle_accept_key(chunk_key, _writer)
                    return self.address, None
                except:  # noqa: E722
                    _handle_reject_key(chunk_key, *sys.exc_info())
                    raise

        promise.all_(promises).then(lambda *_: self.tell_promise(callback))

    def _wait_unfinished_writing(self, session_id, data_key):
        try:
            self._writing_futures[(session_id, data_key)].result()
            del self._writing_futures[(session_id, data_key)]
        except KeyError:
            pass

    @log_unhandled
    def receive_data_part(self, session_id, chunk_keys, end_marks, *data_parts):
        """
        Receive data part from sender
        :param session_id: session id
        :param chunk_keys: chunk keys
        :param end_marks: array with same number of boolean elements as chunk keys.
                          if one element is True, the corresponding data in data_parts
                          is the last part of the chunk.
        :param data_parts: data parts to be written
        """
        try:
            finished_keys, finished_meta_keys, finished_metas = [], [], []
            for chunk_key, data_part, end_mark in zip(chunk_keys, data_parts, end_marks):
                self._wait_unfinished_writing(session_id, chunk_key)
                session_chunk_key = (session_id, chunk_key)
                try:
                    data_meta = self._data_metas[session_chunk_key]  # type: ReceiverDataMeta

                    # if error occurred, interrupts
                    if data_meta.status == ReceiveStatus.ERROR:
                        raise data_meta.callback_args[1].with_traceback(data_meta.callback_args[2])
                    writer = self._data_writers[session_chunk_key]
                    pool = writer.get_io_pool()
                    self._writing_futures[session_chunk_key] = pool.submit(
                        writer.write, data_part)

                    if end_mark:
                        finished_keys.append(chunk_key)
                        if not isinstance(chunk_key, tuple):
                            finished_meta_keys.append(chunk_key)
                            finished_metas.append(WorkerMeta(chunk_size=data_meta.chunk_size,
                                                             workers=(self.address,)))
                except:  # noqa: E722
                    self._stop_transfer_with_exc(session_id, chunk_keys, sys.exc_info())
                    raise

            if finished_keys:
                for chunk_key in finished_keys:
                    session_chunk_key = (session_id, chunk_key)
                    data_meta = self._data_metas[session_chunk_key]  # type: ReceiverDataMeta

                    self._wait_unfinished_writing(session_id, chunk_key)
                    # update transfer speed stats
                    if self._status_ref:
                        time_delta = time.time() - data_meta.start_time
                        self._status_ref.update_mean_stats(
                            'net_transfer_speed', data_meta.chunk_size * 1.0 / time_delta,
                            _tell=True, _wait=False)
                    self._data_writers[session_chunk_key].close()
                    data_meta.status = ReceiveStatus.RECEIVED
                    logger.debug('Transfer for data %s finished.', chunk_key)
                    del self._data_writers[session_chunk_key]

                self._invoke_finish_callbacks(session_id, finished_keys)
            if finished_meta_keys:
                self.get_meta_client().batch_set_chunk_meta(session_id, finished_meta_keys, finished_metas)
        finally:
            del data_parts

    def _is_receive_running(self, session_id, chunk_key):
        receive_done_statuses = (ReceiveStatus.ERROR, ReceiveStatus.RECEIVED)
        try:
            return self._data_metas[(session_id, chunk_key)].status not in receive_done_statuses
        except KeyError:
            return False

    @log_unhandled
    def cancel_receive(self, session_id, chunk_keys, exc_info=None):
        """
        Cancel data receive by returning an ExecutionInterrupted
        :param session_id: session id
        :param chunk_keys: chunk keys
        :param exc_info: exception to raise
        """
        receiving_keys = []
        for k in chunk_keys:
            if self._is_receive_running(session_id, k):
                receiving_keys.append(k)
            self._wait_unfinished_writing(session_id, k)

        logger.debug('Transfer for %r cancelled.', chunk_keys)

        if exc_info is None:
            exc_info = build_exc_info(ExecutionInterrupted)

        self._stop_transfer_with_exc(session_id, receiving_keys, exc_info)

    @log_unhandled
    def notify_dead_senders(self, dead_workers):
        """
        When some peer workers are dead, corresponding receivers will be cancelled
        :param dead_workers: endpoints of dead workers
        """
        dead_workers = set(dead_workers)
        exc_info = build_exc_info(WorkerDead)
        session_to_keys = defaultdict(set)
        for session_chunk_key in self._data_writers.keys():
            if self._data_metas[session_chunk_key].source_address in dead_workers:
                session_to_keys[session_chunk_key[0]].add(session_chunk_key[1])
        for session_id, data_keys in session_to_keys.items():
            self.ref().cancel_receive(session_id, list(data_keys), exc_info=exc_info, _tell=True)

    @log_unhandled
    def handle_receive_timeout(self, session_id, chunk_keys):
        if not any(self._is_receive_running(session_id, k) for k in chunk_keys):
            # if transfer already finishes, no needs to report timeout
            return
        logger.debug('Transfer for %r timed out, cancelling.', chunk_keys)
        self._stop_transfer_with_exc(session_id, chunk_keys, build_exc_info(TimeoutError))

    def _stop_transfer_with_exc(self, session_id, chunk_keys, exc):
        for chunk_key in chunk_keys:
            self._wait_unfinished_writing(session_id, chunk_key)

        if not isinstance(exc[1], ExecutionInterrupted):
            logger.exception('Error occurred in receiving %r. Cancelling transfer.',
                             chunk_keys, exc_info=exc)

        for chunk_key in chunk_keys:
            session_chunk_key = (session_id, chunk_key)

            # stop and close data writer
            try:
                # transfer is not finished yet, we need to clean up unfinished stuffs
                self._data_writers[session_chunk_key].close(finished=False)
                del self._data_writers[session_chunk_key]
            except KeyError:
                # transfer finished and writer cleaned, no need to clean up
                pass

            try:
                data_meta = self._data_metas[session_chunk_key]  # type: ReceiverDataMeta
                data_meta.status = ReceiveStatus.ERROR
            except KeyError:
                pass

        self._invoke_finish_callbacks(session_id, chunk_keys, *exc, **dict(_accept=False))

    def _invoke_finish_callbacks(self, session_id, chunk_keys, *args, **kwargs):
        # invoke registered callbacks for chunk
        for k in chunk_keys:
            try:
                data_meta = self._data_metas.pop((session_id, k))  # type: ReceiverDataMeta
            except KeyError:
                continue

            if data_meta.transfer_event_id is not None and self._events_ref is not None:
                self._events_ref.close_event(data_meta.transfer_event_id, _tell=True, _wait=False)
            if not kwargs.get('_accept', True):
                if not data_meta.callback_args or data_meta.callback_args[0] is ExecutionInterrupted:
                    data_meta.callback_args = args
                    data_meta.callback_kwargs = kwargs
                else:
                    args = data_meta.callback_args
                    kwargs = data_meta.callback_kwargs
            else:
                data_meta.callback_args = args
                data_meta.callback_kwargs = kwargs
        kwargs['_tell'] = True
        if self._receiver_manager_ref:
            self._receiver_manager_ref.notify_keys_finish(session_id, chunk_keys, *args, **kwargs)


class ResultCopyActor(WorkerActor):
    def start_copy(self, session_id, chunk_key, targets):
        locations = [v[1] for v in self.storage_client.get_data_locations(session_id, [chunk_key])[0]]
        if set(locations).intersection(targets):
            return
        ev = self.ctx.event()
        self.storage_client.copy_to(session_id, [chunk_key], targets) \
            .then(lambda *_: ev.set())
        return ev


class ResultSenderActor(WorkerActor):
    """
    Actor handling sending result to user client
    """
    def __init__(self):
        super().__init__()
        self._result_copy_ref = None
        self._serialize_pool = None

    def post_create(self):
        super().post_create()
        self._serialize_pool = self.ctx.threadpool(1)
        self._result_copy_ref = self.ctx.create_actor(ResultCopyActor, uid=ResultCopyActor.default_uid())

    def pre_destroy(self):
        self._result_copy_ref.destroy()
        super().pre_destroy()

    def fetch_batch_data(self, session_id, chunk_keys, index_objs=None, compression_type=None):
        results = []
        if index_objs is not None:
            for chunk_key, index_obj in zip(chunk_keys, index_objs):
                results.append(self.fetch_data(session_id, chunk_key, index_obj, compression_type=compression_type))
        else:
            for chunk_key in chunk_keys:
                results.append(self.fetch_data(session_id, chunk_key, compression_type=compression_type))
        return results

    def fetch_data(self, session_id, chunk_key, index_obj=None, compression_type=None):
        if compression_type is None:
            compression_type = dataserializer.CompressType(options.worker.transfer_compression)
        if index_obj is None:
            if options.vineyard.socket:
                target_devs = [DataStorageDevice.VINEYARD, DataStorageDevice.DISK]  # pragma: no cover
            else:
                target_devs = [DataStorageDevice.SHARED_MEMORY, DataStorageDevice.DISK]
            ev = self._result_copy_ref.start_copy(session_id, chunk_key, target_devs)
            if ev:
                ev.wait(options.worker.prepare_data_timeout)

            reader = self.storage_client.create_reader(
                session_id, chunk_key, target_devs, packed=True,
                packed_compression=compression_type, _promise=False)

            with reader:
                pool = reader.get_io_pool()
                return pool.submit(reader.read).result()
        else:
            try:
                if options.vineyard.socket:
                    memory_device = DataStorageDevice.VINEYARD  # pragma: no cover
                else:
                    memory_device = DataStorageDevice.SHARED_MEMORY
                value = self.storage_client.get_object(
                    session_id, chunk_key, [memory_device], _promise=False)
            except IOError:
                reader = self.storage_client.create_reader(
                    session_id, chunk_key, [DataStorageDevice.DISK], packed=False, _promise=False)
                with reader:
                    pool = reader.get_io_pool()
                    value = dataserializer.deserialize(pool.submit(reader.read).result())

            try:
                sliced_value = value.iloc[tuple(index_obj)]
            except AttributeError:
                sliced_value = value[tuple(index_obj)]

            return self._serialize_pool.submit(
                dataserializer.dumps, sliced_value, compress=compression_type).result()


def put_remote_chunk(session_id, chunk_key, data, receiver_manager_ref):
    """
    Put a chunk to target machine using given receiver_ref
    """
    from .dataio import ArrowBufferIO
    buf = dataserializer.serialize(data).to_buffer()
    receiver_ref, _ = receiver_manager_ref.create_data_writers(
        session_id, [chunk_key], [buf.size], None, ensure_cached=False, use_promise=False)
    receiver_ref = receiver_manager_ref.ctx.actor_ref(receiver_ref)
    block_size = options.worker.transfer_block_size

    reader = None
    try:
        reader = ArrowBufferIO(buf, 'r', block_size=block_size)
        futures = []
        while True:
            next_part = reader.read(block_size)
            is_last = not next_part or len(next_part) < block_size
            [f.result() for f in futures]
            futures.append(receiver_ref.receive_data_part(
                session_id, [chunk_key], [is_last], next_part, _wait=False))
            if is_last:
                [f.result() for f in futures]
                break
    except:  # noqa: E722
        receiver_ref.cancel_receive(session_id, [chunk_key])
        raise
    finally:
        if reader:
            reader.close()
        del reader
