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

import functools
import logging
import os
import sys
import time
import zlib
from collections import defaultdict

import pandas as pd
import pyarrow

from .. import promise
from ..compat import six, Enum, TimeoutError  # pylint: disable=W0622
from ..config import options
from ..serialize import dataserializer
from ..errors import *
from ..utils import log_unhandled, build_exc_info
from .dataio import FileBufferIO, ArrowBufferIO
from .events import EventContext, EventCategory, EventLevel, ProcedureEventType
from .spill import build_spill_file_name, get_spill_data_size, read_spill_file
from .utils import WorkerActor, ExpiringCache

logger = logging.getLogger(__name__)


class ReceiveStatus(Enum):
    NOT_STARTED = 0
    RECEIVING = 1
    RECEIVED = 2
    ERROR = 3


class SenderActor(WorkerActor):
    """
    Actor handling sending data to ReceiverActors in other workers
    """
    def __init__(self):
        super(SenderActor, self).__init__()
        self._dispatch_ref = None
        self._events_ref = None
        self._mem_quota_ref = None

        self._serialize_pool = None

    def post_create(self):
        from .dispatcher import DispatchActor
        from .events import EventsActor
        from .quota import MemQuotaActor

        super(SenderActor, self).post_create()

        self._events_ref = self.ctx.actor_ref(EventsActor.default_uid())
        if not self.ctx.has_actor(self._events_ref):
            self._events_ref = None

        self._dispatch_ref = self.promise_ref(DispatchActor.default_uid())
        self._dispatch_ref.register_free_slot(self.uid, 'sender')
        self._mem_quota_ref = self.promise_ref(MemQuotaActor.default_uid())

        self._serialize_pool = self.ctx.threadpool(1)

    def _read_data_size(self, session_id, chunk_key):
        """
        Obtain decompressed data size for a certain chunk
        :param session_id: session id
        :param chunk_key: chunk key
        :return: size of data
        """
        nbytes = None
        try:
            nbytes = self._chunk_store.get_actual_size(session_id, chunk_key)
        except KeyError:
            pass
        if nbytes is None:
            try:
                nbytes = get_spill_data_size(chunk_key)
            except KeyError:
                raise DependencyMissing('Dependency %s not met on sending.' % chunk_key)
        return nbytes

    def _collect_receiver_refs(self, chunk_key, target_endpoints, target_slots, timeout):
        from .dispatcher import DispatchActor

        if isinstance(target_endpoints, six.string_types):
            target_endpoints, target_slots = [target_endpoints], [target_slots]
        if target_slots is None:
            target_slots = [None] * len(target_endpoints)

        refs, slot_futures = [], []
        for ep, slot in zip(target_endpoints, target_slots):
            if slot is None:
                dispatch_ref = self.ctx.actor_ref(DispatchActor.default_uid(), address=ep)
                slot_futures.append((ep, dispatch_ref.get_hash_slot('receiver', chunk_key, _wait=False)))
            else:
                refs.append(self.promise_ref(slot, address=ep))
        refs.extend([self.promise_ref(f.result(timeout), address=ep)
                     for ep, f in slot_futures])
        return refs

    @promise.reject_on_exception
    @log_unhandled
    def send_data(self, session_id, chunk_key, target_endpoints, target_slots=None,
                  ensure_cached=True, compression=None, timeout=None, callback=None):
        """
        Send data to other workers
        :param session_id: session id
        :param chunk_key: chunk to be sent
        :param target_endpoints: endpoints to receive this chunk
        :param target_slots: slots for receivers, None if not fixed
        :param ensure_cached: if True, make sure the data is in the shared storage of the target worker
        :param compression: compression type when transfer in network
        :param timeout: timeout of data sending
        :param callback: promise callback
        """
        already_started = set()
        wait_refs = []
        data_size = self._read_data_size(session_id, chunk_key)
        compression = compression or dataserializer.CompressType(options.worker.transfer_compression)

        try:
            receiver_refs = self._collect_receiver_refs(
                chunk_key, target_endpoints, target_slots, timeout)
        except:  # noqa: E722
            self._dispatch_ref.register_free_slot(self.uid, 'sender', _tell=True)
            raise

        @log_unhandled
        def _handle_created(ref, address, status):
            # filter out endpoints already transferred or already started transfer
            if status == ReceiveStatus.RECEIVED:
                already_started.add(address)
            elif status == ReceiveStatus.RECEIVING:
                already_started.add(address)
                wait_refs.append(ref.register_finish_callback(
                    session_id, chunk_key, _timeout=timeout, _promise=True))

        @log_unhandled
        def _finalize(*_):
            self._dispatch_ref.register_free_slot(self.uid, 'sender', _tell=True)
            self.tell_promise(callback, data_size)

        @log_unhandled
        def _handle_rejection(*exc):
            logger.exception('Transfer chunk %s to %r failed', chunk_key, target_endpoints, exc_info=exc)
            self._dispatch_ref.register_free_slot(self.uid, 'sender', _tell=True)
            for ref in receiver_refs:
                ref.cancel_receive(session_id, chunk_key, _tell=True, _wait=False)
            self.tell_promise(callback, *exc, **dict(_accept=False))

        try:
            create_write_promises = []
            for ref in receiver_refs:
                # register transfer actions
                create_write_promises.append(
                    ref.create_data_writer(
                        session_id, chunk_key, data_size, self.ref(), ensure_cached=ensure_cached,
                        timeout=timeout, _timeout=timeout, _promise=True
                    ).then(functools.partial(_handle_created, ref))
                )

            if create_write_promises:
                promise.all_(create_write_promises) \
                    .then(lambda *_: self._compress_and_send(
                        session_id, chunk_key,
                        [ref for ref in receiver_refs if ref.address not in already_started],
                        compression=compression, timeout=timeout,
                    )) \
                    .then(lambda *_: promise.all_(wait_refs)) \
                    .then(_finalize, _handle_rejection)
            else:
                # nothing to send, the slot can be released
                self._dispatch_ref.register_free_slot(self.uid, 'sender', _tell=True, _wait=False)
                return
        except:  # noqa: E722
            _handle_rejection(*sys.exc_info())
            return

    @log_unhandled
    def _compress_and_send(self, session_id, chunk_key, target_refs, compression, timeout=None):
        """
        Compress and send data to receivers in chunked manner
        :param session_id: session id
        :param chunk_key: chunk key
        :param target_refs: refs to send data to
        :param compression: compression type when transfer in network
        """
        # start compress and send data into targets
        logger.debug('Data writer for chunk %s allocated at targets, start transmission', chunk_key)
        block_size = options.worker.transfer_block_size
        reader = None

        # filter out endpoints we need to send to
        try:
            if not target_refs:
                self._dispatch_ref.register_free_slot(self.uid, 'sender', _tell=True, _wait=False)
                return

            # try load data from plasma store
            buf = None
            try:
                buf = self._chunk_store.get_buffer(session_id, chunk_key)
                # create a stream compressor from shared buffer
                reader = ArrowBufferIO(
                    buf, 'r', compress_out=compression, block_size=block_size)
            except KeyError:
                pass
            finally:
                del buf
            if reader is None:
                # no reader created from plasma store, we load directly from spill
                file_name = build_spill_file_name(chunk_key)
                if not file_name:
                    raise SpillNotConfigured('Spill not configured')
                reader = FileBufferIO(
                    open(file_name, 'rb'), 'r', compress_out=compression, block_size=block_size)

            futures = []
            checksum = 0
            with EventContext(self._events_ref, EventCategory.PROCEDURE, EventLevel.NORMAL,
                              ProcedureEventType.NETWORK, self.uid):
                while True:
                    # read a data part from reader we defined above
                    next_chunk = self._serialize_pool.submit(reader.read, block_size).result()
                    is_last = not next_chunk or len(next_chunk) < block_size
                    # make sure all previous transfers finished
                    [f.result(timeout=timeout) for f in futures]
                    checksum = zlib.crc32(next_chunk, checksum)
                    futures = []
                    for ref in target_refs:
                        # we perform async transfer and wait after next part is loaded and compressed
                        futures.append(ref.receive_data_part(
                            session_id, chunk_key, next_chunk, checksum, is_last=is_last, _wait=False))
                    if is_last:
                        [f.result(timeout=timeout) for f in futures]
                        break
        except:  # noqa: E722
            for ref in target_refs:
                ref.cancel_receive(session_id, chunk_key, _tell=True, _wait=False)
            raise
        finally:
            if reader:
                reader.close()
            del reader


class ReceiverDataMeta(object):
    __slots__ = 'start_time', 'chunk_size', 'write_shared', 'checksum', \
                'source_address', 'transfer_event_id', 'status', 'callback_args', \
                'callback_kwargs'

    def __init__(self, start_time=None, chunk_size=None, write_shared=True, checksum=0,
                 source_address=None, transfer_event_id=None, status=None,
                 callback_args=None, callback_kwargs=None):
        self.start_time = start_time or time.time()
        self.chunk_size = chunk_size
        self.write_shared = write_shared
        self.checksum = checksum
        self.source_address = source_address
        self.status = status
        self.transfer_event_id = transfer_event_id
        self.callback_args = callback_args or ()
        self.callback_kwargs = callback_kwargs or {}


class ReceiverActor(WorkerActor):
    """
    Actor handling receiving data from a SenderActor
    """
    def __init__(self):
        super(ReceiverActor, self).__init__()
        self._chunk_holder_ref = None
        self._mem_quota_ref = None
        self._dispatch_ref = None
        self._events_ref = None
        self._status_ref = None

        self._finish_callbacks = defaultdict(list)
        self._data_writers = dict()
        self._data_meta_cache = ExpiringCache()

        self._serialize_pool = None

    def post_create(self):
        from .chunkholder import ChunkHolderActor
        from .quota import MemQuotaActor
        from .events import EventsActor
        from .status import StatusActor
        from .dispatcher import DispatchActor

        super(ReceiverActor, self).post_create()
        self._chunk_holder_ref = self.promise_ref(ChunkHolderActor.default_uid())
        self._mem_quota_ref = self.promise_ref(MemQuotaActor.default_uid())

        self._events_ref = self.ctx.actor_ref(EventsActor.default_uid())
        if not self.ctx.has_actor(self._events_ref):
            self._events_ref = None

        self._status_ref = self.ctx.actor_ref(StatusActor.default_uid())
        if not self.ctx.has_actor(self._status_ref):
            self._status_ref = None

        self._dispatch_ref = self.promise_ref(DispatchActor.default_uid())
        self._dispatch_ref.register_free_slot(self.uid, 'receiver')

        self._serialize_pool = self.ctx.threadpool(1)

    @log_unhandled
    def check_status(self, session_id, chunk_key):
        """
        Check if data exists or is being transferred in the target worker
        :param session_id: session id
        :param chunk_key: chunk key
        """
        from .spill import build_spill_file_name
        session_chunk_key = (session_id, chunk_key)

        if self._chunk_holder_ref.is_stored(chunk_key):
            # data in plasma
            return ReceiveStatus.RECEIVED
        if session_chunk_key in self._data_writers:
            # data still being transferred
            return ReceiveStatus.RECEIVING
        fn = build_spill_file_name(chunk_key)
        if fn and os.path.exists(fn) and session_chunk_key not in self._data_writers:
            # data in plasma store
            return ReceiveStatus.RECEIVED
        return ReceiveStatus.NOT_STARTED

    @promise.reject_on_exception
    @log_unhandled
    def register_finish_callback(self, session_id, chunk_key, callback):
        """
        Register a promise callback to handle transfer termination.
        If the chunk has already been transferred or error occurred,
        the existing results are sent.
        :param session_id: session id
        :param chunk_key: chunk key
        :param callback: promise callback
        """
        session_chunk_key = (session_id, chunk_key)
        if session_chunk_key not in self._data_meta_cache:
            self._finish_callbacks[session_chunk_key].append(callback)
            return
        data_meta = self._data_meta_cache[session_chunk_key]  # type: ReceiverDataMeta
        if data_meta.status in (ReceiveStatus.RECEIVED, ReceiveStatus.ERROR):
            # invoke callback directly when transfer finishes
            self.tell_promise(callback, *data_meta.callback_args, **data_meta.callback_kwargs)
        else:
            self._finish_callbacks[session_chunk_key].append(callback)

    @promise.reject_on_exception
    @log_unhandled
    def create_data_writer(self, session_id, chunk_key, data_size, sender_ref,
                           ensure_cached=True, timeout=0, use_promise=True, callback=None):
        """
        Create a data writer for subsequent data transfer. The writer can either work on
        shared storage or spill.
        :param session_id: session id
        :param chunk_key: chunk key
        :param data_size: uncompressed data size
        :param sender_ref: ActorRef of SenderActor
        :param ensure_cached: if True, the data should be stored in shared memory, otherwise spill is acceptable
        :param timeout: timeout if the chunk receiver does not close
        :param use_promise: if True, we use promise callback to notify accomplishment of writer creation,
            otherwise the function returns directly and when sill is needed, a StorageFull will be raised instead.
        :param callback: promise callback
        """
        sender_address = None if sender_ref is None else sender_ref.address

        logger.debug('Begin creating transmission data writer for chunk %s from %s',
                     chunk_key, sender_address)
        session_chunk_key = (session_id, chunk_key)

        data_status = self.check_status(session_id, chunk_key)

        if data_status == ReceiveStatus.RECEIVING:
            # data transfer already started
            logger.debug('Chunk %s already started transmission', chunk_key)
            if callback:
                self.tell_promise(callback, self.address, ReceiveStatus.RECEIVING)
            return self.address, ReceiveStatus.RECEIVING

        # build meta data for data transfer
        if session_chunk_key in self._data_meta_cache:
            if data_status == ReceiveStatus.RECEIVED:
                # already received: callback directly
                logger.debug('Chunk %s already received', chunk_key)
                if callback:
                    self.tell_promise(callback, self.address, ReceiveStatus.RECEIVED)
                self._invoke_finish_callbacks(session_id, chunk_key)
                return self.address, ReceiveStatus.RECEIVED
            else:
                del self._data_meta_cache[session_chunk_key]

        self._data_meta_cache[session_chunk_key] = ReceiverDataMeta(
            chunk_size=data_size, source_address=sender_address,
            write_shared=True, status=ReceiveStatus.RECEIVING)

        # configure timeout callback
        if timeout:
            self.ref().handle_receive_timeout(session_id, chunk_key, _delay=timeout, _tell=True)

        @log_unhandled
        def _handle_accept(result):
            address, state = result
            if state == ReceiveStatus.RECEIVED:
                self._invoke_finish_callbacks(session_id, chunk_key)
            if callback is not None:
                self.tell_promise(callback, address, state)

        @log_unhandled
        def _handle_reject(*exc):
            logger.debug('Rejecting %s from putting into plasma.', chunk_key)
            self._stop_transfer_with_exc(session_id, chunk_key, exc)
            if callback:
                self.tell_promise(callback, *exc, **dict(_accept=False))

        if use_promise:
            promise.finished() \
                .then(lambda *_: self._create_writer(session_id, chunk_key, ensure_cached=ensure_cached)) \
                .then(_handle_accept, _handle_reject)
        else:
            try:
                result = self._create_writer(session_id, chunk_key, ensure_cached=ensure_cached,
                                             use_promise=False)
                _handle_accept(result)
                return result
            except:  # noqa: E722
                _handle_reject(*sys.exc_info())
                raise

    @log_unhandled
    def _create_writer(self, session_id, chunk_key, ensure_cached=True, spill_times=1,
                       use_promise=True):
        """
        Create data writer for chunk
        :param session_id: session id
        :param chunk_key: chunk key
        :param ensure_cached: True if we need to make sure that data are written into the shared cache
        :param spill_times: spill scale
        :param use_promise: if True, we use promise callback to notify accomplishment of writer creation,
            otherwise the function returns directly and when sill is needed, a StorageFull will be raised instead.
        :return:
        """
        block_size = options.worker.transfer_block_size
        disk_compression = dataserializer.CompressType(options.worker.disk_compression)

        # actual create data writer
        session_chunk_key = (session_id, chunk_key)
        data_meta = self._data_meta_cache[session_chunk_key]  # type: ReceiverDataMeta
        data_size = data_meta.chunk_size
        buf = None
        try:
            # attempt to create data chunk on shared store
            buf = self._chunk_store.create(session_id, chunk_key, data_size)
            data_meta.start_time = time.time()
            logger.debug('Successfully created data writer with %s bytes in plasma for chunk %s',
                         data_size, chunk_key)
            # create a writer for the chunk
            self._data_writers[session_chunk_key] = ArrowBufferIO(buf, 'w', block_size=block_size)
            if self._events_ref is not None:
                data_meta.transfer_event_id = self._events_ref.add_open_event(
                    EventCategory.PROCEDURE, EventLevel.NORMAL, ProcedureEventType.NETWORK, self.uid)
            return self.address, None
        except (KeyError, StoreKeyExists):
            if self.check_status(session_id, chunk_key) != ReceiveStatus.RECEIVED:
                raise
            # data already registered
            logger.debug('Chunk %s already registered', chunk_key)
            self._invoke_finish_callbacks(session_id, chunk_key)
            return self.address, ReceiveStatus.RECEIVED
        except StoreFull:
            # no space left in the shared store
            if ensure_cached:
                # if promise is disabled, we have to raise an error
                if not use_promise:
                    raise
                # spill and try again
                return self._chunk_holder_ref.spill_size(data_size, spill_times, _promise=True) \
                    .then(lambda *_: self._create_writer(session_id, chunk_key, ensure_cached=True,
                                                         spill_times=min(spill_times + 1, 1024)))
            else:
                # create a writer for spill
                logger.debug('Writing data %s directly into spill.', chunk_key)
                data_meta.write_shared = False
                self._chunk_holder_ref.spill_size(data_size, _tell=True)
                spill_file_name = build_spill_file_name(chunk_key, writing=True)
                try:
                    spill_file = FileBufferIO(
                        open(spill_file_name, 'wb'), 'w', compress_in=disk_compression,
                        block_size=block_size)
                    data_meta.start_time = time.time()
                    self._data_writers[session_chunk_key] = spill_file
                    if self._events_ref is not None:
                        data_meta.transfer_event_id = self._events_ref.add_open_event(
                            EventCategory.PROCEDURE, EventLevel.NORMAL, ProcedureEventType.NETWORK, self.uid)
                except (KeyError, IOError):
                    if self.check_status(session_id, chunk_key) == ReceiveStatus.RECEIVED:
                        logger.debug('Chunk %s already stored', chunk_key)
                        return self.address, ReceiveStatus.RECEIVED
                    raise ObjectNotInPlasma([chunk_key])
                return self.address, None
        finally:
            del buf

    @log_unhandled
    def receive_data_part(self, session_id, chunk_key, data_part, checksum, is_last=False):
        """
        Receive data part from sender
        :param session_id: session id
        :param chunk_key: chunk key
        :param data_part: data to be written
        :param checksum: checksum up to now
        :param is_last: if True, after writing this chunk, a
        """
        session_chunk_key = (session_id, chunk_key)
        try:
            data_meta = self._data_meta_cache[session_chunk_key]  # type: ReceiverDataMeta
            meta_future = None

            if data_part:
                if is_last:
                    # we assume (and also with a very high probability)
                    # that data transfer will succeed, hence we set chunk meta
                    # before writing
                    meta_future = self.get_meta_client().set_chunk_meta(
                        session_id, chunk_key, size=data_meta.chunk_size, workers=(self.address,),
                        _wait=False
                    )
                # check if checksum matches
                local_checksum = zlib.crc32(data_part, data_meta.checksum)
                if local_checksum != checksum:
                    raise ChecksumMismatch
                data_meta.checksum = local_checksum

                # if error occurred, interrupts
                if data_meta.status == ReceiveStatus.ERROR:
                    six.reraise(*data_meta.callback_args)
                    return  # pragma: no cover
                self._serialize_pool.submit(self._data_writers[session_chunk_key].write, data_part).result()
                if meta_future:
                    meta_future.result()

            if is_last:
                # update transfer speed stats
                if self._status_ref:
                    time_delta = time.time() - data_meta.start_time
                    self._status_ref.update_mean_stats(
                        'net_transfer_speed', data_meta.chunk_size * 1.0 / time_delta,
                        _tell=True, _wait=False)

                if not isinstance(chunk_key, tuple):
                    self.get_meta_client().set_chunk_meta(
                        session_id, chunk_key, size=data_meta.chunk_size, workers=(self.address,))

                if data_meta.write_shared:
                    # seal data on shared store
                    self._chunk_store.seal(session_id, chunk_key)
                    self._chunk_holder_ref.register_chunk(session_id, chunk_key)
                else:
                    # move spill data to 'ready' place
                    src_dir = build_spill_file_name(chunk_key, writing=True)
                    dest_dir = build_spill_file_name(chunk_key, writing=False)
                    os.rename(src_dir, dest_dir)

                self._data_writers[session_chunk_key].close()
                del self._data_writers[session_chunk_key]

                data_meta.status = ReceiveStatus.RECEIVED
                self._invoke_finish_callbacks(session_id, chunk_key)
                logger.debug('Transfer for data %s finished.', chunk_key)
        except:  # noqa: E722
            self._stop_transfer_with_exc(session_id, chunk_key, sys.exc_info())

    def _is_receive_running(self, session_id, chunk_key):
        session_chunk_key = (session_id, chunk_key)
        if session_chunk_key not in self._data_meta_cache:
            return False
        if self._data_meta_cache[session_chunk_key].status in (ReceiveStatus.ERROR, ReceiveStatus.RECEIVED):
            # already terminated, we do nothing
            return False
        return True

    @log_unhandled
    def cancel_receive(self, session_id, chunk_key, exc_info=None):
        """
        Cancel data receive by returning an ExecutionInterrupted
        :param session_id: session id
        :param chunk_key: chunk key
        :param exc_info: exception to raise
        """
        logger.debug('Transfer for %s cancelled.', chunk_key)
        if not self._is_receive_running(session_id, chunk_key):
            return

        if exc_info is None:
            exc_info = build_exc_info(ExecutionInterrupted)

        self._stop_transfer_with_exc(session_id, chunk_key, exc_info)

    @log_unhandled
    def notify_dead_senders(self, dead_workers):
        """
        When some peer workers are dead, corresponding receivers will be cancelled
        :param dead_workers: endpoints of dead workers
        """
        dead_workers = set(dead_workers)
        exc_info = build_exc_info(WorkerDead)
        for session_chunk_key in self._data_writers.keys():
            if self._data_meta_cache[session_chunk_key].source_address in dead_workers:
                self.ref().cancel_receive(*session_chunk_key, **dict(exc_info=exc_info, _tell=True))

    @log_unhandled
    def handle_receive_timeout(self, session_id, chunk_key):
        if not self._is_receive_running(session_id, chunk_key):
            # if transfer already finishes, no needs to report timeout
            return
        logger.debug('Transfer for %s timed out, cancelling.', chunk_key)
        self._stop_transfer_with_exc(session_id, chunk_key, build_exc_info(TimeoutError))

    def _stop_transfer_with_exc(self, session_id, chunk_key, exc):
        if not isinstance(exc[1], ExecutionInterrupted):
            logger.exception('Error occurred in receiving %s. Cancelling transfer.',
                             chunk_key, exc_info=exc)

        session_chunk_key = (session_id, chunk_key)

        # stop and close data writer
        try:
            self._data_writers[session_chunk_key].close()
            del self._data_writers[session_chunk_key]
            # transfer is not finished yet, we need to clean up unfinished stuffs
            has_write_failure = True
        except KeyError:
            # transfer finished and writer cleaned, no need to clean up
            has_write_failure = False

        data_meta = self._data_meta_cache[session_chunk_key]  # type: ReceiverDataMeta
        # clean up unfinished transfers
        if data_meta.write_shared:
            try:
                self._chunk_store.seal(session_id, chunk_key)
            except KeyError:
                pass
            if has_write_failure:
                self._chunk_store.delete(session_id, chunk_key)
        else:
            src_dir = build_spill_file_name(chunk_key, writing=True)
            if has_write_failure and os.path.exists(src_dir):
                os.unlink(src_dir)

        data_meta.status = ReceiveStatus.ERROR
        self._invoke_finish_callbacks(session_id, chunk_key, *exc, **dict(_accept=False))

    def _invoke_finish_callbacks(self, session_id, chunk_key, *args, **kwargs):
        # invoke registered callbacks for chunk
        session_chunk_key = (session_id, chunk_key)
        data_meta = self._data_meta_cache[session_chunk_key]  # type: ReceiverDataMeta
        data_meta.callback_args = args
        data_meta.callback_kwargs = kwargs

        if data_meta.transfer_event_id is not None and self._events_ref is not None:
            self._events_ref.close_event(data_meta.transfer_event_id, _tell=True)

        for cb in self._finish_callbacks[session_chunk_key]:
            kwargs['_wait'] = False
            self.tell_promise(cb, *args, **kwargs)
        if session_chunk_key in self._finish_callbacks:
            del self._finish_callbacks[session_chunk_key]


class ResultSenderActor(WorkerActor):
    """
    Actor handling sending result to user client
    """
    def __init__(self):
        super(ResultSenderActor, self).__init__()
        self._serialize_pool = None

    def post_create(self):
        super(ResultSenderActor, self).post_create()
        self._serialize_pool = self.ctx.threadpool(1)

    def fetch_data(self, session_id, chunk_key, index_obj=None):
        buf = None
        try:
            compression_type = dataserializer.CompressType(options.worker.transfer_compression)
            if self._chunk_store.contains(session_id, chunk_key):
                if index_obj is None:
                    buf = self._chunk_store.get_buffer(session_id, chunk_key)
                    compressed = self._serialize_pool.submit(
                        dataserializer.dumps, buf, compression_type, raw=True).result()
                else:
                    value = self._chunk_store.get(session_id, chunk_key)
                    if isinstance(value, (pd.DataFrame, pd.Series)):
                        sliced_value = value.iloc[index_obj]
                    else:
                        sliced_value = value[index_obj]
                    compressed = self._serialize_pool.submit(
                        dataserializer.dumps, sliced_value, compression_type).result()
            else:
                file_name = build_spill_file_name(chunk_key)
                if not file_name:
                    raise SpillNotConfigured('Spill not configured')
                if index_obj is None:
                    with open(file_name, 'rb') as inf:
                        compressed = self._serialize_pool.submit(inf.read).result()
                else:
                    # TODO: may have potential memory issue
                    def get_data_slices_from_file(file_name, index_obj, compression_type):
                        value = read_spill_file(file_name)
                        if isinstance(value, (pd.DataFrame, pd.Series)):
                            sliced_value = value.iloc[index_obj]
                        else:
                            sliced_value = value[index_obj]
                        return dataserializer.dumps(sliced_value, compression_type)

                    compressed = self._serialize_pool.submit(
                        get_data_slices_from_file, file_name, index_obj, compression_type).result()

        finally:
            del buf
        return compressed


def put_remote_chunk(session_id, chunk_key, data, receiver_ref):
    """
    Put a chunk to target machine using given receiver_ref
    """
    from .dataio import ArrowBufferIO
    buf = pyarrow.serialize(data).to_buffer()
    receiver_ref.create_data_writer(session_id, chunk_key, buf.size, None,
                                    ensure_cached=False, use_promise=False)

    block_size = options.worker.transfer_block_size

    try:
        reader = ArrowBufferIO(buf, 'r', block_size=block_size)
        checksum = 0
        futures = []
        while True:
            next_chunk = reader.read(block_size)
            is_last = not next_chunk or len(next_chunk) < block_size
            [f.result() for f in futures]
            checksum = zlib.crc32(next_chunk, checksum)
            futures.append(receiver_ref.receive_data_part(
                session_id, chunk_key, next_chunk, checksum, is_last=is_last, _wait=False))
            if is_last:
                [f.result() for f in futures]
                break
    except:
        receiver_ref.cancel_receive(session_id, chunk_key)
        raise
    finally:
        if reader:
            reader.close()
        del reader
