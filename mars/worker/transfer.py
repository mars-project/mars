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

import logging
import os
import sys
import time
import zlib
from collections import defaultdict, OrderedDict

from .. import promise
from ..compat import six, Enum
from ..config import options
from ..serialize import dataserializer
from ..errors import *
from ..utils import log_unhandled
from .dataio import FileBufferIO, ArrowBufferIO
from .spill import build_spill_file_name
from .utils import WorkerActor

logger = logging.getLogger(__name__)


class ReceiveStatus(Enum):
    NOT_STARTED = 0
    RECEIVING = 1
    RECEIVED = 2
    ERROR = 2


class SenderActor(WorkerActor):
    """
    Actor handling sending data to ReceiverActors in other workers
    """
    def __init__(self):
        super(SenderActor, self).__init__()
        self._dispatch_ref = None
        self._mem_quota_ref = None

        self._serialize_pool = None

    def post_create(self):
        from .dispatcher import DispatchActor
        from .quota import MemQuotaActor

        super(SenderActor, self).post_create()
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
        if self._chunk_store.contains(session_id, chunk_key):
            try:
                nbytes = self._chunk_store.get_actual_size(session_id, chunk_key)
            except KeyError:
                pass
        if nbytes is None:
            file_name = build_spill_file_name(chunk_key)
            if not file_name:
                raise SpillNotConfigured('Spill not configured')
            if not os.path.exists(file_name):
                raise DependencyMissing('Dependency %s not met on sending.' % chunk_key)
            with open(file_name, 'rb') as inf:
                nbytes = self._serialize_pool.submit(dataserializer.peek_serialized_size, inf).result()
        return nbytes

    @promise.reject_on_exception
    @log_unhandled
    def send_data(self, session_id, chunk_key, target_endpoints, ensure_cached=True,
                  compression=None, timeout=0, callback=None):
        """
        Send data to other workers
        :param session_id: session id
        :param chunk_key: chunk to be sent
        :param target_endpoints: endpoints to receive this chunk
        :param ensure_cached: if True, make sure the data is in the shared storage of the target worker
        :param compression: compression type when transfer in network
        :param timeout: timeout of data sending
        :param callback: promise callback
        """
        from .dispatcher import DispatchActor
        compression = compression or dataserializer.CompressType(options.worker.transfer_compression)

        remote_receiver_refs = []
        already_started = set()
        finish_promises = []
        if isinstance(target_endpoints, six.string_types):
            target_endpoints = [target_endpoints]

        try:
            logger.debug('Begin sending data %s into endpoints %s', chunk_key, target_endpoints)
            # collect receiver actors and quota actors in remote workers
            for ep in target_endpoints:
                dispatch_ref = self.promise_ref(DispatchActor.default_uid(), address=ep)
                uid = dispatch_ref.get_hash_slot('receiver', chunk_key)

                receiver_ref = self.promise_ref(uid, address=ep)
                remote_status = receiver_ref.check_status(session_id, chunk_key)
                if remote_status == ReceiveStatus.RECEIVED:
                    # data already been sent, no need to transfer any more
                    continue
                elif remote_status == ReceiveStatus.RECEIVING:
                    # data under transfer, we only need to listen to the transfer progress
                    finish_promises.append(
                        receiver_ref.register_finish_callback(session_id, chunk_key, _promise=True))
                    continue
                remote_receiver_refs.append(receiver_ref)

            if not remote_receiver_refs and not finish_promises:
                # no transfer needed, we exit and release slot resource
                logger.debug('No data needed to transfer for chunk %s, invoke callback directly', chunk_key)
                self._dispatch_ref.register_free_slot(self.uid, 'sender', _tell=True)
                self.tell_promise(callback, None)
                return
            data_size = self._read_data_size(session_id, chunk_key)
        except:
            self._dispatch_ref.register_free_slot(self.uid, 'sender', _tell=True)
            raise

        @log_unhandled
        def _handle_creation(address, status):
            # filter out endpoints already transferred or already started transfer
            if status in (ReceiveStatus.RECEIVING, ReceiveStatus.RECEIVED):
                already_started.add(address)

        @log_unhandled
        def _compress_and_send():
            # start compress and send data into targets
            logger.debug('Data writer for chunk %s allocated at targets, start transmission', chunk_key)
            block_size = options.worker.transfer_block_size
            reader = None

            # filter out endpoints we need to send to
            active_receiver_refs = [ref for ref in remote_receiver_refs
                                    if ref.address not in already_started]
            try:
                if not active_receiver_refs:
                    self._dispatch_ref.register_free_slot(self.uid, 'sender', _tell=True)
                    return

                # try load data from plasma store
                if self._chunk_store.contains(session_id, chunk_key):
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
                while True:
                    # read a data part from reader we defined above
                    next_chunk = self._serialize_pool.submit(reader.read, block_size).result()
                    # make sure all previous transfers finished
                    [f.result() for f in futures]
                    if not next_chunk:
                        # no further data to read, we close and finish the transfer
                        reader.close()
                        for ref in active_receiver_refs:
                            ref.finish_receive(session_id, chunk_key, checksum, _tell=True)
                        break
                    checksum = zlib.crc32(next_chunk, checksum)
                    futures = []
                    for ref in active_receiver_refs:
                        # we perform async transfer and wait after next part is loaded and compressed
                        futures.append(ref.receive_data_part(
                            session_id, chunk_key, next_chunk, checksum, _wait=False))
            except:
                for ref in active_receiver_refs:
                    ref.cancel_receive(session_id, chunk_key)
                raise
            finally:
                if reader:
                    reader.close()
                del reader

        @log_unhandled
        def _finalize(*_):
            self._dispatch_ref.register_free_slot(self.uid, 'sender', _tell=True)
            self.tell_promise(callback, data_size)

        @log_unhandled
        def _handle_rejection(*exc):
            for ref in remote_receiver_refs:
                ref.cancel_receive(session_id, chunk_key, _tell=True)
            self._dispatch_ref.register_free_slot(self.uid, 'sender', _tell=True)
            self.tell_promise(callback, *exc, **dict(_accept=False))

        promises = []
        for ref in remote_receiver_refs:
            # register transfer actions
            promises.append(
                ref.create_data_writer(
                    session_id, chunk_key, data_size, self.ref(), ensure_cached=ensure_cached,
                    _timeout=timeout, _promise=True
                ).then(_handle_creation)
            )
            # register finish listener
            finish_promises.append(ref.register_finish_callback(session_id, chunk_key, _promise=True))

        if promises:
            promise.all_(promises).then(_compress_and_send).catch(_handle_rejection)
        else:
            # nothing to send, the slot can be released
            self._dispatch_ref.register_free_slot(self.uid, 'sender', _tell=True)
        # wait for all finish listeners returned
        promise.all_(finish_promises).then(_finalize).catch(_handle_rejection)


class ReceiverActor(WorkerActor):
    """
    Actor handling receiving data from a SenderActor
    """
    def __init__(self):
        super(ReceiverActor, self).__init__()
        self._chunk_holder_ref = None
        self._mem_quota_ref = None
        self._dispatch_ref = None
        self._status_ref = None

        self._finish_callbacks = defaultdict(list)
        self._data_writers = dict()
        self._data_meta_cache = OrderedDict()

        self._serialize_pool = None

    def post_create(self):
        from .chunkholder import ChunkHolderActor
        from .quota import MemQuotaActor
        from .status import StatusActor
        from .dispatcher import DispatchActor

        super(ReceiverActor, self).post_create()
        self._chunk_holder_ref = self.promise_ref(ChunkHolderActor.default_uid())
        self._mem_quota_ref = self.promise_ref(MemQuotaActor.default_uid())

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

        if self._chunk_store.contains(session_id, chunk_key):
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
        data_meta = self._data_meta_cache[session_chunk_key]
        if data_meta['status'] in (ReceiveStatus.RECEIVED, ReceiveStatus.ERROR):
            # invoke callback directly when transfer finishes
            self.tell_promise(callback, *data_meta['callback_args'], **data_meta['callback_kwargs'])
        else:
            self._finish_callbacks[session_chunk_key].append(callback)

    @promise.reject_on_exception
    @log_unhandled
    def create_data_writer(self, session_id, chunk_key, data_size, sender_ref,
                           ensure_cached=True, callback=None):
        """
        Create a data writer for subsequent data transfer. The writer can either work on
        shared storage or spill.
        :param session_id: session id
        :param chunk_key: chunk key
        :param data_size: uncompressed data size
        :param sender_ref: ActorRef of SenderActor
        :param ensure_cached: if True, the data should be stored in shared memory, otherwise spill is acceptable
        :param callback: promise callback
        """
        logger.debug('Begin creating transmission data writer for chunk %s from %s',
                     chunk_key, sender_ref.address)
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
                logger.debug('Chunk %s already started received', chunk_key)
                if callback:
                    self.tell_promise(callback, self.address, ReceiveStatus.RECEIVED)
                self._invoke_finish_callbacks(session_id, chunk_key)
                return self.address, ReceiveStatus.RECEIVED
            else:
                del self._data_meta_cache[session_chunk_key]

        self._data_meta_cache[session_chunk_key] = dict(
            transfer_start_time=time.time(),
            chunk_size=data_size,
            write_shared=True,
            checksum=0,
            status=ReceiveStatus.RECEIVING,
            callback_args=(),
            callback_kwargs={},
        )

        @log_unhandled
        def _handle_reject(*exc):
            logger.debug('Rejecting %s from putting into plasma.', chunk_key)
            self._stop_transfer_with_exc(session_id, chunk_key, exc)
            if callback:
                self.tell_promise(callback, *exc, **dict(_accept=False))

        @log_unhandled
        def _create_writer(spill_times=1):
            # actual create data writer
            block_size = options.worker.transfer_block_size
            disk_compression = dataserializer.CompressType(options.worker.disk_compression)
            buf = None
            try:
                # attempt to create data chunk on shared store
                buf = self._chunk_store.create(session_id, chunk_key, data_size)
                self._data_meta_cache[session_chunk_key]['transfer_start_time'] = time.time()
                logger.debug('Successfully created data writer with %s bytes in plasma for chunk %s',
                             data_size, chunk_key)
                # create a writer for the chunk
                self._data_writers[session_chunk_key] = ArrowBufferIO(
                    buf, 'w', block_size=block_size)
                if callback:
                    self.tell_promise(callback, self.address, None)
            except (KeyError, StoreKeyExists):
                if self.check_status(session_id, chunk_key) != ReceiveStatus.RECEIVED:
                    raise
                # data already registered
                if callback:
                    logger.debug('Chunk %s already registered', chunk_key)
                    self.tell_promise(callback, self.address, ReceiveStatus.RECEIVED)
                    self._invoke_finish_callbacks(session_id, chunk_key)
            except StoreFull:
                # no space left in the shared store
                if ensure_cached:
                    # spill and try again
                    self._chunk_holder_ref.spill_size(data_size, spill_times, _promise=True) \
                        .then(lambda *_: _create_writer(min(spill_times + 1, 1024))) \
                        .catch(_handle_reject)
                else:
                    # create a writer for spill
                    logger.debug('Writing data %s directly into spill.', chunk_key)
                    self._data_meta_cache[session_chunk_key]['write_shared'] = False
                    self._chunk_holder_ref.spill_size(data_size, _tell=True)
                    spill_file_name = build_spill_file_name(chunk_key, writing=True)
                    try:
                        spill_file = FileBufferIO(
                            open(spill_file_name, 'wb'), 'w', compress_in=disk_compression,
                            block_size=block_size)
                        self._data_meta_cache[session_chunk_key]['transfer_start_time'] = time.time()
                        self._data_writers[session_chunk_key] = spill_file
                    except (KeyError, IOError):
                        if self.check_status(session_id, chunk_key) == ReceiveStatus.RECEIVED:
                            logger.debug('Chunk %s already stored', chunk_key)
                            if callback:
                                self.tell_promise(callback, self.address, ReceiveStatus.RECEIVED)
                            self._invoke_finish_callbacks(session_id, chunk_key)
                            return
                        raise ObjectNotInPlasma([chunk_key])
                    if callback:
                        self.tell_promise(callback, self.address, None)
            finally:
                del buf

        promise.Promise(done=True) \
            .then(lambda *_: _create_writer()) \
            .catch(_handle_reject)

    @log_unhandled
    def receive_data_part(self, session_id, chunk_key, data_part, checksum):
        """
        Receive data part from sender
        :param session_id: session id
        :param chunk_key: chunk key
        :param data_part: data to be written
        :param checksum: checksum up to now
        """
        session_chunk_key = (session_id, chunk_key)
        try:
            data_meta = self._data_meta_cache[session_chunk_key]

            # check if checksum matches
            local_checksum = zlib.crc32(data_part, data_meta['checksum'])
            if local_checksum != checksum:
                raise ChecksumMismatch
            data_meta['checksum'] = local_checksum

            # if error occurred, interrupts
            if data_meta['status'] == ReceiveStatus.ERROR:
                six.reraise(*self._data_meta_cache[session_chunk_key]['callback_args'])
                return
            self._serialize_pool.submit(self._data_writers[session_chunk_key].write, data_part).result()
        except:  # noqa: E722
            self._stop_transfer_with_exc(session_id, chunk_key, sys.exc_info())

    @log_unhandled
    def finish_receive(self, session_id, chunk_key, checksum):
        """
        Finish data receiving and seal data
        :param session_id: session id
        :param chunk_key: chunk key
        :param checksum: checksum of compressed data
        """
        try:
            logger.debug('Finishing transfer for data %s.', chunk_key)
            session_chunk_key = (session_id, chunk_key)

            data_meta = self._data_meta_cache[session_chunk_key]
            # if checksum mismatch, raises
            if checksum != data_meta['checksum']:
                raise ChecksumMismatch

            # update transfer speed stats
            if self._status_ref:
                time_delta = time.time() - data_meta['transfer_start_time']
                self._status_ref.update_mean_stats(
                    'net_transfer_speed', data_meta['chunk_size'] * 1.0 / time_delta,
                    _tell=True, _wait=False)

            if data_meta['write_shared']:
                # seal data on shared store
                self._chunk_store.seal(session_id, chunk_key)
                self._chunk_holder_ref.register_chunk(session_id, chunk_key)
            else:
                # move spill data to 'ready' place
                src_dir = build_spill_file_name(chunk_key, writing=True)
                dest_dir = build_spill_file_name(chunk_key, writing=False)
                os.rename(src_dir, dest_dir)

            self.get_meta_client().set_chunk_meta(
                session_id, chunk_key, size=data_meta['chunk_size'], workers=(self.address,))

            self._data_writers[session_chunk_key].close()
            del self._data_writers[session_chunk_key]

            data_meta['status'] = ReceiveStatus.RECEIVED
            self._invoke_finish_callbacks(session_id, chunk_key)
        except:  # noqa: E722
            self._stop_transfer_with_exc(session_id, chunk_key, sys.exc_info())

    @log_unhandled
    def cancel_receive(self, session_id, chunk_key):
        """
        Cancel data receive by returning an ExecutionInterrupted
        :param session_id: session id
        :param chunk_key: chunk key
        """
        logger.debug('Transfer for %s cancelled.', chunk_key)
        session_chunk_key = (session_id, chunk_key)
        if session_chunk_key not in self._data_meta_cache:
            return

        if self._data_meta_cache[session_chunk_key]['status'] in (ReceiveStatus.ERROR, ReceiveStatus.RECEIVED):
            # already terminated, we do nothing
            return

        try:
            raise ExecutionInterrupted
        except ExecutionInterrupted:
            self._stop_transfer_with_exc(session_id, chunk_key, sys.exc_info())

    def _stop_transfer_with_exc(self, session_id, chunk_key, exc):
        try:
            six.reraise(*exc)
        except ExecutionInterrupted:
            pass
        except:
            logger.exception('Exception occurred in transferring %s. Cancelling transfer.', chunk_key)

        session_chunk_key = (session_id, chunk_key)

        # stop and close data writer
        if session_chunk_key in self._data_writers:
            self._data_writers[session_chunk_key].close()
            del self._data_writers[session_chunk_key]

        data_meta = self._data_meta_cache[session_chunk_key]
        # clean up unfinished transfers
        if data_meta['write_shared']:
            try:
                self._chunk_store.seal(session_id, chunk_key)
            except KeyError:
                pass
            self._chunk_store.delete(session_id, chunk_key)
        else:
            src_dir = build_spill_file_name(chunk_key, writing=True)
            if os.path.exists(src_dir):
                os.unlink(src_dir)

        self._data_meta_cache['status'] = ReceiveStatus.ERROR
        self._invoke_finish_callbacks(session_id, chunk_key, *exc, **dict(_accept=False))

    def _invoke_finish_callbacks(self, session_id, chunk_key, *args, **kwargs):
        # invoke registered callbacks for chunk
        session_chunk_key = (session_id, chunk_key)
        data_meta = self._data_meta_cache[session_chunk_key]
        data_meta['callback_args'] = args
        data_meta['callback_kwargs'] = kwargs

        for cb in self._finish_callbacks[session_chunk_key]:
            self.tell_promise(cb, *args, **kwargs)
        if session_chunk_key in self._finish_callbacks:
            del self._finish_callbacks[session_chunk_key]

        # remove outdated metadata
        clean_keys = []
        last_finish_time = time.time() - options.worker.callback_preserve_time
        for k in self._data_meta_cache:
            if self._data_meta_cache[k]['transfer_start_time'] < last_finish_time:
                clean_keys.append(k)
            else:
                break
        for k in clean_keys:
            del self._data_meta_cache[k]


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

    def fetch_data(self, session_id, chunk_key):
        buf = None
        try:
            if self._chunk_store.contains(session_id, chunk_key):
                buf = self._chunk_store.get_buffer(session_id, chunk_key)
                compression_type = dataserializer.CompressType(options.worker.transfer_compression)
                compressed = self._serialize_pool.submit(
                    dataserializer.dumps, buf, compression_type, raw=True).result()
            else:
                file_name = build_spill_file_name(chunk_key)
                if not file_name:
                    raise SpillNotConfigured('Spill not configured')
                with open(file_name, 'rb') as inf:
                    compressed = self._serialize_pool.submit(inf.read).result()
        finally:
            del buf
        return compressed
