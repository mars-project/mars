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

from .. import promise
from ..config import options
from ..serialize import dataserializer
from ..utils import mod_hash, log_unhandled, calc_data_size
from ..errors import StoreFull, StoreKeyExists, SpillNotConfigured
from .utils import WorkerActor

logger = logging.getLogger(__name__)


def parse_spill_dirs(dir_str):
    """
    Parse paths from a:b to list while resolving asterisks in path
    """
    import glob
    final_dirs = []
    for pattern in dir_str.split(os.path.pathsep):
        sub_patterns = pattern.split(os.path.sep)
        if not sub_patterns:
            continue
        pos = 0
        while pos < len(sub_patterns) and '*' not in sub_patterns[pos]:
            pos += 1
        if pos == len(sub_patterns):
            final_dirs.append(pattern)
            continue
        left_pattern = os.path.sep.join(sub_patterns[:pos + 1])
        for match in glob.glob(left_pattern):
            final_dirs.append(os.path.sep.join([match] + sub_patterns[pos + 1:]))
    return sorted(final_dirs)


def build_spill_file_name(chunk_key, dirs=None, writing=False):
    """
    Build spill file name from chunk key. Path is selected given hash of the chunk key
    :param chunk_key: chunk key
    """
    dirs = dirs or options.worker.spill_directory
    if not dirs:
        return None
    if not isinstance(dirs, list):
        dirs = parse_spill_dirs(dirs)
    spill_dir = dirs[mod_hash(chunk_key, len(dirs))]
    if writing:
        spill_dir = os.path.join(spill_dir, 'writing')
    if not os.path.exists(spill_dir):
        os.makedirs(spill_dir)
    return os.path.join(spill_dir, chunk_key)


def read_spill_file(chunk_key):
    """
    Read spill file of chunk key via gevent thread pool input_pool
    :param chunk_key: chunk key
    :return: mars object
    """
    file_name = build_spill_file_name(chunk_key)
    if not file_name:
        raise SpillNotConfigured('Spill not configured')
    with open(file_name, 'rb') as file_obj:
        return dataserializer.load(file_obj)


def write_spill_file(chunk_key, data):
    """
    Write mars object into spill
    :param chunk_key: chunk key
    :param data: data of the chunk
    """
    src_file_name = build_spill_file_name(chunk_key, writing=True)
    dest_file_name = build_spill_file_name(chunk_key, writing=False)
    if not src_file_name:
        raise SpillNotConfigured('Spill not configured')
    if not os.path.exists(dest_file_name):
        with open(src_file_name, 'wb') as file_obj:
            dataserializer.dump(data, file_obj, dataserializer.COMPRESS_FLAG_LZ4)
        os.rename(src_file_name, dest_file_name)


def spill_exists(chunk_key):
    file_name = build_spill_file_name(chunk_key)
    if not file_name:
        return False
    return os.path.exists(file_name)


class SpillActor(WorkerActor):
    """
    Actor handling spill read and write in single disk partition
    """
    def __init__(self):
        super(SpillActor, self).__init__()
        if not isinstance(options.worker.spill_directory, list):
            options.worker.spill_directory = options.worker.spill_directory.split(os.path.pathsep)
        self._spill_dirs = options.worker.spill_directory
        self._chunk_holder_ref = None
        self._mem_quota_ref = None
        self._status_ref = None
        self._dispatch_ref = None

        self._input_pool = None
        self._compress_pool = None
        self._output_pool = None

    def post_create(self):
        from .chunkholder import ChunkHolderActor
        from .quota import MemQuotaActor
        from .status import StatusActor
        from .dispatcher import DispatchActor

        super(SpillActor, self).post_create()
        self._chunk_holder_ref = self.promise_ref(ChunkHolderActor.default_name())
        self._mem_quota_ref = self.promise_ref(MemQuotaActor.default_name())

        self._status_ref = self.ctx.actor_ref(StatusActor.default_name())
        if not self.ctx.has_actor(self._status_ref):
            self._status_ref = None

        self._dispatch_ref = self.promise_ref(DispatchActor.default_name())
        self._dispatch_ref.register_free_slot(self.uid, 'spill')

        self._input_pool = self.ctx.threadpool(1)
        self._compress_pool = self.ctx.threadpool(1)
        self._output_pool = self.ctx.threadpool(1)

    @promise.reject_on_exception
    @log_unhandled
    def spill(self, session_id, chunk_key, callback):
        """
        Spill chunk to disk
        :param session_id: session id
        :param chunk_key: chunk key
        :param callback: promise callback
        """
        file_name = build_spill_file_name(chunk_key)
        if not file_name:
            raise SpillNotConfigured('Spill not configured')
        if not os.path.exists(file_name):
            data = None
            try:
                data = self._chunk_store.get(session_id, chunk_key)

                logger.debug('Start spilling chunk %s in %s', chunk_key, self.uid)
                start_time = time.time()
                self._output_pool.submit(write_spill_file, chunk_key, data).result()

                if self._status_ref:
                    self._status_ref.update_mean_stats(
                        'disk_write_speed', calc_data_size(data) * 1.0 / (time.time() - start_time),
                        _tell=True, _wait=False)
            except KeyError:
                if self._chunk_holder_ref.is_stored(chunk_key):
                    raise
            finally:
                del data
        self.tell_promise(callback, file_name)

    @promise.reject_on_exception
    @log_unhandled
    def load(self, session_id, chunk_key, callback=None):
        """
        Load spilled chunk from disk into shared storage
        :param session_id: session id
        :param chunk_key: chunk key
        :param callback: promise callback
        """
        if self._chunk_store.contains(session_id, chunk_key):
            # if already loaded, just register
            ref = None
            try:
                ref = self._chunk_store.get(session_id, chunk_key)
                self._chunk_holder_ref.register_chunk(session_id, chunk_key)
                logger.debug('Chunk %s already loaded in plasma', chunk_key)
                self.tell_promise(callback)
                return
            except KeyError:
                pass
            finally:
                del ref

        logger.debug('Start loading chunk %s from spill', chunk_key)
        file_name = build_spill_file_name(chunk_key)
        if not file_name:
            raise SpillNotConfigured('Spill not configured')

        with open(file_name, 'rb') as inpf:
            data_size = dataserializer.peek_serialized_size(inpf)

        @log_unhandled
        def _handle_rejection(*exc):
            self.tell_promise(callback, *exc, **dict(_accept=False))

        @log_unhandled
        def _try_put_chunk(spill_times=1):
            buf = None
            sealed = False
            try:
                start_time = time.time()
                logger.debug('Creating data writer for chunk %s in plasma', chunk_key)
                buf = self._chunk_store.create(session_id, chunk_key, data_size)
                logger.debug('Successfully created data writer for chunk %s in plasma', chunk_key)

                with open(file_name, 'rb') as inpf,\
                        dataserializer.DecompressBufferWriter(buf) as writer:
                    block_size = options.worker.transfer_block_size
                    compress_future = None
                    while True:
                        read_future = self._input_pool.submit(inpf.read, block_size)
                        if compress_future is not None:
                            compress_future.result()
                        if not read_future.result():
                            break
                        compress_future = self._compress_pool.submit(writer.write, read_future.value)

                self._chunk_store.seal(session_id, chunk_key)
                sealed = True
                if self._status_ref:
                    self._status_ref.update_mean_stats(
                        'disk_read_speed', data_size * 1.0 / (time.time() - start_time),
                        _tell=True, _wait=False)

                self._chunk_holder_ref.register_chunk(session_id, chunk_key)
                logger.debug('Chunk %s loaded from spill to plasma', chunk_key)
                self.tell_promise(callback)
            except (KeyError, StoreKeyExists):
                if not self._chunk_holder_ref.is_stored(chunk_key):
                    raise
                logger.debug('Chunk %s already loaded into plasma', chunk_key)
                self.tell_promise(callback)
            except StoreFull:
                # failed to put into shared cache: spill and retry
                logger.debug('Failed to create data writer for chunk %s in plasma', chunk_key)
                self._chunk_holder_ref.spill_size(data_size, spill_times, _promise=True) \
                    .then(lambda *_: _try_put_chunk(min(spill_times + 1, 1024))) \
                    .catch(_handle_rejection)
            finally:
                if buf is not None and not sealed:
                    self._chunk_store.seal(session_id, chunk_key)
                del buf
        _try_put_chunk()

    @promise.reject_on_exception
    @log_unhandled
    def delete(self, chunk_key):
        """
        Delete chunk data in spill
        :param chunk_key: chunk key
        """
        file_name = build_spill_file_name(chunk_key)
        if not file_name:
            raise SpillNotConfigured('Spill not configured')
        if sys.platform == 'win32':
            CREATE_NO_WINDOW = 0x08000000
            self.ctx.popen(['del', file_name], creationflags=CREATE_NO_WINDOW)
        else:
            self.ctx.popen(['rm', '-f', file_name])
