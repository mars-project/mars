# Copyright 1999-2019 Alibaba Group Holding Ltd.
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

import os
import shutil
import sys
import time

from ...config import options
from ...serialize import dataserializer
from ...errors import SpillNotConfigured
from ...utils import mod_hash
from ..dataio import FileBufferIO
from ..status import StatusActor
from ..utils import parse_spill_dirs
from .core import StorageHandler, BytesStorageMixin, BytesStorageIO, \
    DataStorageDevice, wrap_promised, register_storage_handler_cls


def _build_file_name(session_id, data_key, dirs=None, writing=False):
    """
    Build spill file name from chunk key. Path is selected given hash of the chunk key
    :param data_key: chunk key
    """
    if isinstance(data_key, tuple):
        data_key = '@'.join(data_key)
    dirs = dirs or options.worker.spill_directory
    if not dirs:  # pragma: no cover
        raise SpillNotConfigured('Spill not configured')
    if not isinstance(dirs, list):
        dirs = parse_spill_dirs(dirs)
    spill_dir = os.path.join(dirs[mod_hash((session_id, data_key), len(dirs))], str(session_id))
    if writing:
        spill_dir = os.path.join(spill_dir, 'writing')
    if not os.path.exists(spill_dir):
        try:
            os.makedirs(spill_dir)
        except OSError:  # pragma: no cover
            if not os.path.exists(spill_dir):
                raise
    return os.path.join(spill_dir, data_key)


class DiskIO(BytesStorageIO):
    storage_type = DataStorageDevice.DISK

    def __init__(self, session_id, data_key, mode='r', nbytes=None, compress=None,
                 packed=False, storage_ctx=None, status_ref=None):
        super(DiskIO, self).__init__(session_id, data_key, mode=mode,
                                     storage_ctx=storage_ctx)
        block_size = options.worker.copy_block_size

        self._raw_buf = self._buf = None
        self._nbytes = nbytes
        self._compress = compress or dataserializer.CompressType.NONE
        self._total_time = 0
        self._status_ref = status_ref

        filename = self._dest_filename = self._filename = _build_file_name(session_id, data_key)
        if self.is_writable:
            if os.path.exists(self._dest_filename):
                exist_devs = self._storage_ctx.manager_ref.get_data_locations(session_id, data_key) or ()
                if (0, DataStorageDevice.DISK) in exist_devs:
                    raise IOError('File for data (%s, %s) already exists.' % (session_id, data_key))

            filename = self._filename = _build_file_name(session_id, data_key, writing=True)
            buf = self._raw_buf = open(filename, 'wb')

            if packed:
                self._buf = FileBufferIO(
                    buf, 'w', compress_in=compress, block_size=block_size)
            else:
                dataserializer.write_file_header(buf, dataserializer.file_header(
                    dataserializer.SERIAL_VERSION, nbytes, compress
                ))
                self._buf = dataserializer.open_compression_file(buf, compress)
        elif self.is_readable:
            buf = self._raw_buf = open(filename, 'rb')

            header = dataserializer.read_file_header(buf)
            self._nbytes = header.nbytes
            self._offset = 0

            if packed:
                buf.seek(0, os.SEEK_SET)
                self._buf = FileBufferIO(
                    buf, 'r', compress_out=compress, block_size=block_size)
                self._total_bytes = os.path.getsize(filename)
            else:
                compress = self._compress = header.compress
                self._buf = dataserializer.open_decompression_file(buf, compress)
                self._total_bytes = self._nbytes
        else:  # pragma: no cover
            raise NotImplementedError('Mode %r not supported' % mode)

    @property
    def nbytes(self):
        return self._nbytes

    @property
    def filename(self):
        return self._dest_filename

    def read(self, size=-1):
        start = time.time()
        buf = self._buf.read(size)
        self._total_time += time.time() - start
        return buf

    if sys.version_info[0] < 3:  # pragma: no cover
        _read = read

        def read(self, size=-1):  # pylint: disable=E0102
            buf = self._read(size)
            return bytes(buf)

    def write(self, d):
        start = time.time()
        try:
            d.write_to(self._buf)
        except AttributeError:
            self._buf.write(d)
        self._total_time += time.time() - start

    def close(self, finished=True):
        if self._closed:
            return

        self._buf.close()
        if self._raw_buf is not self._buf:
            self._raw_buf.close()
        self._raw_buf = self._buf = None

        try:
            transfer_speed = self._nbytes * 1.0 / self._total_time
        except ZeroDivisionError:
            transfer_speed = None

        if self.is_writable:
            status_key = 'disk_write_speed'
            if finished:
                shutil.move(self._filename, self._dest_filename)
                self.register(self._nbytes)
            else:
                os.unlink(self._filename)
        else:
            status_key = 'disk_read_speed'

        if self._status_ref and transfer_speed is not None:
            self._status_ref.update_mean_stats(status_key, transfer_speed, _tell=True, _wait=False)

        super(DiskIO, self).close(finished=finished)


class DiskHandler(StorageHandler, BytesStorageMixin):
    storage_type = DataStorageDevice.DISK

    def __init__(self, storage_ctx):
        super(DiskHandler, self).__init__(storage_ctx)
        self._compress = dataserializer.CompressType(options.worker.disk_compression)
        self._status_ref = self._storage_ctx.actor_ref(StatusActor.default_uid())
        if not self._storage_ctx.has_actor(self._status_ref):
            self._status_ref = None

    @wrap_promised
    def create_bytes_reader(self, session_id, data_key, packed=False, packed_compression=None,
                            _promise=False):
        return DiskIO(session_id, data_key, 'r', packed=packed, compress=packed_compression,
                      storage_ctx=self._storage_ctx, status_ref=self._status_ref)

    @wrap_promised
    def create_bytes_writer(self, session_id, data_key, total_bytes, packed=False,
                            packed_compression=None, _promise=False):
        return DiskIO(session_id, data_key, 'w', total_bytes, compress=self._compress,
                      packed=packed, storage_ctx=self._storage_ctx, status_ref=self._status_ref)

    def load_from_bytes_io(self, session_id, data_key, src_handler):
        runner_promise = self.transfer_in_global_runner(session_id, data_key, src_handler)
        if runner_promise:
            return runner_promise
        return src_handler.create_bytes_reader(session_id, data_key, _promise=True) \
            .then(lambda reader: self.create_bytes_writer(
                    session_id, data_key, reader.nbytes, _promise=True)
                  .then(lambda writer: self._copy_bytes_data(reader, writer),
                        lambda *exc: self.pass_on_exc(reader.close, exc)))

    @staticmethod
    def _get_serialized_data_size(serialized_obj):
        if hasattr(serialized_obj, 'total_bytes'):
            return serialized_obj.total_bytes
        else:
            return len(serialized_obj)

    def load_from_object_io(self, session_id, data_key, src_handler):
        runner_promise = self.transfer_in_global_runner(session_id, data_key, src_handler)
        if runner_promise:
            return runner_promise
        return src_handler.get_object(session_id, data_key, serialized=True, _promise=True) \
            .then(lambda obj_data: self.create_bytes_writer(
                    session_id, data_key, self._get_serialized_data_size(obj_data), _promise=True)
                  .then(lambda writer: self._copy_object_data(obj_data, writer)))

    def delete(self, session_id, data_key, _tell=False):
        file_name = _build_file_name(session_id, data_key)
        if sys.platform == 'win32':  # pragma: no cover
            CREATE_NO_WINDOW = 0x08000000
            self._actor_ctx.popen(['del', file_name], creationflags=CREATE_NO_WINDOW)
        else:
            self._actor_ctx.popen(['rm', '-f', file_name])
        self.unregister_data(session_id, data_key, _tell=_tell)


register_storage_handler_cls(DataStorageDevice.DISK, DiskHandler)
