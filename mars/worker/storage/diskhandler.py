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
import os
import subprocess
import sys
import time

from ... import promise
from ...config import options
from ...serialize import dataserializer
from ...errors import SpillNotConfigured, StorageDataExists
from ...utils import mod_hash
from ..dataio import FileBufferIO
from ..events import EventsActor, EventCategory, EventLevel, ProcedureEventType
from ..status import StatusActor
from ..utils import parse_spill_dirs
from .core import StorageHandler, BytesStorageMixin, BytesStorageIO, \
    DataStorageDevice, wrap_promised, register_storage_handler_cls
from .diskmerge import DiskFileMergerActor


def _get_file_dir_id(session_id, data_key):
    dirs = options.worker.spill_directory
    return mod_hash((session_id, data_key), len(dirs))


def _build_file_name_by_key(session_id, data_key):
    """
    Build spill file name from chunk key. Path is selected given hash of the chunk key
    :param data_key: chunk key
    """
    if isinstance(data_key, tuple):
        data_key = '@'.join(data_key)
    dirs = options.worker.spill_directory
    spill_dir = os.path.join(dirs[_get_file_dir_id(session_id, data_key)], str(session_id))
    if not os.path.exists(spill_dir):
        try:
            os.makedirs(spill_dir)
        except OSError:  # pragma: no cover
            if not os.path.exists(spill_dir):
                raise
    return os.path.join(spill_dir, data_key)


class RestrictedFile:
    def __init__(self, buf, file_end=0):
        self._buf = buf
        self._file_end = file_end

    def read(self, size):
        if self._file_end is not None:
            max_size = self._file_end - self._buf.tell()
            if size == -1:
                size = max_size
            size = max(min(size, max_size), 0)
        return self._buf.read(size)

    def tell(self):
        return self._buf.tell()

    def close(self):
        self._buf.close()


class DiskIO(BytesStorageIO):
    storage_type = DataStorageDevice.DISK

    def __init__(self, session_id, data_key, mode='r', nbytes=None, compress=None,
                 packed=False, handler=None, merge_filename=None, offset=None, file_length=None):
        super().__init__(session_id, data_key, mode=mode, handler=handler)
        block_size = options.worker.copy_block_size
        dirs = options.worker.spill_directory = parse_spill_dirs(options.worker.spill_directory)
        if not dirs:  # pragma: no cover
            raise SpillNotConfigured

        self._raw_buf = self._buf = None
        self._nbytes = nbytes
        self._compress = compress or dataserializer.CompressType.NONE
        self._total_time = 0
        self._event_id = None
        self._offset = offset or 0
        self._file_end = file_length + offset if file_length is not None else None
        self._merge_file = False

        if merge_filename is not None:
            filename = self._filename = merge_filename
            self._merge_file = True
        else:
            filename = self._filename = _build_file_name_by_key(session_id, data_key)

        if self.is_writable:
            if merge_filename is None and os.path.exists(self._filename):
                exist_devs = self._storage_ctx.manager_ref.get_data_locations(session_id, [data_key])[0]
                if (0, DataStorageDevice.DISK) in exist_devs:
                    self._closed = True
                    raise StorageDataExists(f'File for data ({session_id}, {data_key}) already exists')
                else:
                    os.unlink(self._filename)

            buf = self._raw_buf = open(filename, 'ab')
            if packed:
                self._buf = FileBufferIO(
                    buf, 'w', compress_in=compress, block_size=block_size, managed=False)
            else:
                dataserializer.write_file_header(buf, dataserializer.file_header(
                    dataserializer.SerialType.ARROW, dataserializer.SERIAL_VERSION, nbytes, compress
                ))
                self._buf = dataserializer.open_compression_file(buf, compress)
        elif self.is_readable:
            buf = self._raw_buf = open(filename, 'rb')
            buf.seek(self._offset, os.SEEK_SET)

            header = dataserializer.read_file_header(buf)
            self._nbytes = header.nbytes

            if packed:
                self._buf = FileBufferIO(
                    buf, 'r', compress_out=compress, block_size=block_size, header=header,
                    file_end=self._file_end, managed=False)
                self._total_bytes = os.path.getsize(filename)
            else:
                compress = self._compress = header.compress
                if merge_filename:
                    self._raw_buf = RestrictedFile(self._raw_buf, self._file_end)
                self._buf = dataserializer.open_decompression_file(self._raw_buf, compress)
                self._total_bytes = self._nbytes
        else:  # pragma: no cover
            raise NotImplementedError(f'Mode {mode} not supported')
        if self._handler.events_ref:
            self._event_id = self._handler.events_ref.add_open_event(
                EventCategory.PROCEDURE, EventLevel.NORMAL, ProcedureEventType.DISK_IO,
                self._handler.storage_ctx.host_actor.uid
            )

    @property
    def nbytes(self):
        return self._nbytes

    @property
    def filename(self):
        return self._filename

    def get_io_pool(self, pool_name=None):
        file_dir_id = _get_file_dir_id(self._session_id, self._data_key)
        return super().get_io_pool(f'{pool_name or ""}__{file_dir_id}')

    def read(self, size=-1):
        start = time.time()
        buf = self._buf.read(size)
        self._total_time += time.time() - start
        return buf

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

        if self._raw_buf is not self._buf:
            self._buf.close()
        last_offset = self._raw_buf.tell()
        self._raw_buf.close()
        self._raw_buf = self._buf = None

        transfer_speed = None
        if finished and abs(self._total_time) > 1e-6:
            transfer_speed = self._nbytes * 1.0 / self._total_time

        if self.is_writable:
            status_key = 'disk_write_speed'
            if finished:
                self.register(self._nbytes)

                if self._merge_file:
                    self._handler.disk_merger_ref.release_file_writer(
                        self._session_id, self._data_key, self._filename, self._offset,
                        last_offset, _tell=True)
            else:
                os.unlink(self._filename)
        else:
            status_key = 'disk_read_speed'
            if self._merge_file:
                self._handler.disk_merger_ref.release_file_reader(
                    self._session_id, self._data_key, _tell=True)

        if self._handler.status_ref and transfer_speed is not None:
            self._handler.status_ref.update_mean_stats(status_key, transfer_speed, _tell=True, _wait=False)
        if self._event_id:
            self._handler.events_ref.close_event(self._event_id, _tell=True, _wait=False)

        super().close(finished=finished)


class DiskHandler(StorageHandler, BytesStorageMixin):
    storage_type = DataStorageDevice.DISK

    def __init__(self, storage_ctx, proc_id=None):
        super().__init__(storage_ctx, proc_id=proc_id)
        self._compress = dataserializer.CompressType(options.worker.disk_compression)

        self._status_ref = self._storage_ctx.actor_ref(StatusActor.default_uid())
        if not self._storage_ctx.has_actor(self._status_ref):
            self._status_ref = None

        self._events_ref = self._storage_ctx.actor_ref(EventsActor.default_uid())
        if not self._storage_ctx.has_actor(self._events_ref):
            self._events_ref = None

        self._disk_merger_ref = self._storage_ctx.actor_ref(DiskFileMergerActor.default_uid())
        if not self._storage_ctx.has_actor(self._disk_merger_ref):
            self._disk_merger_ref = None
        else:
            self._disk_merger_ref = self.host_actor.promise_ref(self._disk_merger_ref)

    @property
    def status_ref(self):
        return self._status_ref

    @property
    def events_ref(self):
        return self._events_ref

    @property
    def disk_merger_ref(self):
        return self._disk_merger_ref

    @wrap_promised
    def create_bytes_reader(self, session_id, data_key, packed=False, packed_compression=None,
                            with_merger_lock=False, _promise=False):
        merge_filename, offset, file_len = None, None, None
        if self._disk_merger_ref is not None:
            [data_meta] = self._disk_merger_ref.get_file_metas(session_id, [data_key])
            if data_meta is not None:
                merge_filename, offset = data_meta.filename, data_meta.start
                file_len = data_meta.end - data_meta.start
        if merge_filename is None:
            return DiskIO(
                session_id, data_key, 'r', packed=packed, compress=packed_compression,
                handler=self)
        else:
            return self._disk_merger_ref.await_file_reader(
                session_id, data_key, with_lock=with_merger_lock, _promise=True) \
                .then(lambda *_: DiskIO(
                    session_id, data_key, 'r', packed=packed, compress=packed_compression,
                    handler=self, merge_filename=merge_filename, offset=offset, file_length=file_len))

    @wrap_promised
    def create_bytes_writer(self, session_id, data_key, total_bytes, packed=False,
                            packed_compression=None, auto_register=True, pin_token=None,
                            with_merger_lock=False, _promise=False):
        if _promise and self._disk_merger_ref is not None \
                and total_bytes < options.worker.filemerger.max_accept_size:
            return self._disk_merger_ref.await_file_writer(
                session_id, with_lock=with_merger_lock, _promise=True) \
                .then(lambda filename, offset: DiskIO(
                    session_id, data_key, 'w', total_bytes, compress=self._compress,
                    packed=packed, handler=self, merge_filename=filename, offset=offset))
        else:
            return DiskIO(session_id, data_key, 'w', total_bytes, compress=self._compress,
                          packed=packed, handler=self)

    def load_from_bytes_io(self, session_id, data_keys, src_handler, pin_token=None):
        def _fallback(*_):
            return promise.all_(
                src_handler.create_bytes_reader(session_id, k, _promise=True)
                .then(lambda reader: self.create_bytes_writer(
                    session_id, k, reader.nbytes, with_merger_lock=True, _promise=True)
                      .then(lambda writer: self._copy_bytes_data(reader, writer),
                            lambda *exc: self.pass_on_exc(reader.close, exc)))
                for k in data_keys)

        return self.transfer_in_runner(session_id, data_keys, src_handler, _fallback)

    @staticmethod
    def _get_serialized_data_size(serialized_obj):
        try:
            if hasattr(serialized_obj, 'total_bytes'):
                return int(serialized_obj.total_bytes)
            else:
                return len(serialized_obj)
        finally:
            del serialized_obj

    def load_from_object_io(self, session_id, data_keys, src_handler, pin_token=None):
        data_dict = dict()

        def _load_single_data(key):
            data_size = self._get_serialized_data_size(data_dict[key])
            return self.create_bytes_writer(session_id, key, data_size, with_merger_lock=True, _promise=True) \
                .then(lambda writer: self._copy_object_data(data_dict.pop(key), writer),
                      lambda *exc: self.pass_on_exc(functools.partial(data_dict.pop, key), exc))

        def _load_all_data(objs):
            data_dict.update(zip(data_keys, objs))
            objs[:] = []
            return promise.all_(_load_single_data(k) for k in data_keys) \
                .catch(lambda *exc: self.pass_on_exc(data_dict.clear, exc))

        def _fallback(*_):
            return src_handler.get_objects(session_id, data_keys, serialize=True, _promise=True) \
                .then(_load_all_data, lambda *exc: self.pass_on_exc(data_dict.clear, exc))

        return self.transfer_in_runner(session_id, data_keys, src_handler, _fallback)

    def delete(self, session_id, data_keys, _tell=False):
        if self._disk_merger_ref is None:
            filenames, managed_keys = [], set()
        else:
            filenames, managed_keys = self._disk_merger_ref.delete_file_metas(session_id, data_keys)
            managed_keys = set(managed_keys)

        filenames.extend(_build_file_name_by_key(session_id, k) for k in data_keys
                         if k not in managed_keys)
        del_pool = self.get_io_pool()

        for idx in range(0, len(filenames), 10):
            cmd = ['rm', '-f'] if sys.platform != 'win32' else ['del']
            cmd += filenames[idx:idx + 10]
            kw = dict() if sys.platform != 'win32' else dict(creationflags=0x08000000)  # CREATE_NO_WINDOW
            del_pool.submit(subprocess.Popen, cmd, **kw)
        self.unregister_data(session_id, data_keys, _tell=_tell)


register_storage_handler_cls(DataStorageDevice.DISK, DiskHandler)
