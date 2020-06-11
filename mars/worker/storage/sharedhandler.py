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

from ... import promise
from ...config import options
from ...errors import StorageFull, StorageDataExists
from ...serialize import dataserializer
from ..dataio import ArrowBufferIO
from .core import StorageHandler, BytesStorageMixin, ObjectStorageMixin, \
    SpillableStorageMixin, BytesStorageIO, DataStorageDevice, wrap_promised, \
    register_storage_handler_cls


class SharedStorageIO(BytesStorageIO):
    storage_type = DataStorageDevice.SHARED_MEMORY
    filename = None

    def __init__(self, session_id, data_key, mode='w', shared_store=None,
                 nbytes=None, packed=False, compress=None, auto_register=True,
                 pin_token=None, handler=None):
        from .objectholder import SharedHolderActor

        super().__init__(session_id, data_key, mode=mode, handler=handler)
        self._shared_buf = None
        self._shared_store = shared_store
        self._offset = 0
        self._nbytes = nbytes
        self._holder_ref = self._storage_ctx.actor_ctx.actor_ref(SharedHolderActor.default_uid())
        self._compress = compress or dataserializer.CompressType.NONE
        self._packed = packed
        self._auto_register = auto_register
        self._pin_token = pin_token

        block_size = options.worker.copy_block_size

        if self.is_writable:
            self._shared_buf = shared_store.create(session_id, data_key, nbytes)
            if packed:
                self._buf = ArrowBufferIO(self._shared_buf, 'w', block_size=block_size)
            else:
                import pyarrow
                self._buf = pyarrow.FixedSizeBufferWriter(self._shared_buf)
                self._buf.set_memcopy_threads(6)
        elif self.is_readable:
            self._shared_buf = shared_store.get_buffer(session_id, data_key)
            if packed:
                self._buf = ArrowBufferIO(
                    self._shared_buf, 'r', compress_out=compress, block_size=block_size)
            else:
                self._mv = memoryview(self._shared_buf)
                self._nbytes = len(self._shared_buf)
        else:
            raise NotImplementedError

    def __del__(self):
        self._mv = None
        self._buf = self._shared_buf = None

    @property
    def nbytes(self):
        return self._nbytes

    def get_shared_buffer(self):
        return self._shared_buf

    def read(self, size=-1):
        if self._packed:
            return self._buf.read(size)
        else:
            if size < 0:
                size = self._nbytes
            right_pos = min(self._nbytes, self._offset + size)
            ret = self._mv[self._offset:right_pos]
            self._offset = right_pos
            return ret

    def write(self, d):
        return self._buf.write(d)

    def close(self, finished=True):
        if self._closed:
            return

        if self.is_writable and self._shared_buf is not None:
            self._shared_store.seal(self._session_id, self._data_key)
            if finished:
                if self._auto_register:
                    self._holder_ref.put_objects_by_keys(
                        self._session_id, [self._data_key], pin_token=self._pin_token)
            else:
                self._shared_store.delete(self._session_id, self._data_key)

        self._mv = None
        self._buf = self._shared_buf = None
        super().close(finished=finished)


class SharedStorageHandler(StorageHandler, BytesStorageMixin, ObjectStorageMixin,
                           SpillableStorageMixin):
    storage_type = DataStorageDevice.SHARED_MEMORY

    def __init__(self, storage_ctx, proc_id=None):
        from .objectholder import SharedHolderActor

        StorageHandler.__init__(self, storage_ctx, proc_id=proc_id)
        self._shared_store = storage_ctx.shared_store
        self._holder_ref = storage_ctx.promise_ref(SharedHolderActor.default_uid())

    @wrap_promised
    def create_bytes_reader(self, session_id, data_key, packed=False, packed_compression=None,
                            _promise=False):
        return SharedStorageIO(session_id, data_key, 'r', self._shared_store, packed=packed,
                               compress=packed_compression, handler=self)

    @wrap_promised
    def create_bytes_writer(self, session_id, data_key, total_bytes, packed=False,
                            packed_compression=None, auto_register=True, pin_token=None,
                            _promise=False):
        return SharedStorageIO(session_id, data_key, 'w', self._shared_store,
                               nbytes=total_bytes, packed=packed, auto_register=auto_register,
                               handler=self, pin_token=pin_token)

    @wrap_promised
    def get_objects(self, session_id, data_keys, serialize=False, _promise=False):
        if serialize:
            return [self._shared_store.get_buffer(session_id, k) for k in data_keys]
        else:
            return [self._shared_store.get(session_id, k) for k in data_keys]

    @wrap_promised
    def put_objects(self, session_id, data_keys, objs, sizes=None, serialize=False,
                    pin_token=None, _promise=False):
        succ_keys, succ_shapes = [], []
        affected_keys = []
        request_size, capacity = 0, 0

        objs = [self._deserial(obj) if serialize else obj for obj in objs]
        obj_refs = []
        obj = None
        try:
            for key, obj in zip(data_keys, objs):
                shape = getattr(obj, 'shape', None)
                try:
                    obj_refs.append(self._shared_store.put(session_id, key, obj))
                    succ_keys.append(key)
                    succ_shapes.append(shape)
                except StorageFull as ex:
                    affected_keys.extend(ex.affected_keys)
                    request_size += ex.request_size
                    capacity = ex.capacity
                except StorageDataExists:
                    if self.location in self.storage_ctx.get_data_locations(session_id, [key])[0]:
                        succ_keys.append(key)
                        succ_shapes.append(shape)
                    else:
                        raise
            self._holder_ref.put_objects_by_keys(session_id, succ_keys, shapes=succ_shapes, pin_token=pin_token)
            if affected_keys:
                raise StorageFull(request_size=request_size, capacity=capacity,
                                  affected_keys=affected_keys)
        finally:
            del obj
            objs[:] = []
            obj_refs[:] = []

    def load_from_bytes_io(self, session_id, data_keys, src_handler, pin_token=None):
        is_success = True
        shared_bufs = []
        failed_keys = set()
        storage_full_sizes = [0, 0]

        def _handle_writer_create_fail(key, reader, exc_info):
            nonlocal is_success
            reader.close()
            failed_keys.add(key)
            exc = exc_info[1]
            if isinstance(exc, StorageFull):
                storage_full_sizes[0] += exc.request_size
                storage_full_sizes[1] = exc.capacity
            else:
                is_success = False
                shared_bufs[:] = []
                self.pass_on_exc(reader.close, exc_info)

        def _on_file_close(key, _reader, writer, finished):
            if finished:
                if is_success:
                    shared_bufs.append(writer.get_shared_buffer())
            else:
                failed_keys.add(key)

        def _finalize_load(*exc_info):
            try:
                success_keys = [k for k in data_keys if k not in failed_keys]
                if success_keys:
                    self._holder_ref.put_objects_by_keys(session_id, success_keys, pin_token=pin_token)
                if exc_info:
                    raise exc_info[1].with_traceback(exc_info[2]) from None
                if failed_keys:
                    raise StorageFull(request_size=storage_full_sizes[0], capacity=storage_full_sizes[1],
                                      affected_keys=list(failed_keys))
            finally:
                shared_bufs[:] = []

        def _load_single_key(k):
            on_close = functools.partial(_on_file_close, k)
            handle_worker_create_fail = functools.partial(_handle_writer_create_fail, k)
            return src_handler.create_bytes_reader(session_id, k, _promise=True) \
                .then(lambda reader: self.create_bytes_writer(
                    session_id, k, reader.nbytes, auto_register=False, _promise=True)
                        .then(lambda writer: self._copy_bytes_data(reader, writer, on_close),
                              lambda *exc: handle_worker_create_fail(reader, exc)))

        def _fallback(*_):
            return promise.all_(_load_single_key(k) for k in data_keys) \
                .then(lambda *_: _finalize_load(), _finalize_load)

        return self.transfer_in_runner(session_id, data_keys, src_handler, _fallback)

    def load_from_object_io(self, session_id, data_keys, src_handler, pin_token=None):
        def _fallback(*_):
            ser_needed = src_handler.storage_type not in \
                         (DataStorageDevice.SHARED_MEMORY, DataStorageDevice.PROC_MEMORY)
            return self._batch_load_objects(
                session_id, data_keys,
                lambda k: src_handler.get_objects(session_id, k, serialize=ser_needed, _promise=True),
                serialize=ser_needed, pin_token=pin_token, batch_get=True)

        return self.transfer_in_runner(session_id, data_keys, src_handler, _fallback)

    def delete(self, session_id, data_keys, _tell=False):
        self._holder_ref.delete_objects(session_id, data_keys, _tell=_tell)
        self.unregister_data(session_id, data_keys, _tell=_tell)

    def spill_size(self, size, multiplier=1):
        return self._holder_ref.spill_size(size, multiplier, _promise=True)

    def lift_data_keys(self, session_id, data_keys):
        self._holder_ref.lift_data_keys(session_id, data_keys, _tell=True)

    def pin_data_keys(self, session_id, data_keys, token):
        return self._holder_ref.pin_data_keys(session_id, data_keys, token)

    def unpin_data_keys(self, session_id, data_keys, token, _tell=False):
        return self._holder_ref.unpin_data_keys(session_id, data_keys, token, _tell=_tell)


register_storage_handler_cls(DataStorageDevice.SHARED_MEMORY, SharedStorageHandler)
