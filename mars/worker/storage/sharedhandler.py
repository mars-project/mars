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

import pyarrow

from ...config import options
from ...serialize import dataserializer
from ..dataio import ArrowBufferIO
from .core import StorageHandler, BytesStorageMixin, ObjectStorageMixin, \
    SpillableStorageMixin, BytesStorageIO, DataStorageDevice, wrap_promised, \
    register_storage_handler_cls


class SharedStorageIO(BytesStorageIO):
    storage_type = DataStorageDevice.SHARED_MEMORY
    filename = None

    def __init__(self, session_id, data_key, mode='w', shared_store=None,
                 nbytes=None, packed=False, compress=None, handler=None):
        from .objectholder import SharedHolderActor

        super(SharedStorageIO, self).__init__(session_id, data_key, mode=mode,
                                              handler=handler)
        self._shared_buf = None
        self._shared_store = shared_store
        self._offset = 0
        self._nbytes = nbytes
        self._holder_ref = self._storage_ctx.actor_ctx.actor_ref(SharedHolderActor.default_uid())
        self._compress = compress or dataserializer.CompressType.NONE
        self._packed = packed

        block_size = options.worker.copy_block_size

        if self.is_writable:
            self._shared_buf = shared_store.create(session_id, data_key, nbytes)
            if packed:
                self._buf = ArrowBufferIO(self._shared_buf, 'w', block_size=block_size)
            else:
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

    @property
    def nbytes(self):
        return self._nbytes

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
                # make sure data is not spilled before registration
                pin_token = self._holder_ref.put_object_by_key(
                    self._session_id, self._data_key, pin=True)
                self.register(self._nbytes)
                self._holder_ref.unpin_data_keys(self._session_id, [self._data_key], pin_token, _tell=True)
            else:
                self._shared_store.delete(self._session_id, self._data_key)

        self._mv = None
        self._buf = self._shared_buf = None
        super(SharedStorageIO, self).close(finished=finished)


class SharedStorageHandler(StorageHandler, BytesStorageMixin, ObjectStorageMixin,
                           SpillableStorageMixin):
    storage_type = DataStorageDevice.SHARED_MEMORY

    def __init__(self, storage_ctx):
        from .objectholder import SharedHolderActor

        StorageHandler.__init__(self, storage_ctx)
        self._shared_store = storage_ctx.shared_store
        self._holder_ref = storage_ctx.promise_ref(SharedHolderActor.default_uid())

    @wrap_promised
    def create_bytes_reader(self, session_id, data_key, packed=False, packed_compression=None,
                            _promise=False):
        return SharedStorageIO(session_id, data_key, 'r', self._shared_store, packed=packed,
                               compress=packed_compression, handler=self)

    @wrap_promised
    def create_bytes_writer(self, session_id, data_key, total_bytes, packed=False,
                            packed_compression=None, _promise=False):
        return SharedStorageIO(session_id, data_key, 'w', self._shared_store,
                               nbytes=total_bytes, packed=packed, handler=self)

    @wrap_promised
    def get_object(self, session_id, data_key, serialized=False, _promise=False):
        if serialized:
            return self._shared_store.get_buffer(session_id, data_key)
        else:
            return self._shared_store.get(session_id, data_key)

    @wrap_promised
    def put_object(self, session_id, data_key, obj, serialized=False, _promise=False):
        obj = self._deserial(obj) if serialized else obj
        shape = getattr(obj, 'shape', None)

        buf = None
        try:
            buf = self._shared_store.put(session_id, data_key, obj)
            # make sure data is not spilled before registration
            pin_token = self._holder_ref.put_object_by_key(session_id, data_key, pin=True)
        finally:
            del obj, buf

        data_size = self._shared_store.get_actual_size(session_id, data_key)
        self.register_data(session_id, data_key, data_size, shape=shape)
        self._holder_ref.unpin_data_keys(session_id, [data_key], pin_token, _tell=True)

    def load_from_bytes_io(self, session_id, data_key, src_handler):
        def _fallback(*_):
            return src_handler.create_bytes_reader(session_id, data_key, _promise=True) \
                .then(lambda reader: self.create_bytes_writer(
                    session_id, data_key, reader.nbytes, _promise=True)
                      .then(lambda writer: self._copy_bytes_data(reader, writer),
                            lambda *exc: self.pass_on_exc(reader.close, exc)))

        return self.transfer_in_global_runner(session_id, data_key, src_handler, _fallback)

    def load_from_object_io(self, session_id, data_key, src_handler):
        def _load(obj):
            try:
                return self.put_object(session_id, data_key, obj, _promise=True)
            finally:
                del obj

        def _fallback(*_):
            return src_handler.get_object(session_id, data_key, _promise=True) \
                .then(_load)

        return self.transfer_in_global_runner(session_id, data_key, src_handler, _fallback)

    def delete(self, session_id, data_key, _tell=False):
        self._holder_ref.delete_object(session_id, data_key, _tell=_tell)
        self.unregister_data(session_id, data_key, _tell=_tell)

    def spill_size(self, size, multiplier=1):
        return self._holder_ref.spill_size(size, multiplier, _promise=True)

    def lift_data_key(self, session_id, data_key):
        self._holder_ref.lift_data_key(session_id, data_key, _tell=True)

    def pin_data_keys(self, session_id, data_keys, token):
        return self._holder_ref.pin_data_keys(session_id, data_keys, token)

    def unpin_data_keys(self, session_id, data_keys, token, _tell=False):
        return self._holder_ref.unpin_data_keys(session_id, data_keys, token, _tell=_tell)


register_storage_handler_cls(DataStorageDevice.SHARED_MEMORY, SharedStorageHandler)
