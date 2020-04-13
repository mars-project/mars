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

import logging

from ...actors import FunctionActor
from ...config import options
from ...errors import StorageDataExists
from ...serialize import dataserializer
from ..dataio import ArrowComponentsIO
from .core import StorageHandler, ObjectStorageMixin, BytesStorageIO, \
    DataStorageDevice, wrap_promised, register_storage_handler_cls

try:
    import vineyard
except ImportError:
    vineyard = None
try:
    import pyarrow
except ImportError:
    pyarrow = None


logger = logging.getLogger(__name__)


class VineyardKeyMapActor(FunctionActor):
    @classmethod
    def default_uid(cls):
        return 'w:0:' + cls.__name__

    def __init__(self):
        super().__init__()
        self._mapping = dict()

    def put(self, session_id, chunk_key, obj_id):
        session_chunk_key = (session_id, chunk_key)
        if session_chunk_key in self._mapping:
            raise StorageDataExists(session_chunk_key)
        self._mapping[session_chunk_key] = obj_id

    def get(self, session_id, chunk_key):
        return self._mapping.get((session_id, chunk_key))

    def delete(self, session_id, chunk_key):
        try:
            del self._mapping[(session_id, chunk_key)]
        except KeyError:
            pass

    def batch_delete(self, session_id, chunk_keys):
        for k in chunk_keys:
            self.delete(session_id, k)


class VineyardBytesIO(BytesStorageIO):
    storage_type = DataStorageDevice.VINEYARD

    def __init__(self, vineyard_client, session_id, data_key, data_id, mode='w',
                 nbytes=None, packed=False, compress=None, auto_register=True,
                 pin_token=None, handler=None):
        from .objectholder import SharedHolderActor

        logger.debug('create vineyard bytes IO: mode = %s, packed = %s', mode, packed)

        super().__init__(session_id, data_key, mode=mode, handler=handler)
        self._client = vineyard_client
        self._data_id = data_id
        self._components = None
        self._offset = 0
        self._nbytes = nbytes
        self._holder_ref = self._storage_ctx.actor_ctx.actor_ref(SharedHolderActor.default_uid())
        self._compress = compress or dataserializer.CompressType.NONE
        self._packed = packed
        self._auto_register = auto_register
        self._pin_token = pin_token

        block_size = options.worker.copy_block_size

        if self.is_readable:
            logger.debug('bytes io get: session_id = %s, data_key = %s, data_id = %r, type(data_id) = %r',
                         session_id, data_key, data_id, type(data_id))
            data = self._client.get(data_id)

            self._components = pyarrow.serialize(data, dataserializer.mars_serialize_context()).to_components()
            if packed:
                self._buf = ArrowComponentsIO(
                    self._components, 'r', compress_out=compress, block_size=block_size)
            else:
                raise NotImplementedError('Unknown how to read vineyard values in a unpacked way')
        else:
            raise NotImplementedError

    def __del__(self):
        self._buf = self._components = None

    @property
    def nbytes(self):
        return self._nbytes

    def read(self, size=-1):
        if self._packed:
            return self._buf.read(size)
        else:
            raise NotImplementedError('Unknown how to read vineyard values in a unpacked way')

    def close(self, finished=True):
        if self._closed:
            return
        self._buf = self._components = None
        super().close(finished=finished)


class VineyardHandler(StorageHandler, ObjectStorageMixin):
    storage_type = DataStorageDevice.VINEYARD

    def __init__(self, storage_ctx, proc_id=None):
        StorageHandler.__init__(self, storage_ctx, proc_id=proc_id)
        self._client = vineyard.connect(options.vineyard.socket)
        logger.debug('find mapper ref: %s', VineyardKeyMapActor.default_uid())
        self._mapper_ref = self._storage_ctx.actor_ctx.actor_ref(VineyardKeyMapActor.default_uid())
        logger.debug('find mapper ref done: %s', VineyardKeyMapActor.default_uid())

    def _new_object_id(self, session_id, data_key, data_id):
        logger.debug('mapper put: session_id = %s, data_key = %s, data_id = %r', session_id, data_key, data_id)
        self._mapper_ref.put(session_id, data_key, data_id)

    def _get_object_id(self, session_id, data_key):
        obj_id = self._mapper_ref.get(session_id, data_key)
        logger.debug('mapper get: session_id = %s, data_key = %s, obj_id = %r', session_id, data_key, obj_id)
        if obj_id is None:
            raise KeyError((session_id, data_key))
        return obj_id

    @wrap_promised
    def create_bytes_reader(self, session_id, data_key, packed=False, packed_compression=None,
                            _promise=False):
        data_id = self._get_object_id(session_id, data_key)
        logger.debug('create vineyard bytes reader: data_id = %s', data_id)
        return VineyardBytesIO(self._client, session_id, data_key, data_id, 'r', packed=packed,
                               compress=packed_compression, handler=self)

    @wrap_promised
    def create_bytes_writer(self, session_id, data_key, total_bytes, packed=False,
                            packed_compression=None, auto_register=True, pin_token=None,
                            _promise=False):
        raise NotImplementedError('vineyard bytes writer hasn\'t been implemented')

    @wrap_promised
    def get_objects(self, session_id, data_keys, serialize=False, _promise=False):
        data_ids = [self._get_object_id(session_id, data_key) for data_key in data_keys]
        return self._client.get_object(data_ids)

    @wrap_promised
    def put_objects(self, session_id, data_keys, objs, sizes=None, serialize=False,
                    pin_token=None, _promise=False):
        sizes, shapes = [], []
        for data_key, obj in zip(data_keys, objs):
            data_id, size, shape = self._client.put_object(obj)
            sizes.append(size)
            shapes.append(shape)
            self._new_object_id(session_id, data_key, data_id)
        self.register_data(session_id, data_keys, sizes, shapes)

    def load_from_bytes_io(self, session_id, data_keys, src_handler, pin_token=None):
        def _read_serialized(reader):
            with reader:
                return reader.get_io_pool().submit(reader.read).result()

        def _fallback(*_):
            return self._batch_load_objects(
                session_id, data_keys,
                lambda k: src_handler.create_bytes_reader(session_id, k, _promise=True).then(_read_serialized),
                serialize=True
            )

        return self.transfer_in_runner(session_id, data_keys, src_handler, _fallback)

    def load_from_object_io(self, session_id, data_keys, src_handler, pin_token=None):
        def _fallback(*_):
            return self._batch_load_objects(
                session_id, data_keys,
                lambda k: src_handler.get_objects(session_id, k, _promise=True), batch_get=True)

        return self.transfer_in_runner(session_id, data_keys, src_handler, _fallback)

    def delete(self, session_id, data_keys, _tell=False):
        data_ids = [self._get_object_id(session_id, data_key)
                    for data_key in data_keys]
        self._client.delete(data_ids, deep=True)
        self.unregister_data(session_id, data_keys, _tell=_tell)


register_storage_handler_cls(DataStorageDevice.VINEYARD, VineyardHandler)
