#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from typing import Any, Dict, List, Tuple

import pyarrow as pa
try:
    import vineyard
    from vineyard._C import ObjectMeta
    from vineyard.core import default_builder_context, default_resolver_context
    from vineyard.data.utils import from_json, to_json
    from vineyard.deploy.local import start_vineyardd
except ImportError:
    vineyard = None

from ..lib import sparse
from .base import StorageBackend, StorageLevel, ObjectInfo
from .core import BufferWrappedFileObject, StorageFileObject


def mars_sparse_matrix_builder(client, value, builder, **kw):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::SparseMatrix<%s>' % value.dtype.name
    meta['shape_'] = to_json(value.shape)
    meta.add_member('spmatrix', builder.run(client, value.spmatrix, **kw))
    return client.create_metadata(meta)


def mars_sparse_matrix_resolver(obj, resolver):
    meta = obj.meta
    shape = from_json(meta['shape_'])
    spmatrix = resolver.run(obj.member('spmatrix'))
    return sparse.matrix.SparseMatrix(spmatrix, shape=shape)


if vineyard is not None:
    default_builder_context.register(sparse.matrix.SparseMatrix, mars_sparse_matrix_builder)
    default_resolver_context.register('vineyard::SparseMatrix', mars_sparse_matrix_resolver)


class VineyardFileObject(BufferWrappedFileObject):
    def __init__(self, vineyard_client, object_id, mode, size=None):
        self._client = vineyard_client
        self._object_id = object_id
        super().__init__(mode, size=size)

    def _write_init(self):
        self._buffer = buf = self._client.create_blob(self._size)
        file = self._file = pa.FixedSizeBufferWriter(buf.buffer)
        file.set_memcopy_threads(6)

    def _read_init(self):
        self._buffer = buf = self._client.get_object(self._object_id)
        self._mv = memoryview(buf)
        self._size = len(buf)

    def _write_close(self):
        self._object_id = self._buffer.seal(self._client).id

    def _read_close(self):
        pass


class VineyardObjectInfo(ObjectInfo):
    __slots__ = "buffer",

    def __init__(self,
                 size: int = None,
                 device: int = None,
                 object_id: Any = None,
                 buffer: memoryview = None):
        super().__init__(size=size, device=device,
                         object_id=object_id)
        self.buffer = buffer


class VineyardStorage(StorageBackend):
    def __init__(self, vineyard_socket=None):
        self._client = vineyard.connect(vineyard_socket)

    @property
    def name(self) -> str:
        return 'vineyard'

    @classmethod
    async def setup(cls,**kwargs) -> Tuple[Dict, Dict]:
        etcd_endpoints = kwargs.pop('etcd_endpoints', None)
        vineyard_size = kwargs.pop('vineyard_size', '256M')
        vineyard_socket = kwargs.pop('vineyard_socket', '/tmp/vineyard.sock')
        vineyardd_path = kwargs.pop('vineyardd_path', '/usr/local/bin/vineyardd')

        if kwargs:
            raise TypeError(f'VineyardStorage got unexpected config: {",".join(kwargs)}')

        vineyard_store = start_vineyardd(
            etcd_endpoints,
            vineyardd_path,
            vineyard_size,
            vineyard_socket)
        return dict(vineyard_socket=vineyard_store.__enter__()[1]), dict(vineyard_store=vineyard_store)

    @staticmethod
    async def teardown(**kwargs):
        vineyard_store = kwargs.get('vineyard_store')
        vineyard_store.__exit__(None, None, None)

    @property
    def level(self):
        return StorageLevel.MEMORY

    async def get(self, object_id, **kwarg) -> object:
        return self._client.get(object_id)

    async def put(self, obj) -> ObjectInfo:
        object_id = self._client.put(obj)
        size = self._client.get_meta(object_id).nbytes
        return ObjectInfo(size=size, device='memory', object_id=object_id)

    async def delete(self, object_id):
        self._client.delete([object_id], deep=True)

    async def object_info(self, object_id) -> ObjectInfo:
        size = self._client.get_meta(object_id).nbytes
        return ObjectInfo(size=size, object_id=object_id)

    async def open_writer(self, size=None) -> StorageFileObject:
        if size is None:  # pragma: no cover
            raise ValueError('size must be provided for vineyard backend')

        vineyard_writer = VineyardFileObject(self._client, None, size=size, mode='w')
        return StorageFileObject(vineyard_writer, object_id=None)

    async def open_reader(self, object_id) -> StorageFileObject:
        vineyard_reader = VineyardFileObject(self._client, object_id, mode='r')
        return StorageFileObject(vineyard_reader, object_id=object_id)

    async def list(self) -> List:
        raise NotImplementedError
