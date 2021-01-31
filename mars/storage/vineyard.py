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

from typing import Union, Dict, List

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
from .core import StorageBackend, StorageLevel, ObjectInfo, StorageFileObject


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


class VineyardFileObject:
    def __init__(self, vineyard_client, object_id=None, mode='w', size=None):
        self._client = vineyard_client
        self._object_id = object_id
        self._offset = 0
        self._size = size
        self._is_readable = 'r' in mode
        self._is_writable = 'w' in mode
        self._closed = False

        if self._is_writable:
            self._blob = self._client.create_blob(size)
            self._buf = pa.FixedSizeBufferWriter(self._blob.buffer)
            self._buf.set_memcopy_threads(6)
        elif self._is_readable:
            self._buf = self._client.get_object(object_id)
            self._mv = memoryview(self._buf)
            self._size = len(self._buf)
        else:  # pragma: no cover
            raise NotImplementedError

    @property
    def size(self):
        return self._size

    @property
    def object_id(self):
        return self._object_id

    @property
    def closed(self):
        return self._closed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def read(self, size=-1):
        if size < 0:
            size = self._size
        right_pos = min(self._size, self._offset + size)
        ret = self._mv[self._offset:right_pos]
        self._offset = right_pos
        return ret

    def write(self, d):
        if self._is_writable:
            return self._buf.write(d)
        return

    def close(self):
        if self._closed:
            return

        self._closed = True
        if self._is_writable:
            self._object_id = self._blob.seal(self._client).id

        self._mv = None
        self._buf = None


class VineyardStorage(StorageBackend):
    def __init__(self, vineyard_socket=None):
        self._client = vineyard.connect(vineyard_socket)

    @property
    def name(self) -> str:
        return 'vineyard'

    @classmethod
    async def setup(cls,**kwargs) -> Tuple[Dict, Dict]:
        etcd_endpoints = kwargs.get('etcd_endpoints', None)
        size = kwargs.get('size', '256M')
        socket = kwargs.get('socket', '/tmp/vineyard.sock')
        vineyardd_path = kwargs.get('vineyardd_path', '/usr/local/bin/vineyardd')
        vineyard_store = start_vineyardd(etcd_endpoints, vineyardd_path, size, socket)
        return dict(vineyard_socket=vineyard_store.__enter__()[1]), dict(vineyard_store=vineyard_store)

    @staticmethod
    async def teardown(**kwargs) -> None:
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

    async def create_writer(self, size=None) -> StorageFileObject:
        vineyard_writer = VineyardFileObject(self._client, size=size, mode='w')
        return StorageFileObject(vineyard_writer, object_id=None)

    async def open_reader(self, object_id) -> StorageFileObject:
        vineyard_reader = VineyardFileObject(self._client, object_id, mode='r')
        return StorageFileObject(vineyard_reader, object_id=object_id)

    async def list(self) -> List:
        raise NotImplementedError
