#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2021 Alibaba Group Holding Ltd.
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

import asyncio
from io import UnsupportedOperation
import logging
import sys
from typing import Dict, List, Optional, Tuple

from ..lib import sparse
from ..resource import virtual_memory
from ..utils import implements, lazy_import, calc_size_by_str
from .base import StorageBackend, StorageLevel, ObjectInfo, register_storage_backend
from .core import BufferWrappedFileObject, StorageFileObject

vineyard = lazy_import("vineyard")
pyarrow = lazy_import("pyarrow")

if sys.platform.startswith('win'):
    vineyard = None

logger = logging.getLogger(__name__)


# Setup support for mars datatypes on vineyard

def mars_sparse_matrix_builder(client, value, builder, **kw):
    meta = vineyard.ObjectMeta()
    meta['typename'] = 'vineyard::SparseMatrix<%s>' % value.dtype.name
    meta['shape_'] = vineyard.data.utils.to_json(value.shape)
    meta.add_member('spmatrix', builder.run(client, value.spmatrix, **kw))
    return client.create_metadata(meta)


def mars_sparse_matrix_resolver(obj, resolver) -> sparse.SparseNDArray:
    meta = obj.meta
    shape = vineyard.data.utils.from_json(meta['shape_'])
    spmatrix = resolver.run(obj.member('spmatrix'))
    return sparse.matrix.SparseMatrix(spmatrix, shape=shape)


if vineyard is not None:
    vineyard.core.default_builder_context.register(sparse.matrix.SparseMatrix, mars_sparse_matrix_builder)
    vineyard.core.default_resolver_context.register('vineyard::SparseMatrix', mars_sparse_matrix_resolver)


class VineyardFileObject(BufferWrappedFileObject):
    def __init__(self, vineyard_client, object_id,
                 mode: str, size: Optional[int] = None):
        self._client = vineyard_client
        self._file = None

        self._reader = None
        self._writer = None

        if size is None:
            size = -1  # unknown estimated size.

        super().__init__(object_id, mode, size=size)

    def _read_init(self):
        self._reader = vineyard.data.pickle.PickledReader(self._client.get(self._object_id))
        self._size = self._reader.store_size

    def _write_init(self):
        self._writer = vineyard.data.pickle.PickledWriter(self._size)

    @property
    def buffer(self):
        raise UnsupportedOperation("VineyardFileObject doesn't support the direct 'buffer' property")

    def read(self, size=-1):
        if not self._initialized:
            self._read_init()
            self._initialized = True
        return self._reader.read(size)

    def write(self, content: bytes):
        if not self._initialized:
            self._write_init()
            self._initialized = True
        return self._writer.write(content)

    def _read_close(self):
        self._reader = None

    def _write_close(self):
        self._writer.close()
        self._object_id = self._client.put(self._writer.value)
        self._writer = None


@register_storage_backend
class VineyardStorage(StorageBackend):
    name = 'vineyard'

    def __init__(self,
                 vineyard_size: int,
                 vineyard_socket: str = None):
        self._size = vineyard_size
        self._client = vineyard.connect(vineyard_socket)

    @classmethod
    @implements(StorageBackend.setup)
    async def setup(cls, **kwargs) -> Tuple[Dict, Dict]:
        loop = asyncio.get_running_loop()
        etcd_endpoints = kwargs.pop('etcd_endpoints', None)
        vineyard_size = kwargs.pop('vineyard_size', '1Gi')
        vineyard_socket = kwargs.pop('vineyard_socket', None)
        vineyardd_path = kwargs.pop('vineyardd_path', None)

        if kwargs:
            raise TypeError(f'VineyardStorage got unexpected config: {",".join(kwargs)}')

        vineyard_size = calc_size_by_str(vineyard_size, virtual_memory().total)
        vineyard_store = vineyard.deploy.local.start_vineyardd(
            etcd_endpoints,
            vineyardd_path,
            vineyard_size,
            vineyard_socket,
            rpc=False)
        vineyard_socket = (await loop.run_in_executor(
            None, vineyard_store.__enter__))[1]
        init_params = dict(vineyard_size=vineyard_size,
                           vineyard_socket=vineyard_socket)
        teardown_params = dict(vineyard_store=vineyard_store)
        return init_params, teardown_params

    @staticmethod
    @implements(StorageBackend.teardown)
    async def teardown(**kwargs):
        vineyard_store = kwargs.get('vineyard_store')
        vineyard_store.__exit__(None, None, None)

    @property
    @implements(StorageBackend.level)
    def level(self) -> StorageLevel:
        return StorageLevel.MEMORY

    @property
    @implements(StorageBackend.size)
    def size(self) -> Optional[int]:
        return self._size

    @implements(StorageBackend.get)
    async def get(self, object_id, **kwargs) -> object:
        if kwargs:  # pragma: no cover
            raise NotImplementedError('Got unsupported args: {",".join(kwargs)}')

        return self._client.get(object_id)

    @implements(StorageBackend.put)
    async def put(self, obj, importance: int = 0) -> ObjectInfo:
        object_id = self._client.put(obj)
        size = self._client.get_meta(object_id).nbytes
        return ObjectInfo(size=size, object_id=object_id)

    @implements(StorageBackend.delete)
    async def delete(self, object_id):
        self._client.delete([object_id], deep=True)

    @implements(StorageBackend.object_info)
    async def object_info(self, object_id) -> ObjectInfo:
        size = self._client.get_meta(object_id).nbytes
        return ObjectInfo(size=size, object_id=object_id)

    @implements(StorageBackend.open_writer)
    async def open_writer(self, size=None) -> StorageFileObject:
        vineyard_writer = VineyardFileObject(self._client, None, size=size, mode='w')
        return StorageFileObject(vineyard_writer, object_id=None)

    @implements(StorageBackend.open_reader)
    async def open_reader(self, object_id) -> StorageFileObject:
        vineyard_reader = VineyardFileObject(self._client, object_id, mode='r')
        return StorageFileObject(vineyard_reader, object_id=object_id)

    @implements(StorageBackend.list)
    async def list(self) -> List:
        # FIXME: vineyard's list_objects not equal to plasma
        raise NotImplementedError
