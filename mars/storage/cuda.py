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

import ctypes
import os
import uuid
from typing import Tuple, Dict, List, Optional, Union

from ..serialization import serialize, deserialize
from ..utils import lazy_import, implements
from .base import StorageBackend, StorageLevel, ObjectInfo, register_storage_backend
from .core import BufferWrappedFileObject, StorageFileObject

import numpy as np
import pandas as pd

cupy = lazy_import('cupy', globals=globals())
cudf = lazy_import('cudf', globals=globals())


class CudaObjectId:
    __slots__ = 'headers_list', 'ptrs_list', 'object_id', 'is_tuple'

    def __init__(self,
                 headers_list: List[Dict],
                 ptrs_list: List[List[int]],
                 object_id: str,
                 is_tuple: bool = False):
        self.headers_list = headers_list
        self.ptrs_list = ptrs_list
        self.object_id = object_id
        self.is_tuple = is_tuple


class CudaFileObject(BufferWrappedFileObject):
    def __init__(self, object_id: CudaObjectId, mode: str,
                 size: Optional[int] = None,
                 cuda_buffer: Optional[object] = None):
        self._cuda_buffer = cuda_buffer
        self._cupy_memory = None
        super().__init__(object_id, mode, size=size)

    def _read_init(self):
        from cudf.core import Buffer
        from cupy.cuda.memory import UnownedMemory

        ptr = self._object_id.ptrs[0]
        self._size = self._object_id.headers['size']
        self._buffer = Buffer(ptr, self._size)
        self._cupy_memory = UnownedMemory(ptr, self._size, self._buffer)

    def _write_init(self):
        from cupy.cuda.memory import UnownedMemory

        self._buffer = self._cuda_buffer
        self._cupy_memory = UnownedMemory(self._buffer.ptr, self._size, self._buffer)

    def write(self, content):
        from cupy.cuda import MemoryPointer

        if not self._initialized:
            self._write_init()
            self._initialized = True

        cupy_pointer = MemoryPointer(self._cupy_memory, self._offset)

        if isinstance(content, bytes):
            content_length = len(content)
            source_mem = np.frombuffer(content, dtype='uint8').ctypes.data_as(ctypes.c_void_p)
        else:
            content_length = content.mem.size
            source_mem = content
        cupy_pointer.copy_from(source_mem, content_length)
        self._offset += content_length

    def read(self, size=-1):
        from cudf.core import Buffer
        from cupy.cuda import MemoryPointer

        if not self._initialized:
            self._read_init()
            self._initialized = True
        size = self._size if size < 0 else size
        cupy_pointer = MemoryPointer(self._cupy_memory, self._offset)
        self._offset += size
        return Buffer(cupy_pointer.ptr, size=size)

    def _read_close(self):
        self._cupy_memory = None

    def _write_close(self):
        self._cupy_memory = None


@register_storage_backend
class CudaStorage(StorageBackend):
    name = 'cuda'

    def __init__(self, size=None):
        self._size = size
        self._id_to_buffers = dict()

    @classmethod
    @implements(StorageBackend.setup)
    async def setup(cls, **kwargs) -> Tuple[Dict, Dict]:
        size = kwargs.pop('size', None)
        if kwargs:  # pragma: no cover
            raise TypeError(f'CudaStorage got unexpected config: {",".join(kwargs)}')

        return dict(size=size), dict()

    @staticmethod
    @implements(StorageBackend.teardown)
    async def teardown(**kwargs):
        pass

    @property
    @implements(StorageBackend.level)
    def level(self):
        return StorageLevel.GPU

    @property
    @implements(StorageBackend.size)
    def size(self) -> Union[int, None]:
        return self._size

    @staticmethod
    def _to_cuda(obj):  # pragma: no cover
        if isinstance(obj, np.ndarray):
            return cupy.asarray(obj)
        elif isinstance(obj, pd.DataFrame):
            return cudf.DataFrame.from_pandas(obj)
        elif isinstance(obj, pd.Series):
            return cudf.Series.from_pandas(obj)
        return obj

    @implements(StorageBackend.get)
    async def get(self, object_id: CudaObjectId, **kwargs) -> object:
        from cudf.core.buffer import Buffer
        from rmm import DeviceBuffer

        if kwargs:  # pragma: no cover
            raise NotImplementedError('Got unsupported args: {",".join(kwargs)}')

        headers_list = object_id.headers_list
        ptrs_list = object_id.ptrs_list
        objs = []
        for headers, ptrs in zip(headers_list, ptrs_list):
            data_type = headers.pop('data_type')
            if data_type == 'cupy':
                ptr = ptrs[0]
                size = headers['lengths'][0]
                cuda_buf = DeviceBuffer(ptr=ptr, size=size)
                buffers = [cuda_buf]
            elif data_type == 'cudf':
                buffers = [Buffer(ptr, length, DeviceBuffer(ptr=ptr, size=length))
                           for ptr, length in zip(ptrs, headers['lengths'])]
            else:
                raise TypeError(f'Unknown data type {data_type}')
            objs.append(deserialize(headers, buffers))
        if object_id.is_tuple:
            return tuple(objs)
        else:
            return objs[0]

    @implements(StorageBackend.put)
    async def put(self, obj, importance=0) -> ObjectInfo:
        is_tuple = isinstance(obj, (tuple, list))
        objs = obj if is_tuple else [obj]
        size = 0
        buffers_list = []
        headers_list = []
        ptrs_list = []
        for obj in objs:
            obj = self._to_cuda(obj)
            headers, buffers = serialize(obj)
            if isinstance(obj, cupy.ndarray):
                headers['data_type'] = 'cupy'
                ptrs = [b.data.ptr for b in buffers]
            elif isinstance(obj, (cudf.DataFrame, cudf.Series, cudf.Index)):
                headers['data_type'] = 'cudf'
                ptrs = [b.ptr for b in buffers]
            else:  # pragma: no cover
                raise TypeError(f'Unsupported data type: {type(obj)}')
            size += sum(getattr(buf, 'nbytes', len(buf)) for buf in buffers)
            buffers_list.append(buffers)
            headers_list.append(headers)
            ptrs_list.append(ptrs)
        string_id = str(uuid.uuid4())
        object_id = CudaObjectId(headers_list, ptrs_list, string_id, is_tuple)
        self._id_to_buffers[string_id] = buffers_list
        return ObjectInfo(size=size, object_id=object_id)

    @implements(StorageBackend.delete)
    async def delete(self, object_id: CudaObjectId):
        del self._id_to_buffers[object_id.object_id]

    @implements(StorageBackend.object_info)
    async def object_info(self, object_id: CudaObjectId) -> ObjectInfo:
        size = sum(sum(headers['lengths']) for headers in object_id.headers_list)
        return ObjectInfo(size=size, object_id=object_id)

    @implements(StorageBackend.open_writer)
    async def open_writer(self, size=None) -> StorageFileObject:
        from cudf.core.buffer import Buffer

        cuda_buffer = Buffer.empty(size)
        headers = dict(size=size)
        string_id = str(uuid.uuid4())
        object_id = CudaObjectId([headers], [[cuda_buffer.ptr]], string_id)
        self._id_to_buffers[string_id] = cuda_buffer
        cuda_writer = CudaFileObject(object_id, cuda_buffer=cuda_buffer, mode='w', size=size)
        return StorageFileObject(cuda_writer, object_id=object_id)

    @implements(StorageBackend.open_reader)
    async def open_reader(self, object_id) -> StorageFileObject:
        cuda_reader = CudaFileObject(object_id, mode='r')
        return StorageFileObject(cuda_reader, object_id=object_id)

    @implements(StorageBackend.list)
    async def list(self) -> List:  # pragma: no cover
        raise NotImplementedError("Cuda storage doesn't support `list` method.")
