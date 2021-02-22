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

import ctypes
import os
from typing import Tuple, Dict, List, Union, Optional

from ..serialization import serialize, deserialize
from ..utils import lazy_import, implements
from .base import StorageBackend, StorageLevel, ObjectInfo
from .core import BufferWrappedFileObject, StorageFileObject

import numpy as np
import pandas as pd
try:
    from cupy.cuda import MemoryPointer
    from cudf.core import Buffer
except ImportError:  # pragma: no cover
    MemoryPointer = None
    Buffer = None

cupy = lazy_import('cupy', globals=globals())
cudf = lazy_import('cudf', globals=globals())


class CudaObjectId:
    __slots__ = 'headers', 'ptrs'

    def __init__(self, headers: Dict, ptrs: List[int]):
        self.headers = headers
        self.ptrs = ptrs


class CudaFileObject(BufferWrappedFileObject):
    def __init__(self, object_id: CudaObjectId, mode: str,
                 size: Optional[int] = None,
                 cuda_buffer: Optional[Buffer] = None):
        self._object_id = object_id
        self._cuda_buffer = cuda_buffer
        self._cupy_memory = None
        super().__init__(mode, size=size)

    def _read_init(self):
        from cupy.cuda.memory import UnownedMemory

        ptr = self._object_id.ptrs[0]
        self._size = self._object_id.headers['size']
        self._buffer = Buffer(ptr, self._size)
        self._cupy_memory = UnownedMemory(ptr, self._size, self._buffer)

    def _write_init(self):
        from cupy.cuda.memory import UnownedMemory

        self._buffer = self._cuda_buffer
        self._cupy_memory = UnownedMemory(self._buffer.ptr, self._size, self._buffer)

    def write(self, content: Union[bytes, MemoryPointer]):
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

    def read(self, size=-1) -> Buffer:
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


class CudaStorage(StorageBackend):
    def __init__(self, **kw):
        if kw:  # pragma: no cover
            raise TypeError(f'CudaStorage got unexpected arguments: {",".join(kw)}')
        self._id_to_buffers = dict()

    @classmethod
    @implements(StorageBackend.setup)
    async def setup(cls, **kwargs) -> Tuple[Dict, Dict]:
        if kwargs:  # pragma: no cover
            raise TypeError(f'CudaStorage got unexpected config: {",".join(kwargs)}')

        return dict(), dict()

    @staticmethod
    @implements(StorageBackend.teardown)
    async def teardown(**kwargs):
        pass

    @property
    @implements(StorageBackend.level)
    def level(self):
        return StorageLevel.GPU

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

        headers = object_id.headers
        ptrs = object_id.ptrs
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
        return deserialize(headers, buffers)

    @implements(StorageBackend.put)
    async def put(self, obj, importance=0) -> ObjectInfo:
        obj = self._to_cuda(obj)
        headers, buffers = serialize(obj)
        if isinstance(obj, cupy.ndarray):
            device = obj.device.id
            headers['data_type'] = 'cupy'
            ptrs = [b.data.ptr for b in buffers]
        elif isinstance(obj, (cudf.DataFrame, cudf.Series, cudf.Index)):
            device = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',', 1)[0])
            headers['data_type'] = 'cudf'
            ptrs = [b.ptr for b in buffers]
        else:  # pragma: no cover
            raise TypeError(f'Unsupported data type: {type(obj)}')

        headers['device'] = device
        object_id = CudaObjectId(headers, ptrs)
        size = sum(getattr(buf, 'nbytes', len(buf)) for buf in buffers)
        self._id_to_buffers[object_id] = buffers
        return ObjectInfo(size=size, object_id=object_id, device=device)

    @implements(StorageBackend.delete)
    async def delete(self, object_id):
        del self._id_to_buffers[object_id]

    @implements(StorageBackend.object_info)
    async def object_info(self, object_id: CudaObjectId) -> ObjectInfo:
        size = sum(object_id.headers['lengths'])
        return ObjectInfo(size=size, object_id=object_id, device=object_id.headers['device'])

    @implements(StorageBackend.open_writer)
    async def open_writer(self, size=None) -> StorageFileObject:
        from cudf.core.buffer import Buffer

        cuda_buffer = Buffer.empty(size)
        headers = dict(size=size)
        object_id = CudaObjectId(headers, [cuda_buffer.ptr])
        self._id_to_buffers[object_id] = cuda_buffer
        cuda_writer = CudaFileObject(object_id, cuda_buffer=cuda_buffer, mode='w', size=size)
        return StorageFileObject(cuda_writer, object_id=object_id)

    @implements(StorageBackend.open_reader)
    async def open_reader(self, object_id) -> StorageFileObject:
        cuda_reader = CudaFileObject(object_id, mode='r')
        return StorageFileObject(cuda_reader, object_id=object_id)

    @implements(StorageBackend.list)
    async def list(self) -> List:  # pragma: no cover
        raise NotImplementedError("Cuda storage doesn't support `list` method.")
