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
import pickle
import uuid
from typing import Tuple, Dict, List, Union

from ..serialization import serialize, deserialize
from ..utils import lazy_import, implements
from .base import StorageBackend, StorageLevel, ObjectInfo, register_storage_backend
from .core import StorageFileObject

import numpy as np
import pandas as pd

cupy = lazy_import('cupy', globals=globals())
cudf = lazy_import('cudf', globals=globals())


_id_to_buffers = dict()


class CudaObjectId:
    __slots__ = 'headers_list', 'ptrs_list', 'object_id', 'is_tuple'

    def __init__(self,
                 headers_list: List[Dict] = None,
                 ptrs_list: List[List[int]] = None,
                 object_id: str = None,
                 is_tuple: bool = False):
        self.headers_list = headers_list
        self.ptrs_list = ptrs_list
        self.object_id = object_id
        self.is_tuple = is_tuple


class CudaFileObject:
    def __init__(self,
                 mode: str,
                 object_id: CudaObjectId = None,
                 size: int = None):
        self._mode = mode
        self._object_id = object_id
        self._size = size
        self._closed = False
        self._cuda_buffers = None
        self._cuda_header = None
        self._offset = None
        # for read
        self._has_read_headers = None
        # for write
        self._has_write_headers = None
        self._cur_buffer_index = None
        if 'r' in mode:
            assert object_id is not None
            self._initialize_read()
        elif 'w' in mode:
            self._initialize_write()

    @property
    def object_id(self):
        return self._object_id

    @property
    def mode(self):
        return self._mode

    def _initialize_read(self):
        from cudf.core import Buffer
        from cupy.cuda.memory import UnownedMemory

        self._offset = 0
        self._has_read_headers = False
        is_tuple = self._object_id.is_tuple
        headers_list = self._object_id.headers_list
        self._cuda_header = (is_tuple, headers_list)
        self._cuda_buffers = []
        for headers, ptrs in zip(headers_list, self._object_id.ptrs_list):
            for length, ptr in zip(headers['lengths'], ptrs):
                self._cuda_buffers.append(UnownedMemory(ptr, length, Buffer(ptr, length)))

    def _initialize_write(self):
        self._had_write_headers = False
        self._cur_buffer_index = 0
        self._cuda_buffers = []
        self._offset = 0

    def read(self, size: int):
        # we read cuda_header first and then read cuda buffers one by one,
        # the return value's size is not exactly the specified size.
        from cudf.core import Buffer
        from cupy.cuda import MemoryPointer

        if not self._has_read_headers:
            self._has_read_headers = True
            return pickle.dumps(self._cuda_header)
        if len(self._cuda_buffers) == 0:
            return ''
        cur_buf = self._cuda_buffers[0]
        if size >= cur_buf.size - self._offset:
            # current buf read to end
            cupy_pointer = MemoryPointer(cur_buf, self._offset)
            content = Buffer(cupy_pointer.ptr, size=cur_buf.size - self._offset)
            self._offset = 0
            self._cuda_buffers.pop(0)
            return content
        else:
            cupy_pointer = MemoryPointer(cur_buf, self._offset)
            self._offset += size
            return Buffer(cupy_pointer.ptr, size=size)

    def _write_headers(self):
        from cudf.core import Buffer

        for headers in self._cuda_header[1]:
            for length in headers['lengths']:
                self._cuda_buffers.append(Buffer.empty(length))

    def write(self, content):
        from cupy.cuda import MemoryPointer
        from cupy.cuda.memory import UnownedMemory

        if not self._has_write_headers:
            self._cuda_header = pickle.loads(content)
            self._has_write_headers = True
            self._write_headers()
            return
        cur_buf = self._cuda_buffers[self._cur_buffer_index]
        cur_cupy_memory = UnownedMemory(cur_buf.ptr, len(cur_buf), cur_buf)
        cupy_pointer = MemoryPointer(cur_cupy_memory, self._offset)

        if isinstance(content, bytes):
            content_length = len(content)
            source_mem = np.frombuffer(content, dtype='uint8').ctypes.data_as(ctypes.c_void_p)
        else:
            source_mem = MemoryPointer(UnownedMemory(content.ptr, len(content), content), 0)
            content_length = source_mem.mem.size
        cupy_pointer.copy_from(source_mem, content_length)
        if content_length + self._offset >= cur_cupy_memory.size:
            self._cur_buffer_index += 1
            self._offset = 0
        else:
            self._offset += content_length

    def _read_close(self):
        self._offset = None
        self._cuda_buffers = None
        self._cuda_header = None
        self._has_read_headers = None

    def _write_close(self):
        # generate object id
        is_tuple, headers_list = self._cuda_header
        ptr_idx = 0
        ptrs_list = []
        for headers in headers_list:
            ptrs = []
            for _ in headers['lengths']:
                ptrs.append(self._cuda_buffers[ptr_idx].ptr)
                ptr_idx += 1
            ptrs_list.append(ptrs)

        string_id = str(uuid.uuid4())
        self._object_id = CudaObjectId(
            headers_list, ptrs_list, string_id, is_tuple)
        # hold cuda buffers
        _id_to_buffers[string_id] = self._cuda_buffers

        self._has_write_headers = None
        self._cur_buffer_index = None
        self._cuda_buffers = None
        self._cuda_header = None
        self._offset = None

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._mode == 'w':
            self._write_close()
        else:
            self._read_close()


@register_storage_backend
class CudaStorage(StorageBackend):
    name = 'cuda'

    def __init__(self, size=None):
        self._size = size

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
            data_type = headers.get('data_type')
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
        _id_to_buffers[string_id] = buffers_list
        return ObjectInfo(size=size, object_id=object_id)

    @implements(StorageBackend.delete)
    async def delete(self, object_id: CudaObjectId):
        if object_id.object_id in _id_to_buffers:
            del _id_to_buffers[object_id.object_id]

    @implements(StorageBackend.object_info)
    async def object_info(self, object_id: CudaObjectId) -> ObjectInfo:
        size = sum(sum(headers['lengths']) for headers in object_id.headers_list)
        return ObjectInfo(size=size, object_id=object_id)

    @implements(StorageBackend.open_writer)
    async def open_writer(self, size=None) -> StorageFileObject:
        cuda_writer = CudaFileObject(object_id=None, mode='w', size=size)
        return StorageFileObject(cuda_writer, object_id=None)

    @implements(StorageBackend.open_reader)
    async def open_reader(self, object_id) -> StorageFileObject:
        cuda_reader = CudaFileObject(mode='r', object_id=object_id)
        return StorageFileObject(cuda_reader, object_id=object_id)

    @implements(StorageBackend.list)
    async def list(self) -> List:  # pragma: no cover
        raise NotImplementedError("Cuda storage doesn't support `list` method.")
