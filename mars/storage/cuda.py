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
import pickle
import uuid
from io import BytesIO
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


class CudaFileObject:
    def __init__(self,
                 mode: str,
                 object_id: str,
                 size: int = None):
        self._mode = mode
        self._object_id = object_id
        self._size = size
        self._closed = False
        self._buffers = None
        self._headers = None
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
        self._buffers = []
        headers, buffers = _id_to_buffers[self._object_id]
        self._headers = headers = headers.copy()
        buffer_types = []
        for buf in buffers:
            if isinstance(buf, cupy.ndarray):
                ptr, size = buf.data.ptr, buf.size
                self._buffers.append(UnownedMemory(ptr, size, Buffer(ptr, size)))
                buffer_types.append(['cuda', size])
            elif isinstance(buf, Buffer):
                ptr, size = buf.ptr, buf.size
                if size == 0:
                    # empty buffer cannot construct a UnownedMemory
                    self._buffers.append(None)
                else:
                    self._buffers.append(UnownedMemory(ptr, size, Buffer(ptr, size)))
                buffer_types.append(['cuda', size])
            else:
                size = getattr(buf, 'size', len(buf))
                self._buffers.append(buf)
                buffer_types.append(['memory', size])
        headers['buffer_types'] = buffer_types

    def _initialize_write(self):
        self._had_write_headers = False
        self._cur_buffer_index = 0
        self._buffers = []
        self._offset = 0

    def read(self, size: int):
        # we read cuda_header first and then read cuda buffers one by one,
        # the return value's size is not exactly the specified size.
        from cudf.core import Buffer
        from cupy.cuda import MemoryPointer
        from cupy.cuda.memory import UnownedMemory

        if not self._has_read_headers:
            self._has_read_headers = True
            return pickle.dumps(self._headers)
        if len(self._buffers) == 0:
            return ''
        cur_buf = self._buffers[0]
        # current buf read to end
        if cur_buf is None:
            # empty cuda buffer
            content = Buffer.empty(0)
            self._offset = 0
            self._buffers.pop(0)
            return content
        elif size >= cur_buf.size - self._offset:
            if isinstance(cur_buf, UnownedMemory):
                cupy_pointer = MemoryPointer(cur_buf, self._offset)
                content = Buffer(cupy_pointer.ptr, size=cur_buf.size - self._offset)
            else:
                content = cur_buf[self._offset: self._offset + size]
            self._offset = 0
            self._buffers.pop(0)
            return content
        else:
            if isinstance(cur_buf, UnownedMemory):
                cupy_pointer = MemoryPointer(cur_buf, self._offset)
                self._offset += size
                return Buffer(cupy_pointer.ptr, size=size)
            else:
                self._offset += size
                return cur_buf[self._offset, self._offset + size]

    def write(self, content):
        from cudf.core import Buffer
        from cupy.cuda import MemoryPointer
        from cupy.cuda.memory import UnownedMemory

        if not self._has_write_headers:
            self._headers = headers = pickle.loads(content)
            buffer_types = headers['buffer_types']
            for buffer_type, size in buffer_types:
                if buffer_type == 'cuda':
                    self._buffers.append(Buffer.empty(size))
                else:
                    self._buffers.append(BytesIO())
            self._has_write_headers = True
            return

        cur_buf = self._buffers[self._cur_buffer_index]
        cur_buf_size = self._headers['buffer_types'][self._cur_buffer_index][1]
        if isinstance(cur_buf, Buffer):
            cur_cupy_memory = UnownedMemory(cur_buf.ptr, len(cur_buf), cur_buf)
            cupy_pointer = MemoryPointer(cur_cupy_memory, self._offset)

            if isinstance(content, bytes):
                content_length = len(content)
                source_mem = np.frombuffer(content, dtype='uint8').ctypes.data_as(ctypes.c_void_p)
            else:
                source_mem = MemoryPointer(UnownedMemory(content.ptr, len(content), content), 0)
                content_length = source_mem.mem.size
            cupy_pointer.copy_from(source_mem, content_length)
        else:
            content_length = len(content)
            cur_buf.write(content)
        if content_length + self._offset >= cur_buf_size:
            if isinstance(cur_buf, BytesIO):
                self._buffers[self._cur_buffer_index] = cur_buf.getvalue()
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
        headers = self._headers
        headers.pop('buffer_types')
        # hold cuda buffers

        _id_to_buffers[self._object_id] = headers, self._buffers

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
    async def get(self, object_id: str, **kwargs) -> object:
        from cudf.core import Buffer
        from rmm import DeviceBuffer

        headers, buffers = _id_to_buffers[object_id]
        new_buffers = []
        for buf in buffers:
            if isinstance(buf, cupy.ndarray):
                new_buffers.append(DeviceBuffer(ptr=buf.data.ptr, size=buf.size))
            elif isinstance(buf, cudf.core.Buffer):
                new_buffers.append(Buffer(buf.ptr, buf.size,
                                          DeviceBuffer(ptr=buf.ptr, size=buf.size)))
            else:
                new_buffers.append(buf)
        return deserialize(headers, new_buffers)

    @implements(StorageBackend.put)
    async def put(self, obj, importance=0) -> ObjectInfo:
        string_id = str(uuid.uuid4())
        headers, buffers = serialize(obj)
        size = sum(buf.size for buf in buffers if
                   isinstance(buf, (cupy.ndarray, cudf.core.Buffer)))
        _id_to_buffers[string_id] = headers, buffers
        return ObjectInfo(size=size, object_id=string_id)

    @implements(StorageBackend.delete)
    async def delete(self, object_id: str):
        if object_id in _id_to_buffers:
            del _id_to_buffers[object_id]

    @implements(StorageBackend.object_info)
    async def object_info(self, object_id: str) -> ObjectInfo:
        size = sum(buf.size for buf in _id_to_buffers[object_id][1]
                   if isinstance(buf, (cupy.ndarray, cudf.core.Buffer)))
        return ObjectInfo(size=size, object_id=object_id)

    @implements(StorageBackend.open_writer)
    async def open_writer(self, size=None) -> StorageFileObject:
        object_id = str(uuid.uuid4())
        cuda_writer = CudaFileObject(object_id=object_id, mode='w', size=size)
        return StorageFileObject(cuda_writer, object_id=object_id)

    @implements(StorageBackend.open_reader)
    async def open_reader(self, object_id) -> StorageFileObject:
        cuda_reader = CudaFileObject(mode='r', object_id=object_id)
        return StorageFileObject(cuda_reader, object_id=object_id)

    @implements(StorageBackend.list)
    async def list(self) -> List:  # pragma: no cover
        raise NotImplementedError("Cuda storage doesn't support `list` method.")
