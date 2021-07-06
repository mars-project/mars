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
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import psutil
import pyarrow as pa

from ..resource import virtual_memory
from ..serialization import AioSerializer, AioDeserializer
from ..utils import implements, dataslots, calc_size_by_str, lazy_import
from .base import StorageBackend, StorageLevel, ObjectInfo, register_storage_backend
from .core import BufferWrappedFileObject, StorageFileObject
from .errors import DataNotExist

plasma = lazy_import('pyarrow.plasma', globals=globals(), rename='plasma')
if sys.platform.startswith('win'):
    plasma = None

PAGE_SIZE = 64 * 1024


class PlasmaFileObject(BufferWrappedFileObject):
    def __init__(self,
                 plasma_client: "plasma.PlasmaClient",
                 object_id: Any,
                 mode: str,
                 size: Optional[int] = None):
        self._plasma_client = plasma_client
        self._file = None
        super().__init__(object_id, mode, size=size)

    @property
    def buffer(self):
        return getattr(self, '_buffer', None)

    def _write_init(self):
        self._buffer = buf = self._plasma_client.create(self._object_id, self._size)
        file = self._file = pa.FixedSizeBufferWriter(buf)
        file.set_memcopy_threads(6)

    def _read_init(self):
        self._buffer = buf = self._plasma_client.get_buffers([self._object_id])[0]
        self._mv = memoryview(buf)
        self._size = len(buf)

    def write(self, content: bytes):
        if not self._initialized:
            self._write_init()
            self._initialized = True

        return self._file.write(content)

    def _write_close(self):
        try:
            self._plasma_client.seal(self._object_id)
        except plasma.PlasmaObjectNotFound:
            pass
        self._file = None

    def _read_close(self):
        pass


class PlasmaStorageFileObject(StorageFileObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._buffer = None

    async def close(self):
        self._buffer = self._file.buffer
        await super().close()


@dataslots
@dataclass
class PlasmaObjectInfo(ObjectInfo):
    buffer: memoryview = None
    plasma_socket: str = None

    @classmethod
    @lru_cache(5)
    def _get_plasma_client(cls, socket):
        return plasma.connect(socket)

    def __getstate__(self):
        return self.size, self.device, self.object_id, self.plasma_socket

    def __setstate__(self, state):
        self.size, self.device, self.object_id, self.plasma_socket = state
        client = self._get_plasma_client(self.plasma_socket)
        [self.buffer] = client.get_buffers([self.object_id])


def get_actual_capacity(plasma_client: "plasma.PlasmaClient") -> int:
    """
    Get actual capacity of plasma store

    Parameters
    ----------
    plasma_client: PlasmaClient
        Plasma client.

    Returns
    -------
    size: int
        Actual storage size in bytes
    """
    store_limit = plasma_client.store_capacity()

    left_size = store_limit
    alloc_fraction = 1
    while True:
        allocate_size = int(left_size * alloc_fraction / PAGE_SIZE) * PAGE_SIZE
        try:
            obj_id = plasma.ObjectID.from_random()
            buf = [plasma_client.create(obj_id, allocate_size)]
            plasma_client.seal(obj_id)
            del buf[:]
            break
        except plasma.PlasmaStoreFull:  # pragma: no cover
            alloc_fraction *= 0.99
        finally:
            plasma_client.evict(allocate_size)
    return allocate_size


@register_storage_backend
class PlasmaStorage(StorageBackend):
    name = 'plasma'

    def __init__(self,
                 plasma_socket: str = None,
                 plasma_directory: str = None,
                 capacity: int = None,
                 check_dir_size: bool = True):
        self._plasma_socket = plasma_socket
        self._client = plasma.connect(plasma_socket)
        self._plasma_directory = plasma_directory
        self._capacity = capacity
        self._check_dir_size = check_dir_size

    @classmethod
    @implements(StorageBackend.setup)
    async def setup(cls, **kwargs) -> Tuple[Dict, Dict]:
        loop = asyncio.get_running_loop()
        store_memory = kwargs.pop('store_memory')
        plasma_directory = kwargs.pop('plasma_directory', None)
        check_dir_size = kwargs.pop('check_dir_size', True)

        if kwargs:
            raise TypeError(f'PlasmaStorage got unexpected config: {",".join(kwargs)}')

        store_memory = int(calc_size_by_str(store_memory, virtual_memory().total) * 0.95)
        plasma_store = plasma.start_plasma_store(
            store_memory, plasma_directory=plasma_directory)
        plasma_socket = (await loop.run_in_executor(
            None, plasma_store.__enter__))[0]
        init_params = dict(plasma_socket=plasma_socket,
                           plasma_directory=plasma_directory,
                           check_dir_size=check_dir_size)
        client = plasma.connect(plasma_socket)
        actual_capacity = await loop.run_in_executor(
            None, get_actual_capacity, client)
        init_params['capacity'] = actual_capacity
        teardown_params = dict(plasma_store=plasma_store)
        return init_params, teardown_params

    @staticmethod
    @implements(StorageBackend.teardown)
    async def teardown(**kwargs):
        plasma_store = kwargs.get('plasma_store')
        plasma_store.__exit__(None, None, None)

    @property
    @implements(StorageBackend.level)
    def level(self) -> StorageLevel:
        return StorageLevel.MEMORY

    @property
    @implements(StorageBackend.size)
    def size(self) -> Optional[int]:
        return self._capacity

    def _check_plasma_limit(self, size: int):
        used_size = psutil.disk_usage(self._plasma_directory).used
        total = psutil.disk_usage(self._plasma_directory).total
        if used_size + size > total * 0.95:  # pragma: no cover
            raise plasma.PlasmaStoreFull

    def _generate_object_id(self):
        while True:
            new_id = plasma.ObjectID.from_random()
            if not self._client.contains(new_id):
                return new_id

    @implements(StorageBackend.get)
    async def get(self, object_id, **kwargs) -> object:
        if kwargs:  # pragma: no cover
            raise NotImplementedError('Got unsupported args: {",".join(kwargs)}')

        if not self._client.contains(object_id):  # pragma: no cover
            raise DataNotExist(f'Data {object_id} not exists')

        plasma_file = PlasmaFileObject(self._client, object_id, mode='r')

        async with StorageFileObject(plasma_file, object_id) as f:
            deserializer = AioDeserializer(f)
            return await deserializer.run()

    @implements(StorageBackend.put)
    async def put(self, obj, importance=0) -> ObjectInfo:
        object_id = self._generate_object_id()

        serializer = AioSerializer(obj)
        buffers = await serializer.run()
        buffer_size = sum(getattr(buf, 'nbytes', len(buf))
                          for buf in buffers)

        plasma_file = PlasmaFileObject(self._client, object_id,
                                       mode='w', size=buffer_size)
        async with StorageFileObject(plasma_file, object_id) as f:
            for buffer in buffers:
                await f.write(buffer)

        return PlasmaObjectInfo(size=buffer_size, object_id=object_id,
                                buffer=plasma_file.buffer,
                                plasma_socket=self._plasma_socket)

    @implements(StorageBackend.delete)
    async def delete(self, object_id):
        self._client.delete([object_id])

    @implements(StorageBackend.object_info)
    async def object_info(self, object_id) -> ObjectInfo:
        buf = self._client.get_buffers([object_id])[0]
        return PlasmaObjectInfo(size=buf.size, object_id=object_id,
                                buffer=buf, plasma_socket=self._plasma_socket)

    @implements(StorageBackend.open_writer)
    async def open_writer(self, size=None) -> StorageFileObject:
        if size is None:  # pragma: no cover
            raise ValueError('size must be provided for plasma backend')

        new_id = self._generate_object_id()
        plasma_writer = PlasmaFileObject(self._client, new_id, size=size, mode='w')
        return PlasmaStorageFileObject(plasma_writer, object_id=new_id)

    @implements(StorageBackend.open_reader)
    async def open_reader(self, object_id) -> StorageFileObject:
        if not self._client.contains(object_id):  # pragma: no cover
            raise DataNotExist(f'Data {object_id} not exists')
        plasma_reader = PlasmaFileObject(self._client, object_id, mode='r')
        return PlasmaStorageFileObject(plasma_reader, object_id=object_id)

    @implements(StorageBackend.list)
    async def list(self) -> List:
        return list(self._client.list())
