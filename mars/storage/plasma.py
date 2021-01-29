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

import psutil
import struct
from io import BytesIO
from typing import Union, Dict, List

import pyarrow as pa
from pyarrow import plasma

from ..serialization import serialize, deserialize, serialize_header, deserialize_header
from ..serialization.core import HEADER_LENGTH
from .core import StorageBackend, StorageLevel, ObjectInfo, FileObject


PAGE_SIZE = 64 * 1024


class PlasmaFileObject:
    def __init__(self, plasma_client, object_id, mode='w', size=None):
        self._plasma_client = plasma_client
        self._object_id = object_id
        self._offset = 0
        self._size = size
        self._is_readable = 'r' in mode
        self._is_writable = 'w' in mode
        self._closed = False

        if self._is_writable:
            buf = self._plasma_client.create(object_id, size)
            self._buf = pa.FixedSizeBufferWriter(buf)
            self._buf.set_memcopy_threads(6)
        elif self._is_readable:
            [self._buf] = self._plasma_client.get_buffers([object_id])
            self._mv = memoryview(self._buf)
            self._size = len(self._buf)
        else:  # pragma: no cover
            raise NotImplementedError

    @property
    def size(self):
        return self._size

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
        return self._buf.write(d)

    def close(self):
        if self._closed:
            return

        if self._is_writable:
            self._plasma_client.seal(self._object_id)

        self._mv = None
        self._buf = None


def get_actual_capacity(plasma_client):
    """
    Get actual capacity of plasma store
    :return: actual storage size in bytes
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


class PlasmaStorage(StorageBackend):
    def __init__(self, plasma_socket=None, plasma_directory=None, capacity=None, plasma_store=None):
        self._client = plasma.connect(plasma_socket)
        self._plasma_directory = plasma_directory
        self._capacity = capacity
        self._plasma_store = plasma_store

    @classmethod
    async def setup(cls, **kwargs) -> Union[Dict, None]:
        store_memory = kwargs.get('store_memory')
        plasma_directory = kwargs.get('plasma_directory')

        plasma_store = plasma.start_plasma_store(store_memory,
                                                 plasma_directory=plasma_directory)
        params = dict(plasma_socket=plasma_store.__enter__()[0],
                      plasma_directory=plasma_directory,
                      plasma_store=plasma_store)
        client = plasma.connect(params['plasma_socket'])
        actual_capacity = get_actual_capacity(client)
        params['capacity'] = actual_capacity
        return params

    @staticmethod
    async def teardown(**kwargs) -> None:
        plasma_store = kwargs.get('plasma_store')
        plasma_store.__exit__(None, None, None)

    @property
    def level(self):
        return StorageLevel.MEMORY

    async def _check_plasma_limit(self, size):
        used_size = psutil.disk_usage(self._plasma_directory).used
        totol = psutil.disk_usage(self._plasma_directory).total
        if used_size + size > totol * 0.95:  # pragma: no cover
            raise plasma.PlasmaStoreFull

    def _generate_object_id(self):
        while True:
            new_id = plasma.ObjectID.from_random()
            if not self._client.contains(new_id):
                return new_id

    async def get(self, object_id, **kwargs) -> object:
        [buf] = self._client.get_buffers([object_id])
        length, = struct.unpack('<Q', buf[2:HEADER_LENGTH])
        header, buf_lengths = deserialize_header(buf[HEADER_LENGTH: HEADER_LENGTH + length])
        buffers = []
        start = HEADER_LENGTH + length
        for l in buf_lengths:
            buffers.append(buf[start: start + l])
            start += l
        return deserialize(header, buffers)

    async def put(self, obj, importance=0) -> ObjectInfo:
        sio = BytesIO()
        new_id = self._generate_object_id()
        serialized = serialize(obj)
        header_bytes = serialize_header(serialized)
        _, buffers = serialized
        buffer_length = sum([getattr(b, 'nbytes', len(b)) for b in buffers])
        # reserve one byte for compress information
        sio.write(struct.pack('<H', 0))
        # header length
        sio.write(struct.pack('<Q', len(header_bytes)))
        sio.write(header_bytes)
        header_buf = sio.getvalue()

        total_bytes = buffer_length + len(header_buf)
        await self._check_plasma_limit(total_bytes)

        buffer = self._client.create(new_id, total_bytes)
        stream = pa.FixedSizeBufferWriter(buffer)
        stream.set_memcopy_threads(6)
        stream.write(header_buf)
        for buf in buffers:
            stream.write(buf)
        self._client.seal(new_id)
        return ObjectInfo(size=total_bytes,
                          device='memory', object_id=new_id)

    async def delete(self, object_id):
        self._client.delete([object_id])

    async def object_info(self, object_id) -> ObjectInfo:
        [buf] = self._client.get_buffers([object_id])
        return ObjectInfo(size=buf.size, device='memory', object_id=object_id)

    async def create_writer(self, size=None) -> FileObject:
        new_id = self._generate_object_id()
        plasma_writer = PlasmaFileObject(self._client, new_id, size=size, mode='w')
        return FileObject(plasma_writer, object_id=new_id)

    async def open_reader(self, object_id) -> FileObject:
        plasma_reader = PlasmaFileObject(self._client, object_id, mode='r')
        return FileObject(plasma_reader, object_id=object_id)

    async def list(self) -> List:
        return list(self._client.list())
