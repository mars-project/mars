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

import pyarrow as pa
from pyarrow import plasma

from ..serialize import dataserializer
from .core import StorageBackend, StorageLevel, ObjectInfo


PAGE_SIZE = 64 * 1024


class PlasmaObjectInfo(ObjectInfo):
    def __init__(self, size=None, device=None, object_id=None):
        self._object_id = object_id
        super().__init__(size=size, device=device)

    @property
    def object_id(self):
        return self._object_id


class PlasmaFileObject:
    def __init__(self, plasma_client, mode='w'):
        self._shared_buf = None
        get_buffer = plasma_client
        self._offset = 0
        self._is_readable = 'r' in mode
        self._is_writable = 'w' in mode

        if self._is_writable:
            self._shared_buf = shared_store.create(session_id, data_key, nbytes)
            if packed:
                self._buf = ArrowBufferIO(self._shared_buf, 'w', block_size=block_size)
            else:
                import pyarrow
                self._buf = pyarrow.FixedSizeBufferWriter(self._shared_buf)
                self._buf.set_memcopy_threads(6)
        elif self._is_readable:
            self._shared_buf = get_buffer.get_buffer(session_id, data_key)
            if packed:
                self._buf = ArrowBufferIO(
                    self._shared_buf, 'r', compress_out=compress, block_size=block_size)
            else:
                self._mv = memoryview(self._shared_buf)
                self._nbytes = len(self._shared_buf)
        else:
            raise NotImplementedError

    def __del__(self):
        self._mv = None
        self._buf = self._shared_buf = None

    @property
    def nbytes(self):
        return self._nbytes

    def get_shared_buffer(self):
        return self._shared_buf

    def read(self, size=-1):
        if self._packed:
            return self._buf.read(size)
        else:
            if size < 0:
                size = self._nbytes
            right_pos = min(self._nbytes, self._offset + size)
            ret = self._mv[self._offset:right_pos]
            self._offset = right_pos
            return ret

    def write(self, d):
        return self._buf.write(d)

    def close(self, finished=True):
        if self._closed:
            return

        if self.is_writable and self._shared_buf is not None:
            self._shared_store.seal(self._session_id, self._data_key)
            if finished:
                if self._auto_register:
                    self._holder_ref.put_objects_by_keys(
                        self._session_id, [self._data_key], pin_token=self._pin_token)
            else:
                self._shared_store.delete(self._session_id, self._data_key)

        self._mv = None
        self._buf = self._shared_buf = None
        super().close(finished=finished)




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
        except plasma.PlasmaStoreFull:
            alloc_fraction *= 0.99
        finally:
            plasma_client.evict(allocate_size)
    return allocate_size


class PlasmaStore(StorageBackend):
    def __init__(self, plasma_socket, plasma_directory):
        self._client = plasma.connect(plasma_socket)
        self._plasma_directory = plasma_directory
        self._actual_capacity = get_actual_capacity(self._client)

    @classmethod
    def init(cls, store_memory=None, plasma_directory=None):

        plasma_store = plasma.start_plasma_store(store_memory,
                                                 plasma_directory=plasma_directory)
        return dict(plasma_socket=plasma_store.__enter__()[0],
                    plasma_directory=plasma_directory)

    @property
    def level(self):
        return StorageLevel.MEMORY

    def _check_plasma_limit(self, size):
        used_size = psutil.disk_usage(self._plasma_directory).used
        if used_size + size > self._actual_capacity:
            raise plasma.PlasmaStoreFull

    def get(self, object_id, **kwarg):
        return self._client.get(object_id)

    def put(self, obj, importance=0):
        while True:
            new_id = plasma.ObjectID.from_random()
            if not self._client.contains(new_id):
                break

        serialized = dataserializer.serialize(obj)
        self._check_plasma_limit(serialized.total_bytes)
        buffer = self._client.create(new_id, serialized.total_bytes)
        stream = pa.FixedSizeBufferWriter(buffer)
        stream.set_memcopy_threads(6)
        serialized.write_to(stream)
        self._client.seal(new_id)
        return PlasmaObjectInfo(size=serialized.total_bytes,
                                device='memory', object_id=new_id)

    def delete(self, object_id):
        self._client.delete(object_id)

    def info(self, object_id):
        size = self._fs.stat['size']
        return ObjectInfo(size=size, device='disk')

    def create_writer(self):
        path = self._generate_path()
        return FileObject(self._fs.open(path, 'wb'))

    def open_reader(self, object_id):
        return FileObject(self._fs.open(object_id, 'rb'))

