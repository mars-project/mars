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

import os
import uuid
from typing import Dict, List, Tuple

from ..aio import AioFilesystem
from ..serialization import AioSerializer, AioDeserializer
from ..utils import mod_hash
from .base import StorageBackend, ObjectInfo
from .core import StorageFileObject


class FileSystemStorage(StorageBackend):
    def __init__(self, fs=None, root_dirs=None, level=None):
        self._fs = AioFilesystem(fs)
        self._root_dirs = root_dirs
        self._level = level

    @classmethod
    async def setup(cls, **kwargs) -> Tuple[Dict, Dict]:
        fs = kwargs.get('fs')
        root_dirs = kwargs.get('root_dirs')
        level = kwargs.get('level')
        for d in root_dirs:
            if not fs.exists(d):
                fs.mkdir(d)
        params = dict(fs=fs, root_dirs=root_dirs, level=level)
        return params, params

    @staticmethod
    async def teardown(**kwargs):
        fs = kwargs.get('fs')
        root_dirs = kwargs.get('root_dirs')
        for d in root_dirs:
            fs.delete(d, recursive=True)

    @property
    def level(self):
        return self._level

    def _generate_path(self):
        file_name = str(uuid.uuid4())
        selected_index = mod_hash(file_name, len(self._root_dirs))
        selected_dir = self._root_dirs[selected_index]
        return os.path.join(selected_dir, file_name)

    async def get(self, object_id, **kwargs) -> object:
        file = await self._fs.open(object_id, 'rb')
        async with file as f:
            deserializer = AioDeserializer(f)
            return await deserializer.run()

    async def put(self, obj, importance=0) -> ObjectInfo:
        serializer = AioSerializer(obj)
        buffers = await serializer.run()
        buffer_size = sum(getattr(buf, 'nbytes', len(buf))
                          for buf in buffers)

        path = self._generate_path()
        file = await self._fs.open(path, 'wb')
        async with file as f:
            for buffer in buffers:
                await f.write(buffer)

        return ObjectInfo(size=buffer_size, object_id=path)

    async def delete(self, object_id):
        await self._fs.delete(object_id)

    async def list(self) -> List:
        file_list = []
        for d in self._root_dirs:
            file_list.extend(list(await self._fs.ls(d)))
        return file_list

    async def object_info(self, object_id) -> ObjectInfo:
        stat = await self._fs.stat(object_id)
        return ObjectInfo(size=stat['size'], object_id=object_id)

    async def open_writer(self, size=None) -> StorageFileObject:
        path = self._generate_path()
        file = await self._fs.open(path, 'wb')
        return StorageFileObject(file, file.name)

    async def open_reader(self, object_id) -> StorageFileObject:
        file = await self._fs.open(object_id, 'rb')
        return StorageFileObject(file, file.name)
