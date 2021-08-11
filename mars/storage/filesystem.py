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

import os
import uuid
from typing import Dict, List, Optional, Tuple

from ..lib.aio import AioFilesystem
from ..lib.filesystem import FileSystem, get_fs
from ..serialization import AioSerializer, AioDeserializer
from ..utils import mod_hash, implements
from .base import StorageBackend, ObjectInfo, StorageLevel, register_storage_backend
from .core import StorageFileObject


@register_storage_backend
class FileSystemStorage(StorageBackend):
    name = 'filesystem'

    def __init__(self,
                 fs: FileSystem,
                 root_dirs: List[str],
                 level: StorageLevel,
                 size: int):
        self._fs = AioFilesystem(fs)
        self._root_dirs = root_dirs
        self._level = level
        self._size = size

    @classmethod
    @implements(StorageBackend.setup)
    async def setup(cls, **kwargs) -> Tuple[Dict, Dict]:
        root_dirs = kwargs.pop('root_dirs')
        level = kwargs.pop('level')
        size = kwargs.pop('size', None)
        fs = kwargs.pop('fs', None)
        if kwargs:  # pragma: no cover
            raise TypeError(f'FileSystemStorage got unexpected config: {",".join(kwargs)}')

        if isinstance(root_dirs, str):
            root_dirs = root_dirs.split(':')
        if isinstance(level, str):
            level = StorageLevel.from_str(level)

        if fs is None:
            fs = get_fs(root_dirs[0])

        for d in root_dirs:
            if not fs.exists(d):
                fs.mkdir(d)
        params = dict(fs=fs, root_dirs=root_dirs, level=level, size=size)
        return params, params

    @staticmethod
    @implements(StorageBackend.teardown)
    async def teardown(**kwargs):
        fs = kwargs.get('fs')
        root_dirs = kwargs.get('root_dirs')
        for d in root_dirs:
            fs.delete(d, recursive=True)

    @property
    @implements(StorageBackend.level)
    def level(self) -> StorageLevel:
        return self._level

    @property
    @implements(StorageBackend.size)
    def size(self) -> Optional[int]:
        return self._size

    def _generate_path(self):
        file_name = str(uuid.uuid4())
        selected_index = mod_hash(file_name, len(self._root_dirs))
        selected_dir = self._root_dirs[selected_index]
        return os.path.join(selected_dir, file_name)

    @implements(StorageBackend.get)
    async def get(self, object_id, **kwargs) -> object:
        if kwargs:  # pragma: no cover
            raise NotImplementedError('Got unsupported args: {",".join(kwargs)}')

        file = await self._fs.open(object_id, 'rb')
        async with file as f:
            deserializer = AioDeserializer(f)
            return await deserializer.run()

    @implements(StorageBackend.put)
    async def put(self, obj, importance: int = 0) -> ObjectInfo:
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

    @implements(StorageBackend.delete)
    async def delete(self, object_id):
        await self._fs.delete(object_id)

    @implements(StorageBackend.list)
    async def list(self) -> List:
        file_list = []
        for d in self._root_dirs:
            file_list.extend(list(await self._fs.ls(d)))
        return file_list

    @implements(StorageBackend.object_info)
    async def object_info(self, object_id) -> ObjectInfo:
        stat = await self._fs.stat(object_id)
        return ObjectInfo(size=stat['size'], object_id=object_id)

    @implements(StorageBackend.open_writer)
    async def open_writer(self, size=None) -> StorageFileObject:
        path = self._generate_path()
        file = await self._fs.open(path, 'wb')
        return StorageFileObject(file, file.name)

    @implements(StorageBackend.open_reader)
    async def open_reader(self, object_id) -> StorageFileObject:
        file = await self._fs.open(object_id, 'rb')
        return StorageFileObject(file, file.name)


@register_storage_backend
class DiskStorage(FileSystemStorage):
    name = 'disk'

    @classmethod
    @implements(StorageBackend.setup)
    async def setup(cls, **kwargs) -> Tuple[Dict, Dict]:
        kwargs['level'] = StorageLevel.DISK
        return await super().setup(**kwargs)
