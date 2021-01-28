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
from typing import Union, Dict, List
import uuid

from ..serialize import dataserializer
from ..utils import mod_hash
from .core import StorageBackend, FileObject, ObjectInfo


class FileSystemStorage(StorageBackend):
    def __init__(self, fs=None, root_dirs=None, level=None):
        self._fs = fs
        self._root_dirs = root_dirs
        self._level = level

    @classmethod
    async def setup(cls, **kwargs) -> Union[Dict, None]:
        fs = kwargs.get('fs')
        root_dirs = kwargs.get('root_dirs')
        level = kwargs.get('level')
        for d in root_dirs:
            if not fs.exists(d):
                fs.mkdir(d)
        return dict(fs=fs, root_dirs=root_dirs, level=level)

    @staticmethod
    async def teardown(**kwargs) -> None:
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
        bytes_object = self._fs.open(object_id, 'rb').read()
        return dataserializer.loads(bytes_object)

    async def put(self, obj, importance=0) -> ObjectInfo:
        path = self._generate_path()
        bytes_object = dataserializer.dumps(obj)
        with self._fs.open(path, 'wb') as f:
            f.write(bytes_object)
            size = f.tell()
        return ObjectInfo(size=size, device='disk', object_id=path)

    async def delete(self, object_id):
        os.remove(object_id)

    async def list(self) -> List:
        file_list = []
        for d in self._root_dirs:
            file_list.extend(list(self._fs.ls(d)))
        return file_list

    async def object_info(self, object_id) -> ObjectInfo:
        size = self._fs.stat(object_id)['size']
        return ObjectInfo(size=size, device='disk', object_id=object_id)

    async def create_writer(self, size=None) -> FileObject:
        path = self._generate_path()
        return FileObject(self._fs.open(path, 'wb'))

    async def open_reader(self, object_id) -> FileObject:
        return FileObject(self._fs.open(object_id, 'rb'))
