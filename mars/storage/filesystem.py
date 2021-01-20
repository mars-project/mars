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

from ..serialize import dataserializer
from ..utils import mod_hash
from .core import StorageBackend, FileObject, ObjectInfo, StorageLevel


class FileSystemStorage(StorageBackend):
    def __init__(self, fs, root_dirs):
        self._fs = fs
        self._root_dirs = root_dirs

    @property
    def level(self):
        return StorageLevel.DISK

    def _generate_path(self):
        file_name = str(uuid.uuid4())
        selected_index = mod_hash(file_name, len(self._root_dirs))
        selected_dir = self._root_dirs[selected_index]
        return os.path.join(selected_dir, file_name)

    def get(self, object_id, **kwarg):
        bytes_object = self._fs.open(object_id, 'rb').read()
        return dataserializer.loads(bytes_object)

    def put(self, obj, importance=0):
        path = self._generate_path()
        bytes_object = dataserializer.dumps(obj)
        with self._fs.open(path, 'wb') as f:
            f.write(bytes_object)
            size = f.tell()
        return ObjectInfo(size=size, device='disk', object_id=path)

    def delete(self, object_id):
        os.remove(object_id)

    def info(self, object_id):
        size = self._fs.stat(object_id)['size']
        return ObjectInfo(size=size, device='disk', object_id=object_id)

    def create_writer(self, size=None):
        path = self._generate_path()
        return FileObject(self._fs.open(path, 'wb'))

    def open_reader(self, object_id):
        return FileObject(self._fs.open(object_id, 'rb'))
