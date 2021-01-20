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

from abc import ABC, abstractmethod
from typing import Union, Dict

from enum import Enum


class StorageLevel(Enum):
    GPU = 1 << 0
    MEMORY = 1 << 1
    DISK = 1 << 2
    REMOTE = 1 << 3

    def __or__(self, other: "StorageLevel"):
        return self.value | other.value


class ObjectInfo:
    def __init__(self, size=None, device=None, object_id=None):
        self._size = size
        self._device = device
        self._object_id = object_id

    @property
    def size(self):
        return self._size

    @property
    def device(self):
        return self._device

    @property
    def object_id(self):
        return self._object_id


class FileObject:
    def __init__(self, file_obj, object_id=None):
        self._file_obj = file_obj
        self._object_id = object_id

    @property
    def object_id(self):
        return self._object_id or self._file_obj.name

    def __getattr__(self, item):
        return getattr(self._file_obj, item)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._file_obj.__exit__(exc_type, exc_val, exc_tb)


class StorageBackend(ABC):
    @classmethod
    def setup(cls, **kwargs) -> Union[Dict, None]:
        """
        Setup environments, for example, start plasma store for plasma backend.
        :return: parameters for initialization.
        """
        pass

    @classmethod
    def teardown(cls, **kwargs) -> None:
        """
        Clean up the environments.
        :param kwargs: parameters for clean up.
        :return: None
        """
        pass

    @property
    def level(self):
        raise NotImplementedError

    @abstractmethod
    def get(self, object_id, **kwarg) -> object:
        """
        Get object by key. For some backends,
        `columns` or `slice` can pass to get part of data.
        :param object_id: object id
        :return: Object
        """
        pass

    @abstractmethod
    def put(self, obj, importance=0) -> ObjectInfo:
        """
        Put object into storage with object_id.
        :param obj: object to put
        :param importance: the priority to spill when storage is full
        :return: object information including size, raw_size, device
        """
        pass

    @abstractmethod
    def delete(self, object_id):
        """
        Delete object from storage by object_id.
        :param object_id: object id
        :return: None
        """
        pass

    def migrate(self, object_id, destination, device=None):
        """
        Migrating object from local to other worker.
        :param object_id: object id.
        :param destination: target worker.
        :param device: device for store.
        :return: None
        """
        pass

    @abstractmethod
    def info(self, object_id) -> ObjectInfo:
        """
        Get information about stored object.
        :param object_id: object id
        :return: object info including size and device
        """
        pass

    @abstractmethod
    def create_writer(self, size=None) -> FileObject:
        """
        Return a file-like object for writing.
        :param size: maximum size in bytes
        :return: file-like object
        """
        pass

    @abstractmethod
    def open_reader(self, object_id) -> FileObject:
        """
        Return a file-like object for reading.
        :param object_id: object id
        :return: file-like object
        """
        pass

    def pin(self, object_id):
        """
        Pin the data.
        :param object_id: object id
        :return: None
        """
        pass

    def unpin(self, object_id):
        """
        Unpin the data.
        :param object_id: object id
        :return: None
        """
        pass
