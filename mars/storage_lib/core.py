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

from enum import Enum
from collections import namedtuple


class StorageLevel(Enum):
    MEMORY = 1 << 0
    DISK = 1 << 1
    REMOTE = 1 << 2

    def __or__(self, other: "StorageLevel"):
        return self.value | other.value


ObjectInfo = namedtuple('size', 'device')


class StorageLib:
    __slots__ = ('_level',)

    def __init__(self, level):
        self._level = level

    def get(self, object_key, **kwarg):
        """
        Get object by key.
        :param object_key: object key
        :return: Object
        """
        raise NotImplementedError

    def put(self, obj, object_key, importance=0):
        """
        Put object into storage with object_key.
        :param obj:
        :param object_key:
        :param importance:
        :return: None
        """
        raise NotImplementedError

    def delete(self, object_key):
        """
        Delete object from storage by object_key.
        :param object_key:
        :return: None
        """
        raise NotImplementedError

    def migrate(self, object_keys, destinations):
        """
        Migrating objects from local to other workers.
        :param object_keys: list of object keys.
        :param destinations: list of target workers.
        :return: None
        """
        raise NotImplementedError

    def info(self, object_key) -> ObjectInfo:
        """
        Get information about stored object.
        :param object_key: object key
        :return: object info including size and device
        """
        raise NotImplementedError

    def create_writer(self, path):
        """
        Return a file-like object for writing.
        :param path: file path
        :return: file-like object
        """
        raise NotImplementedError

    def open_reader(self, path):
        """
        Return a file-like object for reading.
        :param path: file path
        :return: file-like object
        """
        raise NotImplementedError

    def pin(self, object_key):
        """
        Pin the data.
        :param object_key:  object key to pin the data
        :return: None
        """
        raise NotImplementedError

    def unpin(self, object_key):
        """
        Unpin the data.
        :param object_key: object key to unpin the data
        :return: None
        """
        raise NotImplementedError
