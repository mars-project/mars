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

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import Executor
from typing import Any, Optional, Union

from ..lib.aio import AioFileObject


class StorageFileObject(AioFileObject):
    def __init__(self,
                 file: Any,
                 object_id: Any,
                 loop: asyncio.BaseEventLoop = None,
                 executor: Executor = None):
        self.object_id = object_id
        super().__init__(file, loop=loop, executor=executor)


class BufferWrappedFileObject(ABC):
    def __init__(self,
                 mode: str,
                 size: Optional[int] = None):
        # check arguments
        assert mode in ('w', 'r'), 'mode must be "w" or "r"'
        if mode == 'w' and size is None:  # pragma: no cover
            raise ValueError('size must be provided to write')

        self._size = size
        self._mode = mode

        self._offset = 0
        self._initialized = False
        self._closed = False

        self._mv = None
        self._buffer = None

    @abstractmethod
    def _read_init(self):
        """
        Initialization for read purpose.
        """

    @abstractmethod
    def _write_init(self):
        """
        Initialization for write purpose.
        """

    @property
    def buffer(self):
        return self._buffer

    def read(self, size=-1):
        if not self._initialized:
            self._read_init()
            self._initialized = True

        offset = self._offset
        size = self._size if size < 0 else size
        end = min(self._size, offset + size)
        result = self._mv[offset: end]
        self._offset = end
        return result

    def write(self, content: Union[bytes, memoryview]):
        if not self._initialized:
            self._write_init()
            self._initialized = True

        offset = self._offset
        content_length = getattr(content, 'nbytes', len(content))
        new_offset = offset + content_length
        self._mv[offset: new_offset] = content
        self._offset = new_offset

    @abstractmethod
    def _read_close(self):
        """
        Close for read.
        """

    @abstractmethod
    def _write_close(self):
        """
        Close for write.
        """

    def close(self):
        if self._closed:
            return

        if self._mode == 'w':
            self._write_close()
        else:
            self._read_close()
        self._mv = None
        self._buffer = None
