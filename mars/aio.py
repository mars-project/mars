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
import functools
from concurrent.futures import Executor
from typing import Any, Type


def _make_async_method(attr):
    async def method(self, *args, **kwargs):
        func = functools.partial(getattr(self._file, attr), *args, **kwargs)
        return await self._loop.run_in_executor(self._executor, func)

    return method


def delegate_to_executor(*attrs):
    def wrap_cls(cls: Type):
        for attr in attrs:
            setattr(cls, attr, _make_async_method(attr))
        return cls

    return wrap_cls


@delegate_to_executor(
    "close",
    "flush",
    "isatty",
    "read",
    "read1",
    "readinto",
    "readline",
    "readlines",
    "seek",
    "seekable",
    "tell",
    "truncate",
    "writable",
    "write",
    "writelines",
)
class AioFileObject:
    def __init__(self,
                 file: Any,
                 loop: asyncio.BaseEventLoop = None,
                 executor: Executor = None):
        if loop is None:
            loop = asyncio.get_event_loop()

        self._file = file
        self._loop = loop
        self._executor = executor

    def __aiter__(self):
        return self

    async def __anext__(self):
        """Simulate normal file iteration."""
        line = await self.readline()
        if line:
            return line
        else:
            raise StopAsyncIteration

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        self._file = None
