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


import asyncio
import functools
from concurrent.futures import Executor
from typing import Any, Type


def _make_delegate_method(attr):
    async def method(self, *args, **kwargs):
        func = functools.partial(getattr(self._file, attr), *args, **kwargs)
        return await self._loop.run_in_executor(self._executor, func)

    return method


def _make_proxy_method(attr):
    def method(self, *args, **kwargs):
        return getattr(self._file, attr)(*args, **kwargs)

    return method


def _make_proxy_property(attr):
    def proxy_property(self):
        return getattr(self._file, attr)

    return property(proxy_property)


def delegate_to_executor(*attrs):
    def wrap_cls(cls: Type):
        for attr in attrs:
            setattr(cls, attr, _make_delegate_method(attr))
        return cls

    return wrap_cls


def proxy_method_directly(*attrs):
    def wrap_cls(cls: Type):
        for attr in attrs:
            setattr(cls, attr, _make_proxy_method(attr))
        return cls

    return wrap_cls


def proxy_property_directly(*attrs):
    def wrap_cls(cls):
        for attr in attrs:
            setattr(cls, attr, _make_proxy_property(attr))
        return cls

    return wrap_cls


class AioBase:
    def __init__(self,
                 file: Any,
                 loop: asyncio.BaseEventLoop = None,
                 executor: Executor = None):
        if loop is None:
            loop = asyncio.get_event_loop()
        if isinstance(file, AioBase):
            file = file._file

        self._file = file
        self._loop = loop
        self._executor = executor
