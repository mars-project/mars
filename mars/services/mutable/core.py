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


class MutableTensorInfo:
    """
    Why `MutableTensorInfo`?

    We need a cluster to transfer meta information of mutable tensor, between
    server and client, as over the HTTP web session.

    Thus we design an internal-only `MutableTensorInfo` type as a container
    for those information.

    A `MutableTensor` can be initialized from

        - a info, which contains the metadata
        - a `mutable_api`, which will be used to request the backend API
        - a `loop`, which will be used to execute `__setitem__` (and `__getitem__`)
          synchronously to make the API more user-friendly.
    """
    def __init__(self, shape, dtype, name, default_value):
        self._shape = shape
        self._dtype = dtype
        self._name = name
        self._default_value = default_value

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def name(self):
        return self._name

    @property
    def default_value(self):
        return self._default_value


class MutableTensor:
    def __init__(self, info, mutable_api, loop):
        self._info = info
        self._mutable_api = mutable_api
        self._loop = loop

    @classmethod
    def create(cls,
               info: "MutableTensorInfo",
               mutable_api,  # no type signature, to avoid cycle imports
               loop: asyncio.AbstractEventLoop) -> "MutableTensor":
        return MutableTensor(info, mutable_api, loop)

    @property
    def shape(self):
        return self._info.shape

    @property
    def dtype(self):
        return self._info.dtype

    @property
    def name(self):
        return self._info.name

    @property
    def default_value(self):
        return self._info.default_value

    async def read(self, index, timestamp=None):
        return await self._mutable_api.read(self.name, index, timestamp)

    async def write(self, index, value, timestamp=None):
        return await self._mutable_api.write(self.name, index, value, timestamp)

    def __getitem__(self, index):
        coro = self.read(index)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    def __setitem__(self, index, value):
        coro = self.write(index, value)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    async def seal(self, timestamp=None):
        return await self._mutable_api.seal_mutable_tensor(self.name, timestamp)
