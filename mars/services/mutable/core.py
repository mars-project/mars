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
    def create(
        cls,
        info: "MutableTensorInfo",
        mutable_api,  # no type signature, to avoid cycle imports
        loop: asyncio.AbstractEventLoop,
    ) -> "MutableTensor":
        return MutableTensor(info, mutable_api, loop)

    @property
    def shape(self):
        """
        Get the shape the mutable tensor.

        Returns
        -------
            Tuple
        """
        return self._info.shape

    @property
    def dtype(self):
        """
        Get the dtype the mutable tensor.

        Returns
        -------
            np.dtype or str
        """
        return self._info.dtype

    @property
    def name(self):
        """
        Get the dtype the mutable tensor.

        Returns
        -------
            str
        """
        return self._info.name

    @property
    def default_value(self):
        """
        Get the dtype the mutable tensor.

        Returns
        -------
            int or float
        """
        return self._info.default_value

    async def read(self, index, timestamp=None):
        """
        Read value from mutable tensor.

        Parameters
        ----------
        index:
            Index to read from the tensor.

        timestamp: optional
            Timestamp to read value that happened before then.
        """
        return await self._mutable_api.read(self.name, index, timestamp)

    async def write(self, index, value, timestamp=None):
        """
        Write value to mutable tensor.

        Parameters
        ----------
        index:
            Index to write to the tensor.

        value:
            The value that will be filled into the mutable tensor according to `index`.

        timestamp: optional
            Timestamp to associated with the newly touched value.
        """
        return await self._mutable_api.write(self.name, index, value, timestamp)

    def __getitem__(self, index):
        """
        Read value from mutable tensor with a synchronous API.

        Parameters
        ----------
        index:
            Index to read from the tensor.
        """
        coro = self.read(index)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    def __setitem__(self, index, value):
        """
        Write value to mutable tensor with a synchronous API.

        Parameters
        ----------
        index:
            Index to write to the tensor.

        value:
            The value that will be filled into the mutable tensor according to `index`.
        """
        coro = self.write(index, value)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    async def seal(self, timestamp=None):
        """
        Seal the mutable tensor by name.

        Parameters
        ----------
        timestamp: optional
            Operations that happened before timestamp will be sealed, and later ones will be discard.

        Returns
        -------
            object
        """
        return await self._mutable_api.seal_mutable_tensor(self.name, timestamp)
