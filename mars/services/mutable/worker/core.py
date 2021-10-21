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

import bisect
from collections import defaultdict
import sys
from typing import List, Union

import numpy as np

from .... import oscar as mo
from ....typing import ChunkType


class MutableTensorChunkActor(mo.Actor):
    def __init__(
        self,
        session_id: str,
        manager_address: str,
        chunks: List,
        dtype: Union[np.dtype, str],
        default_value: Union[int, float] = 0,
    ) -> None:
        self._session_id = session_id
        self._manager_address = manager_address
        self._chunks = chunks
        self._dtype = dtype
        self._default_value = default_value

        self._storage_api = None
        self._meta_api = None

        self._index_to_chunk = None

    @classmethod
    def gen_uid(cls, session_id: str, name: str, index: int):
        return f"mutable-tensor-chunk-{session_id}-{name}-{index}"

    async def __post_create__(self):
        from ...storage import StorageAPI
        from ...meta import MetaAPI

        self._storage_api = await StorageAPI.create(self._session_id, self.address)
        self._meta_api = await MetaAPI.create(self._session_id, self._manager_address)

        self._index_to_chunk = {
            chunk.index: MutableTensorChunk(
                chunk,
                self._manager_address,
                self.address,
                default_value=self._default_value,
            )
            for chunk in self._chunks
        }

    async def write(self, chunk_index, records):
        chunk: MutableTensorChunk = self._index_to_chunk[chunk_index]
        await chunk.write(records)

    async def read(self, chunk_index, records, chunk_value_shape, timestamp):
        chunk: MutableTensorChunk = self._index_to_chunk[chunk_index]
        return await chunk.read(records, chunk_value_shape, timestamp)

    async def seal(self, timestamp):
        for _, chunk in self._index_to_chunk.items():
            chunk_data = await chunk.seal(timestamp)
            await self._storage_api.put(chunk.chunk.key, chunk_data)
            await self._meta_api.set_chunk_meta(
                chunk.chunk, bands=[(self.address, "numa-0")]
            )


class MutableTensorChunk:
    def __init__(
        self,
        chunk: ChunkType,
        manager_address: str,
        worker_address: str,
        default_value: Union[int, float] = 0,
    ) -> None:
        self._chunk = chunk
        self._manager_address = manager_address
        self._worker_address = worker_address
        self._default_value = default_value

        self._records = defaultdict(list)

    @property
    def chunk(self):
        return self._chunk

    async def write(self, records):
        for flat_index, value, ts in records:
            self._records[flat_index].append((ts, value))

    async def read(self, records, chunk_value_shape, timestamp):
        result = np.full(shape=chunk_value_shape, fill_value=self._default_value)
        for flat_index, value_index in records:
            if flat_index not in self._records:
                continue
            # Find the newest one.
            #
            # FIXME Python doesn't have things like SortedDict or SortedList,
            # we trigger a `sorted` here to ensure the correct semantic and try
            # to be as efficient as possible.
            self._records[flat_index].sort()
            # bitsect will compare on first element in the tuple.
            index = bisect.bisect_right(
                self._records[flat_index], (timestamp, sys.float_info.max)
            )
            if index == 0:
                continue
            result[value_index] = self._records[flat_index][index - 1][
                1
            ]  # take the value
        return result

    async def seal(self, timestamp):
        result = np.full(self._chunk.shape, self._default_value)
        for flat_index, values in self._records.items():
            if flat_index not in self._records:
                continue
            # compute value
            values.sort()
            index = bisect.bisect_right(values, (timestamp, sys.float_info.max))
            if index == 0:
                continue
            # compute value index
            value_index = np.unravel_index(flat_index, self._chunk.shape)
            result[value_index] = values[index - 1][1]  # take the value
        return result
