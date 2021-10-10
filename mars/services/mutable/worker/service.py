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

from typing import List, Union

import numpy as np

from .... import oscar as mo
from .core import Chunk


class MutableTensorChunkActor(mo.Actor):
    def __init__(self,
                session_id: str,
                manager_address: str,
                chunks: List,
                dtype: Union[np.dtype, str],
                default_value: Union[int, float] = 0) -> None:
        self._session_id = session_id
        self._manager_address = manager_address
        self._chunks = chunks
        self._dtype = dtype
        self._default_value = default_value

        self._index_to_chunk = None
        self._storage_api = None
        self._meta_api = None

    @classmethod
    def gen_uid(cls, name: str, index: int):
        return f'mutable-tensor-chunk-{name}-{index}'

    async def __post_create__(self):
        from ...storage import StorageAPI
        from ...meta import MetaAPI
        self._storage_api = await StorageAPI.create(self._session_id, self.address)
        self._meta_api = await MetaAPI.create(self._session_id, self._manager_address)

        self._index_to_chunk = {chunk.index: Chunk(chunk, self._manager_address, self.address,
                                                   default_value=self._default_value)
                                for chunk in self._chunks}

    async def write(self, chunk_index, records):
        chunk: Chunk = self._index_to_chunk[chunk_index]
        await chunk.write(records)

    async def read(self, chunk_index, records, chunk_value_shape, timestamp):
        chunk: Chunk = self._index_to_chunk[chunk_index]
        return await chunk.read(records, chunk_value_shape, timestamp)

    async def seal(self, timestamp):
        for _, chunk in self._index_to_chunk.items():
            chunk_data = await chunk.seal(timestamp)
            await self._storage_api.put(chunk.chunk.key, chunk_data)
            await self._meta_api.set_chunk_meta(chunk.chunk, bands=[(self.address, 'numa-0')])
