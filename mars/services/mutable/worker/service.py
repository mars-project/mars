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

from collections import OrderedDict
import time

from .... import oscar as mo
from .core import Chunk


class MutableTensorChunkActor(mo.Actor):
    def __init__(self, session_id, manager_address, chunklist: OrderedDict, name: str, default_value=0) -> None:
        self.idx_chunk = OrderedDict()
        self._session_id = session_id
        self._manager_address = manager_address
        self._chunk_list = chunklist
        self._name = name
        self._default_value = default_value

    async def __post_create__(self):
        from ...storage import StorageAPI
        from ...meta import MetaAPI
        self._storage_api = await StorageAPI.create(self._session_id, self.address)
        self._meta_api = await MetaAPI.create(self._session_id, self._manager_address)
        for k, v in self._chunk_list.items():
            _chunk = Chunk(k, *v, self.address, self._storage_api, self._default_value)
            self.idx_chunk[k] = _chunk

    async def __on_receive__(self, message):
        return await super().__on_receive__(message)

    async def write(self, index, relatepos, value, version_time=time.time()):
        chunk: Chunk = self.idx_chunk[index]
        await chunk.write(tuple(relatepos), value, version_time)

    async def read(self, index, relatepos, version_time=None):
        chunk: Chunk = self.idx_chunk[index]
        result = await chunk.read(tuple(relatepos), version_time)
        return result

    async def seal(self):
        for k, v in self._chunk_list.items():
            chunk_key = v[1]
            await self._meta_api.set_chunk_meta(chunk_key, bands=[(self.address, 'numa-0')])
            chunk: Chunk = self.idx_chunk[k]
            chunkdata = await chunk.seal()
            await self._storage_api.put(chunk_key.key, chunkdata)
