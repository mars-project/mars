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
from mars.core.graph.builder import chunk

from .... import oscar as mo
from .core import Chunk


class MutableTensorChunkActor(mo.Actor):
    def __init__(self, chunklist: OrderedDict, name:str, default_value=0) -> None:
        self.idx_chunk = OrderedDict()
        self._chunk_list = chunklist
        self._name = name
        self._default_value = default_value

    async def __post_create__(self):
        from ...storage import StorageAPI
        self.storage_api = await StorageAPI.create(self._name, self.address)
        for k, v in self._chunk_list.items():
            _chunk = Chunk(k, v, self.address, self.storage_api, self._default_value)
            await _chunk.initstorage()
            self.idx_chunk[k] = _chunk

    async def __on_receive__(self, message):
        return await super().__on_receive__(message)

    async def write(self, index, relatepos, value):
        chunk: Chunk = self.idx_chunk[index]
        await chunk.write(tuple(relatepos), value)

    async def read(self, index, relatepos):
        chunk: Chunk = self.idx_chunk[index]
        result = await chunk.read(tuple(relatepos))
        return result
