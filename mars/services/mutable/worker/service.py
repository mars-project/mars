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
from typing import OrderedDict
from .... import oscar as mo
from .core import Chunk

class MutableTensorChunkActor(mo.Actor):
    def __init__(self, chunklist: OrderedDict, value=None) -> None:
        self.idx_chunk = OrderedDict()
        for k,v in chunklist.items():
            self.idx_chunk[k] = Chunk(v,value)

    async def __post_create__(self):
        pass

    async def __on_receive__(self, message):
        return await super().__on_receive__(message)

    async def write(self, index,relatepos,value):
        chunk:Chunk = self.idx_chunk[index]
        chunk.write(tuple(relatepos),value)

    async def read(self, index, relatepos):
        chunk:Chunk = self.idx_chunk[index]
        return chunk.read(tuple(relatepos))