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

import sys
import itertools
from typing import List, Union, Dict, Tuple
from collections import OrderedDict

from .... import oscar as mo
from .... import tensor as mt
from ....utils import build_fetch
from ....core import tile
from ...meta import MetaAPI
from ....tensor.utils import decide_chunk_sizes
from ..worker.service import MutableTensorChunkActor


class MutableTensorActor(mo.Actor):
    def __init__(self,
                session_id: str,
                shape: tuple,
                dtype: str,
                chunk_size: Union[int, tuple],
                worker_pools: Dict[Tuple[str, str], int],
                name: str,
                default_value : Union[int, float]=0):
        self._shape = shape
        self._session_id = session_id
        self._dtype = dtype
        self._work_pools = worker_pools
        self._name = name
        self._chunk_size = chunk_size
        self._chunks = []
        self._default_value = default_value
        self._chunk_to_actors = []
        self._chunkactors_lastindex = []
        self._nsplits = decide_chunk_sizes(self._shape, self._chunk_size, sys.getsizeof(int))
        self._sealed = None

    async def __post_create__(self):
        self._meta_api = await MetaAPI.create(self._session_id, self.address)
        await self.assign_chunks()

    async def assign_chunks(self):
        worker_address = []
        for k in self._work_pools.items():
            worker_address.append(str(k[0][0]))

        tensor = build_fetch(tile(mt.random.rand(*self._shape, chunk_size=self._chunk_size, dtype='int')))
        self._sealed = tensor

        worker_number = len(self._work_pools)
        left_worker = worker_number

        chunk_number = 1
        for nsplit in self._nsplits:
            chunk_number *= len(nsplit)
        left_chunk = chunk_number

        chunks_in_list = 0
        chunk_list = OrderedDict()

        for _chunk in tensor.chunks:
            chunk_list[_chunk.index] = ([self._nsplits[i][_chunk.index[i]] for i in range(len(_chunk.index))], _chunk)
            chunks_in_list += 1
            left_chunk -= 1

            if (chunks_in_list == chunk_number//worker_number and left_worker != 1 or left_worker == 1 and left_chunk == 0):
                chunk_ref = await mo.create_actor(MutableTensorChunkActor, self._session_id, self.address, chunk_list, self._name, self._default_value, address=worker_address[left_worker-1])

                chunk_list = OrderedDict()
                chunks_in_list = 0

                left_worker -= 1

                self._chunk_to_actors.append(chunk_ref)
                self._chunkactors_lastindex.append(self.calc_index(_chunk.index))

    def calc_index(self, idx: tuple) -> int:
        pos = 0
        acc = 1
        for it, nsplit in zip(itertools.count(0), reversed(self._nsplits)):
            it = len(idx) - it-1
            pos += acc*(idx[it])
            acc *= len(nsplit)
        return pos

    async def chunk_to_actors(self) -> List[MutableTensorChunkActor]:
        return self._chunk_to_actors

    async def nsplists(self) -> List[tuple]:
        return self._nsplits

    async def lastindex(self) -> List[tuple]:
        return self._chunkactors_lastindex

    async def shape(self) -> tuple:
        return self._shape

    async def name(self) -> str:
        return self._name

    async def chunk_size(self) -> Union[int, tuple]:
        return self._chunk_size

    async def seal(self):
        for chunk_actor in self._chunk_to_actors:
            await chunk_actor.seal()
        for ref in self._chunk_to_actors:
            await mo.destroy_actor(ref)
        return self._sealed
