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
from ....tensor.utils import decide_chunk_sizes
from ..worker.service import MutableTensorChunkActor


class MutableTensorActor(mo.Actor):
    def __init__(self,
                shape: tuple,
                dtype: str,
                chunk_size: Union[int, tuple],
                worker_pools: Dict[Tuple[str, str], int],
                name: str,
                default_value : Union[int, float]=0):
        self._shape = shape
        self._dtype = dtype
        self._work_pools = worker_pools
        self._name = name
        self._chunk_size = chunk_size
        self._chunks = []
        self.default_value = default_value
        self._chunk_to_actors = []
        self._chunkactors_lastindex = []
        self._nsplits = decide_chunk_sizes(self._shape, self._chunk_size, sys.getsizeof(int))

    async def __post_create__(self):
        await self.assign_chunks()

    async def assign_chunks(self):
        leftworker = workernumer = len(self._work_pools)
        worker_address = []
        for k in self._work_pools.items():
            worker_address.append(str(k[0][0]))
        chunknumber = 1
        num = 0
        for nsplit in self._nsplits:
            chunknumber *= len(nsplit)
        leftchunk = chunknumber
        chunk_list = OrderedDict()
        for idx in itertools.product(*(range(len(nsplit)) for nsplit in self._nsplits)):
            chunk_list[idx] = [self._nsplits[i][idx[i]] for i in range(len(idx))]
            num += 1
            leftchunk -= 1
            if (num == chunknumber//workernumer and leftworker != 1 or leftworker == 1 and leftchunk == 0):
                chunk_ref = await mo.create_actor(MutableTensorChunkActor, chunk_list, self._name, self.default_value, address=worker_address[leftworker-1])
                num = 0
                chunk_list = OrderedDict()
                leftworker -= 1
                self._chunk_to_actors.append(chunk_ref)
                pos = self.calc_index(idx)
                self._chunkactors_lastindex.append(pos)

    def calc_index(self, idx: tuple) -> int:
        pos = 0; acc = 1
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

    async def chunk_size(self) -> Union[int, tuple]:
        return self._chunk_size
