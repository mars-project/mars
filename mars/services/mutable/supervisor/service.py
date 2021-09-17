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
from collections import OrderedDict
from ....tensor.utils import split_indexes_into_chunks, decide_chunk_sizes
from .... import oscar as mo
from ..worker.service import MutableTensorChunkActor


class MutableTensorActor(mo.Actor):
    def __init__(self, shape: tuple, dtype: str, chunk_size, worker_pools, name: str = None, default_value=0):
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
            if (num == chunknumber//workernumer and  leftworker != 1  or leftworker == 1 and leftchunk == 0):
                chunk_ref = await mo.create_actor(MutableTensorChunkActor, chunk_list, self.default_value, address=worker_address[leftworker-1])
                num = 0
                chunk_list = OrderedDict()
                leftworker -= 1
                self._chunk_to_actors.append(chunk_ref)
                pos = self.calc_index(idx)
                self._chunkactors_lastindex.append(pos)

    def calc_index(self, idx: tuple):
        pos = 0; acc = 1
        for it, nsplit in zip(itertools.count(0), reversed(self._nsplits)):
            it = len(idx) - it-1
            pos += acc*(idx[it])
            acc *= len(nsplit)
        return pos

    async def write(self, index, value):
        result = split_indexes_into_chunks(self._nsplits, index)
        for idx, v in result[0].items():
            if len(v[0] > 0):
                target_index = 0
                pos = self.calc_index(idx)
                for actor_index, lastindex in zip(itertools.count(0), self._chunkactors_lastindex):
                    if lastindex >= pos:
                        target_index = actor_index
                        break
                chunk_actor = self._chunk_to_actors[target_index]
                v = v.T
                for nidx in v:
                    await chunk_actor.write(idx, nidx, value)

    async def read(self, index):
        result = split_indexes_into_chunks(self._nsplits, index)
        ans_list = []
        for idx, v in result[0].items():
            if len(v[0] > 0):
                target_index = 0
                pos = self.calc_index(idx)
                for actor_index, lastindex in zip(itertools.count(0), self._chunkactors_lastindex):
                    if lastindex >= pos:
                        target_index = actor_index
                        break
                chunk_actor = self._chunk_to_actors[target_index]
                v = v.T
                for nidx in v:
                    val = await chunk_actor.read(idx, nidx)
                    print(idx, nidx, val)
                    ans_list.append(val)
        return ans_list
