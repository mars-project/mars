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
import time
from typing import List, Union

from .... import oscar as mo
from ....tensor.utils import split_indexes_into_chunks, decide_chunk_sizes
from ..utils import normailize_index


class MutableTensor:
    def __init__(self, ref, shape, chunk_size, nsplits, chunk_to_actors, lastindex):
        self._ref = ref
        self._shape = shape
        self._chunk_size = chunk_size
        self._nsplits = nsplits
        self._chunk_to_actors = chunk_to_actors
        self._chunkactors_lastindex = lastindex

    @classmethod
    async def create(self, ref: mo.ActorRef) -> "MutableTensor":
        _shape = await ref.shape()
        _chunk_size = await ref.chunk_size()
        _nsplits = decide_chunk_sizes(_shape, _chunk_size, sys.getsizeof(int))
        _chunk_to_actors = await ref.chunk_to_actors()
        _chunkactors_lastindex = await ref.lastindex()
        return MutableTensor(ref, _shape, _chunk_size, _nsplits, _chunk_to_actors, _chunkactors_lastindex)

    def calc_index(self, idx: tuple) -> int:
        pos = 0; acc = 1
        for it, nsplit in zip(itertools.count(0), reversed(self._nsplits)):
            it = len(idx) - it-1
            pos += acc*(idx[it])
            acc *= len(nsplit)
        return pos

    async def __getitem__(self, index: Union[int, List[int]]):
        '''
        Function
        ----------
        read a single point of the tensor.

        Parameters:
        ----------
        index: List[int]
        If there is need to read n points of the tensor, suppose the dimension of the tensor is m.\n
        for the i_th point, it would be represented by (a[i][0],a[i][1]....a[i][m-1]),i in range(0,n)\n
        then we extract a[i][j] with the same j and put them together.\n
        It's the way to get the index for the parameter, its form is like \n
        [(a[0][0],a[1][0]...a[n-1][0]),(a[0][1],a[1][1]...a[n-1][1])...(a[0][m-1],a[1][m-1]...a[n-1][m-1])]\n
        e.g.
        Assumed the shape of index is (100,200,300) and we want to read the (0,0,0) (10,20,30) (40,50,80)\n
        the index should be ((0,10,40),(0,20,50),(0,30,80))

        Returns
        -------
        the value of the points
        '''
        index = normailize_index(index)
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
                    val = await chunk_actor.read(idx, nidx, time.time())
                    ans_list.append(val)
        return ans_list

    async def write(self, index, value, version_time=time.time()):
        '''
        Function
        ----------
        read a single point of the tensor.

        Parameters:
        ----------
        index: List[int]
        If there is need to read n points of the tensor, suppose the dimension of the tensor is m.\n
        for the i_th point, it would be represented by (a[i][0],a[i][1]....a[i][m-1]),i in range(0,n)\n
        then we extract a[i][j] with the same j and put them together.\n
        It's the way to get the index for the parameter, its form is like \n
        [(a[0][0],a[1][0]...a[n-1][0]),(a[0][1],a[1][1]...a[n-1][1])...(a[0][m-1],a[1][m-1]...a[n-1][m-1])]\n
        e.g.
        Assumed the shape of index is (100,200,300) and we want to read the (0,0,0) (10,20,30) (40,50,80)\n
        the index should be ((0,10,40),(0,20,50),(0,30,80))
        '''
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
                    await chunk_actor.write(idx, nidx, value, version_time)

    async def seal(self):
        result = await self._ref.seal()
        _name = await self._ref.name()
        await mo.destroy_actor(self._ref)
        return result
