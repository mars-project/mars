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
from mars.dataframe.indexing import reset_index
import sys
import itertools
from typing import List, Any, Dict, OrderedDict, Type, Tuple
from mars.core.graph.builder import chunk
from mars.tensor.utils import split_indexes_into_chunks,decide_chunk_sizes
import numpy as np
import random
from ..cluster import ClusterAPI

from ... import oscar as mo


class Chunk:
    def __init__(self, shape,value=None) -> None:
        self._shape = shape
        self._tensor = np.ones(shape)

    def write(self, index, value):
        self._tensor[index] = value
    
    def read(self, index):
        return self._tensor[index]


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

class MutableTensorActor(mo.Actor):
    def __init__(self, shape: tuple, dtype: str, chunksize, name: str = None):
        self._shape = shape
        self._dtype = dtype
        self._name = name
        self._chunksize = chunksize
        self._ops = []
        self._chunks = []
        self._chunk_to_actors = []
        self._chunkactors_lastindex = []
        self._nsplits = decide_chunk_sizes(self._shape,self._chunksize,sys.getsizeof(int))

    async def __post_create__(self):
        return await super().__post_create__()

    def get_scope(self, chunk: Chunk):
        return (chunk._shape[0]-1)*self._shape[1]+chunk._shape[1]

    async def assign_chunks(self):
        leftworker = workernumer = 3;chunknumber = 1;num = 0
        for nsplit in self._nsplits:
            chunknumber *= len(nsplit)
        leftchunk = chunknumber
        chunk_list = OrderedDict()
        for idx in itertools.product(*(range(len(nsplit)) for nsplit in self._nsplits)):
            chunk_list[idx] = [self._nsplits[i][idx[i]] for i in range(len(idx))]
            num += 1;leftchunk -= 1
            if (num == chunknumber//workernumer and  leftworker != 1  or leftworker == 1 and leftchunk == 0):
                chunk_ref = await mo.create_actor(MutableTensorChunkActor,chunk_list,address=self.address)
                num = 0;chunk_list = OrderedDict();leftworker -= 1
                self._chunk_to_actors.append(chunk_ref)
                pos = self.calc_index(idx)
                self._chunkactors_lastindex.append(pos)

    def calc_index(self,idx):
        pos = 0;acc = 1
        for it,nsplit in zip(itertools.count(0),reversed(self._nsplits)):
            it = len(idx) - it-1
            pos += acc*(idx[it])
            acc *= len(nsplit)
        return pos

    async def write(self, index,value):
        result = split_indexes_into_chunks(self._nsplits,index)
        for idx,v in result[0].items():
            if len(v[0] > 0):
                target_index = 0
                pos = self.calc_index(idx)
                for actor_index, lastindex in zip(itertools.count(0),self._chunkactors_lastindex):
                    if lastindex >= pos:
                        target_index = actor_index
                        break
                chunk_actor = self._chunk_to_actors[target_index]
                v = v.T
                for nidx in v:
                    await chunk_actor.write(idx,nidx,value)

    async def read(self, index):
        result = split_indexes_into_chunks(self._nsplits,index)
        ans=[]
        for idx,v in result[0].items():
            if len(v[0] > 0):
                target_index = 0
                pos = self.calc_index(idx)
                for actor_index, lastindex in zip(itertools.count(0),self._chunkactors_lastindex):
                    if lastindex >= pos:
                        target_index = actor_index
                        break
                chunk_actor = self._chunk_to_actors[target_index]
                v = v.T
                for nidx in v:
                    val = await chunk_actor.read(idx,nidx)
                    ans.append(val)
        return ans
        
    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def name(self):
        return self._name


class MutableTensor:
    def __init__(self,
                 ref: mo.ActorRef,
                 loop: asyncio.AbstractEventLoop):
        self._ref = ref
        self._loop = loop

    def __getattr__(self, attr):
        func = getattr(self._ref, attr)

        def wrap(*args, **kwargs):
            coro = func(*args, **kwargs)
            fut = asyncio.run_coroutine_threadsafe(coro, loop=self._loop)
            return fut.result()

        return wrap
