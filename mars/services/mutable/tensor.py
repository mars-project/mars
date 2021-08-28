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
from typing import List, Any, Dict, Type, Tuple
from mars.core.graph.builder import chunk
import numpy as np
import random
from ..cluster import ClusterAPI

from ... import oscar as mo


class Chunk:
    def __init__(self, index, shape) -> None:
        self._index = index
        self._shape = shape
        self._ops = []

    def set(self, index):
        self._ops.append(index)
        print('chunkactorstart index index',self._index,index)


class MutableTensorChunkActor(mo.Actor):
    def __init__(self, indice, chunklist: List[Chunk] = None) -> None:
        self._chunklist = chunklist
        self.startindice=indice

    async def __post_create__(self):
        pass

    async def __on_receive__(self, message):
        return await super().__on_receive__(message)

    async def write(self, index,indice):
        chunk=self._chunklist[indice-self.startindice]
        chunk.set(index)
        if index==(1,10001):
            print(self.startindice)


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

    async def __post_create__(self):
        return await super().__post_create__()

    def get_scope(self, chunk: Chunk):
        return (chunk._shape[0]-1)*self._shape[1]+chunk._shape[1]

    async def assign_chunks(self):
        workernumer = 3
        chunknumber = len(self._chunks)
        for i in range(workernumer):
            ref = await mo.create_actor(MutableTensorChunkActor,
                                        i*(chunknumber//workernumer),
                                        self._chunks[i*(chunknumber//workernumer):(i+1)*(chunknumber//workernumer)]
                                        if i != workernumer-1 else self._chunks[i*(chunknumber//workernumer):], address=self.address)
            self._chunk_to_actors.append(ref)
            if i != workernumer-1:
                chunk = (i+1)*(chunknumber//workernumer)-1
            else:
                chunk = chunknumber-1
            self._chunkactors_lastindex.append(chunk)

    def get_chunks(self):
        tmp = self._chunksize
        if isinstance(tmp, int):
            for lastrow in np.arange(0, self._shape[0], tmp):
                for lastcolumn in np.arange(0, self._shape[1], tmp):
                    chunk = Chunk((lastrow+1, lastcolumn+1),
                                  (min(tmp, self._shape[0]-lastrow),
                                   min(tmp, self._shape[1]-lastcolumn)))
                    self._chunks.append(chunk)

        if isinstance(tmp, tuple):
            if isinstance(tmp[0], tuple):
                # Todo under tuple in tuple situation, get chunks
                pass
            else:
                for lastrow in np.arange(0, self._shape[0], tmp[0]):
                    for lastcolumn in np.arange(0, self._shape[1], tmp[1]):
                        chunk = Chunk((lastrow+1, lastcolumn+1),
                                      (min(tmp[0], self._shape[0]-lastrow),
                                      min(tmp[1], self._shape[1]-lastcolumn)))
                        self._chunks.append(chunk)

    def judge(self,chunk:Chunk,index):
        pass
        if index[0]>chunk._index[0]+chunk._shape[0]-1:
            return False
        if index[0]<chunk._index[0]:
            return True
        if index[1]<=chunk._index[1]+chunk._shape[1]-1:
            return True
        else:
            return False
        

    async def write(self, index):
        #使用两次二分搜索
        #第一次二分找index所在的chunk
        l=-1;r=len(self._chunks)
        while r-l>1:
            mid=(l+r)//2
            if index == (1,10001):
                print(l,r,self._chunks[mid]._index,index)
            #judge：判断index是否在当前chunk或者在这个chunk之前的chunk
            #之前的chunk是基于从上到下，从左到右的顺序
            if self.judge(self._chunks[mid],index):
                r=mid
            else:
                l=mid
        indice = r
        #第二次二分找chunk所在的chunkactor
        l = -1; r = len(self._chunkactors_lastindex)
        while r-l>1:
            mid=(l+r)//2
            if self._chunkactors_lastindex[mid]<indice:
                l=mid
            else:
                r=mid

        ref = self._chunk_to_actors[r]
        if index == (1,10001):
            print("index found",self._chunks[indice]._index,indice)
        
        await ref.write(index,indice)

    def shape(self):
        return self._shape

    def dtype(self):
        return self._dtype

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