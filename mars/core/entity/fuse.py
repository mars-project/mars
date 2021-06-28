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

import numpy as np

from ...serialization.serializables import ReferenceField
from .chunks import ChunkData, Chunk, CHUNK_TYPE


class FuseChunkData(ChunkData):
    __slots__ = '_inited',

    _chunk = ReferenceField('chunk', CHUNK_TYPE,
                            on_serialize=lambda x: x.data if hasattr(x, 'data') else x)

    def __init__(self, *args, **kwargs):
        self._inited = False
        super().__init__(*args, **kwargs)
        self._extra_params = {}
        self._inited = True

    @property
    def chunk(self):
        return self._chunk

    @property
    def composed(self):
        # for compatibility, just return the topological ordering,
        # once we apply optimization on the subgraph,
        # `composed` is not needed any more and should be removed then.
        assert getattr(self._op, 'fuse_graph', None) is not None
        fuse_graph = self._op.fuse_graph
        return list(fuse_graph.topological_iter())

    def __getattr__(self, attr):
        if not self._inited:
            return object.__getattribute__(self, attr)
        if attr in self._extra_params:
            return self._extra_params[attr]
        try:
            return getattr(self._chunk, attr)
        except AttributeError:
            return object.__getattribute__(self, attr)

    def __setattr__(self, attr, value):
        if attr == 'params':
            self._chunk.params = value
        else:
            super().__setattr__(attr, value)

    @property
    def nbytes(self):
        return np.prod(self.shape) * self.dtype.itemsize


class FuseChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (FuseChunkData,)


FUSE_CHUNK_TYPE = (FuseChunkData, FuseChunk)
