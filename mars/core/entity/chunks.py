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

from ...serialization.serializables import FieldTypes, TupleField
from ...utils import tokenize
from .core import EntityData, Entity


class ChunkData(EntityData):
    __slots__ = ()

    # optional fields
    _index = TupleField('index', FieldTypes.uint32)

    def __repr__(self):
        if self.op.stage is None:
            return f'Chunk <op={type(self.op).__name__}, ' \
                   f'key={self.key}>'
        else:
            return f'Chunk <op={type(self.op).__name__}, ' \
                   f'stage={self.op.stage.name}, key={self.key}>'

    @property
    def index(self):
        return getattr(self, '_index', None)

    @property
    def device(self):
        return self.op.device

    def _update_key(self):
        object.__setattr__(self, '_key', tokenize(
            type(self).__name__, *(getattr(self, k, None) for k in self._keys_
                                   if k != '_index')))


class Chunk(Entity):
    _allow_data_type_ = (ChunkData,)


CHUNK_TYPE = (Chunk, ChunkData)
