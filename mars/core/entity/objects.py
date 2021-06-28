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

from typing import Any, Dict

from ...serialization.serializables import FieldTypes, ListField
from .chunks import ChunkData, Chunk
from .core import Entity
from .executable import _ToObjectMixin
from .tileables import TileableData


class ObjectChunkData(ChunkData):
    # chunk whose data could be any serializable
    __slots__ = ()
    type_name = 'Object'

    def __init__(self, op=None, index=None, **kw):
        super().__init__(_op=op, _index=index, **kw)

    @property
    def params(self) -> Dict[str, Any]:
        # params return the properties which useful to rebuild a new chunk
        return {
            'index': self.index,
        }

    @params.setter
    def params(self, new_params: Dict[str, Any]):
        params = new_params.copy()
        params.pop('index', None)  # index not needed to update
        if params:  # pragma: no cover
            raise TypeError(f'Unknown params: {list(params)}')

    @classmethod
    def get_params_from_data(cls, data: Any) -> Dict[str, Any]:
        return dict()


class ObjectChunk(Chunk):
    __slots__ = ()
    _allow_data_type_ = (ObjectChunkData,)
    type_name = 'Object'


class ObjectData(TileableData, _ToObjectMixin):
    __slots__ = ()
    type_name = 'Object'

    # optional fields
    _chunks = ListField('chunks', FieldTypes.reference(ObjectChunkData),
                        on_serialize=lambda x: [it.data for it in x] if x is not None else x,
                        on_deserialize=lambda x: [ObjectChunk(it) for it in x] if x is not None else x)

    def __init__(self, op=None, nsplits=None, chunks=None, **kw):
        super().__init__(_op=op, _nsplits=nsplits, _chunks=chunks, **kw)

    def __repr__(self):
        return f'Object <op={type(self.op).__name__}, key={self.key}>'

    @property
    def params(self):
        # params return the properties which useful to rebuild a new tileable object
        return dict()

    @params.setter
    def params(self, new_params: Dict[str, Any]):
        params = new_params.copy()
        if params:  # pragma: no cover
            raise TypeError(f'Unknown params: {list(params)}')

    def refresh_params(self):
        # refresh params when chunks updated
        # nothing needs to do for Object
        pass


class Object(Entity, _ToObjectMixin):
    __slots__ = ()
    _allow_data_type_ = (ObjectData,)
    type_name = 'Object'


OBJECT_TYPE = (Object, ObjectData)
OBJECT_CHUNK_TYPE = (ObjectChunk, ObjectChunkData)
