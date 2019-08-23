# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

import operator

import numpy as np

from ..compat import Enum
from ..serialize import Int8Field
from ..operands import ShuffleProxy
from ..core import TileableOperandMixin, FuseChunkData, FuseChunk
from ..operands import Operand, ShuffleMap, ShuffleReduce, Fuse
from ..tensor.core import TENSOR_TYPE, CHUNK_TYPE as TENSOR_CHUNK_TYPE
from .core import DATAFRAME_CHUNK_TYPE, SERIES_CHUNK_TYPE, INDEX_CHUNK_TYPE, \
    DATAFRAME_TYPE, SERIES_TYPE, INDEX_TYPE


class ObjectType(Enum):
    dataframe = 1
    series = 2
    index = 3
    scalar = 4


class DataFrameOperandMixin(TileableOperandMixin):
    __slots__ = ()
    _op_module_ = 'dataframe'

    _OBJECT_TYPE_TO_CHUNK_TYPES = {
        ObjectType.dataframe: DATAFRAME_CHUNK_TYPE,
        ObjectType.series: SERIES_CHUNK_TYPE,
        ObjectType.index: INDEX_CHUNK_TYPE,
        ObjectType.scalar: TENSOR_CHUNK_TYPE
    }

    _OBJECT_TYPE_TO_TILEABLE_TYPES = {
        ObjectType.dataframe: DATAFRAME_TYPE,
        ObjectType.series: SERIES_TYPE,
        ObjectType.index: INDEX_TYPE,
        ObjectType.scalar: TENSOR_TYPE
    }

    @classmethod
    def _chunk_types(cls, object_type):
        return cls._OBJECT_TYPE_TO_CHUNK_TYPES[object_type]

    @classmethod
    def _tileable_types(cls, object_type):
        return cls._OBJECT_TYPE_TO_TILEABLE_TYPES[object_type]

    def _create_chunk(self, output_idx, index, **kw):
        object_type = getattr(self, '_object_type', None)
        if object_type is None:
            raise ValueError('object_type should be specified')
        chunk_type, chunk_data_type = self._chunk_types(object_type)
        kw['op'] = self
        kw['index'] = index
        data = chunk_data_type(**kw)
        return chunk_type(data)

    def _create_tileable(self, output_idx, **kw):
        object_type = getattr(self, '_object_type', None)
        if object_type is None:
            raise ValueError('object_type should be specified')
        tileable_type, tileable_data_type = self._tileable_types(object_type)
        kw['op'] = self
        data = tileable_data_type(**kw)
        return tileable_type(data)

    def new_dataframes(self, inputs, shape=None, dtypes=None, index_value=None, columns_value=None,
                       chunks=None, nsplits=None, output_limit=None, kws=None, **kw):
        return self.new_tileables(inputs, shape=shape, dtypes=dtypes, index_value=index_value,
                                  columns_value=columns_value, chunks=chunks, nsplits=nsplits,
                                  output_limit=output_limit, kws=kws, **kw)

    def new_dataframe(self, inputs, shape=None, dtypes=None, index_value=None, columns_value=None, **kw):
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new DataFrame with more than 1 outputs')

        return self.new_dataframes(inputs, shape=shape, dtypes=dtypes,
                                   index_value=index_value, columns_value=columns_value, **kw)[0]

    def new_seriess(self, inputs, shape=None, dtype=None, index_value=None, name=None,
                    chunks=None, nsplits=None, output_limit=None, kws=None, **kw):
        return self.new_tileables(inputs, shape=shape, dtype=dtype, index_value=index_value,
                                  name=name, chunks=chunks, nsplits=nsplits,
                                  output_limit=output_limit, kws=kws, **kw)

    def new_series(self, inputs, shape=None, dtype=None, index_value=None, name=None, **kw):
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new tensor with more than 1 outputs')

        return self.new_seriess(inputs, shape=shape, dtype=dtype,
                                index_value=index_value, name=name, **kw)[0]

    def new_scalars(self, inputs, dtype=None, chunks=None, output_limit=None, kws=None, **kw):
        return self.new_tileables(inputs, shape=(), dtype=dtype, chunks=chunks, nsplits=(),
                                  output_limit=output_limit, kws=kws, **kw)

    def new_scalar(self, inputs, dtype=None, **kw):
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new tensor with more than 1 outputs')

        return self.new_scalars(inputs, dtype=dtype, **kw)[0]

    @staticmethod
    def _merge_shape(*shapes):
        ret = [np.nan, np.nan]
        for shape in shapes:
            for i, s in enumerate(shape):
                if np.isnan(ret[i]) and not np.isnan(s):
                    ret[i] = s
        return tuple(ret)


class DataFrameOperand(Operand):
    _object_type = Int8Field('object_type', on_serialize=operator.attrgetter('value'),
                             on_deserialize=ObjectType)

    @property
    def object_type(self):
        return self._object_type


class DataFrameShuffleProxy(ShuffleProxy, DataFrameOperandMixin):
    _object_type = Int8Field('object_type', on_serialize=operator.attrgetter('value'),
                             on_deserialize=ObjectType)

    def __init__(self, object_type=None, sparse=None, **kwargs):
        super(DataFrameShuffleProxy, self).__init__(_object_type=object_type,
                                                    _sparse=sparse, **kwargs)

    @property
    def object_type(self):
        return self._object_type


class DataFrameShuffleMap(ShuffleMap):
    _object_type = Int8Field('object_type', on_serialize=operator.attrgetter('value'),
                             on_deserialize=ObjectType)

    @property
    def object_type(self):
        return self._object_type


class DataFrameShuffleReduce(ShuffleReduce):
    _object_type = Int8Field('object_type', on_serialize=operator.attrgetter('value'),
                             on_deserialize=ObjectType)

    @property
    def object_type(self):
        return self._object_type


class DataFrameFuseMixin(DataFrameOperandMixin):
    __slots__ = ()

    def _create_chunk(self, output_idx, index, **kw):
        data = FuseChunkData(_index=index, _shape=kw.pop('shape', None), _op=self, **kw)

        return FuseChunk(data)


class DataFrameFuseChunk(Fuse, DataFrameFuseMixin):
    def __init__(self, sparse=False, **kwargs):
        super(DataFrameFuseChunk, self).__init__(_sparse=sparse, **kwargs)

    @property
    def object_type(self):
        return self._operands[-1].object_type
