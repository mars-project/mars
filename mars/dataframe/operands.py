# Copyright 1999-2020 Alibaba Group Holding Ltd.
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
from collections import OrderedDict
from enum import Enum
from functools import reduce

import pandas as pd

from ..core import FuseChunkData, FuseChunk, Base, Entity
from ..operands import Operand, TileableOperandMixin, MapReduceOperand, Fuse
from ..operands import ShuffleProxy, FuseChunkMixin
from ..serialize import Int8Field, AnyField
from ..tensor.core import TENSOR_TYPE, CHUNK_TYPE as TENSOR_CHUNK_TYPE, TensorOrder
from ..tensor.operands import TensorOperandMixin
from ..utils import calc_nsplits
from .core import DATAFRAME_CHUNK_TYPE, SERIES_CHUNK_TYPE, INDEX_CHUNK_TYPE, \
    DATAFRAME_TYPE, SERIES_TYPE, INDEX_TYPE, DATAFRAME_GROUPBY_TYPE, SERIES_GROUPBY_TYPE, \
    DATAFRAME_GROUPBY_CHUNK_TYPE, SERIES_GROUPBY_CHUNK_TYPE, \
    CATEGORICAL_CHUNK_TYPE, CATEGORICAL_TYPE
from .utils import parse_index


class ObjectType(Enum):
    dataframe = 1
    series = 2
    index = 3
    scalar = 4
    tensor = 8
    dataframe_groupby = 5
    series_groupby = 6
    categorical = 7


class DataFrameOperandMixin(TileableOperandMixin):
    __slots__ = ()
    _op_module_ = 'dataframe'

    _OBJECT_TYPE_TO_CHUNK_TYPES = {
        ObjectType.dataframe: DATAFRAME_CHUNK_TYPE,
        ObjectType.series: SERIES_CHUNK_TYPE,
        ObjectType.index: INDEX_CHUNK_TYPE,
        ObjectType.scalar: TENSOR_CHUNK_TYPE,
        ObjectType.tensor: TENSOR_CHUNK_TYPE,
        ObjectType.dataframe_groupby: DATAFRAME_GROUPBY_CHUNK_TYPE,
        ObjectType.series_groupby: SERIES_GROUPBY_CHUNK_TYPE,
        ObjectType.categorical: CATEGORICAL_CHUNK_TYPE,
    }

    _OBJECT_TYPE_TO_TILEABLE_TYPES = {
        ObjectType.dataframe: DATAFRAME_TYPE,
        ObjectType.series: SERIES_TYPE,
        ObjectType.index: INDEX_TYPE,
        ObjectType.scalar: TENSOR_TYPE,
        ObjectType.tensor: TENSOR_TYPE,
        ObjectType.dataframe_groupby: DATAFRAME_GROUPBY_TYPE,
        ObjectType.series_groupby: SERIES_GROUPBY_TYPE,
        ObjectType.categorical: CATEGORICAL_TYPE,
    }

    @classmethod
    def _chunk_types(cls, object_type):
        return cls._OBJECT_TYPE_TO_CHUNK_TYPES[object_type]

    @classmethod
    def _tileable_types(cls, object_type):
        return cls._OBJECT_TYPE_TO_TILEABLE_TYPES[object_type]

    def _create_chunk(self, output_idx, index, **kw):
        object_type = kw.pop('object_type', getattr(self, '_object_type', None))
        if object_type is None:
            raise ValueError('object_type should be specified')
        if isinstance(object_type, (list, tuple)):
            object_type = object_type[output_idx]
        chunk_type, chunk_data_type = self._chunk_types(object_type)
        kw['op'] = self
        kw['index'] = index
        if object_type == ObjectType.scalar:
            # tensor
            kw['order'] = TensorOrder.C_ORDER
        data = chunk_data_type(**kw)
        return chunk_type(data)

    def _create_tileable(self, output_idx, **kw):
        object_type = kw.pop('object_type', getattr(self, '_object_type', None))
        if object_type is None:
            raise ValueError('object_type should be specified')
        if isinstance(object_type, (list, tuple)):
            object_type = object_type[output_idx]
        tileable_type, tileable_data_type = self._tileable_types(object_type)
        kw['op'] = self
        if object_type == ObjectType.scalar:
            # tensor
            kw['order'] = TensorOrder.C_ORDER
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
            raise TypeError('cannot new Series with more than 1 outputs')

        return self.new_seriess(inputs, shape=shape, dtype=dtype,
                                index_value=index_value, name=name, **kw)[0]

    def new_indexes(self, inputs, shape=None, dtype=None, index_value=None, name=None,
                    chunks=None, nsplits=None, output_limit=None, kws=None, **kw):
        return self.new_tileables(inputs, shape=shape, dtype=dtype, index_value=index_value,
                                  name=name, chunks=chunks, nsplits=nsplits,
                                  output_limit=output_limit, kws=kws, **kw)

    def new_index(self, inputs, shape=None, dtype=None, index_value=None, name=None, **kw):
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new Index with more than 1 outputs')

        return self.new_indexes(inputs, shape=shape, dtype=dtype,
                                index_value=index_value, name=name, **kw)[0]

    def new_scalars(self, inputs, dtype=None, chunks=None, output_limit=None, kws=None, **kw):
        return self.new_tileables(inputs, shape=(), dtype=dtype, chunks=chunks, nsplits=(),
                                  output_limit=output_limit, kws=kws, **kw)

    def new_scalar(self, inputs, dtype=None, **kw):
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new tensor with more than 1 outputs')

        return self.new_scalars(inputs, dtype=dtype, **kw)[0]

    def new_categoricals(self, inputs, shape=None, dtype=None, categories_value=None,
                         chunks=None, nsplits=None, output_limit=None, kws=None, **kw):
        return self.new_tileables(inputs, shape=shape, dtype=dtype,
                                  categories_value=categories_value, chunks=chunks,
                                  nsplits=nsplits, output_limit=output_limit,
                                  kws=kws, **kw)

    def new_categorical(self, inputs, shape=None, dtype=None, categories_value=None, **kw):
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new Categorical with more than 1 outputs')

        return self.new_categoricals(inputs, shape=shape, dtype=dtype,
                                     categories_value=categories_value, **kw)[0]

    @classmethod
    def _process_groupby_params(cls, groupby_params):
        new_groupby_params = groupby_params.copy()
        if isinstance(groupby_params['by'], list):
            by = []
            for v in groupby_params['by']:
                if isinstance(v, (Base, Entity)):
                    by.append(cls.concat_tileable_chunks(v).chunks[0])
                else:
                    by.append(v)
            new_groupby_params['by'] = by
        return new_groupby_params

    @classmethod
    def _get_groupby_inputs(cls, groupby, groupby_params):
        inputs = [groupby]
        chunk_inputs = list(groupby.chunks)
        if isinstance(groupby_params['by'], list):
            for chunk_v, v in zip(groupby_params['by'], groupby.op.groupby_params['by']):
                if isinstance(v, (Base, Entity)):
                    inputs.append(v)
                    chunk_inputs.append(chunk_v)
        return inputs, chunk_inputs

    @classmethod
    def concat_tileable_chunks(cls, tileable):
        from .merge.concat import DataFrameConcat, GroupByConcat

        df = tileable
        assert not df.is_coarse()

        if isinstance(df, DATAFRAME_TYPE):
            chunk = DataFrameConcat(object_type=ObjectType.dataframe).new_chunk(
                df.chunks, shape=df.shape, index=(0, 0), dtypes=df.dtypes,
                index_value=df.index_value, columns_value=df.columns_value)
            return DataFrameConcat(object_type=ObjectType.dataframe).new_dataframe(
                [df], shape=df.shape, chunks=[chunk],
                nsplits=tuple((s,) for s in df.shape), dtypes=df.dtypes,
                index_value=df.index_value, columns_value=df.columns_value)
        elif isinstance(df, SERIES_TYPE):
            chunk = DataFrameConcat(object_type=ObjectType.series).new_chunk(
                df.chunks, shape=df.shape, index=(0,), dtype=df.dtype,
                index_value=df.index_value, name=df.name)
            return DataFrameConcat(object_type=ObjectType.series).new_series(
                [df], shape=df.shape, chunks=[chunk],
                nsplits=tuple((s,) for s in df.shape), dtype=df.dtype,
                index_value=df.index_value, name=df.name)
        elif isinstance(df, INDEX_TYPE):
            chunk = DataFrameConcat(object_type=ObjectType.index).new_chunk(
                df.chunks, shape=df.shape, index=(0,), dtype=df.dtype,
                index_value=df.index_value, name=df.name)
            return DataFrameConcat(object_type=ObjectType.index).new_series(
                [df], shape=df.shape, chunks=[chunk],
                nsplits=tuple((s,) for s in df.shape), dtype=df.dtype,
                index_value=df.index_value, name=df.name)
        elif isinstance(df, (DATAFRAME_GROUPBY_TYPE, SERIES_GROUPBY_TYPE)):
            object_type = ObjectType.dataframe_groupby \
                if isinstance(df, DATAFRAME_GROUPBY_TYPE) else ObjectType.series_groupby
            groupby_params = cls._process_groupby_params(df.op.groupby_params)
            inputs, chunk_inputs = cls._get_groupby_inputs(df, groupby_params)
            chunk = GroupByConcat(groups=df.chunks, groupby_params=groupby_params,
                                  object_type=object_type).new_chunk(
                chunk_inputs, **df.params)
            return GroupByConcat(groups=[df], groupby_params=df.op.groupby_params,
                                 object_type=object_type).new_dataframe(
                inputs, chunks=[chunk], **df.params)
        elif isinstance(df, CATEGORICAL_TYPE):
            chunk = DataFrameConcat(object_type=ObjectType.categorical).new_chunk(
                df.chunks, shape=df.shape, index=(0,), dtype=df.dtype,
                categories_value=df.categories_value)
            return DataFrameConcat(object_type=ObjectType.categorical).new_categorical(
                [df], shape=df.shape, chunks=[chunk],
                nsplits=tuple((s,) for s in df.shape), dtype=df.dtype,
                categories_value=df.categories_value)
        elif isinstance(df, TENSOR_TYPE):
            return TensorOperandMixin.concat_tileable_chunks(tileable)
        else:
            raise NotImplementedError

    @classmethod
    def create_tileable_from_chunks(cls, chunks, inputs=None, **kw):
        ndim = chunks[0].ndim
        index_min, index_max = [None] * ndim, [None] * ndim
        for c in chunks:
            for ax, i in enumerate(c.index):
                if index_min[ax] is None:
                    index_min[ax] = i
                else:
                    index_min[ax] = min(i, index_min[ax])
                if index_max[ax] is None:
                    index_max[ax] = i
                else:
                    index_max[ax] = max(i, index_max[ax])

        # gen {chunk index -> shape}
        chunk_index_to_shape = OrderedDict()
        chunk_index_to_chunk = dict()
        for c in chunks:
            new_index = []
            for ax, i in enumerate(c.index):
                new_index.append(i - index_min[ax])
            chunk_index_to_shape[tuple(new_index)] = c.shape
            chunk_index_to_chunk[tuple(new_index)] = c

        nsplits = calc_nsplits(chunk_index_to_shape)
        shape = tuple(sum(ns) for ns in nsplits)
        chunk_shape = tuple(len(ns) for ns in nsplits)
        op = chunks[0].op.copy().reset_key()
        if isinstance(chunks[0], DATAFRAME_CHUNK_TYPE):
            params = cls._calc_dataframe_params(chunk_index_to_chunk, chunk_shape)
            params.update(kw)
            return op.new_dataframe(inputs, shape=shape, chunks=chunks,
                                    nsplits=nsplits, **params)
        else:
            assert isinstance(chunks[0], SERIES_CHUNK_TYPE)
            params = cls._calc_series_params(chunks)
            params.update(kw)
            return op.new_series(inputs, shape=shape, chunks=chunks,
                                 nsplits=nsplits, **params)

    @classmethod
    def _calc_dataframe_params(cls, chunk_index_to_chunks, chunk_shape):
        dtypes = pd.concat([chunk_index_to_chunks[0, i].dtypes
                            for i in range(chunk_shape[1])])
        columns_value = parse_index(dtypes.index, store_data=True)
        pd_indxes = [chunk_index_to_chunks[i, 0].index_value.to_pandas()
                     for i in range(chunk_shape[0])]
        pd_index = reduce(lambda x, y: x.append(y), pd_indxes)
        index_value = parse_index(pd_index)
        return {'dtypes': dtypes, 'columns_value': columns_value,
                'index_value': index_value}

    @classmethod
    def _calc_series_params(cls, chunks):
        pd_indexes = [c.index_value.to_pandas() for c in chunks]
        pd_index = reduce(lambda x, y: x.append(y), pd_indexes)
        index_value = parse_index(pd_index)
        return {'dtype': chunks[0].dtype, 'index_value': index_value}

    def get_fetch_op_cls(self, _):
        from ..operands import ShuffleProxy
        from .fetch import DataFrameFetchShuffle, DataFrameFetch
        if isinstance(self, ShuffleProxy):
            cls = DataFrameFetchShuffle
        else:
            cls = DataFrameFetch

        def _inner(**kw):
            return cls(object_type=self.object_type, **kw)

        return _inner

    def get_fuse_op_cls(self, _):
        return DataFrameFuseChunk


def on_serialize_object_type(object_type):
    if hasattr(object_type, 'value'):
        return object_type.value
    # otherwise, multiple object types
    return tuple(ot.value for ot in object_type)


def on_deserialize_object_type(object_type):
    if isinstance(object_type, tuple):
        return tuple(ObjectType(v) for v in object_type)
    return ObjectType(object_type)


class DataFrameOperand(Operand):
    _object_type = AnyField('object_type', on_serialize=on_serialize_object_type,
                             on_deserialize=on_deserialize_object_type)

    @property
    def object_type(self):
        return self._object_type


class DataFrameShuffleProxy(ShuffleProxy, DataFrameOperandMixin):
    _object_type = Int8Field('object_type', on_serialize=operator.attrgetter('value'),
                             on_deserialize=ObjectType)

    def __init__(self, object_type=None, sparse=None, **kwargs):
        super().__init__(_object_type=object_type, _sparse=sparse, **kwargs)

    @property
    def object_type(self):
        return self._object_type


class DataFrameMapReduceOperand(MapReduceOperand):
    _object_type = Int8Field('object_type', on_serialize=operator.attrgetter('value'),
                             on_deserialize=ObjectType)

    @property
    def object_type(self):
        return self._object_type


class DataFrameFuseChunkMixin(FuseChunkMixin, DataFrameOperandMixin):
    __slots__ = ()

    def _create_chunk(self, output_idx, index, **kw):
        data = FuseChunkData(_index=index, _shape=kw.pop('shape', None), _op=self, **kw)

        return FuseChunk(data)


class DataFrameFuseChunk(Fuse, DataFrameFuseChunkMixin):
    def __init__(self, sparse=False, **kwargs):
        super().__init__(_sparse=sparse, **kwargs)

    @property
    def object_type(self):
        return self._operands[-1].object_type
