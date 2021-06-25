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

from collections import OrderedDict
from functools import reduce

import pandas as pd

from ..core import FuseChunkData, FuseChunk, ENTITY_TYPE, OutputType
from ..core.operand import Operand, TileableOperandMixin, Fuse, \
    ShuffleProxy, FuseChunkMixin
from ..tensor.core import TENSOR_TYPE
from ..tensor.operands import TensorOperandMixin
from ..utils import calc_nsplits
from .core import DATAFRAME_CHUNK_TYPE, SERIES_CHUNK_TYPE, INDEX_CHUNK_TYPE, \
    DATAFRAME_TYPE, SERIES_TYPE, INDEX_TYPE, DATAFRAME_GROUPBY_TYPE, \
    SERIES_GROUPBY_TYPE, CATEGORICAL_TYPE
from .utils import parse_index


class DataFrameOperandMixin(TileableOperandMixin):
    __slots__ = ()
    _op_module_ = 'dataframe'

    def new_dataframes(self, inputs, shape=None, dtypes=None, index_value=None, columns_value=None,
                       chunks=None, nsplits=None, output_limit=None, kws=None, **kw):
        setattr(self, '_output_types', [OutputType.dataframe])
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
        setattr(self, '_output_types', [OutputType.series])
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
        setattr(self, '_output_types', [OutputType.index])
        return self.new_tileables(inputs, shape=shape, dtype=dtype, index_value=index_value,
                                  name=name, chunks=chunks, nsplits=nsplits,
                                  output_limit=output_limit, kws=kws, **kw)

    def new_index(self, inputs, shape=None, dtype=None, index_value=None, name=None, **kw):
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new Index with more than 1 outputs')

        return self.new_indexes(inputs, shape=shape, dtype=dtype,
                                index_value=index_value, name=name, **kw)[0]

    def new_scalars(self, inputs, dtype=None, chunks=None, output_limit=None, kws=None, **kw):
        setattr(self, '_output_types', [OutputType.scalar])
        return self.new_tileables(inputs, shape=(), dtype=dtype, chunks=chunks, nsplits=(),
                                  output_limit=output_limit, kws=kws, **kw)

    def new_scalar(self, inputs, dtype=None, **kw):
        if getattr(self, 'output_limit') != 1:
            raise TypeError('cannot new tensor with more than 1 outputs')

        return self.new_scalars(inputs, dtype=dtype, **kw)[0]

    def new_categoricals(self, inputs, shape=None, dtype=None, categories_value=None,
                         chunks=None, nsplits=None, output_limit=None, kws=None, **kw):
        setattr(self, '_output_types', [OutputType.categorical])
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
                if isinstance(v, ENTITY_TYPE):
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
                if isinstance(v, ENTITY_TYPE):
                    inputs.append(v)
                    chunk_inputs.append(chunk_v)
        return inputs, chunk_inputs

    @classmethod
    def concat_tileable_chunks(cls, tileable):
        from .merge.concat import DataFrameConcat, GroupByConcat

        df = tileable
        assert not df.is_coarse()

        if isinstance(df, DATAFRAME_TYPE):
            chunk = DataFrameConcat(output_types=[OutputType.dataframe]).new_chunk(
                df.chunks, shape=df.shape, index=(0, 0), dtypes=df.dtypes,
                index_value=df.index_value, columns_value=df.columns_value)
            return DataFrameConcat(output_types=[OutputType.dataframe]).new_dataframe(
                [df], shape=df.shape, chunks=[chunk],
                nsplits=tuple((s,) for s in df.shape), dtypes=df.dtypes,
                index_value=df.index_value, columns_value=df.columns_value)
        elif isinstance(df, SERIES_TYPE):
            chunk = DataFrameConcat(output_types=[OutputType.series]).new_chunk(
                df.chunks, shape=df.shape, index=(0,), dtype=df.dtype,
                index_value=df.index_value, name=df.name)
            return DataFrameConcat(output_types=[OutputType.series]).new_series(
                [df], shape=df.shape, chunks=[chunk],
                nsplits=tuple((s,) for s in df.shape), dtype=df.dtype,
                index_value=df.index_value, name=df.name)
        elif isinstance(df, INDEX_TYPE):
            chunk = DataFrameConcat(output_types=[OutputType.index]).new_chunk(
                df.chunks, shape=df.shape, index=(0,), dtype=df.dtype,
                index_value=df.index_value, name=df.name)
            return DataFrameConcat(output_types=[OutputType.index]).new_index(
                [df], shape=df.shape, chunks=[chunk],
                nsplits=tuple((s,) for s in df.shape), dtype=df.dtype,
                index_value=df.index_value, name=df.name)
        elif isinstance(df, (DATAFRAME_GROUPBY_TYPE, SERIES_GROUPBY_TYPE)):
            output_type = OutputType.dataframe_groupby \
                if isinstance(df, DATAFRAME_GROUPBY_TYPE) else OutputType.series_groupby
            groupby_params = cls._process_groupby_params(df.op.groupby_params)
            inputs, chunk_inputs = cls._get_groupby_inputs(df, groupby_params)
            chunk = GroupByConcat(groups=df.chunks, groupby_params=groupby_params,
                                  output_types=[output_type]).new_chunk(
                chunk_inputs, **df.params)
            return GroupByConcat(groups=[df], groupby_params=df.op.groupby_params,
                                 output_types=[output_type]).new_tileable(
                inputs, chunks=[chunk], **df.params)
        elif isinstance(df, CATEGORICAL_TYPE):
            chunk = DataFrameConcat(output_types=[OutputType.categorical]).new_chunk(
                df.chunks, shape=df.shape, index=(0,), dtype=df.dtype,
                categories_value=df.categories_value)
            return DataFrameConcat(output_types=[OutputType.categorical]).new_categorical(
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
        elif isinstance(chunks[0], SERIES_CHUNK_TYPE):
            params = cls._calc_series_index_params(chunks)
            params.update(kw)
            return op.new_series(inputs, shape=shape, chunks=chunks,
                                 nsplits=nsplits, **params)
        else:
            assert isinstance(chunks[0], INDEX_CHUNK_TYPE)
            params = cls._calc_series_index_params(chunks)
            params.update(kw)
            return op.new_index(inputs, shape=shape, chunks=chunks,
                                nsplits=nsplits, **params)

    @classmethod
    def _calc_dataframe_params(cls, chunk_index_to_chunks, chunk_shape):
        dtypes = pd.concat([chunk_index_to_chunks[0, i].dtypes
                            for i in range(chunk_shape[1])
                            if (0, i) in chunk_index_to_chunks])
        columns_value = parse_index(dtypes.index, store_data=True)
        pd_indexes = [chunk_index_to_chunks[i, 0].index_value.to_pandas()
                      for i in range(chunk_shape[0])
                      if (i, 0) in chunk_index_to_chunks]
        pd_index = reduce(lambda x, y: x.append(y), pd_indexes)
        index_value = parse_index(pd_index)
        return {'dtypes': dtypes, 'columns_value': columns_value,
                'index_value': index_value}

    @classmethod
    def _calc_series_index_params(cls, chunks):
        pd_indexes = [c.index_value.to_pandas() for c in chunks]
        pd_index = reduce(lambda x, y: x.append(y), pd_indexes)
        index_value = parse_index(pd_index)
        return {'dtype': chunks[0].dtype, 'index_value': index_value}

    def get_fuse_op_cls(self, _):
        return DataFrameFuseChunk


DataFrameOperand = Operand


class DataFrameShuffleProxy(ShuffleProxy, DataFrameOperandMixin):
    def __init__(self, sparse=None, output_types=None, **kwargs):
        super().__init__(_sparse=sparse, _output_types=output_types, **kwargs)

    @classmethod
    def execute(cls, ctx, op):
        pass


class DataFrameFuseChunkMixin(FuseChunkMixin, DataFrameOperandMixin):
    __slots__ = ()

    def _create_chunk(self, output_idx, index, **kw):
        data = FuseChunkData(_index=index, _shape=kw.pop('shape', None), _op=self, **kw)

        return FuseChunk(data)


class DataFrameFuseChunk(Fuse, DataFrameFuseChunkMixin):
    @property
    def output_types(self):
        return self.outputs[-1].chunk.op.output_types
