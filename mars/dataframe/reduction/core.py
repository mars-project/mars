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

import pandas as pd

import itertools

from ...config import options
from ...compat import reduce
from ...serialize import BoolField, AnyField, DataTypeField, Int32Field
from ..utils import parse_index, build_empty_df
from ..operands import DataFrameOperandMixin, DataFrameOperand, ObjectType
from ..merge import DataFrameConcat


class DataFrameReductionOperand(DataFrameOperand):
    _axis = AnyField('axis')
    _skipna = BoolField('skipna')
    _level = AnyField('level')

    _dtype = DataTypeField('dtype')
    _combine_size = Int32Field('combine_size')

    def __init__(self, axis=None, skipna=None, level=None, dtype=None, gpu=None, sparse=None, **kw):
        super(DataFrameReductionOperand, self).__init__(_axis=axis, _skipna=skipna, _level=level,
                                                        _dtype=dtype, _gpu=gpu, _sparse=sparse, **kw)

    @property
    def axis(self):
        return self._axis

    @property
    def skipna(self):
        return self._skipna

    @property
    def level(self):
        return self._level

    @property
    def dtype(self):
        return self._dtype

    @property
    def combine_size(self):
        return self._combine_size


class ReductionMixin(DataFrameOperandMixin):
    @classmethod
    def get_reduced_df(cls, in_df, axis, level):
        raise NotImplementedError

    @classmethod
    def single_tiles(cls, op):
        df = op.outputs[0]
        chk = op.inputs[0].chunks[0]
        reduced_df = cls.get_reduced_df(chk, axis=op.axis, level=op.level)
        new_chunk_op = op.copy().reset_key()
        index_value = getattr(reduced_df, 'index', None)
        if index_value is None:
            index_value = pd.RangeIndex(0)
        index_value = parse_index(index_value)
        chunk = new_chunk_op.new_chunk(op.inputs[0].chunks, shape=reduced_df.shape, index=chk.index,
                                       index_value=index_value)
        new_op = op.copy()
        nsplits = tuple((s,) for s in chunk.shape)
        return new_op.new_dataframes(op.inputs, df.shape, nsplits=nsplits, chunks=[chunk],
                                     index_value=df.index_value)

    @classmethod
    def convert_to_reduction_chunks(cls, op):
        chunks = []
        for c in op.inputs[0].chunks:
            new_chunk_op = op.copy().reset_key()
            reduced_df = cls.get_reduced_df(c, axis=op.axis, level=op.level)
            index_value = getattr(reduced_df, 'index', None)
            if index_value is None:
                index_value = pd.RangeIndex(0)
            index_value = parse_index(index_value)
            chunks.append(new_chunk_op.new_chunk([c], shape=reduced_df.shape, index=c.index,
                                                 dtype=reduced_df.dtype, index_value=index_value,
                                                 object_type=ObjectType.series))
        return chunks

    @classmethod
    def execute(cls, ctx, op):
        kwargs = dict(axis=op.axis, level=op.level, skipna=op.skipna)
        in_df = ctx[op.inputs[0].key]
        ctx[op.outputs[0].key] = getattr(in_df, getattr(cls, '_func_name'))(**kwargs)


class DataFrameReductionMixin(ReductionMixin):
    @classmethod
    def tile(cls, op):
        df = op.outputs[0]
        in_df = op.inputs[0]
        combine_size = options.tensor.combine_size

        if len(in_df.chunks) == 1:
            return cls.single_tiles(op)

        n_rows, n_cols = in_df.chunk_shape
        reduction_chunks = cls.convert_to_reduction_chunks(op)
        out_chunks = []
        if op.axis is None or op.axis == 0 or op.axis == 'index':
            for col in range(n_cols):
                chunks = [reduction_chunks[i * n_cols + col] for i in range(n_rows)]
                out_chunks.append(cls.tree_reduction(chunks, op, combine_size, col))
            new_op = op.copy()
            nsplits = tuple((c.shape[0],) for c in out_chunks)
            return new_op.new_seriess(op.inputs, df.shape, nsplits=nsplits, chunks=out_chunks,
                                      dtype=df.dtype, index_value=df.index_value,
                                      object_type=ObjectType.series)

    @classmethod
    def tree_reduction(cls, chunks, op, combine_size, idx):
        while len(chunks) > 1:
            new_chunks = []
            for i in range(0, len(chunks), combine_size):
                chks = chunks[i: i + combine_size]
                concat_op = DataFrameConcat(dtype=chks[0].dtype, axis=1, object_type=ObjectType.series)
                chk = concat_op.new_chunk(chks, index=(i,), index_value=chks[0].index_value)
                reduced_shape = (chks[0].shape[0], len(chks))
                new_op = op.copy().reset_key()
                new_op._axis = 'columns'
                new_chunks.append(new_op.new_chunk([chk], shape=reduced_shape, index=(i,),
                                                   index_value=chk.index_value,
                                                   object_type=ObjectType.series))
            chunks = new_chunks
        return chunks[0]

    @classmethod
    def get_reduced_df(cls, in_df, axis, level):
        func_name = getattr(cls, '_func_name')
        empty_df = build_empty_df(in_df.dtypes)
        return getattr(empty_df, func_name)(axis=axis, level=level)

    def __call__(self, df):
        axis = getattr(self, 'axis', None)
        level = getattr(self, 'level', None)

        reduced_df = self.get_reduced_df(df, axis, level)
        # TODO: return DataFrame if specify level
        return self.new_series([df], shape=reduced_df.shape, dtype=reduced_df.dtype,
                               index_value=parse_index(reduced_df.index), object_type=ObjectType.series)


class SeriesReductionMixin(ReductionMixin):
    @classmethod
    def tile(cls, op):
        df = op.outputs[0]
        in_chunks = op.inputs[0].chunks
        combine_size = options.tensor.combine_size

        if len(in_chunks) == 1:
            return cls.single_tiles(op)

        chunks = cls.convert_to_reduction_chunks(op)

        while len(chunks) > 1:
            new_chunks = []
            for i in range(0, len(chunks), combine_size):
                chks = chunks[i: i + combine_size]
                concat_op = DataFrameConcat(dtype=chks[0].dtype, object_type=ObjectType.series)
                length = sum([c.shape[0] for c in chks if len(c.shape) > 0])
                chk = concat_op.new_chunk(chks, shape=(length,), index=(i,),
                                          index_value=parse_index(pd.RangeIndex(length)))
                reduced_shape = cls.get_reduced_df(chk, op.axis, op.level).shape
                new_op = op.copy().reset_key()
                new_chunks.append(new_op.new_chunk([chk], shape=reduced_shape, index=(i,),
                                                   index_value=parse_index(pd.RangeIndex(len(reduced_shape)))))
            chunks = new_chunks

        new_op = op.copy()
        nsplits = tuple((s,) for s in chunks[0].shape)
        return new_op.new_dataframes(op.inputs, df.shape,
                                     nsplits=tuple(tuple(ns) for ns in nsplits),
                                     chunks=chunks, dtype=df.dtype, index_value=df.index_value)

    @classmethod
    def get_reduced_df(cls, series, axis, level):
        index = series.index_value.to_pandas()
        func_name = getattr(cls, '_func_name')
        empty_series = pd.Series(index=index)
        return getattr(empty_series, func_name)(axis=axis, level=level)

    def __call__(self, series):
        axis = getattr(self, 'axis', None)
        level = getattr(self, 'level', None)

        reduced_shape = self.get_reduced_df(series, axis, level).shape
        return self.new_series([series], shape=reduced_shape, dtype=series.dtype,
                               index_value=parse_index(pd.RangeIndex(len(reduced_shape))), name=series.name)
