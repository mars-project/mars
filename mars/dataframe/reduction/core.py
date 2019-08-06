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

import numpy as np
import pandas as pd

from ...config import options
from ...serialize import BoolField, AnyField, DataTypeField, Int32Field
from ..utils import parse_index, build_empty_df
from ..operands import DataFrameOperandMixin, DataFrameOperand, ObjectType, SERIES_CHUNK_TYPE
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
    def _get_reduced_meta(cls, chunk, axis):
        """
        calculate meta information of reduced result.
        :param chunk: chunk will be reduced
        :param axis: axis
        :return:
        For series, return (shape, index, dtype).
        For DataFrame, return (shape, index, dtypes).
        """
        if isinstance(chunk, SERIES_CHUNK_TYPE):
            return (), parse_index(pd.RangeIndex(0)), chunk.dtype
        if axis == 0 or axis == 'index':
            return (1, chunk.shape[1]), parse_index(pd.RangeIndex(1)), chunk.dtypes
        else:
            emtpy_df = build_empty_df(chunk.dtypes)
            func_name = getattr(cls, '_func_name')
            reduced_dtypes = pd.Series(getattr(emtpy_df, func_name)(axis=axis).dtypes)
            return (chunk.shape[0], 1), chunk.index_value, reduced_dtypes

    @classmethod
    def _tile_one_chunk(cls, op):
        df = op.outputs[0]
        chk = op.inputs[0].chunks[0]
        new_chunk_op = op.copy().reset_key()
        reduced_shape, index_value, dtype = cls._get_reduced_meta(chk, axis=op.axis)
        chunk = new_chunk_op.new_chunk(op.inputs[0].chunks, shape=reduced_shape, index=chk.index,
                                       index_value=index_value, dtype=dtype)
        new_op = op.copy()
        nsplits = tuple((s,) for s in chunk.shape)
        return new_op.new_seriess(op.inputs, df.shape, nsplits=nsplits, chunks=[chunk],
                                  index_value=df.index_value, dtype=df.dtype)

    @classmethod
    def _build_reduction_chunks(cls, op):
        chunks = np.empty(op.inputs[0].chunk_shape, dtype=np.object)
        for c in op.inputs[0].chunks:
            new_chunk_op = op.copy().reset_key()
            if isinstance(c, SERIES_CHUNK_TYPE):
                reduced_shape, index_value, dtype = cls._get_reduced_meta(c, axis=op.axis)
                chunks[c.index] = new_chunk_op.new_chunk([c], shape=reduced_shape, index=c.index,
                                                         dtype=dtype, index_value=index_value)
            else:
                new_chunk_op._object_type = ObjectType.dataframe
                reduced_shape, index_value, dtypes = cls._get_reduced_meta(c, axis=op.axis)
                chunks[c.index] = new_chunk_op.new_chunk([c], shape=reduced_shape, index=c.index,
                                                         dtypes=dtypes, index_value=index_value)
        return chunks


class DataFrameReductionMixin(ReductionMixin):
    @classmethod
    def tile(cls, op):
        df = op.outputs[0]
        in_df = op.inputs[0]
        combine_size = options.tensor.combine_size

        if len(in_df.chunks) == 1:
            return cls._tile_one_chunk(op)

        n_rows, n_cols = in_df.chunk_shape
        reduction_chunks = cls._build_reduction_chunks(op)
        out_chunks = []
        if op.axis is None or op.axis == 0 or op.axis == 'index':
            for col in range(n_cols):
                chunks = [reduction_chunks[i, col] for i in range(n_rows)]
                out_chunks.append(cls.tree_reduction(chunks, op, combine_size, col))
        elif op.axis == 1 or op.axis == 'columns':
            for row in range(n_rows):
                chunks = [reduction_chunks[row, i] for i in range(n_cols)]
                out_chunks.append(cls.tree_reduction(chunks, op, combine_size, row))
        new_op = op.copy()
        nsplits = (tuple(c.shape[0] for c in out_chunks),)
        return new_op.new_seriess(op.inputs, df.shape, nsplits=nsplits, chunks=out_chunks,
                                  dtype=df.dtype, index_value=df.index_value)

    @classmethod
    def tree_reduction(cls, chunks, op, combine_size, idx):
        while len(chunks) > combine_size:
            new_chunks = []
            for i in range(0, len(chunks), combine_size):
                chks = chunks[i: i + combine_size]
                concat_op = DataFrameConcat(axis=op.axis, object_type=ObjectType.dataframe)
                if op.axis == 0 or op.axis == 'index':
                    concat_index = parse_index(pd.RangeIndex(len(chks)))
                    concat_dtypes = chks[0].dtypes
                    concat_shape = (sum([c.shape[0] for c in chks]), chks[0].shape[1])
                else:
                    concat_index = chks[0].index
                    concat_dtypes = pd.Series([c.dtypes[0] for c in chks])
                    concat_shape = (chks[0].shape[0], (sum([c.shape[1] for c in chks])))
                chk = concat_op.new_chunk(chks, shape=concat_shape, index=(i,),
                                          dtypes=concat_dtypes, index_value=concat_index)
                reduced_shape, index_value, dtypes = cls._get_reduced_meta(chk, axis=op.axis)
                new_op = op.copy().reset_key()
                new_op._object_type = ObjectType.dataframe
                new_chunks.append(new_op.new_chunk([chk], shape=reduced_shape, index=(i,), dtypes=dtypes,
                                                   index_value=index_value))
            chunks = new_chunks
        concat_op = DataFrameConcat(axis=op.axis, object_type=ObjectType.dataframe)
        chk = concat_op.new_chunk(chunks, index=(idx,))
        empty_df = build_empty_df(chunks[0].dtypes)
        reduced_df = getattr(empty_df, getattr(cls, '_func_name'))(axis=op.axis, level=op.level,
                                                                   numeric_only=op.numeric_only)
        reduced_shape = (np.nan,) if op.axis == 1 or op.axis == 'columns' else reduced_df.shape
        new_op = op.copy().reset_key()
        return new_op.new_chunk([chk], shape=reduced_shape, index=(idx,), dtype=reduced_df.dtype,
                                index_value=parse_index(reduced_df.index))

    @classmethod
    def execute(cls, ctx, op):
        kwargs = dict(axis=op.axis, level=op.level, skipna=op.skipna, numeric_only=op.numeric_only)
        in_df = ctx[op.inputs[0].key]
        res = getattr(in_df, getattr(cls, '_func_name'))(**kwargs)
        if op.object_type == ObjectType.series:
            ctx[op.outputs[0].key] = res
        else:
            if op.axis == 0 or op.axis == 'index':
                ctx[op.outputs[0].key] = pd.DataFrame(res).transpose()
            else:
                ctx[op.outputs[0].key] = pd.DataFrame(res)

    def __call__(self, df):
        axis = getattr(self, 'axis', None)
        level = getattr(self, 'level', None)
        numeric_only = getattr(self, 'numeric_only', None)

        # TODO: enable specify level if we support groupby
        if level is not None:
            raise NotImplementedError('Not support specify level now')
        if axis is None:
            self._axis = 0

        empty_df = build_empty_df(df.dtypes)
        reduced_df = getattr(empty_df, getattr(self, '_func_name'))(axis=axis, level=level,
                                                                    numeric_only=numeric_only)
        reduced_shape = (df.shape[0],) if axis == 1 or axis == 'columns' else reduced_df.shape
        return self.new_series([df], shape=reduced_shape, dtype=reduced_df.dtype,
                               index_value=parse_index(reduced_df.index))


class SeriesReductionMixin(ReductionMixin):
    @classmethod
    def tile(cls, op):
        df = op.outputs[0]
        in_chunks = op.inputs[0].chunks
        combine_size = options.tensor.combine_size

        if len(in_chunks) == 1:
            return cls._tile_one_chunk(op)

        chunks = cls._build_reduction_chunks(op)

        while len(chunks) > 1:
            new_chunks = []
            for i in range(0, len(chunks), combine_size):
                chks = chunks[i: i + combine_size]
                concat_op = DataFrameConcat(object_type=ObjectType.series)
                length = sum([c.shape[0] for c in chks if len(c.shape) > 0])
                chk = concat_op.new_chunk(chks, shape=(length,), index=(i,), dtype=chks[0].dtype,
                                          index_value=parse_index(pd.RangeIndex(length)))
                new_op = op.copy().reset_key()
                new_chunks.append(new_op.new_chunk([chk], shape=(), index=(i,), dtype=chk.dtype,
                                                   index_value=parse_index(pd.RangeIndex(0))))
            chunks = new_chunks

        new_op = op.copy()
        nsplits = tuple((s,) for s in chunks[0].shape)
        return new_op.new_seriess(op.inputs, df.shape,
                                  nsplits=tuple(tuple(ns) for ns in nsplits),
                                  chunks=chunks, dtype=df.dtype, index_value=df.index_value)

    @classmethod
    def execute(cls, ctx, op):
        kwargs = dict(axis=op.axis, level=op.level, skipna=op.skipna)
        in_df = ctx[op.inputs[0].key]
        ctx[op.outputs[0].key] = getattr(in_df, getattr(cls, '_func_name'))(**kwargs)

    def __call__(self, series):
        level = getattr(self, 'level', None)

        # TODO: enable specify level if we support groupby
        if level is not None:
            raise NotImplementedError('Not support specified level now')

        return self.new_series([series], shape=(), dtype=series.dtype,
                               index_value=parse_index(pd.RangeIndex(0)), name=series.name)
