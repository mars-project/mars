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

import numpy as np
import pandas as pd

from ...config import options
from ...serialize import BoolField, AnyField, DataTypeField, Int32Field
from ..utils import parse_index, build_empty_df
from ..operands import DataFrameOperandMixin, DataFrameOperand, ObjectType, DATAFRAME_TYPE
from ..merge import DataFrameConcat


class DataFrameReductionOperand(DataFrameOperand):
    _axis = AnyField('axis')
    _skipna = BoolField('skipna')
    _level = AnyField('level')
    _numeric_only = BoolField('numeric_only')
    _min_count = Int32Field('min_count')

    _calc_with_count = BoolField('calc_with_count')

    _dtype = DataTypeField('dtype')
    _combine_size = Int32Field('combine_size')

    def __init__(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=None, calc_with_count=None,
                 dtype=None, combine_size=None, gpu=None, sparse=None, object_type=None, **kw):
        super(DataFrameReductionOperand, self).__init__(_axis=axis, _skipna=skipna, _level=level,
                                                        _numeric_only=numeric_only,
                                                        _min_count=min_count,
                                                        _calc_with_count=calc_with_count, _dtype=dtype,
                                                        _combine_size=combine_size, _gpu=gpu, _sparse=sparse,
                                                        _object_type=object_type,**kw)

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
    def numeric_only(self):
        return self._numeric_only

    @property
    def min_count(self):
        return self._min_count

    @property
    def calc_with_count(self):
        return self._calc_with_count

    @property
    def dtype(self):
        return self._dtype

    @property
    def combine_size(self):
        return self._combine_size


class DataFrameReductionMixin(DataFrameOperandMixin):
    @classmethod
    def _tile_one_chunk(cls, op):
        df = op.outputs[0]
        chk = op.inputs[0].chunks[0]
        new_chunk_op = op.copy().reset_key()
        chunk = new_chunk_op.new_chunk(op.inputs[0].chunks, shape=df.shape, index=chk.index,
                                       index_value=df.index_value, dtype=df.dtype)
        new_op = op.copy()
        nsplits = tuple((s,) for s in chunk.shape)
        return new_op.new_seriess(op.inputs, df.shape, nsplits=nsplits, chunks=[chunk],
                                  index_value=df.index_value, dtype=df.dtype)

    @classmethod
    def _tree_reduction(cls, chunks, op, combine_size, idx):
        while len(chunks) > combine_size:
            new_chunks = []
            for i in range(0, len(chunks), combine_size):
                chks = chunks[i: i + combine_size]
                for j, c in enumerate(chunks):
                    c._index = (j,)

                # concatenate chunks into one chunk
                concat_op = DataFrameConcat(axis=op.axis, object_type=ObjectType.dataframe)
                if op.axis == 0:
                    concat_index = parse_index(pd.RangeIndex(len(chks)))
                    concat_dtypes = chks[0].dtypes
                    concat_shape = (sum([c.shape[0] for c in chks]), chks[0].shape[1])
                else:
                    concat_index = chks[0].index
                    concat_dtypes = pd.Series([c.dtypes[0] for c in chks])
                    concat_shape = (chks[0].shape[0], (sum([c.shape[1] for c in chks])))
                chk = concat_op.new_chunk(chks, shape=concat_shape, index=(i,),
                                          dtypes=concat_dtypes, index_value=concat_index)

                # do reduction
                if op.axis == 0:
                    reduced_shape = (1, chk.shape[1])
                    index_value = parse_index(pd.RangeIndex(1))
                    dtypes = chk.dtypes
                else:
                    reduced_shape = (chk.shape[0], 1)
                    index_value = chk.index_value
                    dtypes = pd.Series(op.outputs[0].dtype)
                new_op = op.copy().reset_key()
                if op.min_count is not None and op.min_count > 0:
                    new_op._calc_with_count = True
                # all intermediate results' type is dataframe
                new_op._object_type = ObjectType.dataframe
                new_chunks.append(new_op.new_chunk([chk], shape=reduced_shape, index=(i,), dtypes=dtypes,
                                                   index_value=index_value))
            chunks = new_chunks

        concat_op = DataFrameConcat(axis=op.axis, object_type=ObjectType.dataframe)
        chk = concat_op.new_chunk(chunks, index=(idx,))
        empty_df = build_empty_df(chunks[0].dtypes)
        reduced_df = getattr(empty_df, getattr(cls, '_func_name'))(axis=op.axis, level=op.level,
                                                                   numeric_only=op.numeric_only)
        reduced_shape = (np.nan,) if op.axis == 1 else reduced_df.shape
        new_op = op.copy().reset_key()
        return new_op.new_chunk([chk], shape=reduced_shape, index=(idx,), dtype=reduced_df.dtype,
                                index_value=parse_index(reduced_df.index))

    @classmethod
    def _tile_dataframe(cls, op):
        in_df = op.inputs[0]
        df = op.outputs[0]
        combine_size = op.combine_size or options.combine_size

        n_rows, n_cols = in_df.chunk_shape

        chunk_dtypes = []
        if op.numeric_only and op.axis == 0:
            cum_nsplits = np.cumsum((0,) + in_df.nsplits[0])
            for i in range(len(cum_nsplits) - 1):
                chunk_dtypes.append(build_empty_df(
                    in_df.dtypes[cum_nsplits[i]: cum_nsplits[i + 1]]).select_dtypes(np.number).dtypes)

        # build reduction chunks
        reduction_chunks = np.empty(op.inputs[0].chunk_shape, dtype=np.object)
        for c in op.inputs[0].chunks:
            new_chunk_op = op.copy().reset_key()
            if op.min_count is not None and op.min_count > 0:
                new_chunk_op._calc_with_count = True
            new_chunk_op._object_type = ObjectType.dataframe
            if op.axis == 0:
                if op.numeric_only:
                    dtypes = chunk_dtypes[c.index[1]]
                else:
                    dtypes = c.dtypes
                reduced_shape = (1, len(dtypes))
                index_value = parse_index(pd.RangeIndex(1))
                dtypes = c.dtypes
            else:
                reduced_shape = (c.shape[0], 1)
                index_value = c.index_value
                dtypes = pd.Series(op.outputs[0].dtype)
            reduction_chunks[c.index] = new_chunk_op.new_chunk([c], shape=reduced_shape,
                                                               dtypes=dtypes, index_value=index_value)
        # Tree reduction
        out_chunks = []
        if op.axis is None or op.axis == 0:
            for col in range(n_cols):
                chunks = [reduction_chunks[i, col] for i in range(n_rows)]
                out_chunks.append(cls._tree_reduction(chunks, op, combine_size, col))
        elif op.axis == 1:
            for row in range(n_rows):
                chunks = [reduction_chunks[row, i] for i in range(n_cols)]
                out_chunks.append(cls._tree_reduction(chunks, op, combine_size, row))
        new_op = op.copy()
        nsplits = (tuple(c.shape[0] for c in out_chunks),)
        return new_op.new_seriess(op.inputs, df.shape, nsplits=nsplits, chunks=out_chunks,
                                  dtype=df.dtype, index_value=df.index_value)

    @classmethod
    def _tile_series(cls, op):
        df = op.outputs[0]
        combine_size = op.combine_size or options.combine_size

        chunks = np.empty(op.inputs[0].chunk_shape, dtype=np.object)
        for c in op.inputs[0].chunks:
            new_chunk_op = op.copy().reset_key()
            if op.min_count is not None and op.min_count > 0:
                new_chunk_op._calc_with_count = True
            chunks[c.index] = new_chunk_op.new_chunk([c], shape=(), dtype=df.dtype, index_value=df.index_value)

        while len(chunks) > combine_size:
            new_chunks = []
            for i in range(0, len(chunks), combine_size):
                chks = chunks[i: i + combine_size]
                concat_op = DataFrameConcat(object_type=ObjectType.series)
                length = sum([c.shape[0] for c in chks if len(c.shape) > 0])
                chk = concat_op.new_chunk(chks, shape=(length,), index=(i,), dtype=chks[0].dtype,
                                          index_value=parse_index(pd.RangeIndex(length)))
                new_op = op.copy().reset_key()
                if op.min_count is not None and op.min_count > 0:
                    new_op._calc_with_count = True
                new_chunks.append(new_op.new_chunk([chk], shape=(), index=(i,), dtype=chk.dtype,
                                                   index_value=parse_index(pd.RangeIndex(0))))
            chunks = new_chunks

        concat_op = DataFrameConcat(object_type=ObjectType.series)
        length = sum([c.shape[0] for c in chunks if len(c.shape) > 0])
        chk = concat_op.new_chunk(chunks, shape=(length,), index=(0,), dtype=chunks[0].dtype,
                                  index_value=parse_index(pd.RangeIndex(length)))
        chunk_op = op.copy().reset_key()
        chunk = chunk_op.new_chunk([chk], shape=(), index=(0,), dtype=chk.dtype,
                                   index_value=parse_index(pd.RangeIndex(0)))

        new_op = op.copy().reset_key()
        nsplits = tuple((s,) for s in chunk.shape)
        return new_op.new_seriess(op.inputs, df.shape,
                                  nsplits=tuple(tuple(ns) for ns in nsplits),
                                  chunks=[chunk], dtype=df.dtype, index_value=df.index_value)

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        if len(in_df.chunks) == 1:
            return cls._tile_one_chunk(op)
        if isinstance(in_df, DATAFRAME_TYPE):
            return cls._tile_dataframe(op)
        else:
            return cls._tile_series(op)

    @classmethod
    def _execute_reduction(cls, in_data, op, min_count=None):
        kwargs = dict(axis=op.axis, level=op.level, skipna=op.skipna)
        if op.numeric_only is not None:
            kwargs['numeric_only'] = op.numeric_only
        if min_count is not None:
            kwargs['min_count'] = op.min_count
        return getattr(in_data, getattr(cls, '_func_name'))(**kwargs)

    @classmethod
    def _execute_with_count(cls, ctx, op):
        inputs = ctx[op.inputs[0].key]
        if isinstance(inputs, tuple):
            in_data, concat_count = inputs
            count = concat_count.sum(axis=op.axis)
        else:
            in_data = inputs
            count = in_data.notnull().sum(axis=op.axis)
        r = cls._execute_reduction(in_data, op)
        if isinstance(in_data, pd.Series):
            ctx[op.outputs[0].key] = (r, count)
        else:
            # For dataframe, will keep dimensions for intermediate results.
            ctx[op.outputs[0].key] = (pd.DataFrame(r), pd.DataFrame(count)) if op.axis == 1 \
                else (pd.DataFrame(r).transpose(), pd.DataFrame(count).transpose())

    @classmethod
    def _execute_without_count(cls, ctx, op):
        inputs = ctx[op.inputs[0].key]
        if isinstance(inputs, tuple):
            # When specify `min_count`, the terminal chunk has two inputs one of which is for count.
            # The output should be determined by comparing `count` with `min_count`.
            in_data, concat_count = inputs
            count = concat_count.sum(axis=op.axis)
            r = cls._execute_reduction(in_data, op)
            if np.isscalar(r):
                ctx[op.outputs[0].key] = np.nan if count < op.min_count else r
            else:
                r[count < op.min_count] = np.nan
                ctx[op.outputs[0].key] = r
        else:
            # For dataframe, will keep dimensions for intermediate results.
            in_data = inputs
            r = cls._execute_reduction(in_data, op, min_count=op.min_count)
            if isinstance(in_data, pd.Series) or op.object_type == ObjectType.series:
                ctx[op.outputs[0].key] = r
            else:
                ctx[op.outputs[0].key] = pd.DataFrame(r).transpose() if op.axis == 0 else pd.DataFrame(r)

    @classmethod
    def execute(cls, ctx, op):
        if op.calc_with_count:
            cls._execute_with_count(ctx, op)
        else:
            cls._execute_without_count(ctx, op)

    def _call_dataframe(self, df):
        axis = getattr(self, 'axis', None)
        level = getattr(self, 'level', None)
        numeric_only = getattr(self, 'numeric_only', None)
        if axis == 'index':
            axis = 0
        if axis == 'columns':
            axis = 1
        self._axis = axis
        # TODO: enable specify level if we support groupby
        if level is not None:
            raise NotImplementedError('Not support specify level now')
        if axis is None:
            self._axis = 0

        empty_df = build_empty_df(df.dtypes)
        reduced_df = getattr(empty_df, getattr(self, '_func_name'))(axis=axis, level=level,
                                                                    numeric_only=numeric_only)
        reduced_shape = (df.shape[0],) if axis == 1 else reduced_df.shape
        return self.new_series([df], shape=reduced_shape, dtype=reduced_df.dtype,
                               index_value=parse_index(reduced_df.index))

    def _call_series(self, series):
        level = getattr(self, 'level', None)
        axis = getattr(self, 'axis', None)
        if axis == 'index':
            axis = 0
        self._axis = axis
        # TODO: enable specify level if we support groupby
        if level is not None:
            raise NotImplementedError('Not support specified level now')

        return self.new_series([series], shape=(), dtype=series.dtype,
                               index_value=parse_index(pd.RangeIndex(0)), name=series.name)

    def __call__(self, a):
        if isinstance(a, DATAFRAME_TYPE):
            return self._call_dataframe(a)
        else:
            return self._call_series(a)
