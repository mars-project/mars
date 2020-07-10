#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import pandas as pd
import numpy as np

from ... import opcodes as OperandDef
from ...serialize import BoolField, AnyField, StringField
from ..core import IndexValue
from ..operands import DataFrameOperandMixin, DataFrameOperand, DATAFRAME_TYPE, ObjectType
from ..utils import parse_index, build_empty_df, build_empty_series, standardize_range_index


class DataFrameResetIndex(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.RESET_INDEX

    _level = AnyField('level')
    _drop = BoolField('drop')
    _name = StringField('name')
    _col_level = AnyField('col_level')
    _col_fill = AnyField('col_fill')

    def __init__(self, level=None, drop=None, name=None, col_level=None, col_fill=None, object_type=None, **kwargs):
        super().__init__(_level=level, _drop=drop, _name=name, _col_level=col_level,
                         _col_fill=col_fill, _object_type=object_type, **kwargs)

    @property
    def level(self):
        return self._level

    @property
    def drop(self):
        return self._drop

    @property
    def name(self):
        return self._name

    @property
    def col_level(self):
        return self._col_level

    @property
    def col_fill(self):
        return self._col_fill

    @classmethod
    def _tile_series(cls, op):
        out_chunks = []
        out = op.outputs[0]
        is_range_index = out.index_value.has_value()
        cum_range = np.cumsum((0, ) + op.inputs[0].nsplits[0])
        for c in op.inputs[0].chunks:
            if is_range_index:
                index_value = parse_index(pd.RangeIndex(cum_range[c.index[0]], cum_range[c.index[0] + 1]))
            else:
                index_value = out.index_value
            chunk_op = op.copy().reset_key()
            if op.drop:
                out_chunk = chunk_op.new_chunk([c], shape=c.shape, index=c.index, dtype=c.dtype,
                                               name=c.name, index_value=index_value)
            else:
                shape = (c.shape[0], out.shape[1])
                out_chunk = chunk_op.new_chunk([c], shape=shape, index=c.index + (0,), dtypes=out.dtypes,
                                               index_value=index_value, columns_value=out.columns_value)
            out_chunks.append(out_chunk)
        if not is_range_index and isinstance(out.index_value._index_value, IndexValue.RangeIndex):
            out_chunks = standardize_range_index(out_chunks)
        new_op = op.copy()
        if op.drop:
            return new_op.new_seriess(op.inputs, op.inputs[0].shape, nsplits=op.inputs[0].nsplits,
                                      name=out.name, chunks=out_chunks, dtype=out.dtype,
                                      index_value=out.index_value)
        else:
            nsplits = (op.inputs[0].nsplits[0], (out.shape[1],))
            return new_op.new_dataframes(op.inputs, out.shape, nsplits=nsplits, chunks=out_chunks,
                                         index_value=out.index_value, columns_value=out.columns_value,
                                         dtypes=out.dtypes)

    @classmethod
    def _tile_dataframe(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]
        added_columns_num = len(out_df.dtypes) - len(in_df.dtypes)
        out_chunks = []
        index_has_value = out_df.index_value.has_value()
        chunk_has_nan = any(np.isnan(s) for s in in_df.nsplits[0])
        cum_range = np.cumsum((0, ) + in_df.nsplits[0])
        for c in in_df.chunks:
            if index_has_value:
                if chunk_has_nan:
                    index_value = parse_index(pd.RangeIndex(-1))
                else:
                    index_value = parse_index(pd.RangeIndex(cum_range[c.index[0]], cum_range[c.index[0] + 1]))
            else:
                index_value = out_df.index_value
            if c.index[1] == 0:
                chunk_op = op.copy().reset_key()
                dtypes = out_df.dtypes[:(added_columns_num + len(c.dtypes))]
                columns_value = parse_index(dtypes.index)
                new_chunk = chunk_op.new_chunk([c], shape=(c.shape[0], c.shape[1] + added_columns_num),
                                               index=c.index, index_value=index_value,
                                               columns_value=columns_value, dtypes=dtypes)
            else:
                chunk_op = op.copy().reset_key()
                chunk_op._drop = True
                new_chunk = chunk_op.new_chunk([c], shape=c.shape, index_value=index_value,
                                               index=c.index, columns_value=c.columns_value, dtypes=c.dtypes)
            out_chunks.append(new_chunk)
        if not index_has_value or chunk_has_nan:
            if isinstance(out_df.index_value._index_value, IndexValue.RangeIndex):
                out_chunks = standardize_range_index(out_chunks)
        new_op = op.copy()
        columns_splits = list(in_df.nsplits[1])
        columns_splits[0] += added_columns_num
        nsplits = (in_df.nsplits[0], tuple(columns_splits))
        return new_op.new_dataframes(op.inputs, out_df.shape, nsplits=nsplits,
                                     chunks=out_chunks, dtypes=out_df.dtypes,
                                     index_value=out_df.index_value, columns_value=out_df.columns_value)

    @classmethod
    def tile(cls, op):
        if isinstance(op.inputs[0], DATAFRAME_TYPE):
            return cls._tile_dataframe(op)
        else:
            return cls._tile_series(op)

    @classmethod
    def execute(cls, ctx, op):
        in_data = ctx[op.inputs[0].key]
        out = op.outputs[0]

        kwargs = dict()
        if op.name is not None:
            kwargs['name'] = op.name
        if op.col_level is not None:
            kwargs['col_level'] = op.col_level
        if op.col_fill is not None:
            kwargs['col_fill'] = op.col_fill

        r = in_data.reset_index(level=op.level, drop=op.drop, **kwargs)
        if out.index_value.has_value():
            r.index = out.index_value.to_pandas()
        ctx[out.key] = r

    @classmethod
    def _get_out_index(cls, df, out_shape):
        if isinstance(df.index, pd.RangeIndex):
            range_value = -1 if np.isnan(out_shape[0]) else out_shape[0]
            index_value = parse_index(pd.RangeIndex(range_value))
        else:
            index_value = parse_index(df.index)
        return index_value

    def _call_series(self, a):
        if self.drop:
            range_value = -1 if np.isnan(a.shape[0]) else a.shape[0]
            index_value = parse_index(pd.RangeIndex(range_value))
            return self.new_series([a], shape=a.shape, dtype=a.dtype, name=a.name, index_value=index_value)
        else:
            self._object_type = ObjectType.dataframe
            empty_series = build_empty_series(dtype=a.dtype, index=a.index_value.to_pandas()[:0], name=a.name)
            empty_df = empty_series.reset_index(level=self.level, name=self.name)
            shape = (a.shape[0], len(empty_df.dtypes))
            index_value = self._get_out_index(empty_df, shape)
            return self.new_dataframe([a], shape=shape, index_value=index_value,
                                      columns_value=parse_index(empty_df.columns),
                                      dtypes=empty_df.dtypes)

    def _call_dataframe(self, a):
        if self.drop:
            shape = a.shape
            columns_value = a.columns_value
            dtypes = a.dtypes
            range_value = -1 if np.isnan(a.shape[0]) else a.shape[0]
            index_value = parse_index(pd.RangeIndex(range_value))
        else:
            empty_df = build_empty_df(a.dtypes)
            empty_df.index = a.index_value.to_pandas()[:0]
            empty_df = empty_df.reset_index(level=self.level, col_level=self.col_level, col_fill=self.col_fill)
            shape = (a.shape[0], len(empty_df.columns))
            columns_value = parse_index(empty_df.columns, store_data=True)
            dtypes = empty_df.dtypes
            index_value = self._get_out_index(empty_df, shape)
        return self.new_dataframe([a], shape=shape, columns_value=columns_value,
                                  index_value=index_value, dtypes=dtypes)

    def __call__(self, a):
        if isinstance(a, DATAFRAME_TYPE):
            return self._call_dataframe(a)
        else:
            return self._call_series(a)


def df_reset_index(df, level=None, drop=False, col_level=None, col_fill=None):
    op = DataFrameResetIndex(level=level, drop=drop, col_level=col_level,
                             col_fill=col_fill, object_type=ObjectType.dataframe)
    return op(df)


def series_reset_index(series, level=None, drop=False, name=None):
    op = DataFrameResetIndex(level=level, drop=drop, name=name, object_type=ObjectType.series)
    return op(series)
