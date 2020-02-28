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

from ...serialize import BoolField
from ... import opcodes as OperandDef
from ..datasource.dataframe import from_pandas
from ..indexing.iloc import DataFrameIlocGetItem, SeriesIlocGetItem
from ..utils import parse_index, standardize_range_index
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType, \
    DATAFRAME_TYPE, SERIES_TYPE


class DataFrameAppend(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.APPEND

    _ignore_index = BoolField('ignore_index')
    _verify_integrity = BoolField('verify_integrity')
    _sort = BoolField('sort')

    def __init__(self, ignore_index=None, verify_integrity=None, sort=None, object_type=None, **kw):
        super(DataFrameAppend, self).__init__(_ignore_index=ignore_index,
                                              _verify_integrity=verify_integrity,
                                              _sort=sort,
                                              _object_type=object_type, **kw)

    @property
    def ignore_index(self):
        return self._ignore_index

    @property
    def verify_integrity(self):
        return self._verify_integrity

    @property
    def sort(self):
        return self._sort

    @classmethod
    def _tile_dataframe(cls, op):
        out_df = op.outputs[0]
        inputs = op.inputs
        first_df, others = inputs[0], inputs[1:]
        column_splits = first_df.nsplits[1]
        others = [item.rechunk({1: column_splits})._inplace_tile() for item in others]
        out_chunks = []
        nsplits = [[], list(first_df.nsplits[1])]
        row_index = 0
        for df in [first_df] + others:
            for c in df.chunks:
                index = (c.index[0] + row_index, c.index[1])
                iloc_op = DataFrameIlocGetItem(indexes=(slice(None),) * 2)
                out_chunks.append(iloc_op.new_chunk([c], shape=c.shape, index=index,
                                                    dtypes=c.dtypes,
                                                    index_value=c.index_value,
                                                    columns_value=c.columns_value))
            nsplits[0] += df.nsplits[0]
            row_index += len(df.nsplits[0])
        if op.ignore_index:
            out_chunks = standardize_range_index(out_chunks)

        nsplits = tuple(tuple(n) for n in nsplits)
        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, out_df.shape,
                                     nsplits=nsplits, chunks=out_chunks,
                                     dtypes=out_df.dtypes,
                                     index_value=out_df.index_value,
                                     columns_value=out_df.columns_value)

    @classmethod
    def _tile_series(cls, op):
        out_series = op.outputs[0]
        inputs = op.inputs
        first_series, others = inputs[0], inputs[1:]
        out_chunks = []
        nsplits = ()
        row_index = 0
        for series in [first_series] + others:
            for c in series.chunks:
                index = (c.index[0] + row_index,)
                iloc_op = SeriesIlocGetItem(indexes=(slice(None),))
                out_chunks.append(iloc_op.new_chunk([c], shape=c.shape, index=index,
                                                    index_value=c.index_value,
                                                    dtype=c.dtype,
                                                    name=c.name))
            nsplits += series.nsplits[0]
            row_index += len(series.nsplits[0])

        if op.ignore_index:
            out_chunks = standardize_range_index(out_chunks)

        nsplits = (tuple(nsplits),)
        new_op = op.copy()
        return new_op.new_seriess(op.inputs, out_series.shape,
                                  nsplits=nsplits, chunks=out_chunks,
                                  dtype=out_series.dtype,
                                  index_value=out_series.index_value,
                                  name=out_series.name)

    @classmethod
    def tile(cls, op):
        if op.object_type == ObjectType.dataframe:
            return cls._tile_dataframe(op)
        else:
            return cls._tile_series(op)

    def _call_dataframe(self, df, other):
        if isinstance(other, DATAFRAME_TYPE):
            shape = (df.shape[0] + other.shape[0], df.shape[1])
            inputs = [df, other]
            if self.ignore_index:
                index_value = parse_index(pd.RangeIndex(shape[0]))
            else:
                index_value = parse_index(df.index_value.to_pandas().append(
                    other.index_value.to_pandas()))
        elif isinstance(other, list):
            row_length = df.shape[0]
            index = df.index_value.to_pandas()
            for item in other:
                if not isinstance(item, DATAFRAME_TYPE):  # pragma: no cover
                    raise ValueError('Invalid type {} to append'.format(type(item)))
                row_length += item.shape[0]
                index = index.append(item.index_value.to_pandas())
            shape = (row_length, df.shape[1])
            if self.ignore_index:  # pragma: no cover
                index_value = parse_index(pd.RangeIndex(shape[0]))
            else:
                index_value = parse_index(index)
            inputs = [df] + other
        else:  # pragma: no cover
            raise ValueError('Invalid type {} to append'.format(type(other)))
        return self.new_dataframe(inputs, shape=shape, dtypes=df.dtypes,
                                  index_value=index_value,
                                  columns_value=df.columns_value)

    def _call_series(self, df, other):
        if isinstance(other, SERIES_TYPE):
            shape = (df.shape[0] + other.shape[0],)
            inputs = [df, other]
            if self.ignore_index:
                index_value = parse_index(pd.RangeIndex(shape[0]))
            else:
                index_value = parse_index(df.index_value.to_pandas().append(
                    other.index_value.to_pandas()))
        elif isinstance(other, list):
            row_length = df.shape[0]
            index = df.index_value.to_pandas()
            for item in other:
                if not isinstance(item, SERIES_TYPE):  # pragma: no cover
                    raise ValueError('Invalid type {} to append'.format(type(item)))
                row_length += item.shape[0]
                index = index.append(item.index_value.to_pandas())
            shape = (row_length,)
            if self.ignore_index:  # pragma: no cover
                index_value = parse_index(pd.RangeIndex(shape[0]))
            else:
                index_value = parse_index(index)
            inputs = [df] + other
        else:  # pragma: no cover
            raise ValueError('Invalid type {} to append'.format(type(other)))
        return self.new_series(inputs, shape=shape, dtype=df.dtype,
                               index_value=index_value, name=df.name)

    @classmethod
    def execute(cls, ctx, op):
        first, others = ctx[op.inputs[0].key], [ctx[inp.key] for inp in op.inputs[1:]]
        r = first.append(others, verify_integrity=op.verify_integrity,
                         sort=op.sort)
        ctx[op.outputs[0].key] = r

    def __call__(self, df, other):
        if isinstance(df, DATAFRAME_TYPE):
            self._object_type = ObjectType.dataframe
            return self._call_dataframe(df, other)
        else:
            self._object_type = ObjectType.series
            return self._call_series(df, other)


def append(df, other, ignore_index=False, verify_integrity=False, sort=False):
    if verify_integrity or sort:  # pragma: no cover
        raise NotImplementedError('verify_integrity and sort are not supported now')
    if isinstance(other, dict):
        other = from_pandas(pd.DataFrame(dict((k, [v]) for k, v in other.items())))
    op = DataFrameAppend(ignore_index=ignore_index, verify_integrity=verify_integrity, sort=sort)
    return op(df, other)
