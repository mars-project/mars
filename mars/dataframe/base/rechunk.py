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

import itertools

import pandas as pd

from ... import opcodes as OperandDef
from ...core import OutputType
from ...serialization.serializables import KeyField, AnyField, Int32Field, Int64Field
from ...tensor.rechunk.core import get_nsplits, plan_rechunks, compute_rechunk_slices
from ...tensor.utils import calc_sliced_size
from ...utils import has_unknown_shape
from ..initializer import DataFrame as asdataframe, Series as asseries
from ..operands import DataFrameOperand, DataFrameOperandMixin, DATAFRAME_TYPE
from ..utils import indexing_index_value, merge_index_value


class DataFrameRechunk(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.RECHUNK

    _input = KeyField('input')
    _chunk_size = AnyField('chunk_size')
    _threshold = Int32Field('threshold')
    _chunk_size_limit = Int64Field('chunk_size_limit')

    def __init__(self, chunk_size=None, threshold=None, chunk_size_limit=None, output_types=None, **kw):
        super().__init__(_chunk_size=chunk_size, _threshold=threshold,
                         _chunk_size_limit=chunk_size_limit, _output_types=output_types, **kw)

    @property
    def input(self):
        return self._input

    @property
    def chunk_size(self):
        return self._chunk_size

    @property
    def threshold(self):
        return self._threshold

    @property
    def chunk_size_limit(self):
        return self._chunk_size_limit

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, x):
        if isinstance(x, DATAFRAME_TYPE):
            return self.new_dataframe([x], shape=x.shape, dtypes=x.dtypes,
                                      columns_value=x.columns_value, index_value=x.index_value)
        else:
            self.output_types = x.op.output_types
            f = self.new_series if self.output_types[0] == OutputType.series else self.new_index
            return f([x], shape=x.shape, dtype=x.dtype, index_value=x.index_value, name=x.name)

    @classmethod
    def tile(cls, op):
        if has_unknown_shape(*op.inputs):
            yield

        a = op.input
        a = asdataframe(a) if a.ndim == 2 else asseries(a)
        chunk_size = _get_chunk_size(a, op.chunk_size)
        if chunk_size == a.nsplits:
            return [a]

        out = op.outputs[0]
        new_chunk_size = tuple(tuple(s for s in cs if s != 0)
                               for cs in chunk_size)
        if isinstance(out, DATAFRAME_TYPE):
            itemsize = max(getattr(dt, 'itemsize', 8) for dt in out.dtypes)
        else:
            itemsize = out.dtype.itemsize
        steps = plan_rechunks(op.inputs[0], new_chunk_size, itemsize,
                              threshold=op.threshold,
                              chunk_size_limit=op.chunk_size_limit)
        for c in steps:
            out = compute_rechunk(out.inputs[0], c)

        if op.reassign_worker:
            for c in out.chunks:
                c.op.reassign_worker = True

        return [out]


def _get_chunk_size(a, chunk_size):
    if isinstance(a, DATAFRAME_TYPE):
        itemsize = max(getattr(dt, 'itemsize', 8) for dt in a.dtypes)
    else:
        itemsize = a.dtype.itemsize
    return get_nsplits(a, chunk_size, itemsize)


def rechunk(a, chunk_size, threshold=None, chunk_size_limit=None, reassign_worker=False):
    if not any(pd.isna(s) for s in a.shape) and not a.is_coarse():
        if not has_unknown_shape(a):
            # do client check only when no unknown shape,
            # real nsplits will be recalculated inside `tile`
            chunk_size = _get_chunk_size(a, chunk_size)
            if chunk_size == a.nsplits:
                return a

    op = DataFrameRechunk(chunk_size=chunk_size, threshold=threshold,
                          chunk_size_limit=chunk_size_limit,
                          reassign_worker=reassign_worker)
    return op(a)


def _concat_dataframe_meta(to_concat_chunks):
    if to_concat_chunks[0].index_value.to_pandas().empty:
        index_value = to_concat_chunks[0].index_value
    else:
        idx_to_index_value = dict((c.index[0], c.index_value) for c in to_concat_chunks if c.index[1] == 0)
        index_value = merge_index_value(idx_to_index_value)

    idx_to_columns_value = dict((c.index[1], c.columns_value) for c in to_concat_chunks if c.index[0] == 0)
    columns_value = merge_index_value(idx_to_columns_value, store_data=True)

    idx_to_dtypes = dict((c.index[1], c.dtypes) for c in to_concat_chunks if c.index[0] == 0)
    dtypes = pd.concat([v[1] for v in list(sorted(idx_to_dtypes.items()))])
    return index_value, columns_value, dtypes


def _concat_series_index(to_concat_chunks):
    if to_concat_chunks[0].index_value.to_pandas().empty:
        index_value = to_concat_chunks[0].index_value
    else:
        idx_to_index_value = dict((c.index[0], c.index_value) for c in to_concat_chunks)
        index_value = merge_index_value(idx_to_index_value)
    return index_value


def compute_rechunk(a, chunk_size):
    from ..indexing.iloc import DataFrameIlocGetItem, SeriesIlocGetItem
    from ..merge.concat import DataFrameConcat

    result_slices = compute_rechunk_slices(a, chunk_size)
    result_chunks = []
    idxes = itertools.product(*[range(len(c)) for c in chunk_size])
    chunk_slices = itertools.product(*result_slices)
    chunk_shapes = itertools.product(*chunk_size)
    is_dataframe = isinstance(a, DATAFRAME_TYPE)
    for idx, chunk_slice, chunk_shape in zip(idxes, chunk_slices, chunk_shapes):
        to_merge = []
        merge_idxes = itertools.product(*[range(len(i)) for i in chunk_slice])
        for merge_idx, index_slices in zip(merge_idxes, itertools.product(*chunk_slice)):
            chunk_index, chunk_slice = zip(*index_slices)
            old_chunk = a.cix[chunk_index]
            merge_chunk_shape = tuple(calc_sliced_size(s, chunk_slice[0]) for s in old_chunk.shape)
            new_index_value = indexing_index_value(old_chunk.index_value, chunk_slice[0])
            if is_dataframe:
                new_columns_value = indexing_index_value(old_chunk.columns_value, chunk_slice[1], store_data=True)
                merge_chunk_op = DataFrameIlocGetItem(list(chunk_slice), sparse=old_chunk.op.sparse,
                                                      output_types=[OutputType.dataframe])
                merge_chunk = merge_chunk_op.new_chunk([old_chunk], shape=merge_chunk_shape,
                                                       index=merge_idx, index_value=new_index_value,
                                                       columns_value=new_columns_value,
                                                       dtypes=old_chunk.dtypes.iloc[chunk_slice[1]])
            else:
                merge_chunk_op = SeriesIlocGetItem(list(chunk_slice), sparse=old_chunk.op.sparse,
                                                   output_types=a.op.output_types)
                merge_chunk = merge_chunk_op.new_chunk([old_chunk], shape=merge_chunk_shape,
                                                       index=merge_idx, index_value=new_index_value,
                                                       dtype=old_chunk.dtype, name=old_chunk.name)
            to_merge.append(merge_chunk)
        if len(to_merge) == 1:
            chunk_op = to_merge[0].op.copy()
            if is_dataframe:
                out_chunk = chunk_op.new_chunk(to_merge[0].op.inputs, shape=chunk_shape,
                                               index=idx, index_value=to_merge[0].index_value,
                                               columns_value=to_merge[0].columns_value,
                                               dtypes=to_merge[0].dtypes)
            else:
                out_chunk = chunk_op.new_chunk(to_merge[0].op.inputs, shape=chunk_shape,
                                               index=idx, index_value=to_merge[0].index_value,
                                               name=to_merge[0].name, dtype=to_merge[0].dtype)
            result_chunks.append(out_chunk)
        else:
            if is_dataframe:
                chunk_op = DataFrameConcat(output_types=[OutputType.dataframe])
                index_value, columns_value, dtypes = _concat_dataframe_meta(to_merge)
                out_chunk = chunk_op.new_chunk(to_merge, shape=chunk_shape,
                                               index=idx, index_value=index_value,
                                               columns_value=columns_value,
                                               dtypes=dtypes)
            else:
                chunk_op = DataFrameConcat(output_types=a.op.output_types)
                index_value = _concat_series_index(to_merge)
                out_chunk = chunk_op.new_chunk(to_merge, shape=chunk_shape,
                                               index=idx, index_value=index_value,
                                               dtype=to_merge[0].dtype,
                                               name=to_merge[0].name)
            result_chunks.append(out_chunk)

    if is_dataframe:
        op = DataFrameRechunk(chunk_size, output_types=[OutputType.dataframe])
        return op.new_dataframe([a], a.shape, dtypes=a.dtypes, columns_value=a.columns_value,
                                index_value=a.index_value, nsplits=chunk_size, chunks=result_chunks)
    else:
        op = DataFrameRechunk(chunk_size, output_types=a.op.output_types)
        if a.op.output_types[0] == OutputType.index:
            f = op.new_index
        else:
            f = op.new_series
        return f([a], a.shape, dtype=a.dtype, index_value=a.index_value,
                 nsplits=chunk_size, chunks=result_chunks, name=a.name)
