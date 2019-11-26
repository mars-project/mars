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

import itertools

from ... import opcodes as OperandDef
from ...compat import izip, lzip
from ...serialize import KeyField, AnyField, Int32Field, Int64Field
from ...tensor.rechunk.core import get_nsplits, plan_rechunks, compute_rechunk_slices
from ...tensor.utils import calc_sliced_size
from ..operands import DataFrameOperand, DataFrameOperandMixin, DATAFRAME_TYPE, ObjectType
from ..utils import indexing_index_value, parse_index


class DataFrameRechunk(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.RECHUNK

    _input = KeyField('input')
    _chunk_size = AnyField('chunk_size')
    _threshold = Int32Field('threshold')
    _chunk_size_limit = Int64Field('chunk_size_limit')

    def __init__(self, chunk_size=None, threshold=None, chunk_size_limit=None, object_type=ObjectType.dataframe, **kw):
        super(DataFrameRechunk, self).__init__(_chunk_size=chunk_size, _threshold=threshold,
                                               _chunk_size_limit=chunk_size_limit,
                                               _object_type=object_type, **kw)

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
        super(DataFrameRechunk, self)._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, x):
        if isinstance(x, DATAFRAME_TYPE):
            self._object_type = ObjectType.dataframe
            return self.new_dataframe([x], shape=x.shape, dtypes=x.dtypes,
                                      columns_value=x.columns_value, index_value=x.index_value)
        else:
            # TODO: fix it when we support series.iloc
            raise NotImplementedError

    @classmethod
    def tile(cls, op):
        df = op.outputs[0]
        new_chunk_size = op.chunk_size
        itemsize = max(dt.itemsize for dt in df.dtypes)
        steps = plan_rechunks(op.inputs[0], new_chunk_size, itemsize,
                              threshold=op.threshold,
                              chunk_size_limit=op.chunk_size_limit)
        for c in steps:
            df = compute_rechunk(df.inputs[0], c)

        return [df]


def rechunk(df, chunk_size, threshold=None, chunk_size_limit=None):
    itemsize = max(dt.itemsize for dt in df.dtypes)
    chunk_size = get_nsplits(df, chunk_size, itemsize)
    if chunk_size == df.nsplits:
        return df

    op = DataFrameRechunk(chunk_size, threshold, chunk_size_limit)
    return op(df)


def _concat_index_and_columns(to_concat_chunks):
    if to_concat_chunks[0].index_value.to_pandas().empty:
        index_value = to_concat_chunks[0].index_value
    else:
        idx_to_index_value = dict((c.index[0], c.index_value) for c in to_concat_chunks if c.index[1] == 0)
        index = None
        for _, chunk_index in sorted(idx_to_index_value.items()):
            if index is None:
                index = chunk_index.to_pandas()
            else:
                index = index.append(chunk_index.to_pandas())
        index_value = parse_index(index)

    idx_to_columns_value = dict((c.index[1], c.columns_value) for c in to_concat_chunks if c.index[0] == 0)
    columns = None
    for _, chunk_columns in sorted(idx_to_columns_value.items()):
        if columns is None:
            columns = chunk_columns.to_pandas()
        else:
            columns = columns.append(chunk_columns.to_pandas())
    columns_value = parse_index(columns, store_data=True)
    return index_value, columns_value


def compute_rechunk(df, chunk_size):
    from ..indexing import DataFrameIlocGetItem
    from ..merge.concat import DataFrameConcat

    result_slices = compute_rechunk_slices(df, chunk_size)
    result_chunks = []
    idxes = itertools.product(*[range(len(c)) for c in chunk_size])
    chunk_slices = itertools.product(*result_slices)
    chunk_shapes = itertools.product(*chunk_size)
    for idx, chunk_slice, chunk_shape in izip(idxes, chunk_slices, chunk_shapes):
        to_merge = []
        merge_idxes = itertools.product(*[range(len(i)) for i in chunk_slice])
        for merge_idx, index_slices in izip(merge_idxes, itertools.product(*chunk_slice)):
            chunk_index, chunk_slice = lzip(*index_slices)
            old_chunk = df.cix[chunk_index]
            merge_chunk_shape = tuple(calc_sliced_size(s, chunk_slice[0]) for s in old_chunk.shape)
            merge_chunk_op = DataFrameIlocGetItem(chunk_slice, sparse=old_chunk.op.sparse,
                                                  object_type=ObjectType.dataframe)
            new_index_value = indexing_index_value(old_chunk.index_value, chunk_slice[0])
            new_columns_value = indexing_index_value(old_chunk.columns_value, chunk_slice[1], store_data=True)
            merge_chunk = merge_chunk_op.new_chunk([old_chunk], shape=merge_chunk_shape,
                                                   index=merge_idx, index_value=new_index_value,
                                                   columns_value=new_columns_value, dtypes=old_chunk.dtypes)
            to_merge.append(merge_chunk)
        if len(to_merge) == 1:
            chunk_op = to_merge[0].op.copy()
            out_chunk = chunk_op.new_chunk(to_merge[0].op.inputs, shape=chunk_shape,
                                           index=idx, index_value=to_merge[0].index_value,
                                           columns_value=to_merge[0].columns_value,
                                           dtypes=to_merge[0].dtypes)
            result_chunks.append(out_chunk)
        else:
            chunk_op = DataFrameConcat(object_type=ObjectType.dataframe)
            index_value, columns_value = _concat_index_and_columns(to_merge)
            out_chunk = chunk_op.new_chunk(to_merge, shape=chunk_shape,
                                           index=idx, index_value=index_value,
                                           columns_value=columns_value,
                                           dtypes=to_merge[0].dtypes)
            result_chunks.append(out_chunk)

    op = DataFrameRechunk(chunk_size)
    return op.new_dataframe([df], df.shape, dtypes=df.dtypes, columns_value=df.columns_value,
                            index_value=df.index_value, nsplits=chunk_size, chunks=result_chunks)
