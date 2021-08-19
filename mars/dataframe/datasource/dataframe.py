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

from ... import opcodes as OperandDef
from ...config import options
from ...core import OutputType
from ...serialization.serializables import DataFrameField, SeriesField
from ...tensor.utils import get_chunk_slices
from ..utils import decide_dataframe_chunk_sizes, parse_index, is_cudf
from ..operands import DataFrameOperand, DataFrameOperandMixin


class DataFrameDataSource(DataFrameOperand, DataFrameOperandMixin):
    """
    Represents data from pandas DataFrame
    """

    _op_type_ = OperandDef.DATAFRAME_DATA_SOURCE

    data = DataFrameField('data')
    dtypes = SeriesField('dtypes')

    def __init__(self, data=None, dtypes=None, gpu=None, **kw):
        if dtypes is None and data is not None:
            dtypes = data.dtypes
        if gpu is None and is_cudf(data):  # pragma: no cover
            gpu = True
        super().__init__(data=data, dtypes=dtypes, gpu=gpu,
                         _output_types=[OutputType.dataframe], **kw)

    def __call__(self, shape, chunk_size=None):
        return self.new_dataframe(None, shape, dtypes=self.dtypes,
                                  index_value=parse_index(self.data.index),
                                  columns_value=parse_index(self.data.columns,
                                                            store_data=True),
                                  raw_chunk_size=chunk_size)

    @classmethod
    def tile(cls, op: "DataFrameDataSource"):
        df = op.outputs[0]
        raw_df = op.data

        memory_usage = raw_df.memory_usage(index=False, deep=True)
        chunk_size = df.extra_params.raw_chunk_size or options.chunk_size
        chunk_size = decide_dataframe_chunk_sizes(df.shape, chunk_size, memory_usage)
        chunk_size_idxes = (range(len(size)) for size in chunk_size)

        out_chunks = []
        index_values = dict()
        column_values = dict()
        for chunk_shape, chunk_idx in zip(itertools.product(*chunk_size),
                                          itertools.product(*chunk_size_idxes)):
            chunk_op = op.copy().reset_key()
            slc = get_chunk_slices(chunk_size, chunk_idx)
            i_slc, j_slc = slc
            if j_slc == slice(0, df.shape[1]):
                # optimize full slice, it's way more faster
                j_slc = slice(None)
            chunk_op.data = raw_df.iloc[i_slc, j_slc]
            chunk_op.dtypes = chunk_op.data.dtypes
            i, j = chunk_idx
            if i in index_values:
                index_value = index_values[i]
            else:
                index_value = index_values[i] = parse_index(chunk_op.data.index)
            if j in column_values:
                column_value = column_values[j]
            else:
                column_value = column_values[j] = parse_index(chunk_op.data.columns,
                                                              store_data=True)
            out_chunk = chunk_op.new_chunk(None, shape=chunk_shape, index=chunk_idx,
                                           index_value=index_value,
                                           columns_value=column_value,
                                           dtypes=chunk_op.data.dtypes)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(None, df.shape, dtypes=op.dtypes,
                                     index_value=df.index_value,
                                     columns_value=df.columns_value,
                                     chunks=out_chunks, nsplits=chunk_size,
                                     **df.extra_params)

    @classmethod
    def execute(cls, ctx, op):
        ctx[op.outputs[0].key] = op.data


def from_pandas(data, chunk_size=None, gpu=False, sparse=False):
    op = DataFrameDataSource(data=data, gpu=gpu, sparse=sparse)
    return op(data.shape, chunk_size=chunk_size)
