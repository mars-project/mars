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
from ...serialization.serializables import SeriesField, DataTypeField
from ...config import options
from ...core import OutputType
from ...tensor.utils import get_chunk_slices
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index, decide_series_chunk_size, is_cudf


class SeriesDataSource(DataFrameOperand, DataFrameOperandMixin):
    """
    Represents data from pandas Series
    """

    _op_type_ = OperandDef.SERIES_DATA_SOURCE

    data = SeriesField('data')
    dtype = DataTypeField('dtype')

    def __init__(self, data=None, dtype=None, gpu=None, **kw):
        if dtype is None and data is not None:
            dtype = data.dtype
        if gpu is None and is_cudf(data):  # pragma: no cover
            gpu = True
        super().__init__(data=data, dtype=dtype, gpu=gpu,
                         _output_types=[OutputType.series], **kw)

    def __call__(self, shape, chunk_size=None):
        return self.new_series(None, shape=shape, dtype=self.dtype,
                               index_value=parse_index(self.data.index),
                               name=self.data.name, raw_chunk_size=chunk_size)

    @classmethod
    def tile(cls, op: "SeriesDataSource"):
        series = op.outputs[0]
        raw_series = op.data

        memory_usage = raw_series.memory_usage(index=False, deep=True)
        chunk_size = series.extra_params.raw_chunk_size or options.chunk_size
        chunk_size = decide_series_chunk_size(series.shape, chunk_size, memory_usage)
        chunk_size_idxes = (range(len(size)) for size in chunk_size)

        out_chunks = []
        for chunk_shape, chunk_idx in zip(itertools.product(*chunk_size),
                                          itertools.product(*chunk_size_idxes)):
            chunk_op = op.copy().reset_key()
            slc = get_chunk_slices(chunk_size, chunk_idx)
            if is_cudf(raw_series):  # pragma: no cover
                chunk_op.data = raw_series.iloc[slc[0]]
            else:
                chunk_op.data = raw_series.iloc[slc]
            chunk_op.dtype = chunk_op.data.dtype
            out_chunk = chunk_op.new_chunk(None, shape=chunk_shape, dtype=op.dtype, index=chunk_idx,
                                           index_value=parse_index(chunk_op.data.index),
                                           name=series.name)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_seriess(None, series.shape, dtype=op.dtype,
                                  index_value=series.index_value,
                                  name=series.name, chunks=out_chunks, nsplits=chunk_size)

    @classmethod
    def execute(cls, ctx, op):
        ctx[op.outputs[0].key] = op.data


def from_pandas(data, chunk_size=None, gpu=False, sparse=False):
    op = SeriesDataSource(data=data, gpu=gpu, sparse=sparse)
    return op(data.shape, chunk_size=chunk_size)
