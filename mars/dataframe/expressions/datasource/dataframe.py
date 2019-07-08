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

import itertools

from .... import opcodes as OperandDef
from ....serialize import DataFrameField, SeriesField
from ....config import options
from ....compat import izip
from ....tensor.expressions.utils import get_chunk_slices
from ..utils import decide_dataframe_chunk_sizes, parse_index
from ..core import DataFrameOperand, DataFrameOperandMixin, ObjectType

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pass


class DataFrameDataSource(DataFrameOperand, DataFrameOperandMixin):
    """
    Represents data from pandas DataFrame
    """

    _op_type_ = OperandDef.DATAFRAME_DATA_SOURCE

    _data = DataFrameField('data')
    _dtypes = SeriesField('dtypes')

    def __init__(self, data=None, dtypes=None, gpu=None, sparse=None, **kw):
        if dtypes is None and data is not None:
            dtypes = data.dtypes
        super(DataFrameDataSource, self).__init__(_data=data, _dtypes=dtypes,
                                                  _gpu=gpu, _sparse=sparse,
                                                  _object_type=ObjectType.dataframe, **kw)

    @property
    def data(self):
        return self._data

    @property
    def dtypes(self):
        return self._dtypes

    def __call__(self, shape, index_value=None, columns_value=None, chunk_size=None):
        if index_value is None and columns_value is None:
            index_value = parse_index(self._data.index)
            columns_value = parse_index(self._data.columns, store_data=True)

        return self.new_dataframe(None, shape, dtypes=self.dtypes,
                                  index_value=index_value,
                                  columns_value=columns_value,
                                  raw_chunk_size=chunk_size)

    @classmethod
    def tile(cls, op):
        df = op.outputs[0]
        raw_df = pd.DataFrame(op.data)

        memory_usage = raw_df.memory_usage(index=False, deep=True)
        chunk_size = df.extra_params.raw_chunk_size or options.tensor.chunk_size
        chunk_size = decide_dataframe_chunk_sizes(df.shape, chunk_size, memory_usage)
        chunk_size_idxes = (range(len(size)) for size in chunk_size)

        out_chunks = []
        for chunk_shape, chunk_idx in izip(itertools.product(*chunk_size),
                                           itertools.product(*chunk_size_idxes)):
            chunk_op = op.copy().reset_key()
            slc = get_chunk_slices(chunk_size, chunk_idx)
            chunk_op._data = raw_df.iloc[slc]
            chunk_op._dtypes = chunk_op._data.dtypes
            out_chunk = chunk_op.new_chunk(None, shape=chunk_shape, index=chunk_idx,
                                           index_value=parse_index(chunk_op.data.index),
                                           columns_value=parse_index(chunk_op.data.columns,
                                                                     store_data=True),
                                           dtypes=chunk_op._data.dtypes)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(None, df.shape, dtypes=op.dtypes,
                                     index_value=df.index_value,
                                     columns_value=df.columns,
                                     chunks=out_chunks, nsplits=chunk_size)


def from_pandas(data, chunk_size=None, gpu=None, sparse=False):
    op = DataFrameDataSource(data=data, gpu=gpu, sparse=sparse)
    return op(data.shape, chunk_size=chunk_size)


def from_tensor(tensor, chunk_size=None, gpu=None, sparse=False):
    if tensor.ndim != 2:
        raise NotImplementedError('Not support create DataFrame from {0} dims tensor', format(tensor.ndim))

    # fetch the data from tensor firstly
    data = tensor.execute()
    op = DataFrameDataSource(data=data, dtypes=data.dtype, gpu=gpu, sparse=sparse)

    # make index/column value if create DataFrame from tensor
    return op(tensor.shape, index_value=list(range(tensor.shape[0])), columns_value=list(range(tensor.shape[1])),
              chunk_size=chunk_size)
