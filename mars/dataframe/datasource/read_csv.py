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

from io import BytesIO

import pandas as pd
import numpy as np

from ... import opcodes as OperandDef
from ...serialize import StringField, DictField, ListField, Int32Field, Int64Field
from ...filesystem import open_file, file_size
from ..utils import parse_index
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType


class DataFrameReadCSV(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.READ_CSV

    _path = StringField('path')
    _names = ListField('names')
    _sep = StringField('sep')
    _index_col = Int32Field('index_col')
    _compression = StringField('compression')
    _offset = Int64Field('offset')
    _size = Int64Field('size')

    _storage_options = DictField('storage_options')

    def __init__(self, path=None, names=None, sep=None, index_col=None, compression=None,
                 offset=None, size=None, storage_options=None, **kw):
        super(DataFrameReadCSV, self).__init__(_path=path, _names=names, _sep=sep, _index_col=index_col,
                                               _compression=compression, _offset=offset, _size=size,
                                               _storage_options=storage_options,
                                               _object_type=ObjectType.dataframe, **kw)

    @property
    def path(self):
        return self._path

    @property
    def names(self):
        return self._names

    @property
    def sep(self):
        return self._sep

    @property
    def index_col(self):
        return self._index_col

    @property
    def compression(self):
        return self._compression

    @property
    def storage_options(self):
        return self._storage_options

    @classmethod
    def tile(cls, op):
        df = op.outputs[0]

        chunk_bytes = df.extra_params.chunk_bytes
        total_bytes = file_size(op.path)
        offset = 0
        out_chunks = []
        for i in range(int(total_bytes / chunk_bytes) + 1):
            chunk_op = op.copy().reset_key()
            chunk_op._offset = offset
            chunk_op._size = chunk_bytes
            shape = (np.nan, len(df.dtypes))
            new_chunk = chunk_op.new_chunk(None, shape=shape, index=(i,), index_value=df.index_value,
                                           columns_value=df.columns_value, dtypes=df.dtypes)
            out_chunks.append(new_chunk)
            offset += chunk_bytes

        new_op = op.copy()
        nsplits = ((np.nan,) * len(out_chunks), df.shape[1])
        return new_op.new_dataframes(None, df.shape, dtypes=op.dtypes,
                                     index_value=df.index_value,
                                     columns_value=df.columns,
                                     chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        with open_file(op.path, compression=op.compression, storage_options=op.storage_options) as f:
            offset = op.offset
            f.seek(offset)
            if f.tell() == 0:
                start = 0
            else:
                f.readline()
                start = f.tell()
            f.seek(offset + op.size)
            f.readline()
            end = f.tell()
            f.seek(start)
            b = BytesIO(f.read(end - start))
            df = pd.read_csv(b, names=op.names, index_col=op.index_col, dtypes=op.dtypes.to_dict())
        ctx[op.outputs[0].key] = df

    def __call__(self, index_value=None, columns_value=None, dtypes=None, chunk_bytes=None):
        shape = (np.nan, len(dtypes))
        return self.new_dataframe(None, shape, dtypes=dtypes, index_value=index_value,
                                  columns_value=columns_value, chunk_bytes=chunk_bytes)


def read_csv(path, names=None, sep=None, index_col=None, compression=None, header=None, dtype=None,
             chunk_bytes=None, storage_options=None, **kwargs):
    # infer dtypes and columns
    with open_file(path, compression=compression, storage_options=storage_options) as f:
        if header:
            [f.readline() for _ in range(header)]
        header = f.readline()
        first_row = f.readline()
        mini_df = pd.read_csv(BytesIO(header + first_row), index_col=index_col, dtype=dtype, names=names)

    if isinstance(mini_df.index, pd.RangeIndex):
        index_value = parse_index(np.nan)
    else:
        index_value = parse_index(mini_df.index)
    columns_value = parse_index(mini_df.columns)
    if not isinstance(index_col, int):
        index_col = list(mini_df.columns).index(index_col)
    op = DataFrameReadCSV(path=path, names=names, sep=sep, index_col=index_col, compression=compression,
                          storage_options=storage_options, **kwargs)

    return op(index_value=index_value, columns_value=columns_value,
              dtypes=mini_df.dtypes, chunk_bytes=chunk_bytes)




