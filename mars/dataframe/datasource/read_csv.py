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
from ...config import options
from ...serialize import StringField, DictField, ListField, Int32Field, Int64Field
from ...filesystem import open_file, file_size
from ..utils import parse_index
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType


class DataFrameReadCSV(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.READ_CSV

    _path = StringField('path')
    _names = ListField('names')
    _sep = StringField('sep')
    _header = Int32Field('header')
    _index_col = Int32Field('index_col')
    _compression = StringField('compression')
    _offset = Int64Field('offset')
    _size = Int64Field('size')

    _storage_options = DictField('storage_options')

    def __init__(self, path=None, names=None, sep=None, header=None, index_col=None, compression=None,
                 offset=None, size=None, storage_options=None, **kw):
        super(DataFrameReadCSV, self).__init__(_path=path, _names=names, _sep=sep, _header=header,
                                               _index_col=index_col, _compression=compression,
                                               _offset=offset, _size=size,
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
    def header(self):
        return self._header

    @property
    def index_col(self):
        return self._index_col

    @property
    def compression(self):
        return self._compression

    @property
    def offset(self):
        return self._offset

    @property
    def size(self):
        return self._size

    @property
    def storage_options(self):
        return self._storage_options

    @classmethod
    def _tile_compressed(cls, op):
        # Compression does not support break into small parts
        df = op.outputs[0]
        chunk_op = op.copy().reset_key()
        chunk_op._offset = 0
        chunk_op._size = file_size(op.path)
        shape = df.shape
        new_chunk = chunk_op.new_chunk(None, shape=shape, index=(0, 0), index_value=df.index_value,
                                       columns_value=df.columns, dtypes=df.dtypes)
        new_op = op.copy()
        nsplits = ((np.nan,), (df.shape[1],))
        return new_op.new_dataframes(None, df.shape, dtypes=df.dtypes,
                                     index_value=df.index_value,
                                     columns_value=df.columns,
                                     chunks=[new_chunk], nsplits=nsplits)

    @classmethod
    def _find_start_end(cls, f, offset, size):
        f.seek(offset)
        if f.tell() == 0:
            start = 0
        else:
            f.readline()
            start = f.tell()
        f.seek(offset + size)
        f.readline()
        end = f.tell()
        return start, end

    @classmethod
    def tile(cls, op):
        if op.compression:
            return cls._tile_compressed(op)

        df = op.outputs[0]
        chunk_bytes = df.extra_params.chunk_bytes
        total_bytes = file_size(op.path)
        offset = 0
        out_chunks = []
        for i in range(int(total_bytes / chunk_bytes) + 1):
            chunk_op = op.copy().reset_key()
            chunk_op._offset = offset
            chunk_op._size = min(chunk_bytes, total_bytes - offset)
            shape = (np.nan, len(df.dtypes))
            new_chunk = chunk_op.new_chunk(None, shape=shape, index=(i, 0), index_value=df.index_value,
                                           columns_value=df.columns, dtypes=df.dtypes)
            out_chunks.append(new_chunk)
            offset += chunk_bytes

        new_op = op.copy()
        nsplits = ((np.nan,) * len(out_chunks), (df.shape[1],))
        return new_op.new_dataframes(None, df.shape, dtypes=df.dtypes,
                                     index_value=df.index_value,
                                     columns_value=df.columns,
                                     chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        out_df = op.outputs[0]
        with open_file(op.path, compression=op.compression, storage_options=op.storage_options) as f:
            if op.compression is not None:
                df = pd.read_csv(BytesIO(f.read()), header=op.header, names=op.names, index_col=op.index_col,
                                         dtype=out_df.dtypes.to_dict())
            else:
                start, end = cls._find_start_end(f, op.offset, op.size)
                f.seek(start)
                b = BytesIO(f.read(end - start))
                if end == start:
                    # the last chunk may be empty
                    df = pd.DataFrame(columns=out_df.columns.to_pandas())
                else:
                    if out_df.index == (0, 0):
                        # The first chunk contains header
                        df = pd.read_csv(b, header=op.header, names=op.names, index_col=op.index_col,
                                         dtype=out_df.dtypes.to_dict())
                    else:
                        df = pd.read_csv(b, names=op.names, index_col=op.index_col,
                                         dtype=out_df.dtypes.to_dict())
        ctx[out_df.key] = df

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
        head = f.readline()
        first_row = f.readline()
        mini_df = pd.read_csv(BytesIO(head + first_row), index_col=index_col, dtype=dtype, names=names)

    if isinstance(mini_df.index, pd.RangeIndex):
        index_value = parse_index(pd.RangeIndex(0))
    else:
        index_value = parse_index(mini_df.index)
    columns_value = parse_index(mini_df.columns, store_data=True)
    if index_col and not isinstance(index_col, int):
        index_col = list(mini_df.columns).index(index_col)
    names = list(mini_df.columns)
    op = DataFrameReadCSV(path=path, names=names, sep=sep, header=header or 0, index_col=index_col,
                          compression=compression, storage_options=storage_options, **kwargs)
    chunk_bytes = chunk_bytes or options.chunk_store_limit
    return op(index_value=index_value, columns_value=columns_value,
              dtypes=mini_df.dtypes, chunk_bytes=chunk_bytes)




