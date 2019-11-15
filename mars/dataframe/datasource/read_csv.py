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

from io import BytesIO

import pandas as pd
import numpy as np

from ... import opcodes as OperandDef
from ...config import options
from ...utils import parse_readable_size, lazy_import
from ...serialize import StringField, DictField, ListField, Int32Field, Int64Field, AnyField
from ...filesystem import open_file, file_size, glob
from ..utils import parse_index, build_empty_df
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType


cudf = lazy_import('cudf', globals=globals())


def _find_chunk_start_end(f, offset, size):
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


class DataFrameReadCSV(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.READ_CSV

    _path = AnyField('path')
    _names = ListField('names')
    _sep = StringField('sep')
    _header = AnyField('header')
    _index_col = Int32Field('index_col')
    _compression = StringField('compression')
    _offset = Int64Field('offset')
    _size = Int64Field('size')

    _storage_options = DictField('storage_options')

    def __init__(self, path=None, names=None, sep=None, header=None, index_col=None, compression=None,
                 offset=None, size=None, gpu=None, storage_options=None, **kw):
        super(DataFrameReadCSV, self).__init__(_path=path, _names=names, _sep=sep, _header=header,
                                               _index_col=index_col, _compression=compression,
                                               _offset=offset, _size=size, _gpu=gpu,
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
                                       columns_value=df.columns_value, dtypes=df.dtypes)
        new_op = op.copy()
        nsplits = ((np.nan,), (df.shape[1],))
        return new_op.new_dataframes(None, df.shape, dtypes=df.dtypes,
                                     index_value=df.index_value,
                                     columns_value=df.columns_value,
                                     chunks=[new_chunk], nsplits=nsplits)

    @classmethod
    def tile(cls, op):
        if op.compression:
            return cls._tile_compressed(op)

        df = op.outputs[0]
        chunk_bytes = df.extra_params.chunk_bytes
        chunk_bytes = int(parse_readable_size(chunk_bytes)[0])

        paths = op.path if isinstance(op.path, (tuple, list)) else [op.path]

        out_chunks = []
        index_num = 0
        for path in paths:
            total_bytes = file_size(path)
            offset = 0
            for _ in range(int(np.ceil(total_bytes * 1.0 / chunk_bytes))):
                chunk_op = op.copy().reset_key()
                chunk_op._path = path
                chunk_op._offset = offset
                chunk_op._size = min(chunk_bytes, total_bytes - offset)
                shape = (np.nan, len(df.dtypes))
                new_chunk = chunk_op.new_chunk(None, shape=shape, index=(index_num, 0), index_value=df.index_value,
                                               columns_value=df.columns_value, dtypes=df.dtypes)
                out_chunks.append(new_chunk)
                index_num += 1
                offset += chunk_bytes

        new_op = op.copy()
        nsplits = ((np.nan,) * len(out_chunks), (df.shape[1],))
        return new_op.new_dataframes(None, df.shape, dtypes=df.dtypes,
                                     index_value=df.index_value,
                                     columns_value=df.columns_value,
                                     chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        xdf = cudf if op.gpu else pd
        out_df = op.outputs[0]
        csv_kwargs = op.extra_params.copy()

        with open_file(op.path, compression=op.compression, storage_options=op.storage_options) as f:
            if op.compression is not None:
                # As we specify names and dtype, we need to skip header rows
                csv_kwargs['skiprows'] = 1 if op.header == 'infer' else op.header
                df = xdf.read_csv(BytesIO(f.read()), sep=op.sep, names=op.names, index_col=op.index_col,
                                  dtype=out_df.dtypes.to_dict(), **csv_kwargs)
            else:
                start, end = _find_chunk_start_end(f, op.offset, op.size)
                f.seek(start)
                b = BytesIO(f.read(end - start))
                if end == start:
                    # the last chunk may be empty
                    df = build_empty_df(out_df.dtypes)
                else:
                    if start == 0:
                        # The first chunk contains header
                        # As we specify names and dtype, we need to skip header rows
                        csv_kwargs['skiprows'] = 1 if op.header == 'infer' else op.header
                    df = xdf.read_csv(b, sep=op.sep, names=op.names, index_col=op.index_col,
                                      dtype=out_df.dtypes.to_dict(), **csv_kwargs)
        ctx[out_df.key] = df

    def __call__(self, index_value=None, columns_value=None, dtypes=None, chunk_bytes=None):
        shape = (np.nan, len(dtypes))
        return self.new_dataframe(None, shape, dtypes=dtypes, index_value=index_value,
                                  columns_value=columns_value, chunk_bytes=chunk_bytes)


def read_csv(path, names=None, sep=',', index_col=None, compression=None, header='infer', dtype=None,
             chunk_bytes=None, gpu=None, head_bytes='100k', head_lines=None, storage_options=None, **kwargs):
    """
    Read comma-separated values (csv) file(s) into DataFrame.
    :param path: file path(s).
    :param names: List of column names to use. If file contains no header row,
    then you should explicitly pass header=None. Duplicates in this list are not allowed.
    :param sep:Delimiter to use, default is ','.
    :param index_col: Column(s) to use as the row labels of the DataFrame, either given as string name or column index.
    :param compression: For on-the-fly decompression of on-disk data.
    :param header: Row number(s) to use as the column names, and the start of the data.
    :param dtype: Data type for data or columns. E.g. {'a': np.float64, 'b': np.int32, 'c': 'Int64'}
    Use str or object together with suitable na_values settings to preserve and not interpret dtype.
    :param chunk_bytes: Number of chunk bytes.
    :param gpu: If read into cudf DataFrame.
    :param head_bytes: Number of bytes to use in the head of file, mainly for data inference.
    :param head_lines: Number of lines to use in the head of file, mainly for data inference.
    :param storage_options: Options for storage connection.
    :param kwargs:
    :return: Mars DataFrame.
    """
    # infer dtypes and columns
    if isinstance(path, (list, tuple)):
        file_path = path[0]
    else:
        file_path = glob(path)[0]
    with open_file(file_path, compression=compression, storage_options=storage_options) as f:
        if head_lines is not None:
            b = b''.join([f.readline() for _ in range(head_lines)])
        else:
            head_bytes = int(parse_readable_size(head_bytes)[0])
            head_start, head_end = _find_chunk_start_end(f, 0, head_bytes)
            f.seek(head_start)
            b = f.read(head_end - head_start)
        mini_df = pd.read_csv(BytesIO(b), sep=sep, index_col=index_col, dtype=dtype, names=names, header=header)

    if isinstance(mini_df.index, pd.RangeIndex):
        index_value = parse_index(pd.RangeIndex(0))
    else:
        index_value = parse_index(mini_df.index)
    columns_value = parse_index(mini_df.columns, store_data=True)
    if index_col and not isinstance(index_col, int):
        index_col = list(mini_df.columns).index(index_col)
    names = list(mini_df.columns)
    op = DataFrameReadCSV(path=path, names=names, sep=sep, header=header, index_col=index_col,
                          compression=compression, gpu=gpu, storage_options=storage_options, **kwargs)
    chunk_bytes = chunk_bytes or options.chunk_store_limit
    return op(index_value=index_value, columns_value=columns_value,
              dtypes=mini_df.dtypes, chunk_bytes=chunk_bytes)
