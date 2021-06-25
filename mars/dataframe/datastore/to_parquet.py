#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import pandas as pd

from ... import opcodes as OperandDef
from ...core import recursive_tile
from ...lib.filesystem import open_file, get_fs
from ...serialization.serializables import KeyField, AnyField, StringField, ListField, \
    BoolField, DictField
from ...utils import has_unknown_shape
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index
from ..datasource.read_parquet import check_engine

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pq = None
    pa = None


class DataFrameToParquet(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.TO_PARQUET

    _input = KeyField('input')
    _path = AnyField('path')
    _engine = StringField('engine')
    _index = BoolField('index')
    _compression = AnyField('compression')
    _partition_cols = ListField('partition_cols')
    _additional_kwargs = DictField('additional_kwargs')
    _storage_options = DictField('storage_options')

    def __init__(self, path=None, engine=None, index=None, compression=None,
                 partition_cols=None, storage_options=None,
                 additional_kwargs=None, **kw):
        super().__init__(_path=path, _engine=engine, _index=index,
                         _compression=compression, _partition_cols=partition_cols,
                         _storage_options=storage_options,
                         _additional_kwargs=additional_kwargs,
                         **kw)

    @property
    def input(self):
        return self._input

    @property
    def path(self):
        return self._path

    @property
    def engine(self):
        return self._engine

    @property
    def index(self):
        return self._index

    @property
    def compression(self):
        return self._compression

    @property
    def partition_cols(self):
        return self._partition_cols

    @property
    def storage_options(self):
        return self._storage_options

    @property
    def additional_kwargs(self):
        return self._additional_kwargs

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    @classmethod
    def _get_path(cls, path, i):
        if '*' not in path:
            return path
        return path.replace('*', str(i))

    @classmethod
    def tile(cls, op):
        in_df = op.input
        out_df = op.outputs[0]

        # make sure only 1 chunk on the column axis
        if in_df.chunk_shape[1] > 1:
            if has_unknown_shape(in_df):
                yield
            in_df = yield from recursive_tile(
                in_df.rechunk({1: in_df.shape[1]}))

        out_chunks = []
        for chunk in in_df.chunks:
            chunk_op = op.copy().reset_key()
            index_value = parse_index(chunk.index_value.to_pandas()[:0], chunk)
            out_chunk = chunk_op.new_chunk([chunk], shape=(0, 0),
                                           index_value=index_value,
                                           columns_value=out_df.columns_value,
                                           dtypes=out_df.dtypes,
                                           index=chunk.index)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        params = out_df.params.copy()
        params.update(dict(chunks=out_chunks, nsplits=((0,) * in_df.chunk_shape[0], (0,))))
        return new_op.new_tileables([in_df], **params)

    @classmethod
    def execute(cls, ctx, op):
        df = ctx[op.input.key]
        out = op.outputs[0]
        i = op.outputs[0].index[0]
        path = op.path
        has_wildcard = False
        if '*' in path:
            path = path.replace('*', str(i))
            has_wildcard = True

        if op.partition_cols is None:
            if not has_wildcard:
                fs = get_fs(path, op.storage_options)
                path = fs.pathsep.join([path.rstrip(fs.pathsep), f'{i}.parquet'])
            if op.engine == 'fastparquet':
                df.to_parquet(path, engine=op.engine, compression=op.compression,
                              index=op.index, open_with=open_file, **op.additional_kwargs)
            else:
                with open_file(path, mode='wb', storage_options=op.storage_options) as f:
                    df.to_parquet(f, engine=op.engine, compression=op.compression,
                                  index=op.index, **op.additional_kwargs or dict())
        else:
            if op.engine == 'pyarrow':
                pq.write_to_dataset(pa.Table.from_pandas(df), path,
                                    partition_cols=op.partition_cols)
            else:  # pragma: no cover
                raise NotImplementedError('Only support pyarrow engine when '
                                          'specify `partition_cols`.')

        ctx[out.key] = pd.DataFrame()

    def __call__(self, df):
        index_value = parse_index(df.index_value.to_pandas()[:0], df)
        columns_value = parse_index(df.columns_value.to_pandas()[:0], store_data=True)
        return self.new_dataframe([df], shape=(0, 0), dtypes=df.dtypes[:0],
                                  index_value=index_value, columns_value=columns_value)


def to_parquet(df, path, engine='auto', compression='snappy', index=None,
               partition_cols=None, **kwargs):
    """
    Write a DataFrame to the binary parquet format, each chunk will be
    written to a Parquet file.

    Parameters
    ----------
    path : str or file-like object
        If path is a string with wildcard e.g. '/to/path/out-*.parquet',
        `to_parquet` will try to write multiple files, for instance,
        chunk (0, 0) will write data into '/to/path/out-0.parquet'.
        If path is a string without wildcard, we will treat it as a directory.

    engine : {'auto', 'pyarrow', 'fastparquet'}, default 'auto'
        Parquet library to use. The default behavior is to try 'pyarrow',
        falling back to 'fastparquet' if 'pyarrow' is unavailable.

    compression : {'snappy', 'gzip', 'brotli', None}, default 'snappy'
        Name of the compression to use. Use ``None`` for no compression.

    index : bool, default None
        If ``True``, include the dataframe's index(es) in the file output.
        If ``False``, they will not be written to the file.
        If ``None``, similar to ``True`` the dataframe's index(es)
        will be saved. However, instead of being saved as values,
        the RangeIndex will be stored as a range in the metadata so it
        doesn't require much space and is faster. Other indexes will
        be included as columns in the file output.

    partition_cols : list, optional, default None
        Column names by which to partition the dataset.
        Columns are partitioned in the order they are given.
        Must be None if path is not a string.

    **kwargs
        Additional arguments passed to the parquet library.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    >>> df.to_parquet('*.parquet.gzip',
    ...               compression='gzip').execute()  # doctest: +SKIP
    >>> md.read_parquet('*.parquet.gzip').execute()  # doctest: +SKIP
       col1  col2
    0     1     3
    1     2     4

    >>> import io
    >>> f = io.BytesIO()
    >>> df.to_parquet(f).execute()
    >>> f.seek(0)
    0
    >>> content = f.read()
    """
    engine = check_engine(engine)
    op = DataFrameToParquet(path=path, engine=engine, compression=compression, index=index,
                            partition_cols=partition_cols, additional_kwargs=kwargs)
    return op(df)
