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

import os
import pickle
from urllib.parse import urlparse

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None

try:
    import fastparquet
except ImportError:
    fastparquet = None

from ... import opcodes as OperandDef
from ...config import options
from ...lib.filesystem import file_size, get_fs, glob, open_file
from ...serialization.serializables import AnyField, BoolField, DictField, ListField,\
    StringField, Int32Field, Int64Field, BytesField
from ...utils import is_object_dtype
from ..arrays import ArrowStringDtype
from ..operands import OutputType
from ..utils import parse_index, to_arrow_dtypes, contain_arrow_dtype
from .core import IncrementalIndexDatasource, ColumnPruneSupportedDataSourceMixin, \
    IncrementalIndexDataSourceMixin


PARQUET_MEMORY_SCALE = 15
STRING_FIELD_OVERHEAD = 50


def check_engine(engine):
    if engine == 'auto':
        if pa is not None:
            return 'pyarrow'
        elif fastparquet is not None:  # pragma: no cover
            return 'fastparquet'
        else:  # pragma: no cover
            raise RuntimeError('Please install either pyarrow or fastparquet.')
    elif engine == 'pyarrow':
        if pa is None:  # pragma: no cover
            raise RuntimeError('Please install pyarrow first.')
        return engine
    elif engine == 'fastparquet':
        if fastparquet is None:  # pragma: no cover
            raise RuntimeError('Please install fastparquet first.')
        return engine
    else:  # pragma: no cover
        raise RuntimeError('Unsupported engine {} to read parquet.'.format(engine))


def get_engine(engine):
    if engine == 'pyarrow':
        return ArrowEngine()
    elif engine == 'fastparquet':
        return FastpaquetEngine()
    else:  # pragma: no cover
        raise RuntimeError('Unsupported engine {}'.format(engine))


class ParquetEngine:
    def get_row_num(self, f):
        raise NotImplementedError

    def read_dtypes(self, f, **kwargs):
        raise NotImplementedError

    def read_to_pandas(self, f, columns=None, nrows=None,
                       use_arrow_dtype=None, **kwargs):
        raise NotImplementedError

    def read_group_to_pandas(self, f, group_index, columns=None,
                             nrows=None, use_arrow_dtype=None, **kwargs):
        raise NotImplementedError


class ArrowEngine(ParquetEngine):
    def get_row_num(self, f):
        file = pq.ParquetFile(f)
        return file.metadata.num_rows

    def read_dtypes(self, f, **kwargs):
        file = pq.ParquetFile(f)
        return file.schema_arrow.empty_table().to_pandas().dtypes

    @classmethod
    def _table_to_pandas(cls, t, nrows=None, use_arrow_dtype=None):
        if nrows is not None:
            t = t.slice(0, nrows)
        if use_arrow_dtype:
            df = t.to_pandas(
                types_mapper={pa.string(): ArrowStringDtype()}.get)
        else:
            df = t.to_pandas()
        return df

    def read_to_pandas(self, f, columns=None, nrows=None,
                       use_arrow_dtype=None, **kwargs):
        file = pq.ParquetFile(f)
        t = file.read(columns=columns, **kwargs)
        return self._table_to_pandas(t, nrows=nrows,
                                     use_arrow_dtype=use_arrow_dtype)

    def read_group_to_pandas(self, f, group_index, columns=None,
                             nrows=None, use_arrow_dtype=None, **kwargs):
        file = pq.ParquetFile(f)
        t = file.read_row_group(group_index, columns=columns, **kwargs)
        return self._table_to_pandas(t, nrows=nrows,
                                     use_arrow_dtype=use_arrow_dtype)


class FastpaquetEngine(ParquetEngine):
    def get_row_num(self, f):
        file = fastparquet.ParquetFile(f)
        return file.count

    def read_dtypes(self, f, **kwargs):
        file = fastparquet.ParquetFile(f)
        dtypes_dict = file._dtypes()
        return pd.Series(dict((c, dtypes_dict[c]) for c in file.columns))

    def read_to_pandas(self, f, columns=None, nrows=None,
                       use_arrow_dtype=None, **kwargs):
        file = fastparquet.ParquetFile(f)
        df = file.to_pandas(**kwargs)
        if nrows is not None:
            df = df.head(nrows)
        if use_arrow_dtype:
            df = df.astype(to_arrow_dtypes(df.dtypes).to_dict())

        return df


class DataFrameReadParquet(IncrementalIndexDatasource,
                           ColumnPruneSupportedDataSourceMixin,
                           IncrementalIndexDataSourceMixin):
    _op_type_ = OperandDef.READ_PARQUET

    _path = AnyField('path')
    _engine = StringField('engine')
    _columns = ListField('columns')
    _use_arrow_dtype = BoolField('use_arrow_dtype')
    _groups_as_chunks = BoolField('groups_as_chunks')
    _group_index = Int32Field('group_index')
    _read_kwargs = DictField('read_kwargs')
    _incremental_index = BoolField('incremental_index')
    _nrows = Int64Field('nrows')
    _storage_options = DictField('storage_options')
    # for chunk
    _partitions = BytesField('partitions')
    _partition_keys = ListField('partition_keys')
    _num_group_rows = Int64Field('num_group_rows')
    # as read meta may be too time-consuming when number of files is large,
    # thus we only read first file to get row number and raw file size
    _first_chunk_row_num = Int64Field('first_chunk_row_num')
    _first_chunk_raw_bytes = Int64Field('first_chunk_raw_bytes')

    def __init__(self, path=None, engine=None, columns=None, use_arrow_dtype=None,
                 groups_as_chunks=None, group_index=None, incremental_index=None,
                 read_kwargs=None, partitions=None, partition_keys=None,
                 num_group_rows=None, nrows=None,
                 storage_options=None, first_chunk_row_num=None,
                 first_chunk_raw_bytes=None, memory_scale=None, **kw):
        super().__init__(_path=path, _engine=engine, _columns=columns,
                         _use_arrow_dtype=use_arrow_dtype,
                         _groups_as_chunks=groups_as_chunks,
                         _group_index=group_index,
                         _read_kwargs=read_kwargs,
                         _incremental_index=incremental_index,
                         _partitions=partitions,
                         _partition_keys=partition_keys,
                         _num_group_rows=num_group_rows, _nrows=nrows,
                         _storage_options=storage_options,
                         _memory_scale=memory_scale,
                         _first_chunk_row_num=first_chunk_row_num,
                         _first_chunk_raw_bytes=first_chunk_raw_bytes,
                         _output_types=[OutputType.dataframe], **kw)

    @property
    def path(self):
        return self._path

    @property
    def engine(self):
        return self._engine

    @property
    def columns(self):
        return self._columns

    @property
    def use_arrow_dtype(self):
        return self._use_arrow_dtype

    @property
    def groups_as_chunks(self):
        return self._groups_as_chunks

    @property
    def group_index(self):
        return self._group_index

    @property
    def read_kwargs(self):
        return self._read_kwargs

    @property
    def incremental_index(self):
        return self._incremental_index

    @property
    def partitions(self):
        return self._partitions

    @property
    def partition_keys(self):
        return self._partition_keys

    @property
    def num_group_rows(self):
        return self._num_group_rows

    @property
    def storage_options(self):
        return self._storage_options

    @property
    def first_chunk_row_num(self):
        return self._first_chunk_row_num

    @property
    def first_chunk_raw_bytes(self):
        return self._first_chunk_raw_bytes

    def get_columns(self):
        return self._columns

    def set_pruned_columns(self, columns, *, keep_order=None):
        self._columns = columns

    @classmethod
    def _to_arrow_dtypes(cls, dtypes, op):
        if op.use_arrow_dtype is None and not op.gpu and \
                options.dataframe.use_arrow_dtype:  # pragma: no cover
            # check if use_arrow_dtype set on the server side
            dtypes = to_arrow_dtypes(dtypes)
        return dtypes

    @classmethod
    def _tile_partitioned(cls, op: "DataFrameReadParquet"):
        out_df = op.outputs[0]
        shape = (np.nan, out_df.shape[1])
        dtypes = cls._to_arrow_dtypes(out_df.dtypes, op)
        dataset = pq.ParquetDataset(op.path)

        parsed_path = urlparse(op.path)
        if not os.path.exists(op.path) and parsed_path.scheme:
            path_prefix = f'{parsed_path.scheme}://{parsed_path.netloc}'
        else:
            path_prefix = ''

        chunk_index = 0
        out_chunks = []
        first_chunk_row_num, first_chunk_raw_bytes = None, None
        for i, piece in enumerate(dataset.pieces):
            chunk_op = op.copy().reset_key()
            chunk_op._path = chunk_path = path_prefix + piece.path
            chunk_op._partitions = pickle.dumps(dataset.partitions)
            chunk_op._partition_keys = piece.partition_keys
            if i == 0:
                first_chunk_raw_bytes = file_size(chunk_path, op.storage_options)
                first_chunk_row_num = piece.get_metadata().num_rows
            chunk_op._first_chunk_row_num = first_chunk_row_num
            chunk_op._first_chunk_raw_bytes = first_chunk_raw_bytes
            new_chunk = chunk_op.new_chunk(
                None, shape=shape, index=(chunk_index, 0),
                index_value=out_df.index_value,
                columns_value=out_df.columns_value,
                dtypes=dtypes)
            out_chunks.append(new_chunk)
            chunk_index += 1

        new_op = op.copy()
        nsplits = ((np.nan,) * len(out_chunks), (out_df.shape[1],))
        return new_op.new_dataframes(None, out_df.shape, dtypes=dtypes,
                                     index_value=out_df.index_value,
                                     columns_value=out_df.columns_value,
                                     chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def _tile_no_partitioned(cls, op: "DataFrameReadParquet"):
        chunk_index = 0
        out_chunks = []
        out_df = op.outputs[0]

        dtypes = cls._to_arrow_dtypes(out_df.dtypes, op)
        shape = (np.nan, out_df.shape[1])

        paths = op.path if isinstance(op.path, (tuple, list)) else \
            glob(op.path, storage_options=op.storage_options)

        first_chunk_row_num, first_chunk_raw_bytes = None, None
        for i, pth in enumerate(paths):
            if i == 0:
                with open_file(pth, storage_options=op.storage_options) as f:
                    first_chunk_row_num = get_engine(op.engine).get_row_num(f)
                first_chunk_raw_bytes = file_size(pth, storage_options=op.storage_options)

            if op.groups_as_chunks:
                num_row_groups = pq.ParquetFile(pth).num_row_groups
                for group_idx in range(num_row_groups):
                    chunk_op = op.copy().reset_key()
                    chunk_op._path = pth
                    chunk_op._group_index = group_idx
                    chunk_op._first_chunk_row_num = first_chunk_row_num
                    chunk_op._first_chunk_raw_bytes = first_chunk_raw_bytes
                    chunk_op._num_group_rows = num_row_groups
                    new_chunk = chunk_op.new_chunk(
                        None, shape=shape, index=(chunk_index, 0),
                        index_value=out_df.index_value,
                        columns_value=out_df.columns_value,
                        dtypes=dtypes)
                    out_chunks.append(new_chunk)
                    chunk_index += 1
            else:
                chunk_op = op.copy().reset_key()
                chunk_op._path = pth
                chunk_op._first_chunk_row_num = first_chunk_row_num
                chunk_op._first_chunk_raw_bytes = first_chunk_raw_bytes
                new_chunk = chunk_op.new_chunk(
                    None, shape=shape, index=(chunk_index, 0),
                    index_value=out_df.index_value,
                    columns_value=out_df.columns_value,
                    dtypes=dtypes)
                out_chunks.append(new_chunk)
                chunk_index += 1

        new_op = op.copy()
        nsplits = ((np.nan,) * len(out_chunks), (out_df.shape[1],))
        return new_op.new_dataframes(None, out_df.shape, dtypes=dtypes,
                                     index_value=out_df.index_value,
                                     columns_value=out_df.columns_value,
                                     chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def _tile(cls, op):
        if get_fs(op.path, op.storage_options).isdir(op.path):
            return cls._tile_partitioned(op)
        else:
            return cls._tile_no_partitioned(op)

    @classmethod
    def _execute_partitioned(cls, ctx, op: "DataFrameReadParquet"):
        out = op.outputs[0]
        partitions = pickle.loads(op.partitions)
        piece = pq.ParquetDatasetPiece(op.path, partition_keys=op.partition_keys,
                                       open_file_func=open_file)
        table = piece.read(partitions=partitions)
        if op.nrows is not None:
            table = table.slice(0, op.nrows)
        ctx[out.key] = table.to_pandas()

    @classmethod
    def execute(cls, ctx, op: "DataFrameReadParquet"):
        out = op.outputs[0]
        path = op.path

        if op.partitions is not None:
            return cls._execute_partitioned(ctx, op)

        engine = get_engine(op.engine)
        with open_file(path, storage_options=op.storage_options) as f:
            use_arrow_dtype = contain_arrow_dtype(out.dtypes)
            if op.groups_as_chunks:
                df = engine.read_group_to_pandas(f, op.group_index, columns=op.columns,
                                                 nrows=op.nrows,
                                                 use_arrow_dtype=use_arrow_dtype,
                                                 **op.read_kwargs or dict())
            else:
                df = engine.read_to_pandas(f, columns=op.columns,
                                           nrows=op.nrows,
                                           use_arrow_dtype=use_arrow_dtype,
                                           **op.read_kwargs or dict())

            ctx[out.key] = df

    @classmethod
    def estimate_size(cls, ctx, op: "DataFrameReadParquet"):
        first_chunk_row_num = op.first_chunk_row_num
        first_chunk_raw_bytes = op.first_chunk_raw_bytes
        raw_bytes = file_size(op.path, storage_options=op.storage_options)
        if op.num_group_rows:
            raw_bytes = np.ceil(np.divide(raw_bytes, op.num_group_rows)) \
                .astype(np.int64).item()

        estimated_row_num = np.ceil(
            first_chunk_row_num * (raw_bytes / first_chunk_raw_bytes))\
            .astype(np.int64).item()
        phy_size = raw_bytes * (op.memory_scale or PARQUET_MEMORY_SCALE)
        n_strings = len([dt for dt in op.outputs[0].dtypes if is_object_dtype(dt)])
        pd_size = phy_size + n_strings * estimated_row_num * STRING_FIELD_OVERHEAD
        ctx[op.outputs[0].key] = (pd_size, pd_size + phy_size)

    def __call__(self, index_value=None, columns_value=None, dtypes=None):
        shape = (np.nan, len(dtypes))
        return self.new_dataframe(None, shape, dtypes=dtypes, index_value=index_value,
                                  columns_value=columns_value)


def read_parquet(path, engine: str = "auto", columns=None,
                 groups_as_chunks=False, use_arrow_dtype=None,
                 incremental_index=False, storage_options=None,
                 memory_scale=None, **kwargs):
    """
    Load a parquet object from the file path, returning a DataFrame.

    Parameters
    ----------
    path : str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL.
        For file URLs, a host is expected. A local file could be:
        ``file://localhost/path/to/table.parquet``.
        A file URL can also be a path to a directory that contains multiple
        partitioned parquet files. Both pyarrow and fastparquet support
        paths to directories as well as file URLs. A directory path could be:
        ``file://localhost/path/to/tables``.
        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handler (e.g. via builtin ``open`` function)
        or ``StringIO``.
    engine : {'auto', 'pyarrow', 'fastparquet'}, default 'auto'
        Parquet library to use. The default behavior is to try 'pyarrow',
        falling back to 'fastparquet' if 'pyarrow' is unavailable.
    columns : list, default=None
        If not None, only these columns will be read from the file.
    groups_as_chunks : bool, default False
        if True, each row group correspond to a chunk.
        if False, each file correspond to a chunk.
        Only available for 'pyarrow' engine.
    incremental_index: bool, default False
        If index_col not specified, ensure range index incremental,
        gain a slightly better performance if setting False.
    use_arrow_dtype: bool, default None
        If True, use arrow dtype to store columns.
    storage_options: dict, optional
        Options for storage connection.
    memory_scale: int, optional
        Scale that real memory occupation divided with raw file size.
    **kwargs
        Any additional kwargs are passed to the engine.

    Returns
    -------
    Mars DataFrame
    """

    engine_type = check_engine(engine)
    engine = get_engine(engine_type)

    if get_fs(path, storage_options).isdir(path):
        # If path is a directory, we will read as a partitioned datasets.
        if engine_type != 'pyarrow':
            raise TypeError('Only support pyarrow engine when reading from'
                            'partitioned datasets.')
        dataset = pq.ParquetDataset(path)
        dtypes = dataset.schema.to_arrow_schema().empty_table().to_pandas().dtypes
        for partition in dataset.partitions:
            dtypes[partition.name] = pd.CategoricalDtype()
    else:
        if not isinstance(path, list):
            file_path = glob(path, storage_options=storage_options)[0]
        else:
            file_path = path[0]

        with open_file(file_path, storage_options=storage_options) as f:
            dtypes = engine.read_dtypes(f)

        if columns:
            dtypes = dtypes[columns]

        if use_arrow_dtype is None:
            use_arrow_dtype = options.dataframe.use_arrow_dtype
        if use_arrow_dtype:
            dtypes = to_arrow_dtypes(dtypes)

    index_value = parse_index(pd.RangeIndex(-1))
    columns_value = parse_index(dtypes.index, store_data=True)
    op = DataFrameReadParquet(path=path, engine=engine_type, columns=columns,
                              groups_as_chunks=groups_as_chunks,
                              use_arrow_dtype=use_arrow_dtype,
                              read_kwargs=kwargs,
                              incremental_index=incremental_index,
                              storage_options=storage_options,
                              memory_scale=memory_scale)
    return op(index_value=index_value, columns_value=columns_value,
              dtypes=dtypes)
