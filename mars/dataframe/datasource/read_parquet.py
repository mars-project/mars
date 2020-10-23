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

import os
import pickle

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
from ...filesystem import open_file, glob
from ...serialize import AnyField, BoolField, DictField, ListField,\
    StringField, Int32Field, BytesField
from ..arrays import ArrowStringDtype
from ..operands import DataFrameOperandMixin, DataFrameOperand, OutputType
from ..utils import parse_index, to_arrow_dtypes, contain_arrow_dtype, \
    standardize_range_index


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
            raise RuntimeError('Please install pyarrow fisrt.')
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


class ParqueEngine:
    def read_dtypes(self, f, **kwargs):
        raise NotImplementedError

    def read_to_pandas(self, f, columns=None,
                       use_arrow_dtype=None, **kwargs):
        raise NotImplementedError

    def read_group_to_pandas(self, f, group_index, columns=None,
                             use_arrow_dtype=None, **kwargs):
        raise NotImplementedError


class ArrowEngine(ParqueEngine):
    def read_dtypes(self, f, **kwargs):
        file = pq.ParquetFile(f)
        return file.schema_arrow.empty_table().to_pandas().dtypes

    @classmethod
    def _table_to_pandas(cls, t, use_arrow_dtype=None):
        if use_arrow_dtype:
            df = t.to_pandas(
                types_mapper={pa.string(): ArrowStringDtype()}.get)
        else:
            df = t.to_pandas()
        return df

    def read_to_pandas(self, f, columns=None,
                       use_arrow_dtype=None, **kwargs):
        file = pq.ParquetFile(f)
        t = file.read(columns=columns, **kwargs)
        return self._table_to_pandas(t, use_arrow_dtype=use_arrow_dtype)

    def read_group_to_pandas(self, f, group_index, columns=None,
                             use_arrow_dtype=None, **kwargs):
        file = pq.ParquetFile(f)
        t = file.read_row_group(group_index, columns=columns, **kwargs)
        return self._table_to_pandas(t, use_arrow_dtype=use_arrow_dtype)


class FastpaquetEngine(ParqueEngine):
    def read_dtypes(self, f, **kwargs):
        file = fastparquet.ParquetFile(f)
        dtypes_dict = file._dtypes()
        return pd.Series(dict((c, dtypes_dict[c]) for c in file.columns))

    def read_to_pandas(self, f, columns=None,
                       use_arrow_dtype=None, **kwargs):
        file = fastparquet.ParquetFile(f)
        df = file.to_pandas(**kwargs)
        if use_arrow_dtype:
            df = df.astype(to_arrow_dtypes(df.dtypes).to_dict())

        return df


class DataFrameReadParquet(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.READ_PARQUET

    _path = AnyField('path')
    _engine = StringField('engine')
    _columns = ListField('columns')
    _use_arrow_dtype = BoolField('use_arrow_dtype')
    _groups_as_chunks = BoolField('groups_as_chunks')
    _group_index = Int32Field('group_index')
    _read_kwargs = DictField('read_kwargs')
    _incremental_index = BoolField('incremental_index')
    _storage_options = DictField('storage_options')

    # for chunk
    _partitions = BytesField('partitions')
    _partition_keys = ListField('partition_keys')

    def __init__(self, path=None, engine=None, columns=None, use_arrow_dtype=None,
                 groups_as_chunks=None, group_index=None, incremental_index=None,
                 read_kwargs=None, partitions=None, partition_keys=None,
                 storage_options=None, **kw):
        super().__init__(_path=path, _engine=engine, _columns=columns,
                         _use_arrow_dtype=use_arrow_dtype,
                         _groups_as_chunks=groups_as_chunks,
                         _group_index=group_index,
                         _read_kwargs=read_kwargs,
                         _incremental_index=incremental_index,
                         _partitions=partitions,
                         _partition_keys=partition_keys,
                         _storage_options=storage_options,
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
    def storage_options(self):
        return self._storage_options

    @classmethod
    def _to_arrow_dtypes(cls, dtypes, op):
        if op.use_arrow_dtype is None and not op.gpu and \
                options.dataframe.use_arrow_dtype:  # pragma: no cover
            # check if use_arrow_dtype set on the server side
            dtypes = to_arrow_dtypes(dtypes)
        return dtypes

    @classmethod
    def _tile_partitioned(cls, op):
        out_df = op.outputs[0]
        shape = (np.nan, out_df.shape[1])
        dtypes = cls._to_arrow_dtypes(out_df.dtypes, op)
        dataset = pq.ParquetDataset(op.path)

        chunk_index = 0
        out_chunks = []
        for piece in dataset.pieces:
            chunk_op = op.copy().reset_key()
            chunk_op._path = piece.path
            chunk_op._partitions = pickle.dumps(dataset.partitions)
            chunk_op._partition_keys = piece.partition_keys
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
    def _tile_no_partitioned(cls, op):
        chunk_index = 0
        out_chunks = []
        out_df = op.outputs[0]

        dtypes = cls._to_arrow_dtypes(out_df.dtypes, op)
        shape = (np.nan, out_df.shape[1])

        paths = op.path if isinstance(op.path, (tuple, list)) else \
            glob(op.path, storage_options=op.storage_options)

        for pth in paths:
            if op.groups_as_chunks:
                for group_idx in range(pq.ParquetFile(pth).num_row_groups):
                    chunk_op = op.copy().reset_key()
                    chunk_op._path = pth
                    chunk_op._group_index = group_idx
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
                new_chunk = chunk_op.new_chunk(
                    None, shape=shape, index=(chunk_index, 0),
                    index_value=out_df.index_value,
                    columns_value=out_df.columns_value,
                    dtypes=dtypes)
                out_chunks.append(new_chunk)
                chunk_index += 1

        if op.incremental_index:
            out_chunks = standardize_range_index(out_chunks)

        new_op = op.copy()
        nsplits = ((np.nan,) * len(out_chunks), (out_df.shape[1],))
        return new_op.new_dataframes(None, out_df.shape, dtypes=dtypes,
                                     index_value=out_df.index_value,
                                     columns_value=out_df.columns_value,
                                     chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def tile(cls, op):
        if os.path.isdir(op.path):
            return cls._tile_partitioned(op)
        else:
            return cls._tile_no_partitioned(op)

    @classmethod
    def _execute_partitioned(cls, ctx, op):
        out = op.outputs[0]
        partitions = pickle.loads(op.partitions)
        piece = pq.ParquetDatasetPiece(op.path, partition_keys=op.partition_keys,
                                       open_file_func=open_file)
        df = piece.read(partitions=partitions).to_pandas()
        ctx[out.key] = df

    @classmethod
    def execute(cls, ctx, op):
        out = op.outputs[0]
        path = op.path

        if op.partitions is not None:
            return cls._execute_partitioned(ctx, op)

        engine = get_engine(op.engine)
        with open_file(path, storage_options=op.storage_options) as f:
            use_arrow_dtype = contain_arrow_dtype(out.dtypes)
            if op.groups_as_chunks:
                df = engine.read_group_to_pandas(f, op.group_index, columns=op.columns,
                                                 use_arrow_dtype=use_arrow_dtype,
                                                 **op.read_kwargs or dict())
            else:
                df = engine.read_to_pandas(f, columns=op.columns,
                                           use_arrow_dtype=use_arrow_dtype,
                                           **op.read_kwargs or dict())

            ctx[out.key] = df

    def __call__(self, index_value=None, columns_value=None, dtypes=None):
        shape = (np.nan, len(dtypes))
        return self.new_dataframe(None, shape, dtypes=dtypes, index_value=index_value,
                                  columns_value=columns_value)


def read_parquet(path, engine: str = "auto", columns=None,
                 groups_as_chunks=False, use_arrow_dtype=None,
                 incremental_index=False, storage_options=None,
                 **kwargs):
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
        Create a new RangeIndex if csv doesn't contain index columns.
    use_arrow_dtype: bool, default None
        If True, use arrow dtype to store columns.
    storage_options: dict, optional
        Options for storage connection.
    **kwargs
        Any additional kwargs are passed to the engine.

    Returns
    -------
    Mars DataFrame
    """

    engine_type = check_engine(engine)
    engine = get_engine(engine_type)

    if os.path.isdir(path):
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
                              storage_options=storage_options)
    return op(index_value=index_value, columns_value=columns_value,
              dtypes=dtypes)
