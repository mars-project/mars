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

import binascii
import datetime
import pickle
import uuid
from typing import List, Union

import numpy as np
import pandas as pd
import cloudpickle

from ... import opcodes as OperandDef
from ...config import options
from ...core.context import Context
from ...serialization.serializables import StringField, AnyField, BoolField, ListField, \
    Int64Field, Float64Field, BytesField
from ...tensor.utils import normalize_chunk_sizes
from ...typing import OperandType, TileableType
from ..arrays import ArrowStringDtype
from ..operands import OutputType
from ..utils import parse_index, create_sa_connection, to_arrow_dtypes
from .core import IncrementalIndexDatasource, ColumnPruneSupportedDataSourceMixin, \
    IncrementalIndexDataSourceMixin


class DataFrameReadSQL(IncrementalIndexDatasource,
                       ColumnPruneSupportedDataSourceMixin,
                       IncrementalIndexDataSourceMixin):
    _op_type_ = OperandDef.READ_SQL

    _table_or_sql = AnyField('table_or_sql')
    _selectable = BytesField('selectable', on_serialize=pickle.dumps,
                             on_deserialize=pickle.loads)
    _con = AnyField('con')
    _schema = StringField('schema')
    _index_col = AnyField('index_col')
    _coerce_float = BoolField('coerce_float')
    _parse_dates = AnyField('parse_dates')
    _columns = ListField('columns')
    _engine_kwargs = BytesField('engine_kwargs', on_serialize=cloudpickle.dumps,
                                on_deserialize=cloudpickle.loads)
    _row_memory_usage = Float64Field('row_memory_usage')
    _method = StringField('method')
    _incremental_index = BoolField('incremental_index')
    _use_arrow_dtype = BoolField('use_arrow_dtype')
    # for chunks
    _offset = Int64Field('offset')
    _partition_col = StringField('partition_col')
    _num_partitions = Int64Field('num_partitions')
    _low_limit = AnyField('low_limit')
    _high_limit = AnyField('high_limit')
    _left_end = BoolField('left_end')
    _right_end = BoolField('right_end')
    _nrows = Int64Field('nrows')

    def __init__(self, table_or_sql=None, selectable=None, con=None, schema=None,
                 index_col=None, coerce_float=None, parse_dates=None, columns=None,
                 engine_kwargs=None, row_memory_usage=None, method=None,
                 incremental_index=None, use_arrow_dtype=None, offset=None, partition_col=None,
                 num_partitions=None, low_limit=None, high_limit=None, left_end=None,
                 right_end=None, nrows=None, output_types=None, gpu=None, **kw):
        super().__init__(_table_or_sql=table_or_sql, _selectable=selectable, _con=con,
                         _schema=schema, _index_col=index_col, _coerce_float=coerce_float,
                         _parse_dates=parse_dates, _columns=columns,
                         _engine_kwargs=engine_kwargs, _row_memory_usage=row_memory_usage,
                         _method=method, _incremental_index=incremental_index,
                         _use_arrow_dtype=use_arrow_dtype, _offset=offset,
                         _partition_col=partition_col, _num_partitions=num_partitions,
                         _low_limit=low_limit, _left_end=left_end, _right_end=right_end,
                         _high_limit=high_limit, _nrows=nrows, _output_types=output_types,
                         _gpu=gpu, **kw)
        if not self.output_types:
            self._output_types = [OutputType.dataframe]

    @property
    def table_or_sql(self):
        return self._table_or_sql

    @property
    def selectable(self):
        return self._selectable

    @property
    def con(self):
        return self._con

    @property
    def schema(self):
        return self._schema

    @property
    def index_col(self):
        return self._index_col

    @property
    def coerce_float(self):
        return self._coerce_float

    @property
    def parse_dates(self):
        return self._parse_dates

    @property
    def columns(self):
        return self._columns

    @property
    def engine_kwargs(self):
        return self._engine_kwargs

    @property
    def row_memory_usage(self):
        return self._row_memory_usage

    @property
    def method(self):
        return self._method

    @property
    def incremental_index(self):
        return self._incremental_index

    @property
    def use_arrow_dtype(self):
        return self._use_arrow_dtype

    @property
    def offset(self):
        return self._offset

    @property
    def partition_col(self):
        return self._partition_col

    @property
    def num_partitions(self):
        return self._num_partitions

    @property
    def low_limit(self):
        return self._low_limit

    @property
    def high_limit(self):
        return self._high_limit

    @property
    def left_end(self):
        return self._left_end

    @property
    def right_end(self):
        return self._right_end

    @property
    def nrows(self):
        return self._nrows

    def get_columns(self):
        return self._columns

    def set_pruned_columns(self, columns, *, keep_order=None):
        self._columns = columns

    def _get_selectable(self, engine_or_conn, columns=None):
        import sqlalchemy as sa
        from sqlalchemy import sql
        from sqlalchemy.exc import SQLAlchemyError

        # process table_name
        if self._selectable is not None:
            selectable = self._selectable
        else:
            if isinstance(self._table_or_sql, sa.Table):
                selectable = self._table_or_sql
                self._table_or_sql = selectable.name
            else:
                m = sa.MetaData()
                try:
                    selectable = sa.Table(self._table_or_sql, m, autoload=True,
                                          autoload_with=engine_or_conn, schema=self._schema)
                except SQLAlchemyError:
                    temp_name_1 = 't1_' + binascii.b2a_hex(uuid.uuid4().bytes).decode()
                    temp_name_2 = 't2_' + binascii.b2a_hex(uuid.uuid4().bytes).decode()
                    if columns:
                        selectable = sql.text(self._table_or_sql).columns(*[sql.column(c) for c in columns]) \
                            .alias(temp_name_2)
                    else:
                        selectable = sql.select(
                            '*', from_obj=sql.text(f'({self._table_or_sql}) AS {temp_name_1}')) \
                            .alias(temp_name_2)
                    self._selectable = selectable
        return selectable

    def _collect_info(self, engine_or_conn, selectable, columns, test_rows):
        from sqlalchemy import sql

        # fetch test DataFrame
        if columns:
            query = sql.select([sql.column(c) for c in columns], from_obj=selectable).limit(test_rows)
        else:
            query = sql.select(selectable.columns, from_obj=selectable).limit(test_rows)
        test_df = pd.read_sql(query, engine_or_conn, index_col=self._index_col,
                              coerce_float=self._coerce_float,
                              parse_dates=self._parse_dates)
        if len(test_df) == 0:
            self._row_memory_usage = None
        else:
            self._row_memory_usage = \
                test_df.memory_usage(deep=True, index=True).sum() / len(test_df)

        if self._method == 'offset':
            # fetch size
            size = list(engine_or_conn.execute(
                sql.select([sql.func.count()]).select_from(selectable)))[0][0]
            shape = (size, test_df.shape[1])
        else:
            shape = (np.nan, test_df.shape[1])

        return test_df, shape

    def __call__(self, test_rows, chunk_size):
        import sqlalchemy as sa
        from sqlalchemy.sql import elements

        with create_sa_connection(self._con, **(self._engine_kwargs or dict())) as con:
            self._con = str(con.engine.url)
            selectable = self._get_selectable(con)

            # process index_col
            index_col = self._index_col
            if index_col is not None:
                if not isinstance(index_col, (list, tuple)):
                    index_col = (index_col,)
                new_index_col = []
                for col in index_col:
                    if isinstance(col, (sa.Column, elements.Label)):
                        new_index_col.append(col.name)
                    elif isinstance(col, str):
                        new_index_col.append(col)
                    elif col is not None:
                        raise TypeError(f'unknown index_col type: {type(col)}')
                self._index_col = new_index_col

            # process columns
            columns = self._columns or []
            new_columns = []
            for col in columns:
                if isinstance(col, str):
                    new_columns.append(col)
                else:
                    new_columns.append(col.name)
            self._columns = new_columns

            if self._columns:
                collect_cols = self._columns + (self._index_col or [])
            else:
                collect_cols = []
            test_df, shape = self._collect_info(con, selectable, collect_cols, test_rows)

            # reconstruct selectable using known column names
            if not collect_cols:
                self._columns = list(test_df.columns)
                if self._selectable is not None:
                    self._selectable = None
                    self._get_selectable(con, columns=self._columns + (self._index_col or []))

            if self.method == 'partition':
                if not self.index_col or self.partition_col not in self.index_col:
                    part_frame = test_df
                else:
                    part_frame = test_df.index.to_frame()

                if not issubclass(part_frame[self.partition_col].dtype.type, (np.number, np.datetime64)):
                    raise TypeError('Type of partition column should be numeric or datetime, '
                                    f'now it is {test_df[self.partition_col].dtype}')

            if isinstance(test_df.index, pd.RangeIndex):
                index_value = parse_index(pd.RangeIndex(shape[0] if not np.isnan(shape[0]) else -1),
                                          str(selectable), self._con)
            else:
                index_value = parse_index(test_df.index)

            columns_value = parse_index(test_df.columns, store_data=True)

            dtypes = test_df.dtypes
            use_arrow_dtype = self._use_arrow_dtype
            if use_arrow_dtype is None:
                use_arrow_dtype = options.dataframe.use_arrow_dtype
            if use_arrow_dtype:
                dtypes = to_arrow_dtypes(dtypes, test_df=test_df)

            return self.new_dataframe(None, shape=shape, dtypes=dtypes,
                                      index_value=index_value,
                                      columns_value=columns_value,
                                      raw_chunk_size=chunk_size)

    @classmethod
    def _tile_offset(cls, op: 'DataFrameReadSQL'):
        df = op.outputs[0]

        if op.row_memory_usage is not None:
            # Data selected
            chunk_size = df.extra_params.raw_chunk_size or options.chunk_size
            if chunk_size is None:
                chunk_size = (int(options.chunk_store_limit / op.row_memory_usage), df.shape[1])
            row_chunk_sizes = normalize_chunk_sizes(df.shape, chunk_size)[0]
        else:
            # No data selected
            row_chunk_sizes = (0,)
        offsets = np.cumsum((0,) + row_chunk_sizes).tolist()

        out_chunks = []
        for i, row_size in enumerate(row_chunk_sizes):
            chunk_op = op.copy().reset_key()
            chunk_op._row_memory_usage = None  # no need for chunk
            offset = chunk_op._offset = offsets[i]
            if df.index_value.has_value():
                # range index
                index_value = parse_index(
                    df.index_value.to_pandas()[offset: offsets[i + 1]])
            else:
                index_value = parse_index(df.index_value.to_pandas(),
                                          op.table_or_sql or str(op.selectable), op.con, i, row_size)
            out_chunk = chunk_op.new_chunk(None, shape=(row_size, df.shape[1]),
                                           columns_value=df.columns_value,
                                           index_value=index_value, dtypes=df.dtypes,
                                           index=(i, 0))
            out_chunks.append(out_chunk)

        nsplits = (row_chunk_sizes, (df.shape[1],))
        new_op = op.copy()
        return new_op.new_dataframes(None, chunks=out_chunks, nsplits=nsplits,
                                     **df.params)

    def _parse_datetime(self, val):
        if isinstance(self.parse_dates, list):
            return pd.to_datetime(val)
        args = self.parse_dates[self.partition_col]
        args = {'format': args} if isinstance(args, str) else args
        return pd.to_datetime(val, **args)

    @classmethod
    def _tile_partition(cls, op: 'DataFrameReadSQL'):
        df = op.outputs[0]

        selectable = op._get_selectable(None)

        if op._low_limit is None or op._high_limit is None:
            import sqlalchemy as sa
            from sqlalchemy import sql

            engine = sa.create_engine(op.con, **(op.engine_kwargs or dict()))
            try:
                part_col = selectable.columns[op.partition_col]
                range_results = engine.execute(sql.select([sql.func.min(part_col), sql.func.max(part_col)]))

                op._low_limit, op._high_limit = next(range_results)
                if op.parse_dates and op.partition_col in op.parse_dates:
                    op._low_limit = op._parse_datetime(op._low_limit)
                    op._high_limit = op._parse_datetime(op._high_limit)
            finally:
                engine.dispose()

        if isinstance(op._low_limit, (datetime.datetime, np.datetime64, pd.Timestamp)):
            seps = pd.date_range(op._low_limit, op._high_limit, op.num_partitions + 1)
        else:
            seps = np.linspace(op._low_limit, op._high_limit, op.num_partitions + 1, endpoint=True)

        out_chunks = []
        for i, (start, end) in enumerate(zip(seps, seps[1:])):
            chunk_op = op.copy().reset_key()
            chunk_op._row_memory_usage = None  # no need for chunk
            chunk_op._num_partitions = None
            chunk_op._low_limit = start
            chunk_op._high_limit = end
            chunk_op._left_end = i == 0
            chunk_op._right_end = i == op.num_partitions - 1

            if df.index_value.has_value():
                # range index
                index_value = parse_index(-1, chunk_op.key, chunk_op.index_value.key)
            else:
                index_value = parse_index(df.index_value.to_pandas(),
                                          str(selectable), op.con, i)
            out_chunk = chunk_op.new_chunk(None, shape=(np.nan, df.shape[1]),
                                           columns_value=df.columns_value,
                                           index_value=index_value, dtypes=df.dtypes,
                                           index=(i, 0))
            out_chunks.append(out_chunk)

        nsplits = ((np.nan,) * len(out_chunks), (df.shape[1],))
        new_op = op.copy()
        return new_op.new_dataframes(None, chunks=out_chunks, nsplits=nsplits,
                                     **df.params)

    @classmethod
    def tile(cls, op: 'DataFrameReadSQL'):
        if op.method == 'offset':
            return cls._tile_offset(op)
        else:
            return cls._tile_partition(op)

    @classmethod
    def post_tile(cls, op: OperandType, results: List[TileableType]):
        if op.method != 'offset':
            # method `offset` knows shape of each chunk
            # just skip incremental process
            return super().post_tile(op, results)

    @classmethod
    def execute(cls, ctx, op: 'DataFrameReadSQL'):
        import sqlalchemy as sa

        def _adapt_datetime(dt):
            if isinstance(dt, np.datetime64):
                return dt.astype('<M8[ms]').astype(datetime.datetime)
            elif isinstance(dt, pd.Timestamp):
                return dt.to_pydatetime()
            return dt

        out = op.outputs[0]

        engine = sa.create_engine(op.con, **(op.engine_kwargs or dict()))
        try:
            selectable = op._get_selectable(engine)

            columns = [selectable.columns[col] for col in op.columns]
            column_names = set(op.columns)
            if op.index_col:
                for icol in op.index_col:
                    if icol not in column_names:
                        columns.append(selectable.columns[icol])

            # convert to python timestamp in case np / pd time types not handled
            op._low_limit = _adapt_datetime(op._low_limit)
            op._high_limit = _adapt_datetime(op._high_limit)

            query = sa.sql.select(columns)
            if op.method == 'partition':
                part_col = selectable.columns[op.partition_col]
                if op.left_end:
                    query = query.where(part_col < op.high_limit)
                elif op.right_end:
                    query = query.where(part_col >= op.low_limit)
                else:
                    query = query.where((part_col >= op.low_limit) & (part_col < op.high_limit))

            if hasattr(selectable, 'primary_key') and len(selectable.primary_key) > 0:
                # if table has primary key, sort as the order
                query = query.order_by(*list(selectable.primary_key))
            elif op.index_col:
                # if no primary key, sort as the index_col
                query = query.order_by(
                    *[selectable.columns[col] for col in op.index_col])
            else:
                # at last, we sort by all the columns
                query = query.order_by(*columns)

            if op.method == 'offset':
                query = query.limit(out.shape[0])
                if op.offset > 0:
                    query = query.offset(op.offset)

            if op.nrows is not None:
                query = query.limit(op.nrows)

            df = pd.read_sql(query, engine, index_col=op.index_col,
                             coerce_float=op.coerce_float,
                             parse_dates=op.parse_dates)
            if op.method == 'offset' and op.index_col is None and op.offset > 0:
                index = pd.RangeIndex(op.offset, op.offset + out.shape[0])
                if op.nrows is not None:
                    index = index[:op.nrows]
                df.index = index

            use_arrow_dtype = op.use_arrow_dtype
            if use_arrow_dtype is None:
                use_arrow_dtype = options.dataframe.use_arrow_dtype
            if use_arrow_dtype:
                dtypes = to_arrow_dtypes(df.dtypes, test_df=df)
                for i in range(len(dtypes)):
                    dtype = dtypes.iloc[i]
                    if isinstance(dtype, ArrowStringDtype):
                        df.iloc[:, i] = df.iloc[:, i].astype(dtype)

            if out.ndim == 2:
                ctx[out.key] = df
            else:
                # this happens when column pruning results in one single series
                ctx[out.key] = df.iloc[:, 0]
        finally:
            engine.dispose()

    @classmethod
    def post_execute(cls, ctx: Union[dict, Context], op: OperandType):
        if op.method != 'offset':
            # method `offset` knows shape of each chunk
            # just skip incremental process
            return super().post_execute(ctx, op)


def _read_sql(table_or_sql, con, schema=None, index_col=None, coerce_float=True,
              params=None, parse_dates=None, columns=None, chunksize=None,
              incremental_index=False, use_arrow_dtype=None,
              test_rows=None, chunk_size=None,
              engine_kwargs=None, partition_col=None, num_partitions=None,
              low_limit=None, high_limit=None):
    if chunksize is not None:
        raise NotImplementedError('read_sql_query with chunksize not supported')
    method = 'offset' if partition_col is None else 'partition'

    op = DataFrameReadSQL(table_or_sql=table_or_sql, selectable=None, con=con, schema=schema,
                          index_col=index_col, coerce_float=coerce_float,
                          params=params, parse_dates=parse_dates, columns=columns,
                          engine_kwargs=engine_kwargs, incremental_index=incremental_index,
                          use_arrow_dtype=use_arrow_dtype, method=method, partition_col=partition_col,
                          num_partitions=num_partitions, low_limit=low_limit,
                          high_limit=high_limit)
    return op(test_rows, chunk_size)


def read_sql(sql, con, index_col=None, coerce_float=True, params=None, parse_dates=None,
             columns=None, chunksize=None, test_rows=5, chunk_size=None, engine_kwargs=None,
             incremental_index=True, partition_col=None, num_partitions=None, low_limit=None,
             high_limit=None):
    """
    Read SQL query or database table into a DataFrame.

    This function is a convenience wrapper around ``read_sql_table`` and
    ``read_sql_query`` (for backward compatibility). It will delegate
    to the specific function depending on the provided input. A SQL query
    will be routed to ``read_sql_query``, while a database table name will
    be routed to ``read_sql_table``. Note that the delegated function might
    have more specific notes about their functionality not listed here.

    Parameters
    ----------
    sql : str or SQLAlchemy Selectable (select or text object)
        SQL query to be executed or a table name.
    con : SQLAlchemy connectable (engine/connection) or database str URI
        or DBAPI2 connection (fallback mode)'

        Using SQLAlchemy makes it possible to use any DB supported by that
        library. If a DBAPI2 object, only sqlite3 is supported. The user is responsible
        for engine disposal and connection closure for the SQLAlchemy connectable. See
        `here <https://docs.sqlalchemy.org/en/13/core/connections.html>`_
    index_col : str or list of strings, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point, useful for SQL result sets.
    params : list, tuple or dict, optional, default: None
        List of parameters to pass to execute method.  The syntax used
        to pass parameters is database driver dependent. Check your
        database driver documentation for which of the five syntax styles,
        described in PEP 249's paramstyle, is supported.
        Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}.
    parse_dates : list or dict, default: None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times, or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    columns : list, default: None
        List of column names to select from SQL table (only used when reading
        a table).
    chunksize : int, default None
        If specified, return an iterator where `chunksize` is the number of rows
        to include in each chunk. Note that this argument is only kept for
        compatibility. If a non-none value passed, an error will be reported.
    test_rows: int, default 5
        The number of rows to fetch for inferring dtypes.
    chunk_size: : int or tuple of ints, optional
        Specifies chunk size for each dimension.
    engine_kwargs: dict, default None
        Extra kwargs to pass to sqlalchemy.create_engine
    incremental_index: bool, default True
        If index_col not specified, ensure range index incremental,
        gain a slightly better performance if setting False.
    partition_col : str, default None
        Specify name of the column to split the result of the query. If
        specified, the range ``[low_limit, high_limit]`` will be divided
        into ``n_partitions`` chunks with equal lengths. We do not
        guarantee the sizes of chunks be equal. When the value is None,
        ``OFFSET`` and ``LIMIT`` clauses will be used to cut the result
        of the query.
    num_partitions : int, default None
        The number of chunks to divide the result of the query into,
        when ``partition_col`` is specified.
    low_limit : default None
        The lower bound of the range of column ``partition_col``. If not
        specified, a query will be executed to query the minimum of
        the column.
    high_limit : default None
        The higher bound of the range of column ``partition_col``. If not
        specified, a query will be executed to query the maximum of
        the column.

    Returns
    -------
    DataFrame

    See Also
    --------
    read_sql_table : Read SQL database table into a DataFrame.
    read_sql_query : Read SQL query into a DataFrame.
    """
    return _read_sql(table_or_sql=sql, con=con, index_col=index_col, coerce_float=coerce_float,
                     params=params, parse_dates=parse_dates, columns=columns,
                     engine_kwargs=engine_kwargs, incremental_index=incremental_index,
                     chunksize=chunksize, test_rows=test_rows, chunk_size=chunk_size,
                     partition_col=partition_col, num_partitions=num_partitions,
                     low_limit=low_limit, high_limit=high_limit)


def read_sql_table(table_name, con, schema=None, index_col=None, coerce_float=True,
                   parse_dates=None, columns=None, chunksize=None, test_rows=5,
                   chunk_size=None, engine_kwargs=None, incremental_index=True,
                   use_arrow_dtype=None, partition_col=None, num_partitions=None,
                   low_limit=None, high_limit=None):
    """
    Read SQL database table into a DataFrame.

    Given a table name and a SQLAlchemy connectable, returns a DataFrame.
    This function does not support DBAPI connections.

    Parameters
    ----------
    table_name : str
        Name of SQL table in database.
    con : SQLAlchemy connectable or str
        A database URI could be provided as as str.
        SQLite DBAPI connection mode not supported.
    schema : str, default None
        Name of SQL schema in database to query (if database flavor
        supports this). Uses default schema if None (default).
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point. Can result in loss of Precision.
    parse_dates : list or dict, default None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    columns : list, default None
        List of column names to select from SQL table.
    chunksize : int, default None
        If specified, returns an iterator where `chunksize` is the number of
        rows to include in each chunk. Note that this argument is only kept
        for compatibility. If a non-none value passed, an error will be
        reported.
    test_rows: int, default 5
        The number of rows to fetch for inferring dtypes.
    chunk_size: : int or tuple of ints, optional
        Specifies chunk size for each dimension.
    engine_kwargs: dict, default None
        Extra kwargs to pass to sqlalchemy.create_engine
    incremental_index: bool, default True
        If index_col not specified, ensure range index incremental,
        gain a slightly better performance if setting False.
    use_arrow_dtype: bool, default None
        If True, use arrow dtype to store columns.
    partition_col : str, default None
        Specify name of the column to split the result of the query. If
        specified, the range ``[low_limit, high_limit]`` will be divided
        into ``n_partitions`` chunks with equal lengths. We do not
        guarantee the sizes of chunks be equal. When the value is None,
        ``OFFSET`` and ``LIMIT`` clauses will be used to cut the result
        of the query.
    num_partitions : int, default None
        The number of chunks to divide the result of the query into,
        when ``partition_col`` is specified.
    low_limit : default None
        The lower bound of the range of column ``partition_col``. If not
        specified, a query will be executed to query the minimum of
        the column.
    high_limit : default None
        The higher bound of the range of column ``partition_col``. If not
        specified, a query will be executed to query the maximum of
        the column.

    Returns
    -------
    DataFrame
        A SQL table is returned as two-dimensional data structure with labeled
        axes.

    See Also
    --------
    read_sql_query : Read SQL query into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.

    Notes
    -----
    Any datetime values with time zone information will be converted to UTC.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> md.read_sql_table('table_name', 'postgres:///db_name')  # doctest:+SKIP
    """
    return _read_sql(table_or_sql=table_name, con=con, schema=schema, index_col=index_col,
                     coerce_float=coerce_float, parse_dates=parse_dates, columns=columns,
                     engine_kwargs=engine_kwargs, incremental_index=incremental_index,
                     use_arrow_dtype=use_arrow_dtype, chunksize=chunksize,
                     test_rows=test_rows, chunk_size=chunk_size,
                     partition_col=partition_col, num_partitions=num_partitions,
                     low_limit=low_limit, high_limit=high_limit)


def read_sql_query(sql, con, index_col=None, coerce_float=True, params=None, parse_dates=None,
                   chunksize=None, test_rows=5, chunk_size=None, engine_kwargs=None,
                   incremental_index=True, use_arrow_dtype=None,
                   partition_col=None, num_partitions=None,
                   low_limit=None, high_limit=None):
    """
    Read SQL query into a DataFrame.

    Returns a DataFrame corresponding to the result set of the query
    string. Optionally provide an `index_col` parameter to use one of the
    columns as the index, otherwise default integer index will be used.

    Parameters
    ----------
    sql : str SQL query or SQLAlchemy Selectable (select or text object)
        SQL query to be executed.
    con : SQLAlchemy connectable(engine/connection), database str URI,
        or sqlite3 DBAPI2 connection
        Using SQLAlchemy makes it possible to use any DB supported by that
        library.
        If a DBAPI2 object, only sqlite3 is supported.
    index_col : str or list of strings, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point. Useful for SQL result sets.
    params : list, tuple or dict, optional, default: None
        List of parameters to pass to execute method.  The syntax used
        to pass parameters is database driver dependent. Check your
        database driver documentation for which of the five syntax styles,
        described in PEP 249's paramstyle, is supported.
        Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}.
    parse_dates : list or dict, default: None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times, or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    chunksize : int, default None
        If specified, return an iterator where `chunksize` is the number of
        rows to include in each chunk. Note that this argument is only kept
        for compatibility. If a non-none value passed, an error will be
        reported.
    incremental_index: bool, default True
        If index_col not specified, ensure range index incremental,
        gain a slightly better performance if setting False.
    use_arrow_dtype: bool, default None
        If True, use arrow dtype to store columns.
    test_rows: int, default 5
        The number of rows to fetch for inferring dtypes.
    chunk_size: : int or tuple of ints, optional
        Specifies chunk size for each dimension.
    engine_kwargs: dict, default None
        Extra kwargs to pass to sqlalchemy.create_engine
    partition_col : str, default None
        Specify name of the column to split the result of the query. If
        specified, the range ``[low_limit, high_limit]`` will be divided
        into ``n_partitions`` chunks with equal lengths. We do not
        guarantee the sizes of chunks be equal. When the value is None,
        ``OFFSET`` and ``LIMIT`` clauses will be used to cut the result
        of the query.
    num_partitions : int, default None
        The number of chunks to divide the result of the query into,
        when ``partition_col`` is specified.
    low_limit : default None
        The lower bound of the range of column ``partition_col``. If not
        specified, a query will be executed to query the minimum of
        the column.
    high_limit : default None
        The higher bound of the range of column ``partition_col``. If not
        specified, a query will be executed to query the maximum of
        the column.

    Returns
    -------
    DataFrame

    See Also
    --------
    read_sql_table : Read SQL database table into a DataFrame.
    read_sql

    Notes
    -----
    Any datetime values with time zone information parsed via the `parse_dates`
    parameter will be converted to UTC.
    """
    return _read_sql(table_or_sql=sql, con=con, index_col=index_col, coerce_float=coerce_float,
                     params=params, parse_dates=parse_dates, engine_kwargs=engine_kwargs,
                     incremental_index=incremental_index, use_arrow_dtype=use_arrow_dtype,
                     chunksize=chunksize, test_rows=test_rows,
                     chunk_size=chunk_size, partition_col=partition_col, num_partitions=num_partitions,
                     low_limit=low_limit, high_limit=high_limit)
