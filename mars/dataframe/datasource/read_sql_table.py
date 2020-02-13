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

from contextlib import contextmanager

import numpy as np
import pandas as pd
import cloudpickle

from ... import opcodes as OperandDef
from ...config import options
from ...serialize import StringField, AnyField, BoolField, \
    ListField, Int64Field, Float64Field, BytesField
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import parse_index, decide_dataframe_chunk_sizes


class DataFrameReadSQLTable(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.READ_SQL_TABLE

    _table_name = StringField('table_name')
    _con = StringField('con')
    _schema = StringField('schema')
    _index_col = AnyField('index_col')
    _coerce_float = BoolField('coerce_float')
    _parse_dates = AnyField('parse_dates')
    _columns = ListField('columns')
    _chunksize = Int64Field('chunksize')
    _engine_kwargs = BytesField('engine_kwargs', on_serialize=cloudpickle.dumps,
                                on_deserialize=cloudpickle.loads)
    _row_memory_usage = Float64Field('row_memory_usage')
    # for chunks
    _offset = Int64Field('offset')

    def __init__(self, table_name=None, con=None, schema=None, index_col=None,
                 coerce_float=None, parse_dates=None, columns=None, chunksize=None,
                 engine_kwargs=None, row_memory_usage=None, offset=None,
                 object_type=None, gpu=None, **kw):
        super().__init__(_table_name=table_name, _con=con, _schema=schema,
                         _index_col=index_col, _coerce_float=coerce_float,
                         _parse_dates=parse_dates, _columns=columns, _chunksize=chunksize,
                         _engine_kwargs=engine_kwargs, _row_memory_usage=row_memory_usage,
                         _offset=offset, _object_type=object_type, _gpu=gpu, **kw)
        if self._object_type is None:
            self._object_type = ObjectType.dataframe

    @property
    def table_name(self):
        return self._table_name

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
    def chunksize(self):
        return self._chunksize

    @property
    def engine_kwargs(self):
        return self._engine_kwargs

    @property
    def row_memory_usage(self):
        return self._row_memory_usage

    @property
    def offset(self):
        return self._offset

    def _collect_info(self, engine_or_conn, table, columns, test_rows):
        from sqlalchemy import sql

        # fetch test DataFrame
        query = sql.select(columns).limit(test_rows).select_from(table)
        test_df = pd.read_sql(query, engine_or_conn, index_col=self._index_col,
                              coerce_float=self._coerce_float,
                              parse_dates=self._parse_dates)
        self._row_memory_usage = \
            test_df.memory_usage(deep=True, index=True).sum() / test_rows

        # fetch size
        size = list(engine_or_conn.execute(
            sql.select([sql.func.count()]).select_from(table)))[0][0]
        shape = (size, test_df.shape[1])

        return test_df, shape

    @contextmanager
    def _create_con(self):
        import sqlalchemy as sa
        from sqlalchemy.engine import Connection, Engine

        # process con
        con = self._con
        engine = None
        if isinstance(con, Connection):
            self._con = str(con.engine.url)
            # connection create by user
            close = False
            dispose = False
        elif isinstance(con, Engine):
            self._con = str(con.url)
            con = con.connect()
            close = True
            dispose = False
        else:
            engine = sa.create_engine(con, **(self._engine_kwargs or dict()))
            con = engine.connect()
            close = True
            dispose = True

        yield con

        if close:
            con.close()
        if dispose:
            engine.dispose()

    def __call__(self, test_rows, chunk_size):
        import sqlalchemy as sa
        from sqlalchemy.sql import elements

        with self._create_con() as con:
            # process table_name
            if isinstance(self._table_name, sa.Table):
                table = self._table_name
                self._table_name = table.name
            else:
                m = sa.MetaData()
                table = sa.Table(self._table_name, m, autoload=True,
                                 autoload_with=con, schema=self._schema)

            # process index_col
            index_col = self._index_col
            if index_col is not None:
                if not isinstance(index_col, (list, tuple)):
                    index_col = (index_col,)
                new_index_col = []
                sa_index_col = []
                for col in index_col:
                    if isinstance(col, (sa.Column, elements.Label)):
                        new_index_col.append(col.name)
                        sa_index_col.append(col)
                    elif isinstance(col, str):
                        sa_index_col.append(table.columns[col])
                        new_index_col.append(col)
                    elif col is not None:
                        raise TypeError('unknown index_col type: {}'.format(type(col)))
                self._index_col = new_index_col
                index_col = sa_index_col

            # process columns
            columns = self._columns if self._columns is not None else table.columns
            new_columns = []
            sa_columns = []
            for col in columns:
                if isinstance(col, str):
                    new_columns.append(col)
                    sa_columns.append(table.columns[col])
                else:
                    new_columns.append(col.name)
                    sa_columns.append(col)
            self._columns = new_columns
            if self._index_col is not None:
                for icol in index_col:
                    sa_columns.append(icol)

            test_df, shape = self._collect_info(con, table, sa_columns, test_rows)

            if isinstance(test_df.index, pd.RangeIndex):
                index_value = parse_index(pd.RangeIndex(shape[0]))
            else:
                index_value = parse_index(test_df.index)
            columns_value = parse_index(test_df.columns, store_data=True)
            return self.new_dataframe(None, shape=shape, dtypes=test_df.dtypes,
                                      index_value=index_value,
                                      columns_value=columns_value,
                                      raw_chunk_size=chunk_size)

    @classmethod
    def tile(cls, op):
        df = op.outputs[0]

        memory_usage = op.row_memory_usage * df.shape[0]
        chunk_size = df.extra_params.raw_chunk_size or options.chunk_size
        row_chunk_sizes = decide_dataframe_chunk_sizes(df.shape, chunk_size, memory_usage)[0]
        offsets = np.cumsum((0,) + row_chunk_sizes)

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
                                          op.table_name, op.con, i, row_size)
            out_chunk = chunk_op.new_chunk(None, shape=(row_size, df.shape[1]),
                                           columns_value=df.columns_value,
                                           index_value=index_value, dtypes=df.dtypes,
                                           index=(i, 0))
            out_chunks.append(out_chunk)

        nsplits = (row_chunk_sizes, (df.shape[1],))
        new_op = op.copy()
        return new_op.new_dataframes(None, chunks=out_chunks, nsplits=nsplits,
                                     **df.params)

    @classmethod
    def execute(cls, ctx, op):
        import sqlalchemy as sa

        out = op.outputs[0]

        engine = sa.create_engine(op.con, **(op.engine_kwargs or dict()))
        try:
            table = sa.Table(op.table_name, sa.MetaData(), autoload=True,
                             autoload_with=engine, schema=op.schema)

            columns = [table.columns[col] for col in op.columns]
            column_names = set(op.columns)
            if op.index_col:
                for icol in op.index_col:
                    if icol not in column_names:
                        columns.append(table.columns[icol])

            query = sa.sql.select(columns).limit(out.shape[0])
            if op.offset > 0:
                query = query.offset(op.offset)
            query = query.select_from(table)

            df = pd.read_sql(query, engine, index_col=op.index_col,
                             coerce_float=op.coerce_float,
                             parse_dates=op.parse_dates,
                             chunksize=op.chunksize)
            if op.index_col is None and op.offset > 0:
                df.index = pd.RangeIndex(op.offset, op.offset + out.shape[0])
            ctx[out.key] = df
        finally:
            engine.dispose()


def read_sql_table(table_name, con, schema=None, index_col=None, coerce_float=True,
                   parse_dates=None, columns=None, chunksize=None,
                   test_rows=5, chunk_size=None, engine_kwargs=None):
    op = DataFrameReadSQLTable(table_name=table_name, con=con, schema=schema,
                               index_col=index_col, coerce_float=coerce_float,
                               parse_dates=parse_dates, columns=columns,
                               chunksize=chunksize, engine_kwargs=engine_kwargs)
    return op(test_rows, chunk_size)
