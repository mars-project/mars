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

from io import StringIO

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...core import OutputType, recursive_tile
from ...core.operand import OperandStage
from ...lib.filesystem import open_file
from ...serialization.serializables import KeyField, AnyField, StringField, ListField, \
    BoolField, Int32Field, Int64Field, DictField
from ...tensor.core import TensorOrder
from ...tensor.operands import TensorOperand, TensorOperandMixin
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index


class DataFrameToCSV(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.TO_CSV

    _input = KeyField('input')
    _path = AnyField('path')
    _sep = StringField('sep')
    _na_rep = StringField('na_rep')
    _float_format = StringField('float_format')
    _columns = ListField('columns')
    _header = AnyField('header')
    _index = BoolField('index')
    _index_label = AnyField('index_label')
    _mode = StringField('mode')
    _encoding = StringField('encoding')
    _compression = AnyField('compression')
    _quoting = Int32Field('quoting')
    _quotechar = StringField('quotechar')
    _line_terminator = StringField('line_terminator')
    _chunksize = Int64Field('chunksize')
    _date_format = StringField('date_format')
    _doublequote = BoolField('doublequote')
    _escapechar = StringField('escapechar')
    _decimal = StringField('decimal')
    _storage_options = DictField('storage_options')
    # for chunk
    _output_stat = BoolField('output_stat')

    def __init__(self, path=None, sep=None, na_rep=None, float_format=None,
                 columns=None, header=None, index=None, index_label=None,
                 mode=None, encoding=None, compression=None, quoting=None,
                 quotechar=None, line_terminator=None, chunksize=None, date_format=None,
                 doublequote=None, escapechar=None, decimal=None, output_stat=None,
                 storage_options=None, output_types=None, **kw):
        super().__init__(_path=path, _sep=sep, _na_rep=na_rep, _float_format=float_format,
                         _columns=columns, _header=header, _index=index, _index_label=index_label,
                         _mode=mode, _encoding=encoding, _compression=compression, _quoting=quoting,
                         _quotechar=quotechar, _line_terminator=line_terminator, _chunksize=chunksize,
                         _date_format=date_format, _doublequote=doublequote,
                         _escapechar=escapechar, _decimal=decimal, _output_stat=output_stat,
                         _storage_options=storage_options, _output_types=output_types, **kw)

    @property
    def input(self):
        return self._input

    @property
    def path(self):
        return self._path

    @property
    def sep(self):
        return self._sep

    @property
    def na_rep(self):
        return self._na_rep

    @property
    def float_format(self):
        return self._float_format

    @property
    def columns(self):
        return self._columns

    @property
    def header(self):
        return self._header

    @property
    def index(self):
        return self._index

    @property
    def index_label(self):
        return self._index_label

    @property
    def mode(self):
        return self._mode

    @property
    def encoding(self):
        return self._encoding

    @property
    def compression(self):
        return self._compression

    @property
    def quoting(self):
        return self._quoting

    @property
    def quotechar(self):
        return self._quotechar

    @property
    def line_terminator(self):
        return self._line_terminator

    @property
    def chunksize(self):
        return self._chunksize

    @property
    def date_format(self):
        return self._date_format

    @property
    def doublequote(self):
        return self._doublequote

    @property
    def escapechar(self):
        return self._escapechar

    @property
    def decimal(self):
        return self._decimal

    @property
    def storage_options(self):
        return self._storage_options

    @property
    def one_file(self):
        # if wildcard in path, write csv into multiple files
        return '*' not in self._path

    @property
    def output_stat(self):
        return self._output_stat

    @property
    def output_limit(self):
        return 1 if not self.output_stat else 2

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    @classmethod
    def tile(cls, op: 'DataFrameToCSV'):
        in_df = op.input
        out_df = op.outputs[0]

        if in_df.ndim == 2 and in_df.chunk_shape[1] > 1:
            # make sure only 1 chunk on the column axis
            in_df = yield from recursive_tile(in_df.rechunk({1: in_df.shape[1]}))

        one_file = op.one_file

        out_chunks = [], []
        for chunk in in_df.chunks:
            chunk_op = op.copy().reset_key()
            if not one_file:
                index_value = parse_index(chunk.index_value.to_pandas()[:0], chunk)
                if chunk.ndim == 2:
                    out_chunk = chunk_op.new_chunk([chunk], shape=(0, 0),
                                                   index_value=index_value,
                                                   columns_value=out_df.columns_value,
                                                   dtypes=out_df.dtypes,
                                                   index=chunk.index)
                else:
                    out_chunk = chunk_op.new_chunk([chunk], shape=(0,),
                                                   index_value=index_value,
                                                   dtype=out_df.dtype,
                                                   index=chunk.index)
                out_chunks[0].append(out_chunk)
            else:
                chunk_op._output_stat = True
                chunk_op.stage = OperandStage.map
                chunk_op.output_types = [OutputType.scalar] * 2
                # bytes of csv
                kws = [{
                    'shape': (),
                    'dtype': np.dtype(np.str_),
                    'index': chunk.index,
                    'order': TensorOrder.C_ORDER,
                    'output_type': OutputType.scalar,
                    'type': 'csv',
                }, {
                    'shape': (),
                    'dtype': np.dtype(np.intp),
                    'index': chunk.index,
                    'order': TensorOrder.C_ORDER,
                    'output_type': OutputType.scalar,
                    'type': 'stat',
                }]
                chunks = chunk_op.new_chunks([chunk], kws=kws, output_limit=len(kws))
                out_chunks[0].append(chunks[0])
                out_chunks[1].append(chunks[1])

        if not one_file:
            out_chunks = out_chunks[0]
        else:
            stat_chunk = DataFrameToCSVStat(path=op.path, dtype=np.dtype(np.int64),
                                            storage_options=op.storage_options).new_chunk(
                out_chunks[1], shape=(len(out_chunks[0]),), order=TensorOrder.C_ORDER)
            new_out_chunks = []
            for c in out_chunks[0]:
                op = DataFrameToCSV(stage=OperandStage.agg, path=op.path,
                                    storage_options=op.storage_options,
                                    output_types=op.output_types)
                if out_df.ndim == 2:
                    out_chunk = op.new_chunk(
                        [c, stat_chunk], shape=(0, 0), dtypes=out_df.dtypes,
                        index_value=out_df.index_value,
                        columns_value=out_df.columns_value,
                        index=c.index)
                else:
                    out_chunk = op.new_chunk(
                        [c, stat_chunk], shape=(0,), dtype=out_df.dtype,
                        index_value=out_df.index_value, index=c.index)
                new_out_chunks.append(out_chunk)
            out_chunks = new_out_chunks

        new_op = op.copy()
        params = out_df.params.copy()
        if out_df.ndim == 2:
            params.update(dict(chunks=out_chunks, nsplits=((0,) * in_df.chunk_shape[0], (0,))))
        else:
            params.update(dict(chunks=out_chunks, nsplits=((0,) * in_df.chunk_shape[0],)))
        return new_op.new_tileables([in_df], **params)

    def __call__(self, df):
        index_value = parse_index(df.index_value.to_pandas()[:0], df)
        if df.ndim == 2:
            columns_value = parse_index(df.columns_value.to_pandas()[:0], store_data=True)
            return self.new_dataframe([df], shape=(0, 0), dtypes=df.dtypes[:0],
                                      index_value=index_value, columns_value=columns_value)
        else:
            return self.new_series([df], shape=(0,), dtype=df.dtype, index_value=index_value)

    @classmethod
    def _to_csv(cls, op, df, path, header=None):
        if header is None:
            header = op.header
        df.to_csv(path, sep=op.sep, na_rep=op.na_rep, float_format=op.float_format,
                  columns=op.columns, header=header, index=op.index, index_label=op.index_label,
                  mode=op.mode, encoding=op.encoding, compression=op.compression, quoting=op.quoting,
                  quotechar=op.quotechar, line_terminator=op.line_terminator, chunksize=op.chunksize,
                  date_format=op.date_format, doublequote=op.doublequote, escapechar=op.escapechar,
                  decimal=op.decimal)

    @classmethod
    def _execute_map(cls, ctx, op):
        out = op.outputs[0]

        df = ctx[op.input.key]
        sio = StringIO()
        header = op.header if out.index[0] == 0 else False
        # do not output header if index of chunk > 0
        cls._to_csv(op, df, sio, header=header)

        ret = sio.getvalue().encode(op.encoding or 'utf-8')
        ctx[op.outputs[0].key] = ret
        ctx[op.outputs[1].key] = len(ret)

    @classmethod
    def _execute_agg(cls, ctx, op):
        out = op.outputs[0]
        i = out.index[0]
        path = cls._get_path(op.path, i)

        csv_bytes, offsets = [ctx[inp.key] for inp in op.inputs]
        offset_start = offsets[i]

        # write csv bytes into file
        with open_file(path, mode='r+b', storage_options=op.storage_options) as f:
            f.seek(offset_start)
            f.write(csv_bytes)

        ctx[out.key] = pd.DataFrame() if out.ndim == 2 else pd.Series([], dtype=out.dtype)

    @classmethod
    def _get_path(cls, path, i):
        if '*' not in path:
            return path
        return path.replace('*', str(i))

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls._execute_map(ctx, op)
        elif op.stage == OperandStage.agg:
            cls._execute_agg(ctx, op)
        else:
            assert op.stage is None
            df = ctx[op.input.key]
            out = op.outputs[0]
            path = cls._get_path(op.path, op.outputs[0].index[0])
            with open_file(path, mode='w', storage_options=op.storage_options) as f:
                cls._to_csv(op, df, f)
            ctx[out.key] = pd.DataFrame() if out.ndim == 2 else pd.Series([], dtype=out.dtype)


class DataFrameToCSVStat(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.TO_CSV_STAT

    _path = AnyField('path')
    _storage_options = DictField('storage_options')

    def __init__(self, path=None, storage_options=None, dtype=None, **kw):
        super().__init__(_path=path, _storage_options=storage_options,
                         _dtype=dtype, **kw)

    @property
    def path(self):
        return self._path

    @property
    def storage_options(self):
        return self._storage_options

    @classmethod
    def execute(cls, ctx, op):
        sizes = [ctx[inp.key] for inp in op.inputs]
        total_bytes = sum(sizes)
        offsets = np.cumsum([0] + sizes)[:-1]

        # write NULL bytes into file
        with open_file(op.path, mode='wb', storage_options=op.storage_options) as f:
            rest = total_bytes
            while rest > 0:
                # at most 4M
                write_bytes = min(4 * 1024 ** 2, rest)
                f.write(b'\00' * write_bytes)
                rest -= write_bytes

        ctx[op.outputs[0].key] = offsets


def to_csv(df, path, sep=',', na_rep='', float_format=None, columns=None, header=True,
           index=True, index_label=None, mode='w', encoding=None, compression='infer',
           quoting=None, quotechar='"', line_terminator=None, chunksize=None, date_format=None,
           doublequote=True, escapechar=None, decimal='.', storage_options=None):
    r"""
    Write object to a comma-separated values (csv) file.

    Parameters
    ----------
    path : str
        File path.
        If path is a string with wildcard e.g. '/to/path/out-*.csv',
        to_csv will try to write multiple files, for instance,
        chunk (0, 0) will write data into '/to/path/out-0.csv'.
        If path is a string without wildcard,
        all data will be written into a single file.
    sep : str, default ','
        String of length 1. Field delimiter for the output file.
    na_rep : str, default ''
        Missing data representation.
    float_format : str, default None
        Format string for floating point numbers.
    columns : sequence, optional
        Columns to write.
    header : bool or list of str, default True
        Write out the column names. If a list of strings is given it is
        assumed to be aliases for the column names.
    index : bool, default True
        Write row names (index).
    index_label : str or sequence, or False, default None
        Column label for index column(s) if desired. If None is given, and
        `header` and `index` are True, then the index names are used. A
        sequence should be given if the object uses MultiIndex. If
        False do not print fields for index names. Use index_label=False
        for easier importing in R.
    mode : str
        Python write mode, default 'w'.
    encoding : str, optional
        A string representing the encoding to use in the output file,
        defaults to 'utf-8'.
    compression : str or dict, default 'infer'
        If str, represents compression mode. If dict, value at 'method' is
        the compression mode. Compression mode may be any of the following
        possible values: {'infer', 'gzip', 'bz2', 'zip', 'xz', None}. If
        compression mode is 'infer' and `path_or_buf` is path-like, then
        detect compression mode from the following extensions: '.gz',
        '.bz2', '.zip' or '.xz'. (otherwise no compression). If dict given
        and mode is 'zip' or inferred as 'zip', other entries passed as
        additional compression options.
    quoting : optional constant from csv module
        Defaults to csv.QUOTE_MINIMAL. If you have set a `float_format`
        then floats are converted to strings and thus csv.QUOTE_NONNUMERIC
        will treat them as non-numeric.
    quotechar : str, default '\"'
        String of length 1. Character used to quote fields.
    line_terminator : str, optional
        The newline character or character sequence to use in the output
        file. Defaults to `os.linesep`, which depends on the OS in which
        this method is called ('\n' for linux, '\r\n' for Windows, i.e.).
    chunksize : int or None
        Rows to write at a time.
    date_format : str, default None
        Format string for datetime objects.
    doublequote : bool, default True
        Control quoting of `quotechar` inside a field.
    escapechar : str, default None
        String of length 1. Character used to escape `sep` and `quotechar`
        when appropriate.
    decimal : str, default '.'
        Character recognized as decimal separator. E.g. use ',' for
        European data.
    Returns
    -------
    None or str
        If path_or_buf is None, returns the resulting csv format as a
        string. Otherwise returns None.

    See Also
    --------
    read_csv : Load a CSV file into a DataFrame.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'name': ['Raphael', 'Donatello'],
    ...                    'mask': ['red', 'purple'],
    ...                    'weapon': ['sai', 'bo staff']})
    >>> df.to_csv('out.csv', index=False).execute()
    """

    if mode != 'w':  # pragma: no cover
        raise NotImplementedError("only support to_csv with mode 'w' for now")
    op = DataFrameToCSV(path=path, sep=sep, na_rep=na_rep, float_format=float_format,
                        columns=columns, header=header, index=index, index_label=index_label,
                        mode=mode, encoding=encoding, compression=compression, quoting=quoting,
                        quotechar=quotechar, line_terminator=line_terminator, chunksize=chunksize,
                        date_format=date_format, doublequote=doublequote, escapechar=escapechar,
                        decimal=decimal, storage_options=storage_options)
    return op(df)
