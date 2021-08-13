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

from io import BytesIO
from urllib.parse import urlparse

import pandas as pd
import numpy as np
try:
    from pyarrow import NativeFile
except ImportError:  # pragma: no cover
    NativeFile = None

from ... import opcodes as OperandDef
from ...config import options
from ...core import OutputType
from ...lib.filesystem import get_fs, open_file, file_size, glob
from ...serialization.serializables import StringField, DictField, ListField, Int32Field, \
    Int64Field, BoolField, AnyField
from ...utils import parse_readable_size, lazy_import, FixedSizeFileObject
from ..arrays import ArrowStringDtype
from ..utils import parse_index, build_empty_df, to_arrow_dtypes, contain_arrow_dtype
from .core import IncrementalIndexDatasource, ColumnPruneSupportedDataSourceMixin, \
    IncrementalIndexDataSourceMixin


cudf = lazy_import('cudf', globals=globals())


def _find_delimiter(f, block_size=2 ** 16):
    delimiter = b'\n'
    if f.tell() == 0:
        return 0
    while True:
        b = f.read(block_size)
        if not b:
            return f.tell()
        elif delimiter in b:
            return f.tell() - len(b) + b.index(delimiter) + 1


def _find_hdfs_start_end(f, offset, size):
    # As pyarrow doesn't support `readline` operation (https://github.com/apache/arrow/issues/3838),
    # we need to find the start and end of file block manually.

    # Be careful with HdfsFile's seek, it doesn't allow seek beyond EOF.
    loc = min(offset, f.size())
    f.seek(loc)
    start = _find_delimiter(f)
    loc = min(offset + size, f.size())
    f.seek(loc)
    end = _find_delimiter(f)
    return start, end


def _find_chunk_start_end(f, offset, size):
    if NativeFile is not None and isinstance(f, NativeFile):
        return _find_hdfs_start_end(f, offset, size)
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


class DataFrameReadCSV(IncrementalIndexDatasource,
                       ColumnPruneSupportedDataSourceMixin,
                       IncrementalIndexDataSourceMixin):
    _op_type_ = OperandDef.READ_CSV

    _path = AnyField('path')
    _names = ListField('names')
    _sep = StringField('sep')
    _header = AnyField('header')
    _index_col = Int32Field('index_col')
    _compression = StringField('compression')
    _usecols = AnyField('usecols')
    _offset = Int64Field('offset')
    _size = Int64Field('size')
    _nrows = Int64Field('nrows')
    _incremental_index = BoolField('incremental_index')
    _use_arrow_dtype = BoolField('use_arrow_dtype')
    _keep_usecols_order = BoolField('keep_usecols_order')
    _storage_options = DictField('storage_options')

    def __init__(self, path=None, names=None, sep=None, header=None, index_col=None,
                 compression=None, usecols=None, offset=None, size=None, nrows=None,
                 keep_usecols_order=None, incremental_index=None,
                 use_arrow_dtype=None, storage_options=None, **kw):
        super().__init__(_path=path, _names=names, _sep=sep, _header=header,
                         _index_col=index_col, _compression=compression,
                         _usecols=usecols, _offset=offset, _size=size, _nrows=nrows,
                         _incremental_index=incremental_index,
                         _keep_usecols_order=keep_usecols_order,
                         _use_arrow_dtype=use_arrow_dtype,
                         _storage_options=storage_options,
                         _output_types=[OutputType.dataframe], **kw)

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
    def usecols(self):
        return self._usecols

    @property
    def offset(self):
        return self._offset

    @property
    def size(self):
        return self._size

    @property
    def incremental_index(self):
        return self._incremental_index

    @property
    def use_arrow_dtype(self):
        return self._use_arrow_dtype

    @property
    def keep_usecols_order(self):
        return self._keep_usecols_order

    @property
    def storage_options(self):
        return self._storage_options

    def get_columns(self):
        return self._usecols

    def set_pruned_columns(self, columns, *, keep_order=None):
        self._usecols = columns
        self._keep_usecols_order = keep_order

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
    def _tile(cls, op):
        if op.compression:
            return cls._tile_compressed(op)

        df = op.outputs[0]
        chunk_bytes = df.extra_params.chunk_bytes
        chunk_bytes = int(parse_readable_size(chunk_bytes)[0])

        dtypes = df.dtypes
        if op.use_arrow_dtype is None and not op.gpu and \
                options.dataframe.use_arrow_dtype:  # pragma: no cover
            # check if use_arrow_dtype set on the server side
            dtypes = to_arrow_dtypes(df.dtypes)

        path_prefix = ''
        if isinstance(op.path, (tuple, list)):
            paths = op.path
        elif get_fs(op.path, op.storage_options).isdir(op.path):
            parsed_path = urlparse(op.path)
            if parsed_path.scheme.lower() == 'hdfs':
                path_prefix = f'{parsed_path.scheme}://{parsed_path.netloc}'
                paths = get_fs(op.path, op.storage_options).ls(op.path)
            else:
                paths = glob(op.path.rstrip('/') + '/*', storage_options=op.storage_options)
        else:
            paths = glob(op.path, storage_options=op.storage_options)

        out_chunks = []
        index_num = 0
        for path in paths:
            path = path_prefix + path
            total_bytes = file_size(path)
            offset = 0
            for _ in range(int(np.ceil(total_bytes * 1.0 / chunk_bytes))):
                chunk_op = op.copy().reset_key()
                chunk_op._path = path
                chunk_op._offset = offset
                chunk_op._size = min(chunk_bytes, total_bytes - offset)
                shape = (np.nan, len(dtypes))
                index_value = parse_index(df.index_value.to_pandas(), path, index_num)
                new_chunk = chunk_op.new_chunk(None, shape=shape, index=(index_num, 0), index_value=index_value,
                                               columns_value=df.columns_value, dtypes=dtypes)
                out_chunks.append(new_chunk)
                index_num += 1
                offset += chunk_bytes

        new_op = op.copy()
        nsplits = ((np.nan,) * len(out_chunks), (df.shape[1],))
        return new_op.new_dataframes(None, df.shape, dtypes=dtypes,
                                     index_value=df.index_value,
                                     columns_value=df.columns_value,
                                     chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def _pandas_read_csv(cls, f, op):
        csv_kwargs = op.extra_params.copy()
        out_df = op.outputs[0]
        start, end = _find_chunk_start_end(f, op.offset, op.size)
        f.seek(start)
        b = FixedSizeFileObject(f, end - start)
        if hasattr(out_df, 'dtypes'):
            dtypes = out_df.dtypes
        else:
            # Output will be a Series in some optimize rules.
            dtypes = pd.Series([out_df.dtype], index=[out_df.name])
        if end == start:
            # the last chunk may be empty
            df = build_empty_df(dtypes)
            if op.keep_usecols_order and not isinstance(op.usecols, list):
                # convert to Series, if usecols is a scalar
                df = df[op.usecols]
        else:
            if start == 0:
                # The first chunk contains header
                # As we specify names and dtype, we need to skip header rows
                csv_kwargs['skiprows'] = 1 if op.header == 'infer' else op.header
            if op.usecols:
                usecols = op.usecols if isinstance(op.usecols, list) else [op.usecols]
            else:
                usecols = op.usecols
            if contain_arrow_dtype(dtypes):
                # when keep_default_na is True which is default,
                # will replace null value with np.nan,
                # which will cause failure when converting to arrow string array
                csv_kwargs['keep_default_na'] = False
                csv_kwargs['dtype'] = cls._select_arrow_dtype(dtypes)
            df = pd.read_csv(b, sep=op.sep, names=op.names, index_col=op.index_col,
                             usecols=usecols, nrows=op.nrows, **csv_kwargs)
            if op.keep_usecols_order:
                df = df[op.usecols]
        return df

    @classmethod
    def _cudf_read_csv(cls, op):  # pragma: no cover
        if op.usecols:
            usecols = op.usecols if isinstance(op.usecols, list) else [op.usecols]
        else:
            usecols = op.usecols
        csv_kwargs = op.extra_params
        if op.offset == 0:
            df = cudf.read_csv(op.path, byte_range=(op.offset, op.size), sep=op.sep, usecols=usecols, **csv_kwargs)
        else:
            df = cudf.read_csv(op.path, byte_range=(op.offset, op.size), sep=op.sep, names=op.names,
                               usecols=usecols, nrows=op.nrows, **csv_kwargs)

        if op.keep_usecols_order:
            df = df[op.usecols]
        return df

    @classmethod
    def _contains_arrow_dtype(cls, dtypes):
        return any(isinstance(dtype, ArrowStringDtype) for dtype in dtypes)

    @classmethod
    def _select_arrow_dtype(cls, dtypes):
        return dict((c, dtype) for c, dtype in dtypes.items() if
                    isinstance(dtype, ArrowStringDtype))

    @classmethod
    def execute(cls, ctx, op):
        xdf = cudf if op.gpu else pd
        out_df = op.outputs[0]
        csv_kwargs = op.extra_params.copy()

        with open_file(op.path, compression=op.compression, storage_options=op.storage_options) as f:
            if op.compression is not None:
                # As we specify names and dtype, we need to skip header rows
                csv_kwargs['skiprows'] = 1 if op.header == 'infer' else op.header
                dtypes = op.outputs[0].dtypes
                if contain_arrow_dtype(dtypes):
                    # when keep_default_na is True which is default,
                    # will replace null value with np.nan,
                    # which will cause failure when converting to arrow string array
                    csv_kwargs['keep_default_na'] = False
                    csv_kwargs['dtype'] = cls._select_arrow_dtype(dtypes)
                df = xdf.read_csv(f, sep=op.sep, names=op.names, index_col=op.index_col,
                                  usecols=op.usecols, nrows=op.nrows, **csv_kwargs)
                if op.keep_usecols_order:
                    df = df[op.usecols]
            else:
                df = cls._cudf_read_csv(op) if op.gpu else cls._pandas_read_csv(f, op)
        ctx[out_df.key] = df

    def estimate_size(cls, ctx, op):
        phy_size = op.size * (op.memory_scale or 1)
        ctx[op.outputs[0].key] = (phy_size, phy_size * 2)

    def __call__(self, index_value=None, columns_value=None, dtypes=None, chunk_bytes=None):
        shape = (np.nan, len(dtypes))
        return self.new_dataframe(None, shape, dtypes=dtypes, index_value=index_value,
                                  columns_value=columns_value, chunk_bytes=chunk_bytes)


def read_csv(path, names=None, sep=',', index_col=None, compression=None, header='infer',
             dtype=None, usecols=None, nrows=None, chunk_bytes='64M', gpu=None, head_bytes='100k',
             head_lines=None, incremental_index=True, use_arrow_dtype=None,
             storage_options=None, memory_scale=None, **kwargs):
    r"""
    Read a comma-separated values (csv) file into DataFrame.
    Also supports optionally iterating or breaking of the file
    into chunks.

    Parameters
    ----------
    path : str
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be: file://localhost/path/to/table.csv,
        you can also read from external resources using a URL like:
        hdfs://localhost:8020/test.csv.
        If you want to pass in a path object, pandas accepts any ``os.PathLike``.
        By file-like object, we refer to objects with a ``read()`` method, such as
        a file handler (e.g. via builtin ``open`` function) or ``StringIO``.
    sep : str, default ','
        Delimiter to use. If sep is None, the C engine cannot automatically detect
        the separator, but the Python parsing engine can, meaning the latter will
        be used and automatically detect the separator by Python's builtin sniffer
        tool, ``csv.Sniffer``. In addition, separators longer than 1 character and
        different from ``'\s+'`` will be interpreted as regular expressions and
        will also force the use of the Python parsing engine. Note that regex
        delimiters are prone to ignoring quoted data. Regex example: ``'\r\t'``.
    delimiter : str, default ``None``
        Alias for sep.
    header : int, list of int, default 'infer'
        Row number(s) to use as the column names, and the start of the
        data.  Default behavior is to infer the column names: if no names
        are passed the behavior is identical to ``header=0`` and column
        names are inferred from the first line of the file, if column
        names are passed explicitly then the behavior is identical to
        ``header=None``. Explicitly pass ``header=0`` to be able to
        replace existing names. The header can be a list of integers that
        specify row locations for a multi-index on the columns
        e.g. [0,1,3]. Intervening rows that are not specified will be
        skipped (e.g. 2 in this example is skipped). Note that this
        parameter ignores commented lines and empty lines if
        ``skip_blank_lines=True``, so ``header=0`` denotes the first line of
        data rather than the first line of the file.
    names : array-like, optional
        List of column names to use. If the file contains a header row,
        then you should explicitly pass ``header=0`` to override the column names.
        Duplicates in this list are not allowed.
    index_col : int, str, sequence of int / str, or False, default ``None``
      Column(s) to use as the row labels of the ``DataFrame``, either given as
      string name or column index. If a sequence of int / str is given, a
      MultiIndex is used.
      Note: ``index_col=False`` can be used to force pandas to *not* use the first
      column as the index, e.g. when you have a malformed file with delimiters at
      the end of each line.
    usecols : list-like or callable, optional
        Return a subset of the columns. If list-like, all elements must either
        be positional (i.e. integer indices into the document columns) or strings
        that correspond to column names provided either by the user in `names` or
        inferred from the document header row(s). For example, a valid list-like
        `usecols` parameter would be ``[0, 1, 2]`` or ``['foo', 'bar', 'baz']``.
        Element order is ignored, so ``usecols=[0, 1]`` is the same as ``[1, 0]``.
        To instantiate a DataFrame from ``data`` with element order preserved use
        ``pd.read_csv(data, usecols=['foo', 'bar'])[['foo', 'bar']]`` for columns
        in ``['foo', 'bar']`` order or
        ``pd.read_csv(data, usecols=['foo', 'bar'])[['bar', 'foo']]``
        for ``['bar', 'foo']`` order.
        If callable, the callable function will be evaluated against the column
        names, returning names where the callable function evaluates to True. An
        example of a valid callable argument would be ``lambda x: x.upper() in
        ['AAA', 'BBB', 'DDD']``. Using this parameter results in much faster
        parsing time and lower memory usage.
    squeeze : bool, default False
        If the parsed data only contains one column then return a Series.
    prefix : str, optional
        Prefix to add to column numbers when no header, e.g. 'X' for X0, X1, ...
    mangle_dupe_cols : bool, default True
        Duplicate columns will be specified as 'X', 'X.1', ...'X.N', rather than
        'X'...'X'. Passing in False will cause data to be overwritten if there
        are duplicate names in the columns.
    dtype : Type name or dict of column -> type, optional
        Data type for data or columns. E.g. {'a': np.float64, 'b': np.int32,
        'c': 'Int64'}
        Use `str` or `object` together with suitable `na_values` settings
        to preserve and not interpret dtype.
        If converters are specified, they will be applied INSTEAD
        of dtype conversion.
    engine : {'c', 'python'}, optional
        Parser engine to use. The C engine is faster while the python engine is
        currently more feature-complete.
    converters : dict, optional
        Dict of functions for converting values in certain columns. Keys can either
        be integers or column labels.
    true_values : list, optional
        Values to consider as True.
    false_values : list, optional
        Values to consider as False.
    skipinitialspace : bool, default False
        Skip spaces after delimiter.
    skiprows : list-like, int or callable, optional
        Line numbers to skip (0-indexed) or number of lines to skip (int)
        at the start of the file.
        If callable, the callable function will be evaluated against the row
        indices, returning True if the row should be skipped and False otherwise.
        An example of a valid callable argument would be ``lambda x: x in [0, 2]``.
    skipfooter : int, default 0
        Number of lines at bottom of file to skip (Unsupported with engine='c').
    nrows : int, optional
        Number of rows of file to read. Useful for reading pieces of large files.
    na_values : scalar, str, list-like, or dict, optional
        Additional strings to recognize as NA/NaN. If dict passed, specific
        per-column NA values.  By default the following values are interpreted as
        NaN: '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
        '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a',
        'nan', 'null'.
    keep_default_na : bool, default True
        Whether or not to include the default NaN values when parsing the data.
        Depending on whether `na_values` is passed in, the behavior is as follows:
        * If `keep_default_na` is True, and `na_values` are specified, `na_values`
          is appended to the default NaN values used for parsing.
        * If `keep_default_na` is True, and `na_values` are not specified, only
          the default NaN values are used for parsing.
        * If `keep_default_na` is False, and `na_values` are specified, only
          the NaN values specified `na_values` are used for parsing.
        * If `keep_default_na` is False, and `na_values` are not specified, no
          strings will be parsed as NaN.
        Note that if `na_filter` is passed in as False, the `keep_default_na` and
        `na_values` parameters will be ignored.
    na_filter : bool, default True
        Detect missing value markers (empty strings and the value of na_values). In
        data without any NAs, passing na_filter=False can improve the performance
        of reading a large file.
    verbose : bool, default False
        Indicate number of NA values placed in non-numeric columns.
    skip_blank_lines : bool, default True
        If True, skip over blank lines rather than interpreting as NaN values.
    parse_dates : bool or list of int or names or list of lists or dict, default False
        The behavior is as follows:
        * boolean. If True -> try parsing the index.
        * list of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3
          each as a separate date column.
        * list of lists. e.g.  If [[1, 3]] -> combine columns 1 and 3 and parse as
          a single date column.
        * dict, e.g. {'foo' : [1, 3]} -> parse columns 1, 3 as date and call
          result 'foo'
        If a column or index cannot be represented as an array of datetimes,
        say because of an unparsable value or a mixture of timezones, the column
        or index will be returned unaltered as an object data type. For
        non-standard datetime parsing, use ``pd.to_datetime`` after
        ``pd.read_csv``. To parse an index or column with a mixture of timezones,
        specify ``date_parser`` to be a partially-applied
        :func:`pandas.to_datetime` with ``utc=True``. See
        :ref:`io.csv.mixed_timezones` for more.
        Note: A fast-path exists for iso8601-formatted dates.
    infer_datetime_format : bool, default False
        If True and `parse_dates` is enabled, pandas will attempt to infer the
        format of the datetime strings in the columns, and if it can be inferred,
        switch to a faster method of parsing them. In some cases this can increase
        the parsing speed by 5-10x.
    keep_date_col : bool, default False
        If True and `parse_dates` specifies combining multiple columns then
        keep the original columns.
    date_parser : function, optional
        Function to use for converting a sequence of string columns to an array of
        datetime instances. The default uses ``dateutil.parser.parser`` to do the
        conversion. Pandas will try to call `date_parser` in three different ways,
        advancing to the next if an exception occurs: 1) Pass one or more arrays
        (as defined by `parse_dates`) as arguments; 2) concatenate (row-wise) the
        string values from the columns defined by `parse_dates` into a single array
        and pass that; and 3) call `date_parser` once for each row using one or
        more strings (corresponding to the columns defined by `parse_dates`) as
        arguments.
    dayfirst : bool, default False
        DD/MM format dates, international and European format.
    cache_dates : bool, default True
        If True, use a cache of unique, converted dates to apply the datetime
        conversion. May produce significant speed-up when parsing duplicate
        date strings, especially ones with timezone offsets.
        .. versionadded:: 0.25.0
    iterator : bool, default False
        Return TextFileReader object for iteration or getting chunks with
        ``get_chunk()``.
    chunksize : int, optional
        Return TextFileReader object for iteration.
        See the `IO Tools docs
        <https://pandas.pydata.org/pandas-docs/stable/io.html#io-chunking>`_
        for more information on ``iterator`` and ``chunksize``.
    compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}, default 'infer'
        For on-the-fly decompression of on-disk data. If 'infer' and
        `filepath_or_buffer` is path-like, then detect compression from the
        following extensions: '.gz', '.bz2', '.zip', or '.xz' (otherwise no
        decompression). If using 'zip', the ZIP file must contain only one data
        file to be read in. Set to None for no decompression.
    thousands : str, optional
        Thousands separator.
    decimal : str, default '.'
        Character to recognize as decimal point (e.g. use ',' for European data).
    lineterminator : str (length 1), optional
        Character to break file into lines. Only valid with C parser.
    quotechar : str (length 1), optional
        The character used to denote the start and end of a quoted item. Quoted
        items can include the delimiter and it will be ignored.
    quoting : int or csv.QUOTE_* instance, default 0
        Control field quoting behavior per ``csv.QUOTE_*`` constants. Use one of
        QUOTE_MINIMAL (0), QUOTE_ALL (1), QUOTE_NONNUMERIC (2) or QUOTE_NONE (3).
    doublequote : bool, default ``True``
       When quotechar is specified and quoting is not ``QUOTE_NONE``, indicate
       whether or not to interpret two consecutive quotechar elements INSIDE a
       field as a single ``quotechar`` element.
    escapechar : str (length 1), optional
        One-character string used to escape other characters.
    comment : str, optional
        Indicates remainder of line should not be parsed. If found at the beginning
        of a line, the line will be ignored altogether. This parameter must be a
        single character. Like empty lines (as long as ``skip_blank_lines=True``),
        fully commented lines are ignored by the parameter `header` but not by
        `skiprows`. For example, if ``comment='#'``, parsing
        ``#empty\na,b,c\n1,2,3`` with ``header=0`` will result in 'a,b,c' being
        treated as the header.
    encoding : str, optional
        Encoding to use for UTF when reading/writing (ex. 'utf-8'). `List of Python
        standard encodings
        <https://docs.python.org/3/library/codecs.html#standard-encodings>`_ .
    dialect : str or csv.Dialect, optional
        If provided, this parameter will override values (default or not) for the
        following parameters: `delimiter`, `doublequote`, `escapechar`,
        `skipinitialspace`, `quotechar`, and `quoting`. If it is necessary to
        override values, a ParserWarning will be issued. See csv.Dialect
        documentation for more details.
    error_bad_lines : bool, default True
        Lines with too many fields (e.g. a csv line with too many commas) will by
        default cause an exception to be raised, and no DataFrame will be returned.
        If False, then these "bad lines" will dropped from the DataFrame that is
        returned.
    warn_bad_lines : bool, default True
        If error_bad_lines is False, and warn_bad_lines is True, a warning for each
        "bad line" will be output.
    delim_whitespace : bool, default False
        Specifies whether or not whitespace (e.g. ``' '`` or ``'    '``) will be
        used as the sep. Equivalent to setting ``sep='\s+'``. If this option
        is set to True, nothing should be passed in for the ``delimiter``
        parameter.
    low_memory : bool, default True
        Internally process the file in chunks, resulting in lower memory use
        while parsing, but possibly mixed type inference.  To ensure no mixed
        types either set False, or specify the type with the `dtype` parameter.
        Note that the entire file is read into a single DataFrame regardless,
        use the `chunksize` or `iterator` parameter to return the data in chunks.
        (Only valid with C parser).
    float_precision : str, optional
        Specifies which converter the C engine should use for floating-point
        values. The options are `None` for the ordinary converter,
        `high` for the high-precision converter, and `round_trip` for the
        round-trip converter.
    chunk_bytes: int, float or str, optional
        Number of chunk bytes.
    gpu: bool, default False
        If read into cudf DataFrame.
    head_bytes: int, float or str, optional
        Number of bytes to use in the head of file, mainly for data inference.
    head_lines: int, optional
        Number of lines to use in the head of file, mainly for data inference.
    incremental_index: bool, default True
        If index_col not specified, ensure range index incremental,
        gain a slightly better performance if setting False.
    use_arrow_dtype: bool, default None
        If True, use arrow dtype to store columns.
    storage_options: dict, optional
        Options for storage connection.

    Returns
    -------
    DataFrame
        A comma-separated values (csv) file is returned as two-dimensional
        data structure with labeled axes.

    See Also
    --------
    to_csv : Write DataFrame to a comma-separated values (csv) file.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> from mars.lib.filesystem.oss import build_oss_path
    >>> md.read_csv('data.csv')  # doctest: +SKIP
    >>> # read from HDFS
    >>> md.read_csv('hdfs://localhost:8020/test.csv')  # doctest: +SKIP
    >>> # read from OSS
    >>> auth_path = build_oss_path(file_path, access_key_id, access_key_secret, end_point)
    >>> md.read_csv(auth_path)
    """
    # infer dtypes and columns
    if isinstance(path, (list, tuple)):
        file_path = path[0]
    elif get_fs(path, storage_options).isdir(path):
        parsed_path = urlparse(path)
        if parsed_path.scheme.lower() == 'hdfs':
            path_prefix = f'{parsed_path.scheme}://{parsed_path.netloc}'
            file_path = path_prefix + get_fs(path, storage_options).ls(path)[0]
        else:
            file_path = glob(path.rstrip('/') + '/*', storage_options)[0]
    else:
        file_path = glob(path, storage_options)[0]

    with open_file(file_path, compression=compression, storage_options=storage_options) as f:
        if head_lines is not None:
            b = b''.join([f.readline() for _ in range(head_lines)])
        else:
            head_bytes = int(parse_readable_size(head_bytes)[0])
            head_start, head_end = _find_chunk_start_end(f, 0, head_bytes)
            f.seek(head_start)
            b = f.read(head_end - head_start)
        mini_df = pd.read_csv(BytesIO(b), sep=sep, index_col=index_col, dtype=dtype,
                              names=names, header=header)
        if names is None:
            names = list(mini_df.columns)
        else:
            # if names specified, header should be None
            header = None
        if usecols:
            usecols = usecols if isinstance(usecols, list) else [usecols]
            col_index = sorted(mini_df.columns.get_indexer(usecols))
            mini_df = mini_df.iloc[:, col_index]

    if isinstance(mini_df.index, pd.RangeIndex):
        index_value = parse_index(pd.RangeIndex(-1))
    else:
        index_value = parse_index(mini_df.index)
    columns_value = parse_index(mini_df.columns, store_data=True)
    if index_col and not isinstance(index_col, int):
        index_col = list(mini_df.columns).index(index_col)
    op = DataFrameReadCSV(path=path, names=names, sep=sep, header=header, index_col=index_col,
                          usecols=usecols, compression=compression, gpu=gpu,
                          incremental_index=incremental_index, use_arrow_dtype=use_arrow_dtype,
                          storage_options=storage_options, memory_scale=memory_scale,
                          **kwargs)
    chunk_bytes = chunk_bytes or options.chunk_store_limit
    dtypes = mini_df.dtypes
    if use_arrow_dtype is None:
        use_arrow_dtype = options.dataframe.use_arrow_dtype
    if not gpu and use_arrow_dtype:
        dtypes = to_arrow_dtypes(dtypes, test_df=mini_df)
    ret = op(index_value=index_value, columns_value=columns_value,
             dtypes=dtypes, chunk_bytes=chunk_bytes)
    if nrows is not None:
        return ret.head(nrows)
    return ret
