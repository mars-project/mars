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

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from ... import opcodes as OperandDef
from ...core import recursive_tile
from ...lib.version import parse as parse_version
from ...serialization.serializables import AnyField, StringField, ListField
from ...tensor.base import sort
from ..core import DATAFRAME_TYPE, SERIES_TYPE
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import build_empty_df, build_empty_series, parse_index

_need_astype_contiguous = parse_version(pd.__version__) == parse_version('1.3.0')


class DataFrameAstype(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.ASTYPE

    _dtype_values = AnyField('dtype_values')
    _errors = StringField('errors')
    _category_cols = ListField('category_cols')

    def __init__(self, dtype_values=None, errors=None, category_cols=None, output_types=None, **kw):
        super().__init__(_dtype_values=dtype_values, _errors=errors, _category_cols=category_cols,
                         _output_types=output_types, **kw)

    @property
    def dtype_values(self):
        return self._dtype_values

    @property
    def errors(self):
        return self._errors

    @property
    def category_cols(self):
        return self._category_cols

    @classmethod
    def _tile_one_chunk(cls, op):
        c = op.inputs[0].chunks[0]
        chunk_op = op.copy().reset_key()
        chunk_params = op.outputs[0].params.copy()
        chunk_params['index'] = c.index
        out_chunks = [chunk_op.new_chunk([c], **chunk_params)]

        new_op = op.copy()
        return new_op.new_tileables(op.inputs, nsplits=op.inputs[0].nsplits,
                                    chunks=out_chunks, **op.outputs[0].params.copy())

    @classmethod
    def _tile_series_index(cls, op):
        in_series = op.inputs[0]
        out = op.outputs[0]

        unique_chunk = None
        if op.dtype_values == 'category' and isinstance(op.dtype_values, str):
            unique_chunk = (yield from recursive_tile(
                sort(in_series.unique()))).chunks[0]

        chunks = []
        for c in in_series.chunks:
            chunk_op = op.copy().reset_key()
            params = c.params.copy()
            params['dtype'] = out.dtype
            if unique_chunk is not None:
                chunk_op._category_cols = [in_series.name]
                new_chunk = chunk_op.new_chunk([c, unique_chunk], **params)
            else:
                new_chunk = chunk_op.new_chunk([c], **params)
            chunks.append(new_chunk)

        new_op = op.copy()
        return new_op.new_tileables(op.inputs, nsplits=in_series.nsplits,
                                    chunks=chunks, **out.params.copy())

    @classmethod
    def _tile_dataframe(cls, op):
        in_df = op.inputs[0]
        out = op.outputs[0]
        cum_nsplits = np.cumsum((0,) + in_df.nsplits[1])
        out_chunks = []

        if op.dtype_values == 'category':
            # all columns need unique values
            for c in in_df.chunks:
                chunk_op = op.copy().reset_key()
                params = c.params.copy()
                dtypes = out.dtypes[cum_nsplits[c.index[1]]: cum_nsplits[c.index[1] + 1]]
                params['dtypes'] = dtypes
                chunk_op._category_cols = list(c.columns_value.to_pandas())
                unique_chunks = []
                for col in c.columns_value.to_pandas():
                    unique = yield from recursive_tile(sort(in_df[col].unique()))
                    unique_chunks.append(unique.chunks[0])
                new_chunk = chunk_op.new_chunk([c] + unique_chunks, **params)
                out_chunks.append(new_chunk)
        elif isinstance(op.dtype_values, dict) and 'category' in op.dtype_values.values():
            # some columns' types are category
            category_cols = [c for c, v in op.dtype_values.items()
                             if isinstance(v, str) and v == 'category']
            unique_chunks = dict()
            for col in category_cols:
                unique = yield from recursive_tile(
                    sort(in_df[col].unique()))
                unique_chunks[col] = unique.chunks[0]
            for c in in_df.chunks:
                chunk_op = op.copy().reset_key()
                params = c.params.copy()
                dtypes = out.dtypes[cum_nsplits[c.index[1]]: cum_nsplits[c.index[1] + 1]]
                params['dtypes'] = dtypes
                chunk_category_cols = []
                chunk_unique_chunks = []
                for col in c.columns_value.to_pandas():
                    if col in category_cols:
                        chunk_category_cols.append(col)
                        chunk_unique_chunks.append(unique_chunks[col])
                chunk_op._category_cols = chunk_category_cols
                new_chunk = chunk_op.new_chunk([c] + chunk_unique_chunks, **params)
                out_chunks.append(new_chunk)
        else:
            for c in in_df.chunks:
                chunk_op = op.copy().reset_key()
                params = c.params.copy()
                dtypes = out.dtypes[cum_nsplits[c.index[1]]: cum_nsplits[c.index[1] + 1]]
                params['dtypes'] = dtypes
                new_chunk = chunk_op.new_chunk([c], **params)
                out_chunks.append(new_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, nsplits=in_df.nsplits,
                                     chunks=out_chunks, **out.params.copy())

    @classmethod
    def tile(cls, op):
        if len(op.inputs[0].chunks) == 1:
            return cls._tile_one_chunk(op)
        elif isinstance(op.inputs[0], DATAFRAME_TYPE):
            return (yield from cls._tile_dataframe(op))
        else:
            return (yield from cls._tile_series_index(op))

    @classmethod
    def execute(cls, ctx, op):
        in_data = ctx[op.inputs[0].key]
        if not isinstance(op.dtype_values, dict):
            if op.category_cols is not None:
                uniques = [ctx[c.key] for c in op.inputs[1:]]
                dtype = dict((col, CategoricalDtype(unique_values)) for
                             col, unique_values in zip(op.category_cols, uniques))
                ctx[op.outputs[0].key] = in_data.astype(dtype, errors=op.errors)

            elif isinstance(in_data, pd.Index):
                ctx[op.outputs[0].key] = in_data.astype(op.dtype_values)
            else:
                if _need_astype_contiguous and not in_data.values.flags.contiguous:
                    # astype changes the data order in pandas==1.3.0, see pandas#42396
                    in_data = in_data.copy()
                ctx[op.outputs[0].key] = in_data.astype(op.dtype_values, errors=op.errors)
        else:
            selected_dtype = dict((k, v) for k, v in op.dtype_values.items()
                                  if k in in_data.columns)
            if op.category_cols is not None:
                uniques = [ctx[c.key] for c in op.inputs[1:]]
                for col, unique_values in zip(op.category_cols, uniques):
                    selected_dtype[col] = CategoricalDtype(unique_values)
            ctx[op.outputs[0].key] = in_data.astype(selected_dtype, errors=op.errors)

    def __call__(self, df):
        if isinstance(df, DATAFRAME_TYPE):
            empty_df = build_empty_df(df.dtypes)
            new_df = empty_df.astype(self.dtype_values, errors=self.errors)
            dtypes = []
            for dt, new_dt in zip(df.dtypes, new_df.dtypes):
                if new_dt != dt and isinstance(new_dt, CategoricalDtype):
                    dtypes.append(CategoricalDtype())
                else:
                    dtypes.append(new_dt)
            dtypes = pd.Series(dtypes, index=new_df.dtypes.index)
            return self.new_dataframe([df], shape=df.shape, dtypes=dtypes,
                                      index_value=df.index_value,
                                      columns_value=df.columns_value)
        else:
            empty_series = build_empty_series(df.dtype)
            new_series = empty_series.astype(self.dtype_values, errors=self.errors)
            if new_series.dtype != df.dtype:
                dtype = CategoricalDtype() if isinstance(
                    new_series.dtype, CategoricalDtype) else new_series.dtype
            else:  # pragma: no cover
                dtype = df.dtype

            if isinstance(df, SERIES_TYPE):
                return self.new_series([df], shape=df.shape, dtype=dtype,
                                       name=df.name, index_value=df.index_value)
            else:
                new_index = df.index_value.to_pandas().astype(self.dtype_values)
                new_index_value = parse_index(new_index, store_data=df.index_value.has_value())
                return self.new_index([df], shape=df.shape, dtype=dtype,
                                      name=df.name, index_value=new_index_value)


def astype(df, dtype, copy=True, errors='raise'):
    """
    Cast a pandas object to a specified dtype ``dtype``.

    Parameters
    ----------
    dtype : data type, or dict of column name -> data type
        Use a numpy.dtype or Python type to cast entire pandas object to
        the same type. Alternatively, use {col: dtype, ...}, where col is a
        column label and dtype is a numpy.dtype or Python type to cast one
        or more of the DataFrame's columns to column-specific types.
    copy : bool, default True
        Return a copy when ``copy=True`` (be very careful setting
        ``copy=False`` as changes to values then may propagate to other
        pandas objects).
    errors : {'raise', 'ignore'}, default 'raise'
        Control raising of exceptions on invalid data for provided dtype.

        - ``raise`` : allow exceptions to be raised
        - ``ignore`` : suppress exceptions. On error return original object.

    Returns
    -------
    casted : same type as caller

    See Also
    --------
    to_datetime : Convert argument to datetime.
    to_timedelta : Convert argument to timedelta.
    to_numeric : Convert argument to a numeric type.
    numpy.ndarray.astype : Cast a numpy array to a specified type.

    Examples
    --------
    Create a DataFrame:

    >>> import mars.dataframe as md
    >>> df = md.DataFrame(pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}))
    >>> df.dtypes
    col1    int64
    col2    int64
    dtype: object

    Cast all columns to int32:

    >>> df.astype('int32').dtypes
    col1    int32
    col2    int32
    dtype: object

    Cast col1 to int32 using a dictionary:

    >>> df.astype({'col1': 'int32'}).dtypes
    col1    int32
    col2    int64
    dtype: object

    Create a series:

    >>> ser = md.Series(pd.Series([1, 2], dtype='int32'))
    >>> ser.execute()
    0    1
    1    2
    dtype: int32
    >>> ser.astype('int64').execute()
    0    1
    1    2
    dtype: int64

    Convert to categorical type:

    >>> ser.astype('category').execute()
    0    1
    1    2
    dtype: category
    Categories (2, int64): [1, 2]

    Convert to ordered categorical type with custom ordering:

    >>> cat_dtype = pd.api.types.CategoricalDtype(
    ...     categories=[2, 1], ordered=True)
    >>> ser.astype(cat_dtype).execute()
    0    1
    1    2
    dtype: category
    Categories (2, int64): [2 < 1]

    Note that using ``copy=False`` and changing data on a new
    pandas object may propagate changes:

    >>> s1 = md.Series(pd.Series([1, 2]))
    >>> s2 = s1.astype('int64', copy=False)
    >>> s1.execute()  # note that s1[0] has changed too
    0     1
    1     2
    dtype: int64
    """
    if isinstance(dtype, dict):
        keys = list(dtype.keys())
        if isinstance(df, SERIES_TYPE):
            if len(keys) != 1 or keys[0] != df.name:
                raise KeyError('Only the Series name can be used for the key in Series dtype mappings.')
            else:
                dtype = list(dtype.values())[0]
        else:
            for k in keys:
                columns = df.columns_value.to_pandas()
                if k not in columns:
                    raise KeyError('Only a column name can be used for the key in a dtype mappings argument.')
    op = DataFrameAstype(dtype_values=dtype, errors=errors)
    r = op(df)
    if not copy:
        df.data = r.data
        return df
    else:
        return r


def index_astype(ix, dtype, copy=True):
    """
    Create an Index with values cast to dtypes.

    The class of a new Index is determined by dtype. When conversion is
    impossible, a ValueError exception is raised.

    Parameters
    ----------
    dtype : numpy dtype or pandas type
        Note that any signed integer `dtype` is treated as ``'int64'``,
        and any unsigned integer `dtype` is treated as ``'uint64'``,
        regardless of the size.
    copy : bool, default True
        By default, astype always returns a newly allocated object.
        If copy is set to False and internal requirements on dtype are
        satisfied, the original data is used to create a new Index
        or the original Index is returned.

    Returns
    -------
    Index
        Index with values cast to specified dtype.
    """
    return astype(ix, dtype, copy=copy)
