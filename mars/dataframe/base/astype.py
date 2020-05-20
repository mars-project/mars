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

import numpy as np
import pandas as pd


from ... import opcodes as OperandDef
from ...serialize import BoolField, AnyField, StringField
from ...tiles import TilesError
from ..utils import build_empty_df, build_empty_series, parse_index
from ..core import SERIES_TYPE, INDEX_TYPE
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType


class DataFrameAstype(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.ASTYPE

    _dtype_values = AnyField('dtype_values')
    _copy = BoolField('copy')
    _errors = StringField('errors')

    def __init__(self, dtype_values=None, copy=None, errors=None, object_type=None, **kw):
        super().__init__(_dtype_values=dtype_values,
                         _copy=copy, _errors=errors,
                         _object_type=object_type, **kw)

    @property
    def dtype_values(self):
        return self._dtype_values

    @property
    def copy_(self):
        return self._copy

    @property
    def errors(self):
        return self._errors

    @classmethod
    def _tile_series(cls, op):
        in_series = op.inputs[0]
        out = op.outputs[0]
        chunks = []
        for c in in_series.chunks:
            chunk_op = op.copy().reset_key()
            params = c.params.copy()
            params['dtype'] = out.dtype
            new_chunk = chunk_op.new_chunk([c], **params)
            chunks.append(new_chunk)

        new_op = op.copy()
        return new_op.new_seriess(op.inputs, nsplits=in_series.nsplits,
                                  chunks=chunks, **out.params.copy())

    @classmethod
    def _tile_dataframe(cls, op):
        in_df = op.inputs[0]
        out = op.outputs[0]
        cum_nsplits = np.cumsum((0,) + in_df.nsplits[1])
        out_chunks = []
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
        if isinstance(op.inputs[0], SERIES_TYPE):
            return cls._tile_series(op)
        else:
            return cls._tile_dataframe(op)

    @classmethod
    def execute(cls, ctx, op):
        in_data = ctx[op.inputs[0].key]
        if isinstance(op.dtype_values, str):
            ctx[op.outputs[0].key] = in_data.astype(op.dtype_values, copy=op.copy_, errors=op.errors)
        else:
            selected_dtype = dict((k, v) for k, v in op.dtype_values.items()
                                  if k in in_data.columns)
            ctx[op.outputs[0].key] = in_data.astype(selected_dtype, copy=op.copy_, errors=op.errors)

    def __call__(self, df):
        if isinstance(df, SERIES_TYPE):
            self._object_type = ObjectType.series
            empty_series = build_empty_series(df.dtype)
            new_series = empty_series.astype(self.dtype_values)

            return self.new_series([df], shape=df.shape, dtype=new_series.dtype,
                                   name=df.name, index_value=df.index_value)
        else:
            self._object_type = ObjectType.dataframe
            empty_df = build_empty_df(df.dtypes)
            new_df = empty_df.astype(self.dtype_values)
            return self.new_dataframe([df], shape=df.shape, dtypes=new_df.dtypes,
                                      index_value=df.index_value,
                                      columns_value=df.columns_value)


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
    >>> ser.astype(cat_dtype)
    0    1
    1    2
    dtype: category
    Categories (2, int64): [2 < 1]

    Note that using ``copy=False`` and changing data on a new
    pandas object may propagate changes:

    >>> s1 = pd.Series([1, 2])
    >>> s2 = s1.astype('int64', copy=False)
    >>> s2[0] = 10
    >>> s1  # note that s1[0] has changed too
    0    10
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
    op = DataFrameAstype(dtype_values=dtype, copy=copy, errors=errors)
    return op(df)
