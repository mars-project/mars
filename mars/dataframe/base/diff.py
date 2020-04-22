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

import itertools

import numpy as np

from ... import opcodes
from ...serialize import AnyField, Int8Field, Int64Field
from ..core import DATAFRAME_TYPE
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import build_empty_df, build_empty_series, validate_axis
from .shift import DataFrameShift


class DataFrameDiff(DataFrameOperandMixin, DataFrameOperand):
    _op_type_ = opcodes.DATAFRAME_DIFF

    _periods = Int64Field('periods')
    _axis = Int8Field('axis')

    _bool_columns = AnyField('bool_columns')

    @property
    def periods(self):
        return self._periods

    @property
    def axis(self):
        return self._axis

    @property
    def bool_columns(self):
        return self._bool_columns

    def __init__(self, periods=None, axis=None, bool_columns=None, **kw):
        super().__init__(_periods=periods, _axis=axis, _bool_columns=bool_columns, **kw)

    def __call__(self, df_or_series):
        params = df_or_series.params.copy()

        if isinstance(df_or_series, DATAFRAME_TYPE):
            self._object_type = ObjectType.dataframe
            mock_obj = build_empty_df(df_or_series.dtypes)
            params['dtypes'] = mock_obj.diff().dtypes
        else:
            self._object_type = ObjectType.series
            mock_obj = build_empty_series(df_or_series.dtype, name=df_or_series.name)
            params['dtype'] = mock_obj.diff().dtype

        return self.new_tileable([df_or_series], **params)

    @classmethod
    def tile(cls, op):
        in_obj = op.inputs[0]
        out_obj = op.outputs[0]
        axis = op.axis or 0

        if in_obj.chunk_shape[axis] > 1:
            shift_chunks = DataFrameShift(periods=op.periods, axis=axis)(in_obj) \
                ._inplace_tile().chunks
        else:
            shift_chunks = itertools.repeat(None)

        chunks = []
        bool_columns_dict = dict()
        for in_chunk, shift_chunk in zip(in_obj.chunks, shift_chunks):
            params = in_chunk.params.copy()
            if in_chunk.ndim == 2:
                params['dtypes'] = out_obj.dtypes[in_chunk.dtypes.index]
                try:
                    bool_columns = bool_columns_dict[in_chunk.index[1]]
                except KeyError:
                    bool_columns = bool_columns_dict[in_chunk.index[1]] = \
                        [col for col, dt in in_chunk.dtypes.items() if dt == np.dtype(bool)]
            else:
                params['dtype'] = out_obj.dtype
                bool_columns = (in_chunk.dtype == np.dtype(bool))

            new_op = op.copy().reset_key()
            new_op._bool_columns = bool_columns

            if shift_chunk is None:
                chunks.append(new_op.new_chunk([in_chunk], **params))
            else:
                chunks.append(new_op.new_chunk([in_chunk, shift_chunk], **params))

        new_op = op.copy().reset_key()
        return new_op.new_tileables([in_obj], chunks=chunks, nsplits=in_obj.nsplits,
                                    **out_obj.params)

    @classmethod
    def execute(cls, ctx, op):
        in_data = ctx[op.inputs[0].key]
        if len(op.inputs) == 1:
            if in_data.ndim == 2:
                ctx[op.outputs[0].key] = in_data.diff(periods=op.periods, axis=op.axis)
            else:
                ctx[op.outputs[0].key] = in_data.diff(periods=op.periods)
        else:
            in_shift = ctx[op.inputs[1].key]
            result = in_data - in_shift
            if op.bool_columns:
                if in_data.ndim == 2:
                    result.replace({c: {1: True, -1: True, 0: False} for c in op.bool_columns},
                                   inplace=True)
                else:
                    result.replace({1: True, -1: True, 0: False}, inplace=True)
            ctx[op.outputs[0].key] = result


def df_diff(df, periods=1, axis=0):
    """
    First discrete difference of element.
    Calculates the difference of a DataFrame element compared with another
    element in the DataFrame (default is the element in the same column
    of the previous row).

    Parameters
    ----------
    periods : int, default 1
        Periods to shift for calculating difference, accepts negative
        values.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Take difference over rows (0) or columns (1).

    Returns
    -------
    DataFrame

    See Also
    --------
    Series.diff : First discrete difference for a Series.
    DataFrame.pct_change : Percent change over given number of periods.
    DataFrame.shift : Shift index by desired number of periods with an
        optional time freq.

    Notes
    -----
    For boolean dtypes, this uses :meth:`operator.xor` rather than
    :meth:`operator.sub`.

    Examples
    --------
    Difference with previous row

    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'a': [1, 2, 3, 4, 5, 6],
    ...                    'b': [1, 1, 2, 3, 5, 8],
    ...                    'c': [1, 4, 9, 16, 25, 36]})
    >>> df.execute()
       a  b   c
    0  1  1   1
    1  2  1   4
    2  3  2   9
    3  4  3  16
    4  5  5  25
    5  6  8  36

    >>> df.diff().execute()
         a    b     c
    0  NaN  NaN   NaN
    1  1.0  0.0   3.0
    2  1.0  1.0   5.0
    3  1.0  1.0   7.0
    4  1.0  2.0   9.0
    5  1.0  3.0  11.0

    Difference with previous column

    >>> df.diff(axis=1).execute()
        a    b     c
    0 NaN  0.0   0.0
    1 NaN -1.0   3.0
    2 NaN -1.0   7.0
    3 NaN -1.0  13.0
    4 NaN  0.0  20.0
    5 NaN  2.0  28.0

    Difference with 3rd previous row

    >>> df.diff(periods=3).execute()
         a    b     c
    0  NaN  NaN   NaN
    1  NaN  NaN   NaN
    2  NaN  NaN   NaN
    3  3.0  2.0  15.0
    4  3.0  4.0  21.0
    5  3.0  6.0  27.0

    Difference with following row

    >>> df.diff(periods=-1).execute()
         a    b     c
    0 -1.0  0.0  -3.0
    1 -1.0 -1.0  -5.0
    2 -1.0 -1.0  -7.0
    3 -1.0 -2.0  -9.0
    4 -1.0 -3.0 -11.0
    5  NaN  NaN   NaN
    """
    axis = validate_axis(axis, df)
    op = DataFrameDiff(periods=periods, axis=axis)
    return op(df)


def series_diff(series, periods=1):
    """
    First discrete difference of element.
    Calculates the difference of a Series element compared with another
    element in the Series (default is element in previous row).

    Parameters
    ----------
    periods : int, default 1
        Periods to shift for calculating difference, accepts negative
        values.

    Returns
    -------
    Series
        First differences of the Series.

    See Also
    --------
    Series.pct_change :
        Percent change over given number of periods.
    Series.shift :
        Shift index by desired number of periods with an optional time freq.
    DataFrame.diff :
        First discrete difference of object.

    Notes
    -----
    For boolean dtypes, this uses :meth:`operator.xor` rather than
    :meth:`operator.sub`.

    Examples
    --------

    Difference with previous row

    >>> import mars.dataframe as md
    >>> s = md.Series([1, 1, 2, 3, 5, 8])
    >>> s.diff().execute()
    0    NaN
    1    0.0
    2    1.0
    3    1.0
    4    2.0
    5    3.0
    dtype: float64

    Difference with 3rd previous row

    >>> s.diff(periods=3).execute()
    0    NaN
    1    NaN
    2    NaN
    3    2.0
    4    4.0
    5    6.0
    dtype: float64

    Difference with following row

    >>> s.diff(periods=-1).execute()
    0    0.0
    1   -1.0
    2   -1.0
    3   -2.0
    4   -3.0
    5    NaN
    dtype: float64
    """
    op = DataFrameDiff(periods=periods)
    return op(series)
