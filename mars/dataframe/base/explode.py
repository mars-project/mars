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

from ... import opcodes
from ...core import OutputType, recursive_tile
from ...serialization.serializables import AnyField, BoolField
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index, standardize_range_index


class DataFrameExplode(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.EXPLODE

    _column = AnyField('column')
    _ignore_index = BoolField('ignore_field')

    def __init__(self, column=None, ignore_index=None, output_types=None, **kw):
        super().__init__(_column=column, _ignore_index=ignore_index,
                         _output_types=output_types, **kw)

    @property
    def column(self):
        return self._column

    @property
    def ignore_index(self):
        return self._ignore_index

    def _rewrite_params(self, in_obj):
        params = in_obj.params.copy()
        new_shape = list(in_obj.shape)
        new_shape[0] = np.nan
        params['shape'] = tuple(new_shape)

        if self.ignore_index:
            params['index_value'] = parse_index(
                pd.RangeIndex(-1), (in_obj.key, in_obj.index_value.key))
        else:
            params['index_value'] = parse_index(
                None, (in_obj.key, in_obj.index_value.key))
        return params

    def __call__(self, df_or_series):
        return self.new_tileable([df_or_series], **self._rewrite_params(df_or_series))

    @classmethod
    def tile(cls, op: "DataFrameExplode"):
        in_obj = op.inputs[0]

        if in_obj.ndim == 2 and in_obj.chunk_shape[1] > 1:
            # make sure data's second dimension has only 1 chunk
            in_obj = yield from recursive_tile(
                in_obj.rechunk({1: in_obj.shape[1]}))

        chunks = []
        for chunk in in_obj.chunks:
            new_op = op.copy().reset_key()
            chunks.append(new_op.new_chunk([chunk], **op._rewrite_params(chunk)))

        if op.ignore_index:
            chunks = standardize_range_index(chunks)

        new_op = op.copy().reset_key()
        out_params = op.outputs[0].params
        if in_obj.ndim == 2:
            new_nsplits = ((np.nan,) * in_obj.chunk_shape[0], in_obj.nsplits[1])
        else:
            new_nsplits = ((np.nan,) * in_obj.chunk_shape[0],)
        return new_op.new_tileable([in_obj], chunks=chunks, nsplits=new_nsplits, **out_params)

    @classmethod
    def execute(cls, ctx, op: "DataFrameExplode"):
        in_data = ctx[op.inputs[0].key]
        if in_data.ndim == 2:
            ctx[op.outputs[0].key] = in_data.explode(op.column)
        else:
            ctx[op.outputs[0].key] = in_data.explode()


def df_explode(df, column, ignore_index=False):
    """
    Transform each element of a list-like to a row, replicating index values.

    Parameters
    ----------
    column : str or tuple
        Column to explode.
    ignore_index : bool, default False
        If True, the resulting index will be labeled 0, 1, …, n - 1.

    Returns
    -------
    DataFrame
        Exploded lists to rows of the subset columns;
        index will be duplicated for these rows.

    Raises
    ------
    ValueError :
        if columns of the frame are not unique.

    See Also
    --------
    DataFrame.unstack : Pivot a level of the (necessarily hierarchical)
        index labels.
    DataFrame.melt : Unpivot a DataFrame from wide format to long format.
    Series.explode : Explode a DataFrame from list-like columns to long format.

    Notes
    -----
    This routine will explode list-likes including lists, tuples,
    Series, and np.ndarray. The result dtype of the subset rows will
    be object. Scalars will be returned unchanged. Empty list-likes will
    result in a np.nan for that row.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'A': [[1, 2, 3], 'foo', [], [3, 4]], 'B': 1})
    >>> df.execute()
               A  B
    0  [1, 2, 3]  1
    1        foo  1
    2         []  1
    3     [3, 4]  1

    >>> df.explode('A').execute()
         A  B
    0    1  1
    0    2  1
    0    3  1
    1  foo  1
    2  NaN  1
    3    3  1
    3    4  1
    """
    op = DataFrameExplode(column=column, ignore_index=ignore_index,
                          output_types=[OutputType.dataframe])
    return op(df)


def series_explode(series, ignore_index=False):
    """
    Transform each element of a list-like to a row.

    Parameters
    ----------
    ignore_index : bool, default False
        If True, the resulting index will be labeled 0, 1, …, n - 1.

    Returns
    -------
    Series
        Exploded lists to rows; index will be duplicated for these rows.

    See Also
    --------
    Series.str.split : Split string values on specified separator.
    Series.unstack : Unstack, a.k.a. pivot, Series with MultiIndex
        to produce DataFrame.
    DataFrame.melt : Unpivot a DataFrame from wide format to long format.
    DataFrame.explode : Explode a DataFrame from list-like
        columns to long format.

    Notes
    -----
    This routine will explode list-likes including lists, tuples,
    Series, and np.ndarray. The result dtype of the subset rows will
    be object. Scalars will be returned unchanged. Empty list-likes will
    result in a np.nan for that row.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> import mars.dataframe as md
    >>> s = md.Series([[1, 2, 3], 'foo', [], [3, 4]])
    >>> s.execute()
    0    [1, 2, 3]
    1          foo
    2           []
    3       [3, 4]
    dtype: object

    >>> s.explode().execute()
    0      1
    0      2
    0      3
    1    foo
    2    NaN
    3      3
    3      4
    dtype: object
    """
    op = DataFrameExplode(ignore_index=ignore_index, output_types=[OutputType.series])
    return op(series)
