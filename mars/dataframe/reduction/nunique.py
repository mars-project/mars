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

from collections import OrderedDict

import pandas as pd

from ... import opcodes as OperandDef
from ...serialize import BoolField
from ...utils import lazy_import
from .core import DataFrameReductionOperand, DataFrameReductionMixin, ObjectType


cudf = lazy_import('cudf', globals=globals())


class DataFrameNunique(DataFrameReductionOperand, DataFrameReductionMixin):
    _op_type_ = OperandDef.NUNIQUE
    _func_name = 'nunique'

    _dropna = BoolField('dropna')

    def __init__(self, dropna=None, **kw):
        super(DataFrameNunique, self).__init__(_dropna=dropna, **kw)

    @property
    def dropna(self):
        return self._dropna

    @classmethod
    def _execute_map(cls, ctx, op):
        xdf = cudf if op.gpu else pd
        in_data = ctx[op.inputs[0].key]
        if isinstance(in_data, xdf.Series) or op.object_type == ObjectType.series:
            unique_values = in_data.drop_duplicates()
            ctx[op.outputs[0].key] = xdf.Series(unique_values, name=in_data.name)
        else:
            if op.axis == 0:
                df = xdf.DataFrame(OrderedDict((d, [v.drop_duplicates().to_list()])
                                               for d, v in in_data.iteritems()))
            else:
                df = xdf.DataFrame(columns=[0])
                for d, v in in_data.iterrows():
                    df.loc[d] = [v.drop_duplicates().to_list()]
            ctx[op.outputs[0].key] = df

    @classmethod
    def _execute_combine(cls, ctx, op):
        xdf = cudf if op.gpu else pd
        in_data = ctx[op.inputs[0].key]
        if isinstance(in_data, xdf.Series):
            unique_values = in_data.explode().drop_duplicates()
            ctx[op.outputs[0].key] = xdf.Series(unique_values, name=in_data.name)
        else:
            if op.axis == 0:
                df = xdf.DataFrame(OrderedDict((d, [v.explode().drop_duplicates().to_list()])
                                               for d, v in in_data.iteritems()))
            else:
                df = xdf.DataFrame(columns=[0])
                for d, v in in_data.iterrows():
                    df.loc[d] = [v.explode().drop_duplicates().to_list()]
            ctx[op.outputs[0].key] = df

    @classmethod
    def _execute_agg(cls, ctx, op):
        xdf = cudf if op.gpu else pd
        in_data = ctx[op.inputs[0].key]
        if isinstance(in_data, xdf.Series):
            ctx[op.outputs[0].key] = in_data.explode().nunique(dropna=op.dropna)
        else:
            if op.axis == 0:
                ctx[op.outputs[0].key] = xdf.Series(OrderedDict((d, v.explode().nunique(dropna=op.dropna))
                                                    for d, v in in_data.iteritems()))
            else:
                ctx[op.outputs[0].key] = xdf.Series(OrderedDict((d, v.explode().nunique(dropna=op.dropna))
                                                    for d, v in in_data.iterrows()))

    @classmethod
    def _execute_reduction(cls, in_data, op, min_count=None, reduction_func=None):
        kwargs = dict()
        if op.axis is not None:
            kwargs['axis'] = op.axis
        return in_data.nunique(dropna=op.dropna, **kwargs)


def nunique_dataframe(df, axis=0, dropna=True, combine_size=None):
    """
    Count distinct observations over requested axis.

    Return Series with number of distinct observations. Can ignore NaN
    values.

    Parameters
    ----------
    axis : {0 or 'index', 1 or 'columns'}, default 0
        The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for
        column-wise.
    dropna : bool, default True
        Don't include NaN in the counts.
    combine_size : int, optional
        The number of chunks to combine.

    Returns
    -------
    Series

    See Also
    --------
    Series.nunique: Method nunique for Series.
    DataFrame.count: Count non-NA cells for each column or row.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    >>> df.nunique().execute()
    A    3
    B    1
    dtype: int64

    >>> df.nunique(axis=1).execute()
    0    1
    1    2
    2    2
    dtype: int64
    """
    op = DataFrameNunique(axis=axis, dropna=dropna, combine_size=combine_size,
                          object_type=ObjectType.series)
    return op(df)


def nunique_series(df, dropna=True, combine_size=None):
    """
    Return number of unique elements in the object.

    Excludes NA values by default.

    Parameters
    ----------
    dropna : bool, default True
        Don't include NaN in the count.
    combine_size : int, optional
        The number of chunks to combine.

    Returns
    -------
    int

    See Also
    --------
    DataFrame.nunique: Method nunique for DataFrame.
    Series.count: Count non-NA/null observations in the Series.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> s = md.Series([1, 3, 5, 7, 7])
    >>> s.execute()
    0    1
    1    3
    2    5
    3    7
    4    7
    dtype: int64

    >>> s.nunique().execute()
    4
    """
    op = DataFrameNunique(dropna=dropna, combine_size=combine_size,
                          object_type=ObjectType.scalar)
    return op(df)
