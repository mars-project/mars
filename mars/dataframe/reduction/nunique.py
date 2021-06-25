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

import pandas as pd
try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None

from ... import opcodes as OperandDef
from ...core import OutputType
from ...config import options
from ...serialization.serializables import BoolField
from ...utils import lazy_import
from ..arrays import ArrowListArray, ArrowListDtype
from .core import DataFrameReductionOperand, DataFrameReductionMixin, \
    CustomReduction

cudf = lazy_import('cudf', globals=globals())


class NuniqueReduction(CustomReduction):
    pre_with_agg = True

    def __init__(self, name='unique', axis=0, dropna=True, use_arrow_dtype=False,
                 is_gpu=False):
        super().__init__(name, is_gpu=is_gpu)
        self._axis = axis
        self._dropna = dropna
        self._use_arrow_dtype = use_arrow_dtype

    @staticmethod
    def _drop_duplicates_to_arrow(v, explode=False):
        if explode:
            v = v.explode()
        try:
            return ArrowListArray([v.drop_duplicates().to_numpy()])
        except pa.ArrowInvalid:
            # fallback due to diverse dtypes
            return [v.drop_duplicates().to_list()]

    def pre(self, in_data):  # noqa: W0221  # pylint: disable=arguments-differ
        xdf = cudf if self.is_gpu() else pd
        if isinstance(in_data, xdf.Series):
            unique_values = in_data.drop_duplicates()
            return xdf.Series(unique_values, name=in_data.name)
        else:
            if self._axis == 0:
                data = dict()
                for d, v in in_data.iteritems():
                    if not self._use_arrow_dtype or xdf is cudf:
                        data[d] = [v.drop_duplicates().to_list()]
                    else:
                        data[d] = self._drop_duplicates_to_arrow(v)
                df = xdf.DataFrame(data)
            else:
                df = xdf.DataFrame(columns=[0])
                for d, v in in_data.iterrows():
                    if not self._use_arrow_dtype or xdf is cudf:
                        df.loc[d] = [v.drop_duplicates().to_list()]
                    else:
                        df.loc[d] = self._drop_duplicates_to_arrow(v)
            return df

    def agg(self, in_data):  # noqa: W0221  # pylint: disable=arguments-differ
        xdf = cudf if self.is_gpu() else pd
        if isinstance(in_data, xdf.Series):
            unique_values = in_data.explode().drop_duplicates()
            return xdf.Series(unique_values, name=in_data.name)
        else:
            if self._axis == 0:
                data = dict()
                for d, v in in_data.iteritems():
                    if not self._use_arrow_dtype or xdf is cudf:
                        data[d] = [v.explode().drop_duplicates().to_list()]
                    else:
                        v = pd.Series(v.to_numpy())
                        data[d] = self._drop_duplicates_to_arrow(v, explode=True)
                df = xdf.DataFrame(data)
            else:
                df = xdf.DataFrame(columns=[0])
                for d, v in in_data.iterrows():
                    if not self._use_arrow_dtype or xdf is cudf:
                        df.loc[d] = [v.explode().drop_duplicates().to_list()]
                    else:
                        df.loc[d] = self._drop_duplicates_to_arrow(v, explode=True)
            return df

    def post(self, in_data):  # noqa: W0221  # pylint: disable=arguments-differ
        xdf = cudf if self.is_gpu() else pd
        if isinstance(in_data, xdf.Series):
            return in_data.explode().nunique(dropna=self._dropna)
        else:
            in_data_iter = in_data.iteritems() if self._axis == 0 else in_data.iterrows()
            data = dict()
            for d, v in in_data_iter:
                if isinstance(v.dtype, ArrowListDtype):
                    v = xdf.Series(v.to_numpy())
                data[d] = v.explode().nunique(dropna=self._dropna)
            return xdf.Series(data)


class DataFrameNunique(DataFrameReductionOperand, DataFrameReductionMixin):
    _op_type_ = OperandDef.NUNIQUE
    _func_name = 'nunique'

    _dropna = BoolField('dropna')
    _use_arrow_dtype = BoolField('use_arrow_dtype')

    def __init__(self, dropna=None, use_arrow_dtype=None, **kw):
        super().__init__(_dropna=dropna, _use_arrow_dtype=use_arrow_dtype, **kw)

    @property
    def dropna(self):
        return self._dropna

    @property
    def use_arrow_dtype(self):
        return self._use_arrow_dtype

    @classmethod
    def get_reduction_callable(cls, op):
        return NuniqueReduction(name=cls._func_name, axis=op.axis, dropna=op.dropna,
                                use_arrow_dtype=op.use_arrow_dtype, is_gpu=op.is_gpu())


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
                          output_types=[OutputType.series],
                          use_arrow_dtype=options.dataframe.use_arrow_dtype)
    return op(df)


def nunique_series(series, dropna=True, combine_size=None):
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
                          output_types=[OutputType.scalar],
                          use_arrow_dtype=options.dataframe.use_arrow_dtype)
    return op(series)
