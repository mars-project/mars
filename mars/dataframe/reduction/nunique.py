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

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None

from ... import opcodes as OperandDef
from ...core import OutputType
from ...config import options
from ...serialization.serializables import BoolField
from ...utils import lazy_import
from ..arrays import ArrowListArray
from .core import DataFrameReductionOperand, DataFrameReductionMixin, CustomReduction

cp = lazy_import("cupy", globals=globals(), rename="cp")
cudf = lazy_import("cudf", globals=globals())


class NuniqueReduction(CustomReduction):
    pre_with_agg = True
    post_with_agg = True

    def __init__(
        self, name="nunique", axis=0, dropna=True, use_arrow_dtype=False, is_gpu=False
    ):
        super().__init__(name, is_gpu=is_gpu)
        self._axis = axis
        self._dropna = dropna
        self._use_arrow_dtype = use_arrow_dtype

    def _get_modules(self):
        if not self.is_gpu():
            return np, pd
        else:  # pragma: no cover
            return cp, cudf

    def _drop_duplicates(self, value, explode=False, agg=False):
        xp, xdf = self._get_modules()
        use_arrow_dtype = self._use_arrow_dtype and xp is not cp
        if self._use_arrow_dtype and xp is not cp and hasattr(value, "to_numpy"):
            value = value.to_numpy()
        else:
            value = value.values

        if explode:
            if len(value) == 0:
                if not use_arrow_dtype:
                    return [xp.array([], dtype=object)]
                else:
                    return [ArrowListArray([])]
            value = xp.concatenate(value)

        value = xdf.unique(value)

        if not agg:
            if not use_arrow_dtype:
                return [value]
            else:
                try:
                    return ArrowListArray([value])
                except pa.ArrowInvalid:
                    # fallback due to diverse dtypes
                    return [value]
        else:
            if self._dropna:
                return xp.sum(xdf.notna(value))
            return len(value)

    def pre(self, in_data):  # noqa: W0221  # pylint: disable=arguments-differ
        xp, xdf = self._get_modules()
        out_dtype = object if not self._use_arrow_dtype or xp is cp else None
        if isinstance(in_data, xdf.Series):
            unique_values = self._drop_duplicates(in_data)
            return xdf.Series(unique_values, name=in_data.name, dtype=out_dtype)
        else:
            if self._axis == 0:
                data = dict()
                for d, v in in_data.iteritems():
                    data[d] = self._drop_duplicates(v)
                df = xdf.DataFrame(data, copy=False, dtype=out_dtype)
            else:
                df = xdf.DataFrame(columns=[0])
                for d, v in in_data.iterrows():
                    df.loc[d] = self._drop_duplicates(v)
            return df

    def agg(self, in_data):  # noqa: W0221  # pylint: disable=arguments-differ
        xp, xdf = self._get_modules()
        out_dtype = object if not self._use_arrow_dtype or xp is cp else None
        if isinstance(in_data, xdf.Series):
            unique_values = self._drop_duplicates(in_data, explode=True)
            return xdf.Series(unique_values, name=in_data.name, dtype=out_dtype)
        else:
            if self._axis == 0:
                data = dict()
                for d, v in in_data.iteritems():
                    data[d] = self._drop_duplicates(v, explode=True)
                df = xdf.DataFrame(data, copy=False, dtype=out_dtype)
            else:
                df = xdf.DataFrame(columns=[0])
                for d, v in in_data.iterrows():
                    df.loc[d] = self._drop_duplicates(v, explode=True)
            return df

    def post(self, in_data):  # noqa: W0221  # pylint: disable=arguments-differ
        xp, xdf = self._get_modules()
        if isinstance(in_data, xdf.Series):
            return self._drop_duplicates(in_data, explode=True, agg=True)
        else:
            in_data_iter = (
                in_data.iteritems() if self._axis == 0 else in_data.iterrows()
            )
            data = dict()
            for d, v in in_data_iter:
                data[d] = self._drop_duplicates(v, explode=True, agg=True)
            return xdf.Series(data)


class DataFrameNunique(DataFrameReductionOperand, DataFrameReductionMixin):
    _op_type_ = OperandDef.NUNIQUE
    _func_name = "nunique"

    _dropna = BoolField("dropna")
    _use_arrow_dtype = BoolField("use_arrow_dtype")

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
        return NuniqueReduction(
            name=cls._func_name,
            axis=op.axis,
            dropna=op.dropna,
            use_arrow_dtype=op.use_arrow_dtype,
            is_gpu=op.is_gpu(),
        )


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
    op = DataFrameNunique(
        axis=axis,
        dropna=dropna,
        combine_size=combine_size,
        output_types=[OutputType.series],
        use_arrow_dtype=options.dataframe.use_arrow_dtype,
    )
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
    op = DataFrameNunique(
        dropna=dropna,
        combine_size=combine_size,
        output_types=[OutputType.scalar],
        use_arrow_dtype=options.dataframe.use_arrow_dtype,
    )
    return op(series)
