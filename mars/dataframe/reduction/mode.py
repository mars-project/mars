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

from ... import opcodes
from ...core import OutputType
from ...config import options
from ...serialization.serializables import BoolField
from ...utils import lazy_import
from ..arrays import ArrowListDtype
from .core import DataFrameReductionOperand, DataFrameReductionMixin, CustomReduction

cudf = lazy_import("cudf")


class ModeReduction(CustomReduction):
    pre_with_agg = True

    def __init__(
        self, name="mode", axis=0, numeric_only=False, dropna=True, is_gpu=False
    ):
        super().__init__(name, is_gpu=is_gpu)
        self._axis = axis
        self._numeric_only = numeric_only
        self._dropna = dropna

    def pre(self, in_data):  # noqa: W0221  # pylint: disable=arguments-differ
        xdf = cudf if self.is_gpu() else pd
        if isinstance(in_data, xdf.Series):
            return in_data.value_counts(dropna=self._dropna)
        else:
            if self._axis == 0:
                data = dict()
                for d, v in in_data.iteritems():
                    data[d] = [v.value_counts(dropna=self._dropna).to_dict()]
                df = xdf.DataFrame(data)
            else:
                df = xdf.DataFrame(columns=[0])
                for d, v in in_data.iterrows():
                    df.loc[d] = [v.value_counts(dropna=self._dropna).to_dict()]
            return df

    def agg(self, in_data):  # noqa: W0221  # pylint: disable=arguments-differ
        xdf = cudf if self.is_gpu() else pd
        if isinstance(in_data, xdf.Series):
            return in_data.groupby(in_data.index, dropna=self._dropna).sum()
        else:
            if self._axis == 0:
                data = dict()
                for d, v in in_data.iteritems():
                    data[d] = [v.apply(pd.Series).sum().to_dict()]
                df = xdf.DataFrame(data)
            else:
                df = xdf.DataFrame(columns=[0])
                for d, v in in_data.iterrows():
                    df.loc[d] = [v.apply(pd.Series).sum().to_dict()]
            return df

    def post(self, in_data):  # noqa: W0221  # pylint: disable=arguments-differ
        xdf = cudf if self.is_gpu() else pd

        def _handle_series(s):
            summed = s.groupby(s.index, dropna=self._dropna).sum()
            if summed.ndim == 2:
                summed = summed.iloc[:, 0]
            return pd.Series(summed[summed == summed.max()].index)

        if isinstance(in_data, xdf.Series):
            return _handle_series(in_data)
        else:
            in_data_iter = (
                in_data.iteritems() if self._axis == 0 else in_data.iterrows()
            )
            s_list = []
            for d, v in in_data_iter:
                if isinstance(v.dtype, ArrowListDtype):
                    v = xdf.Series(v.to_numpy())
                s = _handle_series(v.apply(pd.Series).T)
                s.name = d
                s_list.append(s)
            res = pd.concat(s_list, axis=1)
            if self._axis == 0:
                return res
            return res.T


class DataFrameMode(DataFrameReductionOperand, DataFrameReductionMixin):
    _op_type_ = opcodes.MODE
    _func_name = "mode"

    numeric_only = BoolField("numeric_only", default=None)
    use_arrow_dtype = BoolField("use_arrow_dtype", default=None)

    @classmethod
    def get_reduction_callable(cls, op: "DataFrameMode"):
        return ModeReduction(
            name=cls._func_name,
            axis=op.axis,
            numeric_only=op.numeric_only,
            dropna=op.skipna,
            is_gpu=op.is_gpu(),
        )

    @property
    def dropna(self) -> bool:
        return self.skipna

    @classmethod
    def tile(cls, op):
        ts = yield from super().tile(op)
        return ts

    def __call__(self, *args, **kwargs):
        t = super().__call__(*args, **kwargs)
        return t


def mode_dataframe(df, axis=0, numeric_only=False, dropna=True, combine_size=None):
    """
    Get the mode(s) of each element along the selected axis.

    The mode of a set of values is the value that appears most often.
    It can be multiple values.

    Parameters
    ----------
    axis : {0 or 'index', 1 or 'columns'}, default 0
        The axis to iterate over while searching for the mode:

        * 0 or 'index' : get mode of each column
        * 1 or 'columns' : get mode of each row.

    numeric_only : bool, default False
        If True, only apply to numeric columns.
    dropna : bool, default True
        Don't consider counts of NaN/NaT.

    Returns
    -------
    DataFrame
        The modes of each column or row.

    See Also
    --------
    Series.mode : Return the highest frequency value in a Series.
    Series.value_counts : Return the counts of values in a Series.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> import mars.dataframe as md
    >>> df = md.DataFrame([('bird', 2, 2),
    ...                    ('mammal', 4, mt.nan),
    ...                    ('arthropod', 8, 0),
    ...                    ('bird', 2, mt.nan)],
    ...                   index=('falcon', 'horse', 'spider', 'ostrich'),
    ...                   columns=('species', 'legs', 'wings'))
    >>> df.execute()
               species  legs  wings
    falcon        bird     2    2.0
    horse       mammal     4    NaN
    spider   arthropod     8    0.0
    ostrich       bird     2    NaN

    By default, missing values are not considered, and the mode of wings
    are both 0 and 2. Because the resulting DataFrame has two rows,
    the second row of ``species`` and ``legs`` contains ``NaN``.

    >>> df.mode().execute()
      species  legs  wings
    0    bird   2.0    0.0
    1     NaN   NaN    2.0

    Setting ``dropna=False`` ``NaN`` values are considered and they can be
    the mode (like for wings).

    >>> df.mode(dropna=False).execute()
      species  legs  wings
    0    bird     2    NaN

    Setting ``numeric_only=True``, only the mode of numeric columns is
    computed, and columns of other types are ignored.

    >>> df.mode(numeric_only=True).execute()
       legs  wings
    0   2.0    0.0
    1   NaN    2.0

    To compute the mode over columns and not rows, use the axis parameter:

    >>> df.mode(axis='columns', numeric_only=True).execute()
               0    1
    falcon   2.0  NaN
    horse    4.0  NaN
    spider   0.0  8.0
    ostrich  2.0  NaN
    """
    op = DataFrameMode(
        axis=axis,
        numeric_only=numeric_only,
        dropna=dropna,
        combine_size=combine_size,
        output_types=[OutputType.series],
        use_arrow_dtype=options.dataframe.use_arrow_dtype,
    )
    return op(df)


def mode_series(series, dropna=True, combine_size=None):
    """
    Return the mode(s) of the Series.

    The mode is the value that appears most often. There can be multiple modes.

    Always returns Series even if only one value is returned.

    Parameters
    ----------
    dropna : bool, default True
        Don't consider counts of NaN/NaT.

    Returns
    -------
    Series
        Modes of the Series in sorted order.
    """
    op = DataFrameMode(
        dropna=dropna,
        combine_size=combine_size,
        output_types=[OutputType.scalar],
        use_arrow_dtype=options.dataframe.use_arrow_dtype,
    )
    return op(series)
