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

from ..utils import validate_axis


def pct_change(df_or_series, periods=1, fill_method='pad',
               limit=None, freq=None, **kwargs):
    """
    Percentage change between the current and a prior element.

    Computes the percentage change from the immediately previous row by
    default. This is useful in comparing the percentage of change in a time
    series of elements.

    Parameters
    ----------
    periods : int, default 1
        Periods to shift for forming percent change.
    fill_method : str, default 'pad'
        How to handle NAs before computing percent changes.
    limit : int, default None
        The number of consecutive NAs to fill before stopping.
    freq : DateOffset, timedelta, or str, optional
        Increment to use from time series API (e.g. 'M' or BDay()).
    **kwargs
        Additional keyword arguments are passed into
        `DataFrame.shift` or `Series.shift`.

    Returns
    -------
    chg : Series or DataFrame
        The same type as the calling object.

    See Also
    --------
    Series.diff : Compute the difference of two elements in a Series.
    DataFrame.diff : Compute the difference of two elements in a DataFrame.
    Series.shift : Shift the index by some number of periods.
    DataFrame.shift : Shift the index by some number of periods.

    Examples
    --------
    **Series**

    >>> import mars.dataframe as md

    >>> s = md.Series([90, 91, 85])
    >>> s.execute()
    0    90
    1    91
    2    85
    dtype: int64

    >>> s.pct_change().execute()
    0         NaN
    1    0.011111
    2   -0.065934
    dtype: float64

    >>> s.pct_change(periods=2).execute()
    0         NaN
    1         NaN
    2   -0.055556
    dtype: float64

    See the percentage change in a Series where filling NAs with last
    valid observation forward to next valid.

    >>> s = md.Series([90, 91, None, 85])
    >>> s.execute()
    0    90.0
    1    91.0
    2     NaN
    3    85.0
    dtype: float64

    >>> s.pct_change(fill_method='ffill').execute()
    0         NaN
    1    0.011111
    2    0.000000
    3   -0.065934
    dtype: float64

    **DataFrame**

    Percentage change in French franc, Deutsche Mark, and Italian lira from
    1980-01-01 to 1980-03-01.

    >>> df = md.DataFrame({
    ...     'FR': [4.0405, 4.0963, 4.3149],
    ...     'GR': [1.7246, 1.7482, 1.8519],
    ...     'IT': [804.74, 810.01, 860.13]},
    ...     index=['1980-01-01', '1980-02-01', '1980-03-01'])
    >>> df.execute()
                    FR      GR      IT
    1980-01-01  4.0405  1.7246  804.74
    1980-02-01  4.0963  1.7482  810.01
    1980-03-01  4.3149  1.8519  860.13

    >>> df.pct_change().execute()
                      FR        GR        IT
    1980-01-01       NaN       NaN       NaN
    1980-02-01  0.013810  0.013684  0.006549
    1980-03-01  0.053365  0.059318  0.061876

    Percentage of change in GOOG and APPL stock volume. Shows computing
    the percentage change between columns.

    >>> df = md.DataFrame({
    ...     '2016': [1769950, 30586265],
    ...     '2015': [1500923, 40912316],
    ...     '2014': [1371819, 41403351]},
    ...     index=['GOOG', 'APPL'])
    >>> df.execute()
              2016      2015      2014
    GOOG   1769950   1500923   1371819
    APPL  30586265  40912316  41403351

    >>> df.pct_change(axis='columns').execute()
          2016      2015      2014
    GOOG   NaN -0.151997 -0.086016
    APPL   NaN  0.337604  0.012002
    """

    axis = validate_axis(kwargs.pop('axis', 0))
    if fill_method is None:
        data = df_or_series
    else:
        data = df_or_series.fillna(method=fill_method, axis=axis, limit=limit)

    rs = data.div(data.shift(periods=periods, freq=freq,
                             axis=axis, **kwargs)) - 1
    if freq is not None:
        # Shift method is implemented differently when freq is not None
        # We want to restore the original index
        rs = rs.loc[~rs.index.duplicated()]
        rs = rs.reindex_like(data)
    return rs
