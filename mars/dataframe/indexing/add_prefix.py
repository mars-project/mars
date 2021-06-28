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

from functools import partial


def df_add_prefix(df, prefix):
    """
    Prefix labels with string `prefix`.

    For DataFrame, the column labels are prefixed.

    Parameters
    ----------
    prefix : str
        The string to add before each label.

    Returns
    -------
    DataFrame
        New DataFrame with updated labels.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> df = md.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})
    >>> df.execute()
        A  B
    0  1  3
    1  2  4
    2  3  5
    3  4  6

    >>> df.add_prefix('col_').execute()
            col_A  col_B
    0       1       3
    1       2       4
    2       3       5
    3       4       6
    """
    f = partial("{prefix}{}".format, prefix=prefix)

    return df.rename(columns=f)


def series_add_prefix(series, prefix):
    """
    Prefix labels with string `prefix`.

    For Series, the row labels are prefixed.

    Parameters
    ----------
    prefix : str
        The string to add before each label.

    Returns
    -------
    Series
        New Series with updated labels.

    Examples
    --------
    >>> import mars.dataframe as md
    >>> s = md.Series([1, 2, 3, 4])
    >>> s.execute()
    0    1
    1    2
    2    3
    3    4
    dtype: int64

    >>> s.add_prefix('item_').execute()
    item_0    1
    item_1    2
    item_2    3
    item_3    4
    dtype: int64
    """
    f = partial("{prefix}{}".format, prefix=prefix)

    return series.rename(index=f)
