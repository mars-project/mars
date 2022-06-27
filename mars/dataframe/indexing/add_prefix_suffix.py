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

import textwrap
from functools import partial


def _get_prefix_suffix_docs(is_prefix: bool):
    if is_prefix:
        action, pos = "prefix", "before"
        r_action = "suffix"
    else:
        action, pos = "suffix", "after"
        r_action = "prefix"

    def mk_col(ch: str, s: str):
        return f"{ch}_{s}" if is_prefix else f"{s}_{ch}"

    doc = f"""
    {action.capitalize()} labels with string `{action}`.

    For Series, the row labels are {action}ed.
    For DataFrame, the column labels are {action}ed.

    Parameters
    ----------
    {action} : str
        The string to add {pos} each label.

    Returns
    -------
    Series or DataFrame
        New Series or DataFrame with updated labels.

    See Also
    --------
    Series.add_{r_action}: Suffix row labels with string `{r_action}`.
    DataFrame.add_{r_action}: Suffix column labels with string `{r_action}`.

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

    >>> s.add_prefix({mk_col('item', '')!r}).execute()
    {mk_col('item', '0')}    1
    {mk_col('item', '1')}    2
    {mk_col('item', '2')}    3
    {mk_col('item', '3')}    4
    dtype: int64

    >>> df = md.DataFrame({{'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]}})
    >>> df.execute()
       A  B
    0  1  3
    1  2  4
    2  3  5
    3  4  6

    >>> df.add_prefix({mk_col('col', '')!r}).execute()
         {mk_col('col', 'A')}  {mk_col('col', 'B')}
    0        1      3
    1        2      4
    2        3      5
    3        4      6
    """
    return textwrap.dedent(doc).strip()


def df_add_prefix(df, prefix):
    f = partial("{prefix}{}".format, prefix=prefix)
    return df.rename(columns=f)


def series_add_prefix(series, prefix):
    f = partial("{prefix}{}".format, prefix=prefix)
    return series.rename(index=f)


def df_add_suffix(df, suffix):
    f = partial("{}{suffix}".format, suffix=suffix)
    return df.rename(columns=f)


def series_add_suffix(series, suffix):
    f = partial("{}{suffix}".format, suffix=suffix)
    return series.rename(index=f)


df_add_prefix.__doc__ = _get_prefix_suffix_docs(True)
series_add_prefix.__doc__ = df_add_prefix.__doc__
df_add_suffix.__doc__ = _get_prefix_suffix_docs(False)
series_add_suffix.__doc__ = df_add_suffix.__doc__
