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

_flex_doc_FRAME = """
Get {desc} of dataframe and other, element-wise (binary operator `{op_name}`).
Equivalent to ``{equiv}``, but with support to substitute a fill_value
for missing data in one of the inputs. With reverse version, `{reverse}`.
Among flexible wrappers (`add`, `sub`, `mul`, `div`, `mod`, `pow`) to
arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

Parameters
----------
other : scalar, sequence, Series, or DataFrame
    Any single or multiple element data structure, or list-like object.
axis : {{0 or 'index', 1 or 'columns'}}
    Whether to compare by the index (0 or 'index') or columns
    (1 or 'columns'). For Series input, axis to match Series index on.
level : int or label
    Broadcast across a level, matching Index values on the
    passed MultiIndex level.
fill_value : float or None, default None
    Fill existing missing (NaN) values, and any new element needed for
    successful DataFrame alignment, with this value before computation.
    If data in both corresponding DataFrame locations is missing
    the result will be missing.

Returns
-------
DataFrame
    Result of the arithmetic operation.

See Also
--------
DataFrame.add : Add DataFrames.
DataFrame.sub : Subtract DataFrames.
DataFrame.mul : Multiply DataFrames.
DataFrame.div : Divide DataFrames (float division).
DataFrame.truediv : Divide DataFrames (float division).
DataFrame.floordiv : Divide DataFrames (integer division).
DataFrame.mod : Calculate modulo (remainder after division).
DataFrame.pow : Calculate exponential power.

Notes
-----
Mismatched indices will be unioned together.

Examples
--------
>>> import mars.dataframe as md
>>> df = md.DataFrame({{'angles': [0, 3, 4],
...                    'degrees': [360, 180, 360]}},
...                   index=['circle', 'triangle', 'rectangle'])
>>> df.execute()
           angles  degrees
circle          0      360
triangle        3      180
rectangle       4      360

Add a scalar with operator version which return the same
results.

>>> (df + 1).execute()
           angles  degrees
circle          1      361
triangle        4      181
rectangle       5      361

>>> df.add(1).execute()
           angles  degrees
circle          1      361
triangle        4      181
rectangle       5      361

Divide by constant with reverse version.

>>> df.div(10).execute()
           angles  degrees
circle        0.0     36.0
triangle      0.3     18.0
rectangle     0.4     36.0

>>> df.rdiv(10).execute()
             angles   degrees
circle          inf  0.027778
triangle   3.333333  0.055556
rectangle  2.500000  0.027778

Subtract a list and Series by axis with operator version.

>>> (df - [1, 2]).execute()
           angles  degrees
circle         -1      358
triangle        2      178
rectangle       3      358

>>> df.sub([1, 2], axis='columns').execute()
           angles  degrees
circle         -1      358
triangle        2      178
rectangle       3      358

>>> df.sub(md.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
...        axis='index').execute()
           angles  degrees
circle         -1      359
triangle        2      179
rectangle       3      359

Multiply a DataFrame of different shape with operator version.

>>> other = md.DataFrame({{'angles': [0, 3, 4]}},
...                      index=['circle', 'triangle', 'rectangle'])
>>> other.execute()
           angles
circle          0
triangle        3
rectangle       4

>>> (df * other).execute()
           angles  degrees
circle          0      NaN
triangle        9      NaN
rectangle      16      NaN

>>> df.mul(other, fill_value=0).execute()
           angles  degrees
circle          0      0.0
triangle        9      0.0
rectangle      16      0.0

Divide by a MultiIndex by level.

>>> df_multindex = md.DataFrame({{'angles': [0, 3, 4, 4, 5, 6],
...                              'degrees': [360, 180, 360, 360, 540, 720]}},
...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
...                                    ['circle', 'triangle', 'rectangle',
...                                     'square', 'pentagon', 'hexagon']])
>>> df_multindex.execute()
             angles  degrees
A circle          0      360
  triangle        3      180
  rectangle       4      360
B square          4      360
  pentagon        5      540
  hexagon         6      720

>>> df.div(df_multindex, level=1, fill_value=0).execute()
             angles  degrees
A circle        NaN      1.0
  triangle      1.0      1.0
  rectangle     1.0      1.0
B square        0.0      0.0
  pentagon      0.0      0.0
  hexagon       0.0      0.0
"""

_flex_doc_SERIES = """
Return {desc} of series and other, element-wise (binary operator `{op_name}`).

Equivalent to ``series {equiv} other``, but with support to substitute a fill_value for
missing data in one of the inputs.

Parameters
----------
other : Series or scalar value
fill_value : None or float value, default None (NaN)
    Fill existing missing (NaN) values, and any new element needed for
    successful Series alignment, with this value before computation.
    If data in both corresponding Series locations is missing
    the result will be missing.
level : int or name
    Broadcast across a level, matching Index values on the
    passed MultiIndex level.

Returns
-------
Series
    The result of the operation.

See Also
--------
Series.{reverse}

Examples
--------
>>> import numpy as np
>>> import mars.dataframe as md
>>> a = md.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
>>> a.execute()
a    1.0
b    1.0
c    1.0
d    NaN
dtype: float64

>>> b = md.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
>>> b.execute()
a    1.0
b    NaN
d    1.0
e    NaN
dtype: float64
"""

_flex_comp_doc_FRAME = """
Get {desc} of dataframe and other, element-wise (binary operator `{op_name}`).
Among flexible wrappers (`eq`, `ne`, `le`, `lt`, `ge`, `gt`) to comparison
operators.

Equivalent to `dataframe {equiv} other` with support to choose axis (rows or columns)
and level for comparison.

Parameters
----------
other : scalar, sequence, Series, or DataFrame
    Any single or multiple element data structure, or list-like object.
axis : {{0 or 'index', 1 or 'columns'}}, default 'columns'
    Whether to compare by the index (0 or 'index') or columns
    (1 or 'columns').
level : int or label
    Broadcast across a level, matching Index values on the passed
    MultiIndex level.

Returns
-------
DataFrame of bool
    Result of the comparison.

See Also
--------
DataFrame.eq : Compare DataFrames for equality elementwise.
DataFrame.ne : Compare DataFrames for inequality elementwise.
DataFrame.le : Compare DataFrames for less than inequality
    or equality elementwise.
DataFrame.lt : Compare DataFrames for strictly less than
    inequality elementwise.
DataFrame.ge : Compare DataFrames for greater than inequality
    or equality elementwise.
DataFrame.gt : Compare DataFrames for strictly greater than
    inequality elementwise.

Notes
-----
Mismatched indices will be unioned together.
`NaN` values are considered different (i.e. `NaN` != `NaN`).

Examples
--------
>>> df = pd.DataFrame({{'cost': [250, 150, 100],
...                    'revenue': [100, 250, 300]}},
...                   index=['A', 'B', 'C'])
>>> df.execute()
   cost  revenue
A   250      100
B   150      250
C   100      300

Comparison with a scalar, using either the operator or method:

>>> (df == 100).execute()
    cost  revenue
A  False     True
B  False    False
C   True    False

>>> df.eq(100).execute()
    cost  revenue
A  False     True
B  False    False
C   True    False

When `other` is a :class:`Series`, the columns of a DataFrame are aligned
with the index of `other` and broadcast:

>>> (df != pd.Series([100, 250], index=["cost", "revenue"])).execute()
    cost  revenue
A   True     True
B   True    False
C  False     True

Use the method to control the broadcast axis:

>>> df.ne(pd.Series([100, 300], index=["A", "D"]), axis='index').execute()
   cost  revenue
A  True    False
B  True     True
C  True     True
D  True     True

When comparing to an arbitrary sequence, the number of columns must
match the number elements in `other`:

>>> (df == [250, 100]).execute()
    cost  revenue
A   True     True
B  False    False
C  False    False

Use the method to control the axis:

>>> df.eq([250, 250, 100], axis='index').execute()
    cost  revenue
A   True    False
B  False     True
C   True    False

Compare to a DataFrame of different shape.

>>> other = pd.DataFrame({{'revenue': [300, 250, 100, 150]}},
...                      index=['A', 'B', 'C', 'D'])
>>> other.execute()
   revenue
A      300
B      250
C      100
D      150

>>> df.gt(other).execute()
    cost  revenue
A  False    False
B  False    False
C  False     True
D  False    False

Compare to a MultiIndex by level.

>>> df_multindex = pd.DataFrame({{'cost': [250, 150, 100, 150, 300, 220],
...                              'revenue': [100, 250, 300, 200, 175, 225]}},
...                             index=[['Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2'],
...                                    ['A', 'B', 'C', 'A', 'B', 'C']])
>>> df_multindex.execute()
      cost  revenue
Q1 A   250      100
   B   150      250
   C   100      300
Q2 A   150      200
   B   300      175
   C   220      225

>>> df.le(df_multindex, level=1).execute()
       cost  revenue
Q1 A   True     True
   B   True     True
   C   True     True
Q2 A  False     True
   B   True    False
   C   True    False
"""


_flex_comp_doc_SERIES = """
Return {desc} of series and other, element-wise (binary operator `{op_name}`).

Equivalent to ``series {equiv} other``, but with support to substitute a fill_value for
missing data in one of the inputs.

Parameters
----------
other : Series or scalar value
fill_value : None or float value, default None (NaN)
    Fill existing missing (NaN) values, and any new element needed for
    successful Series alignment, with this value before computation.
    If data in both corresponding Series locations is missing
    the result will be missing.
level : int or name
    Broadcast across a level, matching Index values on the
    passed MultiIndex level.

Returns
-------
Series
    The result of the operation.

Examples
--------
>>> import numpy as np
>>> import mars.dataframe as md
>>> a = md.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
>>> a.execute()
a    1.0
b    1.0
c    1.0
d    NaN
dtype: float64

>>> b = md.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
>>> b.execute()
a    1.0
b    NaN
d    1.0
e    NaN
dtype: float64
"""


def bin_arithmetic_doc(desc, op_name=None, equiv=None, reverse=None, series_example=None):
    def wrapper(fun):
        nonlocal op_name, reverse
        op_name = op_name or fun.__name__
        if reverse is None:
            reverse = op_name[1:] if op_name.startswith('r') else 'r' + op_name
        fun.__frame_doc__ = _flex_doc_FRAME.format(desc=desc, op_name=op_name, equiv=equiv,
                                                   reverse=reverse)
        fun.__series_doc__ = _flex_doc_SERIES.format(desc=desc, op_name=op_name, equiv=equiv,
                                                     reverse=reverse)
        if series_example is not None:  # pragma: no branch
            fun.__series_doc__ += '\n' + series_example.strip()
        return fun

    return wrapper


def bin_compare_doc(desc, op_name=None, equiv=None, series_example=None):
    def wrapper(fun):
        nonlocal op_name
        op_name = op_name or fun.__name__
        fun.__frame_doc__ = _flex_comp_doc_FRAME.format(desc=desc, op_name=op_name, equiv=equiv)
        fun.__series_doc__ = _flex_comp_doc_SERIES.format(desc=desc, op_name=op_name, equiv=equiv)
        if series_example is not None:  # pragma: no branch
            fun.__series_doc__ += '\n' + series_example.strip()
        return fun

    return wrapper
