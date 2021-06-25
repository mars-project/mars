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

from ... import opcodes as OperandDef
from ...serialization.serializables import Int32Field
from ...utils import classproperty
from .core import DataFrameUnaryUfunc


class DataFrameAround(DataFrameUnaryUfunc):
    _op_type_ = OperandDef.AROUND
    _func_name = 'around'

    _decimals = Int32Field('decimals')

    def __init__(self, decimals=None, output_types=None, **kw):
        super().__init__(_decimals=decimals, output_types=output_types, **kw)

    @classproperty
    def tensor_op_type(self):
        from ...tensor.arithmetic import TensorAround
        return TensorAround

    @property
    def decimals(self):
        return self._decimals

    @classmethod
    def execute(cls, ctx, op):
        df = ctx[op.inputs[0].key]
        func_name = getattr(cls, '_func_name')
        if hasattr(df, func_name):
            ctx[op.outputs[0].key] = getattr(df, func_name)(decimals=op.decimals)
        else:
            ctx[op.outputs[0].key] = getattr(np, func_name)(df, decimals=op.decimals)


def around(df, decimals=0, *args, **kwargs):
    if len(args) > 0:
        raise TypeError('round() takes 0 positional arguments '
                        f'but {len(args)} was given')
    op = DataFrameAround(decimals=decimals, **kwargs)
    return op(df)


around.__frame_doc__ = """
Round a DataFrame to a variable number of decimal places.

Parameters
----------
decimals : int, dict, Series
    Number of decimal places to round each column to. If an int is
    given, round each column to the same number of places.
    Otherwise dict and Series round to variable numbers of places.
    Column names should be in the keys if `decimals` is a
    dict-like, or in the index if `decimals` is a Series. Any
    columns not included in `decimals` will be left as is. Elements
    of `decimals` which are not columns of the input will be
    ignored.
*args
    Additional keywords have no effect but might be accepted for
    compatibility with numpy.
**kwargs
    Additional keywords have no effect but might be accepted for
    compatibility with numpy.

Returns
-------
DataFrame
    A DataFrame with the affected columns rounded to the specified
    number of decimal places.

See Also
--------
numpy.around : Round a numpy array to the given number of decimals.
Series.round : Round a Series to the given number of decimals.

Examples
--------
>>> import mars.dataframe as md
>>> df = md.DataFrame([(.21, .32), (.01, .67), (.66, .03), (.21, .18)],
...                   columns=['dogs', 'cats'])
>>> df.execute()
    dogs  cats
0  0.21  0.32
1  0.01  0.67
2  0.66  0.03
3  0.21  0.18

By providing an integer each column is rounded to the same number
of decimal places

>>> df.round(1).execute()
    dogs  cats
0   0.2   0.3
1   0.0   0.7
2   0.7   0.0
3   0.2   0.2

With a dict, the number of places for specific columns can be
specified with the column names as key and the number of decimal
places as value

>>> df.round({'dogs': 1, 'cats': 0}).execute()
    dogs  cats
0   0.2   0.0
1   0.0   1.0
2   0.7   0.0
3   0.2   0.0

Using a Series, the number of places for specific columns can be
specified with the column names as index and the number of
decimal places as value

>>> decimals = md.Series([0, 1], index=['cats', 'dogs'])
>>> df.round(decimals).execute()
    dogs  cats
0   0.2   0.0
1   0.0   1.0
2   0.7   0.0
3   0.2   0.0
"""
around.__series_doc__ = """
Round each value in a Series to the given number of decimals.

Parameters
----------
decimals : int, default 0
    Number of decimal places to round to. If decimals is negative,
    it specifies the number of positions to the left of the decimal point.

Returns
-------
Series
    Rounded values of the Series.

See Also
--------
numpy.around : Round values of an np.array.
DataFrame.round : Round values of a DataFrame.

Examples
--------
>>> import mars.tensor as mt
>>> import mars.dataframe as md
>>> s = md.Series([0.1, 1.3, 2.7])
>>> s.round().execute()
0    0.0
1    1.0
2    3.0
dtype: float64
"""
