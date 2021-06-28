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
from ...core import ENTITY_TYPE, recursive_tile
from ...serialization.serializables import AnyField, Int32Field, BoolField, StringField
from ...tensor.utils import filter_inputs
from ..core import DATAFRAME_TYPE, SERIES_TYPE
from ..operands import DataFrameOperandMixin, DataFrameOperand
from ..utils import build_df, build_series, validate_axis


class DataFrameWhere(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = opcodes.WHERE

    _input = AnyField('input')
    _cond = AnyField('cond')
    _other = AnyField('other')
    _axis = Int32Field('axis')
    _level = AnyField('level')
    _errors = StringField('errors')
    _try_cast = BoolField('try_cast')
    _replace_true = BoolField('replace_true')

    def __init__(self, input=None, cond=None, other=None,  # pylint: disable=redefined-builtin
                 axis=None, level=None, errors=None, try_cast=None, replace_true=None, **kw):
        super().__init__(_input=input, _cond=cond, _other=other, _axis=axis, _level=level,
                         _errors=errors, _try_cast=try_cast, _replace_true=replace_true, **kw)

    @property
    def input(self):
        return self._input

    @property
    def cond(self):
        return self._cond

    @property
    def other(self):
        return self._other

    @property
    def axis(self):
        return self._axis

    @property
    def level(self):
        return self._level

    @property
    def errors(self):
        return self._errors

    @property
    def try_cast(self):
        return self._try_cast

    @property
    def replace_true(self):
        return self._replace_true

    def __call__(self, df_or_series):
        def _check_input_index(obj, axis=None):
            axis = axis if axis is not None else self.axis
            if isinstance(obj, DATAFRAME_TYPE) \
                    and (
                        df_or_series.columns_value.key != obj.columns_value.key
                        or df_or_series.index_value.key != obj.index_value.key
                    ):
                raise NotImplementedError('Aligning different indices not supported')
            elif isinstance(obj, SERIES_TYPE) \
                    and df_or_series.axes[axis].index_value.key != obj.index_value.key:
                raise NotImplementedError('Aligning different indices not supported')

        _check_input_index(self.cond, axis=0)
        _check_input_index(self.other)

        if isinstance(df_or_series, DATAFRAME_TYPE):
            mock_obj = build_df(df_or_series)
        else:
            mock_obj = build_series(df_or_series)

        if isinstance(self.other, (pd.DataFrame, DATAFRAME_TYPE)):
            mock_other = build_df(self.other)
        elif isinstance(self.other, (pd.Series, SERIES_TYPE)):
            mock_other = build_series(self.other)
        else:
            mock_other = self.other

        result_df = mock_obj.where(np.zeros(mock_obj.shape).astype(bool), other=mock_other,
                                   axis=self.axis, level=self.level, errors=self.errors,
                                   try_cast=self.try_cast)

        inputs = filter_inputs([df_or_series, self.cond, self.other])
        if isinstance(df_or_series, DATAFRAME_TYPE):
            return self.new_dataframe(inputs, shape=df_or_series.shape,
                                      dtypes=result_df.dtypes, index_value=df_or_series.index_value,
                                      columns_value=df_or_series.columns_value)
        else:
            return self.new_series(inputs, shape=df_or_series.shape, name=df_or_series.name,
                                   dtype=result_df.dtype, index_value=df_or_series.index_value)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs)
        self._input = next(inputs_iter)
        if isinstance(self._cond, ENTITY_TYPE):
            self._cond = next(inputs_iter)
        if isinstance(self._other, ENTITY_TYPE):
            self._other = next(inputs_iter)

    @classmethod
    def tile(cls, op: "DataFrameWhere"):
        def rechunk_input(inp, axis=None):
            axis = axis if axis is not None else op.axis
            if isinstance(inp, DATAFRAME_TYPE):
                inp = yield from recursive_tile(
                    inp.rechunk(op.input.nsplits))
            elif isinstance(inp, SERIES_TYPE):
                inp = yield from recursive_tile(
                    inp.rechunk({0: op.input.nsplits[axis]}))
            return inp

        def get_tiled_chunk(obj, index, axis=None):
            if isinstance(obj, DATAFRAME_TYPE):
                return obj.cix[index[0], index[1]]
            elif isinstance(obj, SERIES_TYPE):
                axis = axis if axis is not None else op.axis
                return obj.cix[index[axis], ]
            else:
                return obj

        # TODO support axis alignment for three objects
        cond = yield from rechunk_input(op.cond, axis=0)
        other = yield from rechunk_input(op.other)

        chunks = []
        for c in op.input.chunks:
            cond_chunk = get_tiled_chunk(cond, c.index, axis=0)
            other_chunk = get_tiled_chunk(other, c.index)

            new_op = op.copy().reset_key()
            new_op._cond = cond_chunk
            new_op._other = other_chunk

            inputs = filter_inputs([c, cond_chunk, other_chunk])
            chunks.append(new_op.new_chunk(inputs, **c.params))

        new_op = op.copy().reset_key()
        return new_op.new_tileables(op.inputs, chunks=chunks, nsplits=op.input.nsplits,
                                    **op.input.params)

    @classmethod
    def execute(cls, ctx, op: "DataFrameWhere"):
        out_obj = op.outputs[0]

        input_data = ctx[op.input.key]
        cond = op.cond
        if isinstance(cond, ENTITY_TYPE):
            cond = ctx[cond.key]

        other = op.other
        if isinstance(other, ENTITY_TYPE):
            other = ctx[other.key]

        if op.replace_true:
            ctx[out_obj.key] = input_data.mask(cond, other, axis=op.axis, level=op.level,
                                               errors=op.errors, try_cast=op.try_cast)
        else:
            ctx[out_obj.key] = input_data.where(cond, other, axis=op.axis, level=op.level,
                                                errors=op.errors, try_cast=op.try_cast)


_doc_template = """
Replace values where the condition is {replace_true}.

Parameters
----------
cond : bool Series/DataFrame, array-like, or callable
    Where `cond` is False, keep the original value. Where
    True, replace with corresponding value from `other`.
    If `cond` is callable, it is computed on the Series/DataFrame and
    should return boolean Series/DataFrame or array. The callable must
    not change input Series/DataFrame (though pandas doesn't check it).
other : scalar, Series/DataFrame, or callable
    Entries where `cond` is True are replaced with
    corresponding value from `other`.
    If other is callable, it is computed on the Series/DataFrame and
    should return scalar or Series/DataFrame. The callable must not
    change input Series/DataFrame (though pandas doesn't check it).
inplace : bool, default False
    Whether to perform the operation in place on the data.
axis : int, default None
    Alignment axis if needed.
level : int, default None
    Alignment level if needed.
errors : str, {{'raise', 'ignore'}}, default 'raise'
    Note that currently this parameter won't affect
    the results and will always coerce to a suitable dtype.

    - 'raise' : allow exceptions to be raised.
    - 'ignore' : suppress exceptions. On error return original object.

try_cast : bool, default False
    Try to cast the result back to the input type (if possible).

Returns
-------
Same type as caller

See Also
--------
:func:`DataFrame.{opposite}` : Return an object of same shape as
    self.

Notes
-----
The mask method is an application of the if-then idiom. For each
element in the calling DataFrame, if ``cond`` is ``False`` the
element is used; otherwise the corresponding element from the DataFrame
``other`` is used.

The signature for :func:`DataFrame.where` differs from
:func:`numpy.where`. Roughly ``df1.where(m, df2)`` is equivalent to
``np.where(m, df1, df2)``.

For further details and examples see the ``mask`` documentation in
:ref:`indexing <indexing.where_mask>`.

Examples
--------
>>> import mars.tensor as mt
>>> import mars.dataframe as md
>>> s = md.Series(range(5))
>>> s.where(s > 0).execute()
0    NaN
1    1.0
2    2.0
3    3.0
4    4.0
dtype: float64

>>> s.mask(s > 0).execute()
0    0.0
1    NaN
2    NaN
3    NaN
4    NaN
dtype: float64

>>> s.where(s > 1, 10).execute()
0    10
1    10
2    2
3    3
4    4
dtype: int64

>>> df = md.DataFrame(mt.arange(10).reshape(-1, 2), columns=['A', 'B'])
>>> df.execute()
   A  B
0  0  1
1  2  3
2  4  5
3  6  7
4  8  9
>>> m = df % 3 == 0
>>> df.where(m, -df).execute()
   A  B
0  0 -1
1 -2  3
2 -4 -5
3  6 -7
4 -8  9
>>> df.where(m, -df) == mt.where(m, df, -df).execute()
      A     B
0  True  True
1  True  True
2  True  True
3  True  True
4  True  True
>>> df.where(m, -df) == df.mask(~m, -df).execute()
      A     B
0  True  True
1  True  True
2  True  True
3  True  True
4  True  True
"""


def _where(df_or_series, cond, other=np.nan, inplace=False, axis=None, level=None, errors='raise',
           try_cast=False, replace_true=False):
    if df_or_series.ndim == 2 and getattr(other, 'ndim', 2) == 1 and axis is None:
        raise ValueError('Must specify axis=0 or 1')

    axis = validate_axis(axis or 0, df_or_series)
    op = DataFrameWhere(cond=cond, other=other, axis=axis, level=level, errors=errors,
                        try_cast=try_cast, replace_true=replace_true)
    result = op(df_or_series)
    if inplace:
        df_or_series.data = result.data
    else:
        return result


def where(df_or_series, cond, other=np.nan, inplace=False, axis=None, level=None, errors='raise',
          try_cast=False):
    return _where(df_or_series, cond, other=other, inplace=inplace, axis=axis, level=level,
                  errors=errors, try_cast=try_cast, replace_true=False)


def mask(df_or_series, cond, other=np.nan, inplace=False, axis=None, level=None, errors='raise',
         try_cast=False):
    return _where(df_or_series, cond, other=other, inplace=inplace, axis=axis, level=level,
                  errors=errors, try_cast=try_cast, replace_true=True)


mask.__doc__ = _doc_template.format(replace_true=True, opposite='where')
where.__doc__ = _doc_template.format(replace_true=False, opposite='mask')
