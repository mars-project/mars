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

from ... import opcodes as OperandDef
from ...core import recursive_tile
from ...serialization.serializables import KeyField, AnyField
from ...tensor import tensor as astensor
from ...tensor.core import TENSOR_TYPE
from ...tensor.utils import decide_unify_split, validate_axis
from ..core import DATAFRAME_TYPE, SERIES_TYPE, IndexValue
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import parse_index


class DataFrameDot(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DOT

    _lhs = KeyField('lhs')
    _rhs = AnyField('rhs')

    def __init__(self, output_types=None, lhs=None, rhs=None, **kw):
        super().__init__(_output_types=output_types, _lhs=lhs, _rhs=rhs, **kw)

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._lhs = self._inputs[0]
        self._rhs = self._inputs[1]

    def __call__(self, lhs, rhs):
        if not isinstance(rhs, (DATAFRAME_TYPE, SERIES_TYPE)):
            rhs = astensor(rhs)
            test_rhs = rhs
        else:
            test_rhs = rhs.to_tensor()

        test_ret = lhs.to_tensor().dot(test_rhs)
        if test_ret.ndim == 0:
            if isinstance(lhs, SERIES_TYPE) and isinstance(rhs, TENSOR_TYPE):
                # return tensor
                return test_ret
            return self.new_scalar([lhs, rhs], dtype=test_ret.dtype)
        elif test_ret.ndim == 1:
            if lhs.ndim == 1:
                if hasattr(rhs, 'columns_value'):
                    index_value = rhs.columns_value
                else:
                    # tensor
                    length = -1 if np.isnan(rhs.shape[1]) else rhs.shape[1]
                    pd_index = pd.RangeIndex(length)
                    index_value = parse_index(pd_index, store_data=True)
            else:
                assert rhs.ndim == 1
                index_value = lhs.index_value
            return self.new_series([lhs, rhs], shape=test_ret.shape,
                                   dtype=test_ret.dtype, index_value=index_value)
        else:
            if isinstance(rhs, TENSOR_TYPE):
                dtypes = pd.Series(np.repeat(test_ret.dtype, test_ret.shape[1]),
                                   index=pd.RangeIndex(test_ret.shape[1]))
                columns_value = parse_index(dtypes.index, store_data=True)
            else:
                dtypes = pd.Series(np.repeat(test_ret.dtype, test_ret.shape[1]),
                                   index=rhs.columns_value.to_pandas())
                columns_value = rhs.columns_value
            return self.new_dataframe([lhs, rhs], shape=test_ret.shape,
                                      index_value=lhs.index_value,
                                      columns_value=columns_value,
                                      dtypes=dtypes)

    @classmethod
    def _align(cls, lhs, rhs):
        if isinstance(rhs, TENSOR_TYPE):
            # no need to align when rhs is a tensor
            return lhs, rhs

        is_lhs_range_index = False
        if isinstance(lhs, DATAFRAME_TYPE) and \
                isinstance(lhs.columns_value.value, IndexValue.RangeIndex):
            is_lhs_range_index = True
        if isinstance(lhs, SERIES_TYPE) and \
                isinstance(lhs.index_value.value, IndexValue.RangeIndex):
            is_lhs_range_index = True

        is_rhs_range_index = False
        if isinstance(rhs.index_value.value, IndexValue.RangeIndex):
            is_rhs_range_index = True

        if not is_lhs_range_index or not is_rhs_range_index:
            # TODO: e.g. use rhs.loc[lhs.columns_value.to_pandas()]
            # when lhs is a DataFrame and lhs.columns is not a RangeIndex,
            # so does Series
            raise NotImplementedError

        return lhs, rhs

    @classmethod
    def tile(cls, op):
        from ..datasource.from_tensor import dataframe_from_tensor, series_from_tensor

        lhs, rhs = op.lhs, op.rhs
        lhs, rhs = cls._align(lhs, rhs)
        out = op.outputs[0]

        if any(np.isnan(ns) for ns in lhs.nsplits[-1]):
            yield
        if any(np.isnan(ns) for ns in rhs.nsplits[0]):
            yield

        nsplit = decide_unify_split(lhs.nsplits[-1], rhs.nsplits[0])
        lhs_axis = validate_axis(lhs.ndim, -1)
        lhs = yield from recursive_tile(lhs.rechunk({lhs_axis: nsplit}))
        rhs = yield from recursive_tile(rhs.rechunk({0: nsplit}))

        # delegate computation to tensor
        lhs_tensor = lhs if isinstance(lhs, TENSOR_TYPE) else lhs.to_tensor()
        rhs_tensor = rhs if isinstance(rhs, TENSOR_TYPE) else rhs.to_tensor()
        ret = lhs_tensor.dot(rhs_tensor)

        if isinstance(out, TENSOR_TYPE):
            pass
        elif ret.ndim == 1:
            ret = series_from_tensor(ret)
            if isinstance(lhs, DATAFRAME_TYPE):
                # lhs DataFrame
                ret._index_value = lhs.index_value
            elif isinstance(rhs, DATAFRAME_TYPE):
                # lhs Series, rhs DataFrame
                ret._index_value = rhs.columns_value
        elif ret.ndim == 2:
            ret = dataframe_from_tensor(ret)
            ret._index_value = lhs.index_value
            if isinstance(rhs, DATAFRAME_TYPE):
                ret._columns_value = rhs.columns_value
                ret._dtypes = rhs.dtypes

        tiled = yield from recursive_tile(ret)
        return [tiled]


def dot(df_or_seris, other):
    op = DataFrameDot(lhs=df_or_seris, rhs=other)
    return op(df_or_seris, other)


dot.__frame_doc__ = """
Compute the matrix multiplication between the DataFrame and other.

This method computes the matrix product between the DataFrame and the
values of an other Series, DataFrame or a numpy array.

It can also be called using ``self @ other`` in Python >= 3.5.

Parameters
----------
other : Series, DataFrame or array-like
    The other object to compute the matrix product with.

Returns
-------
Series or DataFrame
    If other is a Series, return the matrix product between self and
    other as a Series. If other is a DataFrame or a numpy.array, return
    the matrix product of self and other in a DataFrame of a np.array.

See Also
--------
Series.dot: Similar method for Series.

Notes
-----
The dimensions of DataFrame and other must be compatible in order to
compute the matrix multiplication. In addition, the column names of
DataFrame and the index of other must contain the same values, as they
will be aligned prior to the multiplication.

The dot method for Series computes the inner product, instead of the
matrix product here.

Examples
--------
Here we multiply a DataFrame with a Series.

>>> import mars.tensor as mt
>>> import mars.dataframe as md
>>> df = md.DataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])
>>> s = md.Series([1, 1, 2, 1])
>>> df.dot(s).execute()
0    -4
1     5
dtype: int64

Here we multiply a DataFrame with another DataFrame.

>>> other = md.DataFrame([[0, 1], [1, 2], [-1, -1], [2, 0]])
>>> df.dot(other).execute()
    0   1
0   1   4
1   2   2

Note that the dot method give the same result as @

>>> (df @ other).execute()
    0   1
0   1   4
1   2   2

The dot method works also if other is an np.array.

>>> arr = mt.array([[0, 1], [1, 2], [-1, -1], [2, 0]])
>>> df.dot(arr).execute()
    0   1
0   1   4
1   2   2

Note how shuffling of the objects does not change the result.

>>> s2 = s.reindex([1, 0, 2, 3])
>>> df.dot(s2).execute()
0    -4
1     5
dtype: int64
"""
dot.__series_doc__ = """
Compute the dot product between the Series and the columns of other.

This method computes the dot product between the Series and another
one, or the Series and each columns of a DataFrame, or the Series and
each columns of an array.

It can also be called using `self @ other` in Python >= 3.5.

Parameters
----------
other : Series, DataFrame or array-like
    The other object to compute the dot product with its columns.

Returns
-------
scalar, Series or numpy.ndarray
    Return the dot product of the Series and other if other is a
    Series, the Series of the dot product of Series and each rows of
    other if other is a DataFrame or a numpy.ndarray between the Series
    and each columns of the numpy array.

See Also
--------
DataFrame.dot: Compute the matrix product with the DataFrame.
Series.mul: Multiplication of series and other, element-wise.

Notes
-----
The Series and other has to share the same index if other is a Series
or a DataFrame.

Examples
--------
>>> import mars.tensor as mt
>>> import mars.dataframe as md
>>> s = md.Series([0, 1, 2, 3])
>>> other = md.Series([-1, 2, -3, 4])
>>> s.dot(other).execute()
8
>>> (s @ other).execute()
8
>>> df = md.DataFrame([[0, 1], [-2, 3], [4, -5], [6, 7]])
>>> s.dot(df).execute()
0    24
1    14
dtype: int64
>>> arr = mt.array([[0, 1], [-2, 3], [4, -5], [6, 7]])
>>> s.dot(arr).execute()
array([24, 14])
"""
