#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from ..datasource import tensor as astensor
from ..arithmetic.multiply import TensorTreeMultiply
from .core import TensorCumReduction, TensorCumReductionMixin


class TensorNanCumprod(TensorCumReduction, TensorCumReductionMixin):
    _op_type_ = OperandDef.NANCUMPROD
    _func_name = 'nancumprod'

    def __init__(self, axis=None, **kw):
        super().__init__(_axis=axis, **kw)

    @staticmethod
    def _get_op_types():
        return TensorNanCumprod, TensorTreeMultiply


def nancumprod(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative product of tensor elements over a given axis treating Not a
    Numbers (NaNs) as one.  The cumulative product does not change when NaNs are
    encountered and leading NaNs are replaced by ones.

    Ones are returned for slices that are all-NaN or empty.

    Parameters
    ----------
    a : array_like
        Input tensor.
    axis : int, optional
        Axis along which the cumulative product is computed.  By default
        the input is flattened.
    dtype : dtype, optional
        Type of the returned tensor, as well as of the accumulator in which
        the elements are multiplied.  If *dtype* is not specified, it
        defaults to the dtype of `a`, unless `a` has an integer dtype with
        a precision less than that of the default platform integer.  In
        that case, the default platform integer is used instead.
    out : Tensor, optional
        Alternative output tensor in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type of the resulting values will be cast if necessary.

    Returns
    -------
    nancumprod : Tensor
        A new array holding the result is returned unless `out` is
        specified, in which case it is returned.

    See Also
    --------
    mt.cumprod : Cumulative product across array propagating NaNs.
    isnan : Show which elements are NaN.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.nancumprod(1).execute()
    array([1])
    >>> mt.nancumprod([1]).execute()
    array([1])
    >>> mt.nancumprod([1, mt.nan]).execute()
    array([ 1.,  1.])
    >>> a = mt.array([[1, 2], [3, mt.nan]])
    >>> mt.nancumprod(a).execute()
    array([ 1.,  2.,  6.,  6.])
    >>> mt.nancumprod(a, axis=0).execute()
    array([[ 1.,  2.],
           [ 3.,  2.]])
    >>> mt.nancumprod(a, axis=1).execute()
    array([[ 1.,  2.],
           [ 3.,  3.]])

    """
    a = astensor(a)
    if dtype is None:
        dtype = np.nancumprod(np.empty((1,), dtype=a.dtype)).dtype
    op = TensorNanCumprod(axis=axis, dtype=dtype)
    return op(a, out=out)
