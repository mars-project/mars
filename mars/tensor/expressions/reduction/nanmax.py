#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2018 Alibaba Group Holding Ltd.
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

from .... import opcodes as OperandDef
from ..datasource import tensor as astensor
from .core import TensorReduction, TensorReductionMixin


class TensorNanMax(TensorReduction, TensorReductionMixin):
    _op_type_ = OperandDef.NANMAX

    def __init__(self, axis=None, dtype=None, keepdims=None, combine_size=None, **kw):
        super(TensorNanMax, self).__init__(_axis=axis, _dtype=dtype, _keepdims=keepdims,
                                           _combine_size=combine_size, **kw)

    @staticmethod
    def _get_op_types():
        return TensorNanMax, TensorNanMax, None


def nanmax(a, axis=None, out=None, keepdims=None, combine_size=None):
    """
    Return the maximum of an array or maximum along an axis, ignoring any
    NaNs.  When all-NaN slices are encountered a ``RuntimeWarning`` is
    raised and NaN is returned for that slice.

    Parameters
    ----------
    a : array_like
        Tensor containing numbers whose maximum is desired. If `a` is not a
        tensor, a conversion is attempted.
    axis : int, optional
        Axis along which the maximum is computed. The default is to compute
        the maximum of the flattened tensor.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.  See
        `doc.ufuncs` for details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        If the value is anything but the default, then
        `keepdims` will be passed through to the `max` method
        of sub-classes of `Tensor`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.
    combine_size: int, optional
        The number of chunks to combine.

    Returns
    -------
    nanmax : Tensor
        A tensor with the same shape as `a`, with the specified axis removed.
        If `a` is a 0-d tensor, or if axis is None, a Tensor scalar is
        returned.  The same dtype as `a` is returned.

    See Also
    --------
    nanmin :
        The minimum value of a tensor along a given axis, ignoring any NaNs.
    amax :
        The maximum value of a tensor along a given axis, propagating any NaNs.
    fmax :
        Element-wise maximum of two tensors, ignoring any NaNs.
    maximum :
        Element-wise maximum of two tensors, propagating any NaNs.
    isnan :
        Shows which elements are Not a Number (NaN).
    isfinite:
        Shows which elements are neither NaN nor infinity.

    amin, fmin, minimum

    Notes
    -----
    Mars uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Positive infinity is treated as a very large number and negative
    infinity is treated as a very small (i.e. negative) number.

    If the input has a integer type the function is equivalent to np.max.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.array([[1, 2], [3, mt.nan]])
    >>> mt.nanmax(a).execute()
    3.0
    >>> mt.nanmax(a, axis=0).execute()
    array([ 3.,  2.])
    >>> mt.nanmax(a, axis=1).execute()
    array([ 2.,  3.])

    When positive infinity and negative infinity are present:

    >>> mt.nanmax([1, 2, mt.nan, mt.NINF]).execute()
    2.0
    >>> mt.nanmax([1, 2, mt.nan, mt.inf]).execute()
    inf

    """
    a = astensor(a)
    op = TensorNanMax(axis=axis, dtype=a.dtype, keepdims=keepdims, combine_size=combine_size)
    return op(a, out=out)
