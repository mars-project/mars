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

from ... import opcodes as OperandDef
from ..datasource import tensor as astensor
from .core import TensorReduction, TensorReductionMixin


class TensorNanMin(TensorReduction, TensorReductionMixin):
    _op_type_ = OperandDef.NANMIN
    _func_name = 'nanmin'

    def __init__(self, axis=None, keepdims=None, combine_size=None, stage=None, **kw):
        stage = self._rewrite_stage(stage)
        super().__init__(_axis=axis, _keepdims=keepdims,
                         _combine_size=combine_size, stage=stage, **kw)


def nanmin(a, axis=None, out=None, keepdims=None, combine_size=None):
    """
    Return minimum of a tensor or minimum along an axis, ignoring any NaNs.
    When all-NaN slices are encountered a ``RuntimeWarning`` is raised and
    Nan is returned for that slice.

    Parameters
    ----------
    a : array_like
        Tensor containing numbers whose minimum is desired. If `a` is not an
        tensor, a conversion is attempted.
    axis : int, optional
        Axis along which the minimum is computed. The default is to compute
        the minimum of the flattened tensor.
    out : Tensor, optional
        Alternate output tensor in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.  See
        `doc.ufuncs` for details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

        If the value is anything but the default, then
        `keepdims` will be passed through to the `min` method
        of sub-classes of `Tensor`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.
    combine_size: int, optional
        The number of chunks to combine.

    Returns
    -------
    nanmin : Tensor
        An tensor with the same shape as `a`, with the specified axis
        removed.  If `a` is a 0-d tensor, or if axis is None, a tensor
        scalar is returned.  The same dtype as `a` is returned.

    See Also
    --------
    nanmax :
        The maximum value of an array along a given axis, ignoring any NaNs.
    amin :
        The minimum value of an array along a given axis, propagating any NaNs.
    fmin :
        Element-wise minimum of two arrays, ignoring any NaNs.
    minimum :
        Element-wise minimum of two arrays, propagating any NaNs.
    isnan :
        Shows which elements are Not a Number (NaN).
    isfinite:
        Shows which elements are neither NaN nor infinity.

    amax, fmax, maximum

    Notes
    -----
    Mars uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Positive infinity is treated as a very large number and negative
    infinity is treated as a very small (i.e. negative) number.

    If the input has a integer type the function is equivalent to mt.min.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.array([[1, 2], [3, mt.nan]])
    >>> mt.nanmin(a).execute()
    1.0
    >>> mt.nanmin(a, axis=0).execute()
    array([ 1.,  2.])
    >>> mt.nanmin(a, axis=1).execute()
    array([ 1.,  3.])

    When positive infinity and negative infinity are present:

    >>> mt.nanmin([1, 2, mt.nan, mt.inf]).execute()
    1.0
    >>> mt.nanmin([1, 2, mt.nan, mt.NINF]).execute()
    -inf

    """
    a = astensor(a)
    op = TensorNanMin(axis=axis, dtype=a.dtype, keepdims=keepdims, combine_size=combine_size)
    return op(a, out=out)
