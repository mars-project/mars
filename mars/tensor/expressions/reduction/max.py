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


class TensorMax(TensorReduction, TensorReductionMixin):
    _op_type_ = OperandDef.MAX

    def __init__(self, axis=None, dtype=None, keepdims=None, combine_size=None, **kw):
        super(TensorMax, self).__init__(_axis=axis, _dtype=dtype, _keepdims=keepdims,
                                        _combine_size=combine_size, **kw)

    @staticmethod
    def _get_op_types():
        return TensorMax, TensorMax, None


def max(a, axis=None, out=None, keepdims=None, combine_size=None):
    """
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate.  By default, flattened input is
        used.

        If this is a tuple of ints, the maximum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : Tensor, optional
        Alternative output tensor in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See `doc.ufuncs` (Section "Output arguments") for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `amax` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-classes `sum` method does not implement `keepdims` any
        exceptions will be raised.
    combine_size: int, optional
        The number of chunks to combine.

    Returns
    -------
    amax : Tensor or scalar
        Maximum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is given, the result is a tensor of dimension
        ``a.ndim - 1``.

    See Also
    --------
    amin :
        The minimum value of a tensor along a given axis, propagating any NaNs.
    nanmax :
        The maximum value of a tensor along a given axis, ignoring any NaNs.
    maximum :
        Element-wise maximum of two tensors, propagating any NaNs.
    fmax :
        Element-wise maximum of two tensors, ignoring any NaNs.
    argmax :
        Return the indices of the maximum values.

    nanmin, minimum, fmin

    Notes
    -----
    NaN values are propagated, that is if at least one item is NaN, the
    corresponding max value will be NaN as well. To ignore NaN values
    (MATLAB behavior), please use nanmax.

    Don't use `amax` for element-wise comparison of 2 arrays; when
    ``a.shape[0]`` is 2, ``maximum(a[0], a[1])`` is faster than
    ``amax(a, axis=0)``.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.arange(4).reshape((2,2))
    >>> a.execute()
    array([[0, 1],
           [2, 3]])
    >>> mt.amax(a).execute()           # Maximum of the flattened array
    3
    >>> mt.amax(a, axis=0).execute()   # Maxima along the first axis
    array([2, 3])
    >>> mt.amax(a, axis=1).execute()   # Maxima along the second axis
    array([1, 3])

    >>> b = mt.arange(5, dtype=float)
    >>> b[2] = mt.NaN
    >>> mt.amax(b).execute()
    nan
    >>> mt.nanmax(b).execute()
    4.0

    """
    a = astensor(a)
    op = TensorMax(axis=axis, dtype=a.dtype, keepdims=keepdims, combine_size=combine_size)
    return op(a, out=out)
