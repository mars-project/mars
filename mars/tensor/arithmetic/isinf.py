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
from ..utils import inject_dtype
from .core import TensorUnaryOp
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='unary')
class TensorIsInf(TensorUnaryOp):
    _op_type_ = OperandDef.ISINF
    _func_name = 'isinf'


@inject_dtype(np.bool_)
def isinf(x, out=None, where=None, **kwargs):
    """
    Test element-wise for positive or negative infinity.

    Returns a boolean array of the same shape as `x`, True where ``x ==
    +/-inf``, otherwise False.

    Parameters
    ----------
    x : array_like
        Input values
    out : Tensor, None, or tuple of Tensor and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated tensor is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.
    **kwargs

    Returns
    -------
    y : bool (scalar) or boolean Tensor
        For scalar input, the result is a new boolean with value True if
        the input is positive or negative infinity; otherwise the value is
        False.

        For tensor input, the result is a boolean tensor with the same shape
        as the input and the values are True where the corresponding
        element of the input is positive or negative infinity; elsewhere
        the values are False.  If a second argument was supplied the result
        is stored there.  If the type of that array is a numeric type the
        result is represented as zeros and ones, if the type is boolean
        then as False and True, respectively.  The return value `y` is then
        a reference to that tensor.

    See Also
    --------
    isneginf, isposinf, isnan, isfinite

    Notes
    -----
    Mars uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754).

    Errors result if the second argument is supplied when the first
    argument is a scalar, or if the first and second arguments have
    different shapes.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.isinf(mt.inf).execute()
    True
    >>> mt.isinf(mt.nan).execute()
    False
    >>> mt.isinf(mt.NINF).execute()
    True
    >>> mt.isinf([mt.inf, -mt.inf, 1.0, mt.nan]).execute()
    array([ True,  True, False, False])

    >>> x = mt.array([-mt.inf, 0., mt.inf])
    >>> y = mt.array([2, 2, 2])
    >>> mt.isinf(x, y).execute()
    array([1, 0, 1])
    >>> y.execute()
    array([1, 0, 1])
    """
    op = TensorIsInf(**kwargs)
    return op(x, out=out, where=where)
