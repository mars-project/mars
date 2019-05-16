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

import numpy as np

from .... import opcodes as OperandDef
from ..utils import infer_dtype
from .core import TensorBinOp, TensorConstant
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='binary_or')
class TensorFMod(TensorBinOp):
    _op_type_ = OperandDef.FMOD

    @classmethod
    def constant_cls(cls):
        return TensorFModConstant


@arithmetic_operand(sparse_mode='binary_or_const')
class TensorFModConstant(TensorConstant):
    _op_type_ = OperandDef.FMOD_CONSTANT


@infer_dtype(np.fmod)
def fmod(x1, x2, out=None, where=None, **kwargs):
    """
    Return the element-wise remainder of division.

    This is the NumPy implementation of the C library function fmod, the
    remainder has the same sign as the dividend `x1`. It is equivalent to
    the Matlab(TM) ``rem`` function and should not be confused with the
    Python modulus operator ``x1 % x2``.

    Parameters
    ----------
    x1 : array_like
      Dividend.
    x2 : array_like
      Divisor.
    out : Tensor, None, or tuple of Tensor and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated tensor is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : Tensor_like
      The remainder of the division of `x1` by `x2`.

    See Also
    --------
    remainder : Equivalent to the Python ``%`` operator.
    divide

    Notes
    -----
    The result of the modulo operation for negative dividend and divisors
    is bound by conventions. For `fmod`, the sign of result is the sign of
    the dividend, while for `remainder` the sign of the result is the sign
    of the divisor. The `fmod` function is equivalent to the Matlab(TM)
    ``rem`` function.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.fmod([-3, -2, -1, 1, 2, 3], 2).execute()
    array([-1,  0, -1,  1,  0,  1])
    >>> mt.remainder([-3, -2, -1, 1, 2, 3], 2).execute()
    array([1, 0, 1, 1, 0, 1])

    >>> mt.fmod([5, 3], [2, 2.]).execute()
    array([ 1.,  1.])
    >>> a = mt.arange(-3, 3).reshape(3, 2)
    >>> a.execute()
    array([[-3, -2],
           [-1,  0],
           [ 1,  2]])
    >>> mt.fmod(a, [2,2]).execute()
    array([[-1,  0],
           [-1,  0],
           [ 1,  0]])
    """
    op = TensorFMod(**kwargs)
    return op(x1, x2, out=out, where=where)
