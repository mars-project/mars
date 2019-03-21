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

from .... import operands
from ..utils import infer_dtype
from .core import TensorBinOp, TensorConstant
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='always_false')
class TensorFloorDiv(operands.FloorDiv, TensorBinOp):
    @classmethod
    def constant_cls(cls):
        return TensorFDivConstant


@arithmetic_operand(sparse_mode='binary_or_const')
class TensorFDivConstant(operands.FDivConstant, TensorConstant):
    @classmethod
    def _is_sparse(cls, x1, x2):
        if hasattr(x1, 'issparse') and x1.issparse():
            if x2 != 0:
                return True
            else:
                raise ZeroDivisionError('float division by zero')
        return False


@infer_dtype(np.floor_divide)
def floordiv(x1, x2, out=None, where=None, **kwargs):
    """
    Return the largest integer smaller or equal to the division of the inputs.
    It is equivalent to the Python ``//`` operator and pairs with the
    Python ``%`` (`remainder`), function so that ``b = a % b + b * (a // b)``
    up to roundoff.

    Parameters
    ----------
    x1 : array_like
        Numerator.
    x2 : array_like
        Denominator.
    out : Tensor, None, or tuple of Tensor and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.
    **kwargs

    Returns
    -------
    y : Tensor
        y = floor(`x1`/`x2`)


    See Also
    --------
    remainder : Remainder complementary to floor_divide.
    divmod : Simultaneous floor division and remainder.
    divide : Standard division.
    floor : Round a number to the nearest integer toward minus infinity.
    ceil : Round a number to the nearest integer toward infinity.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.floor_divide(7,3).execute()
    2
    >>> mt.floor_divide([1., 2., 3., 4.], 2.5).execute()
    array([ 0.,  0.,  1.,  1.])
    """
    op = TensorFloorDiv(**kwargs)
    return op(x1, x2, out=out, where=where)


@infer_dtype(np.floor_divide, reverse=True)
def rfloordiv(x1, x2, **kwargs):
    op = TensorFloorDiv(**kwargs)
    return op.rcall(x1, x2)
