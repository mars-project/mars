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


@arithmetic_operand
class TensorPower(TensorBinOp):
    _op_type_ = OperandDef.POW

    @classmethod
    def _is_sparse(cls, x1, x2):
        return x1.issparse() and not x2.issparse()

    @classmethod
    def constant_cls(cls):
        return TensorPowConstant


@arithmetic_operand
class TensorPowConstant(TensorConstant):
    _op_type_ = OperandDef.POW_CONSTANT

    @classmethod
    def _is_sparse(cls, x1, x2):
        if hasattr(x1, 'issparse') and x1.issparse():
            return True
        return False


@infer_dtype(np.power)
def power(x1, x2, out=None, where=None, **kwargs):
    r"""
    First tensor elements raised to powers from second tensor, element-wise.

    Raise each base in `x1` to the positionally-corresponding power in
    `x2`.  `x1` and `x2` must be broadcastable to the same shape. Note that an
    integer type raised to a negative integer power will raise a ValueError.

    Parameters
    ----------
    x1 : array_like
        The bases.
    x2 : array_like
        The exponents.
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
    y : Tensor
        The bases in `x1` raised to the exponents in `x2`.

    See Also
    --------
    float_power : power function that promotes integers to float

    Examples
    --------
    Cube each element in a list.

    >>> import mars.tensor as mt

    >>> x1 = range(6)
    >>> x1
    [0, 1, 2, 3, 4, 5]
    >>> mt.power(x1, 3).execute()
    array([  0,   1,   8,  27,  64, 125])

    Raise the bases to different exponents.

    >>> x2 = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
    >>> mt.power(x1, x2).execute()
    array([  0.,   1.,   8.,  27.,  16.,   5.])

    The effect of broadcasting.

    >>> x2 = mt.array([[1, 2, 3, 3, 2, 1], [1, 2, 3, 3, 2, 1]])
    >>> x2.execute()
    array([[1, 2, 3, 3, 2, 1],
           [1, 2, 3, 3, 2, 1]])
    >>> mt.power(x1, x2).execute()
    array([[ 0,  1,  8, 27, 16,  5],
           [ 0,  1,  8, 27, 16,  5]])
    """
    op = TensorPower(**kwargs)
    return op(x1, x2, out=out, where=where)


@infer_dtype(np.power, reverse=True)
def rpower(x1, x2, **kwargs):
    op = TensorPower(**kwargs)
    return op.rcall(x1, x2)
