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


@arithmetic_operand(sparse_mode='binary_or')
class TensorLshift(operands.Lshift, TensorBinOp):
    @classmethod
    def constant_cls(cls):
        return TensorLshiftConstant


@arithmetic_operand(sparse_mode='binary_or_const')
class TensorLshiftConstant(operands.RshiftConstant, TensorConstant):
    pass


@infer_dtype(np.left_shift)
def lshift(x1, x2, out=None, where=None, **kwargs):
    """
    Shift the bits of an integer to the left.

    Bits are shifted to the left by appending `x2` 0s at the right of `x1`.
    Since the internal representation of numbers is in binary format, this
    operation is equivalent to multiplying `x1` by ``2**x2``.

    Parameters
    ----------
    x1 : array_like of integer type
        Input values.
    x2 : array_like of integer type
        Number of zeros to append to `x1`. Has to be non-negative.
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
    out : tensor of integer type
        Return `x1` with bits shifted `x2` times to the left.

    See Also
    --------
    right_shift : Shift the bits of an integer to the right.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.left_shift(5, 2).execute()
    20

    >>> mt.left_shift(5, [1,2,3]).execute()
    array([10, 20, 40])
    """
    op = TensorLshift(**kwargs)
    return op(x1, x2, out=out, where=where)


@infer_dtype(np.left_shift, reverse=True)
def rlshift(x1, x2, **kwargs):
    op = TensorLshift(**kwargs)
    return op.rcall(x1, x2)
