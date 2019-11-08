#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2020 Alibaba Group Holding Ltd.
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
from ..utils import infer_dtype
from .core import TensorBinOp
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='binary_or')
class TensorRshift(TensorBinOp):
    _op_type_ = OperandDef.RSHIFT
    _func_name = 'right_shift'


@infer_dtype(np.right_shift)
def rshift(x1, x2, out=None, where=None, **kwargs):
    """
    Shift the bits of an integer to the right.

    Bits are shifted to the right `x2`.  Because the internal
    representation of numbers is in binary format, this operation is
    equivalent to dividing `x1` by ``2**x2``.

    Parameters
    ----------
    x1 : array_like, int
        Input values.
    x2 : array_like, int
        Number of bits to remove at the right of `x1`.
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
    out : Tensor, int
        Return `x1` with bits shifted `x2` times to the right.

    See Also
    --------
    left_shift : Shift the bits of an integer to the left.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> mt.right_shift(10, 1).execute()
    5

    >>> mt.right_shift(10, [1,2,3]).execute()
    array([5, 2, 1])
    """
    op = TensorRshift(**kwargs)
    return op(x1, x2, out=out, where=where)


@infer_dtype(np.right_shift, reverse=True)
def rrshift(x1, x2, **kwargs):
    op = TensorRshift(**kwargs)
    return op.rcall(x1, x2)
