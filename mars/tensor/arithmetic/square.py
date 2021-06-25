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
from ..utils import infer_dtype
from .core import TensorUnaryOp
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='unary')
class TensorSquare(TensorUnaryOp):
    _op_type_ = OperandDef.SQUARE
    _func_name = 'square'


@infer_dtype(np.square)
def square(x, out=None, where=None, **kwargs):
    """
    Return the element-wise square of the input.

    Parameters
    ----------
    x : array_like
        Input data.
    out : Tensor, None, or tuple of tensor and None, optional
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
    out : Tensor
        Element-wise `x*x`, of the same shape and dtype as `x`.
        Returns scalar if `x` is a scalar.

    See Also
    --------
    sqrt
    power

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.square([-1j, 1]).execute()
    array([-1.-0.j,  1.+0.j])
    """
    op = TensorSquare(**kwargs)
    return op(x, out=out, where=where)
