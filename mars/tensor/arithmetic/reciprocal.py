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
class TensorReciprocal(TensorUnaryOp):
    _op_type_ = OperandDef.RECIPROCAL
    _func_name = 'reciprocal'


@infer_dtype(np.reciprocal)
def reciprocal(x, out=None, where=None, **kwargs):
    """
    Return the reciprocal of the argument, element-wise.

    Calculates ``1/x``.

    Parameters
    ----------
    x : array_like
        Input tensor.
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
        Return tensor.

    Notes
    -----
    .. note::
        This function is not designed to work with integers.

    For integer arguments with absolute value larger than 1 the result is
    always zero because of the way Python handles integer division.  For
    integer zero the result is an overflow.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.reciprocal(2.).execute()
    0.5
    >>> mt.reciprocal([1, 2., 3.33]).execute()
    array([ 1.       ,  0.5      ,  0.3003003])
    """
    op = TensorReciprocal(**kwargs)
    return op(x, out=out, where=where)
