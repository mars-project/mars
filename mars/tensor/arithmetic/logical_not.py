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
from .core import TensorUnaryOp
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='unary')
class TensorNot(TensorUnaryOp):
    _op_type_ = OperandDef.NOT
    _func_name = 'logical_not'


@infer_dtype(np.logical_not)
def logical_not(x, out=None, where=None, **kwargs):
    """
    Compute the truth value of NOT x element-wise.

    Parameters
    ----------
    x : array_like
        Logical NOT is applied to the elements of `x`.
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
    y : bool or Tensor of bool
        Boolean result with the same shape as `x` of the NOT operation
        on elements of `x`.

    See Also
    --------
    logical_and, logical_or, logical_xor

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.logical_not(3).execute()
    False
    >>> mt.logical_not([True, False, 0, 1]).execute()
    array([False,  True,  True, False])

    >>> x = mt.arange(5)
    >>> mt.logical_not(x<3).execute()
    array([False, False, False,  True,  True])
    """
    op = TensorNot(**kwargs)
    return op(x, out=out, where=where)
