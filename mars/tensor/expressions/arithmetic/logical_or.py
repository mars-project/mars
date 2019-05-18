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


@arithmetic_operand(sparse_mode='binary_and')
class TensorOr(TensorBinOp):
    _op_type_ = OperandDef.OR

    @classmethod
    def constant_cls(cls):
        return TensorOrConstant


@arithmetic_operand(sparse_mode='binary_or_const')
class TensorOrConstant(TensorConstant):
    _op_type_ = OperandDef.OR_CONSTANT


@infer_dtype(np.logical_or)
def logical_or(x1, x2, out=None, where=None, **kwargs):
    """
    Compute the truth value of x1 OR x2 element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Logical OR is applied to the elements of `x1` and `x2`.
        They have to be of the same shape.
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
    y : Tensor or bool
        Boolean result with the same shape as `x1` and `x2` of the logical
        OR operation on elements of `x1` and `x2`.

    See Also
    --------
    logical_and, logical_not, logical_xor
    bitwise_or

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.logical_or(True, False).execute()
    True
    >>> mt.logical_or([True, False], [False, False]).execute()
    array([ True, False])

    >>> x = mt.arange(5)
    >>> mt.logical_or(x < 1, x > 3).execute()
    array([ True, False, False, False,  True])
    """
    op = TensorOr(**kwargs)
    return op(x1, x2, out=out, where=where)


@infer_dtype(np.logical_or, reverse=True)
def rlogical_or(x1, x2, **kwargs):
    op = TensorOr(**kwargs)
    return op.rcall(x1, x2)
