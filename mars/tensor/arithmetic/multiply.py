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
from functools import reduce

from ... import opcodes as OperandDef
from ..array_utils import device, as_same_device
from ..utils import infer_dtype
from .core import TensorOperand
from .core import TensorBinOp, TensorElementWise
from .utils import arithmetic_operand, tree_op_estimate_size


@arithmetic_operand(sparse_mode='binary_or')
class TensorMultiply(TensorBinOp):
    _op_type_ = OperandDef.MUL
    _func_name = 'multiply'


@infer_dtype(np.multiply)
def multiply(x1, x2, out=None, where=None, **kwargs):
    """
    Multiply arguments element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays to be multiplied.
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
        The product of `x1` and `x2`, element-wise. Returns a scalar if
        both  `x1` and `x2` are scalars.

    Notes
    -----
    Equivalent to `x1` * `x2` in terms of array broadcasting.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.multiply(2.0, 4.0).execute()
    8.0

    >>> x1 = mt.arange(9.0).reshape((3, 3))
    >>> x2 = mt.arange(3.0)
    >>> mt.multiply(x1, x2).execute()
    array([[  0.,   1.,   4.],
           [  0.,   4.,  10.],
           [  0.,   7.,  16.]])
    """
    op = TensorMultiply(**kwargs)
    return op(x1, x2, out=out, where=where)


@infer_dtype(np.multiply, reverse=True)
def rmultiply(x1, x2, **kwargs):
    op = TensorMultiply(**kwargs)
    return op.rcall(x1, x2)


class TensorTreeMultiply(TensorOperand, TensorElementWise):
    _op_type_ = OperandDef.TREE_MULTIPLY

    def __init__(self, dtype=None, sparse=False, **kw):
        super().__init__(_dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def execute(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            ctx[op.outputs[0].key] = reduce(xp.multiply, inputs)

    @classmethod
    def estimate_size(cls, ctx, op):
        tree_op_estimate_size(ctx, op)
