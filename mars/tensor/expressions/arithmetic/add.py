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
from .core import TensorBinOp, TensorConstant, TensorElementWise
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='binary_and')
class TensorAdd(operands.Add, TensorBinOp):
    @classmethod
    def constant_cls(cls):
        return TensorAddConstant


@arithmetic_operand(sparse_mode='binary_or_const')
class TensorAddConstant(operands.AddConstant, TensorConstant):
    pass


@infer_dtype(np.add)
def add(x1, x2, out=None, where=None, **kwargs):
    """
    Add arguments element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        The tensors to be added.  If ``x1.shape != x2.shape``, they must be
        broadcastable to a common shape (which may be the shape of one or
        the other).
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
    add : Tensor or scalar
        The sum of `x1` and `x2`, element-wise.  Returns a scalar if
        both  `x1` and `x2` are scalars.

    Notes
    -----
    Equivalent to `x1` + `x2` in terms of tensor broadcasting.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.add(1.0, 4.0).execute()
    5.0
    >>> x1 = mt.arange(9.0).reshape((3, 3))
    >>> x2 = mt.arange(3.0)
    >>> mt.add(x1, x2).execute()
    array([[  0.,   2.,   4.],
           [  3.,   5.,   7.],
           [  6.,   8.,  10.]])
    """
    op = TensorAdd(**kwargs)
    return op(x1, x2, out=out, where=where)


@infer_dtype(np.add, reverse=True)
def radd(x1, x2, **kwargs):
    op = TensorAdd(**kwargs)
    return op.rcall(x1, x2)


class TensorTreeAdd(operands.TreeAdd, TensorElementWise):
    def __init__(self, dtype=None, sparse=False, **kw):
        super(TensorTreeAdd, self).__init__(_dtype=dtype, _sparse=sparse, **kw)
