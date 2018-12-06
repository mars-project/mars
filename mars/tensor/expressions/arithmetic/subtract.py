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


class TensorSubtract(operands.Subtract, TensorBinOp):
    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super(TensorSubtract, self).__init__(_casting=casting, _err=err,
                                             _dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def _is_sparse(cls, x1, x2):
        return x1.issparse() and x2.issparse()

    @classmethod
    def constant_cls(cls):
        return TensorSubConstant


class TensorSubConstant(operands.SubConstant, TensorConstant):
    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super(TensorSubConstant, self).__init__(_casting=casting, _err=err,
                                                _dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def _is_sparse(cls, x1, x2):
        if hasattr(x1, 'issparse') and x1.issparse() and np.isscalar(x2) and x2 == 0:
            return True
        if hasattr(x2, 'issparse') and x2.issparse() and np.isscalar(x1) and x1 == 0:
            return True
        return False


@infer_dtype(np.subtract)
def subtract(x1, x2, out=None, where=None, **kwargs):
    """
    Subtract arguments, element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        The tensors to be subtracted from each other.
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
        The difference of `x1` and `x2`, element-wise.  Returns a scalar if
        both  `x1` and `x2` are scalars.

    Notes
    -----
    Equivalent to ``x1 - x2`` in terms of tensor broadcasting.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.subtract(1.0, 4.0).execute()
    -3.0

    >>> x1 = mt.arange(9.0).reshape((3, 3))
    >>> x2 = mt.arange(3.0)
    >>> mt.subtract(x1, x2).execute()
    array([[ 0.,  0.,  0.],
           [ 3.,  3.,  3.],
           [ 6.,  6.,  6.]])
    """
    op = TensorSubtract(**kwargs)
    return op(x1, x2, out=out, where=where)


@infer_dtype(np.subtract, reverse=True)
def rsubtract(x1, x2, **kwargs):
    op = TensorSubtract(**kwargs)
    return op.rcall(x1, x2)
