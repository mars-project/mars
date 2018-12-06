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


class TensorHypot(operands.Hypot, TensorBinOp):
    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super(TensorHypot, self).__init__(_casting=casting, _err=err,
                                          _dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def _is_sparse(cls, x1, x2):
        return x1.issparse() and x2.issparse()

    @classmethod
    def constant_cls(cls):
        return TensorHypotConstant


class TensorHypotConstant(operands.HypotConstant, TensorConstant):
    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super(TensorHypotConstant, self).__init__(_casting=casting, _err=err,
                                                  _dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def _is_sparse(cls, x1, x2):
        if hasattr(x1, 'issparse') and x1.issparse() and np.isscalar(x2) and x2 == 0:
            return True
        if hasattr(x2, 'issparse') and x2.issparse() and np.isscalar(x1) and x1 == 0:
            return True
        return False


@infer_dtype(np.hypot)
def hypot(x1, x2, out=None, where=None, **kwargs):
    """
    Given the "legs" of a right triangle, return its hypotenuse.

    Equivalent to ``sqrt(x1**2 + x2**2)``, element-wise.  If `x1` or
    `x2` is scalar_like (i.e., unambiguously cast-able to a scalar type),
    it is broadcast for use with each element of the other argument.
    (See Examples)

    Parameters
    ----------
    x1, x2 : array_like
        Leg of the triangle(s).
    out : Tensor, None, or tuple of Tensor and None, optional
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
    z : Tensor
        The hypotenuse of the triangle(s).

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.hypot(3*mt.ones((3, 3)), 4*mt.ones((3, 3))).execute()
    array([[ 5.,  5.,  5.],
           [ 5.,  5.,  5.],
           [ 5.,  5.,  5.]])

    Example showing broadcast of scalar_like argument:

    >>> mt.hypot(3*mt.ones((3, 3)), [4]).execute()
    array([[ 5.,  5.,  5.],
           [ 5.,  5.,  5.],
           [ 5.,  5.,  5.]])
    """
    op = TensorHypot(**kwargs)
    return op(x1, x2, out=out, where=where)
