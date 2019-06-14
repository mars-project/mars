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

from .... import opcodes as OperandDef
from ....serialize import KeyField
from ...core import Tensor
from ..core import TensorOperand, TensorOperandMixin
from ..datasource import tensor as astensor
from .tensordot import tensordot


class TensorDot(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.DOT

    _a = KeyField('a')
    _b = KeyField('b')

    def __init__(self, dtype=None, sparse=False, **kw):
        super(TensorDot, self).__init__(_dtype=dtype, _sparse=sparse, **kw)

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    def _set_inputs(self, inputs):
        super(TensorDot, self)._set_inputs(inputs)
        self._a, self._b = self._inputs


def dot(a, b, out=None, sparse=None):
    """
    Dot product of two arrays. Specifically,

    - If both `a` and `b` are 1-D arrays, it is inner product of vectors
      (without complex conjugation).

    - If both `a` and `b` are 2-D arrays, it is matrix multiplication,
      but using :func:`matmul` or ``a @ b`` is preferred.

    - If either `a` or `b` is 0-D (scalar), it is equivalent to :func:`multiply`
      and using ``numpy.multiply(a, b)`` or ``a * b`` is preferred.

    - If `a` is an N-D array and `b` is a 1-D array, it is a sum product over
      the last axis of `a` and `b`.

    - If `a` is an N-D array and `b` is an M-D array (where ``M>=2``), it is a
      sum product over the last axis of `a` and the second-to-last axis of `b`::

        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    Parameters
    ----------
    a : array_like
        First argument.
    b : array_like
        Second argument.
    out : Tensor, optional
        Output argument. This must have the exact kind that would be returned
        if it was not used. In particular, it must have the right type, must be
        C-contiguous, and its dtype must be the dtype that would be returned
        for `dot(a,b)`. This is a performance feature. Therefore, if these
        conditions are not met, an exception is raised, instead of attempting
        to be flexible.

    Returns
    -------
    output : Tensor
        Returns the dot product of `a` and `b`.  If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        a tensor is returned.
        If `out` is given, then it is returned.

    Raises
    ------
    ValueError
        If the last dimension of `a` is not the same size as
        the second-to-last dimension of `b`.

    See Also
    --------
    vdot : Complex-conjugating dot product.
    tensordot : Sum products over arbitrary axes.
    einsum : Einstein summation convention.
    matmul : '@' operator as method with out parameter.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.dot(3, 4).execute()
    12

    Neither argument is complex-conjugated:

    >>> mt.dot([2j, 3j], [2j, 3j]).execute()
    (-13+0j)

    For 2-D arrays it is the matrix product:

    >>> a = [[1, 0], [0, 1]]
    >>> b = [[4, 1], [2, 2]]
    >>> mt.dot(a, b).execute()
    array([[4, 1],
           [2, 2]])

    >>> a = mt.arange(3*4*5*6).reshape((3,4,5,6))
    >>> b = mt.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
    >>> mt.dot(a, b)[2,3,2,1,2,2].execute()
    499128
    >>> mt.sum(a[2,3,2,:] * b[1,2,:,2]).execute()
    499128
    """
    a, b = astensor(a), astensor(b)
    if a.isscalar() and b.isscalar():
        ret = a * b
    else:
        ret = tensordot(a, b, axes=((a.ndim - 1,), (b.ndim - 2,)), sparse=sparse)

    if out is None:
        return ret

    # set to out
    if not isinstance(out, Tensor):
        raise ValueError('`out` must be a Tensor, got {0} instead'.format(type(out)))
    out.data = ret.data
    return out
