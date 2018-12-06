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


class TensorTrueDiv(operands.TrueDiv, TensorBinOp):
    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super(TensorTrueDiv, self).__init__(_casting=casting, _err=err,
                                            _dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def _is_sparse(cls, x1, x2):
        return False

    @classmethod
    def constant_cls(cls):
        return TensorTDivConstant


class TensorTDivConstant(operands.TDivConstant, TensorConstant):
    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super(TensorTDivConstant, self).__init__(_casting=casting, _err=err,
                                                 _dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def _is_sparse(cls, x1, x2):
        if hasattr(x1, 'issparse') and x1.issparse():
            if x2 != 0:
                return True
            else:
                raise ZeroDivisionError('float division by zero')
        return False


@infer_dtype(np.true_divide)
def truediv(x1, x2, out=None, where=None, **kwargs):
    """
    Returns a true division of the inputs, element-wise.

    Instead of the Python traditional 'floor division', this returns a true
    division.  True division adjusts the output type to present the best
    answer, regardless of input types.

    Parameters
    ----------
    x1 : array_like
        Dividend tensor.
    x2 : array_like
        Divisor tensor.
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
    out : Tensor
        Result is scalar if both inputs are scalar, tensor otherwise.

    Notes
    -----
    The floor division operator ``//`` was added in Python 2.2 making
    ``//`` and ``/`` equivalent operators.  The default floor division
    operation of ``/`` can be replaced by true division with ``from
    __future__ import division``.

    In Python 3.0, ``//`` is the floor division operator and ``/`` the
    true division operator.  The ``true_divide(x1, x2)`` function is
    equivalent to true division in Python.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.arange(5)
    >>> mt.true_divide(x, 4).execute()
    array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])

    # for python 2
    >>> (x/4).execute()
    array([0, 0, 0, 0, 1])
    >>> (x//4).execute()
    array([0, 0, 0, 0, 1])
    """
    op = TensorTrueDiv(**kwargs)
    return op(x1, x2, out=out, where=where)


@infer_dtype(np.true_divide, reverse=True)
def rtruediv(x1, x2, **kwargs):
    op = TensorTrueDiv(**kwargs)
    return op.rcall(x1, x2)
