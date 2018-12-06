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


class TensorNextafter(operands.Nextafter, TensorBinOp):
    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super(TensorNextafter, self).__init__(_casting=casting, _err=err,
                                              _dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def _is_sparse(cls, x1, x2):
        return x1.issparse() and x2.issparse()

    @classmethod
    def constant_cls(cls):
        return TensorNextafterConstant


class TensorNextafterConstant(operands.NextafterConstant, TensorConstant):
    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super(TensorNextafterConstant, self).__init__(_casting=casting, _err=err,
                                                      _dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def _is_sparse(cls, x1, x2):
        if hasattr(x1, 'issparse') and x1.issparse() and np.isscalar(x2) and x2 == 0:
            return True
        if hasattr(x2, 'issparse') and x2.issparse() and np.isscalar(x1) and x1 == 0:
            return True
        return False


@infer_dtype(np.nextafter)
def nextafter(x1, x2, out=None, where=None, **kwargs):
    """
    Return the next floating-point value after x1 towards x2, element-wise.

    Parameters
    ----------
    x1 : array_like
        Values to find the next representable value of.
    x2 : array_like
        The direction where to look for the next representable value of `x1`.
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
    out : array_like
        The next representable values of `x1` in the direction of `x2`.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> eps = mt.finfo(mt.float64).eps
    >>> (mt.nextafter(1, 2) == eps + 1).execute()
    True
    >>> (mt.nextafter([1, 2], [2, 1]) == [eps + 1, 2 - eps]).execute()
    array([ True,  True])
    """
    op = TensorNextafter(**kwargs)
    return op(x1, x2, out=out, where=where)
