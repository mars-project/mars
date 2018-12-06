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


class TensorBitand(operands.Bitand, TensorBinOp):
    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super(TensorBitand, self).__init__(_casting=casting, _err=err,
                                           _dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def _is_sparse(cls, x1, x2):
        return x1.issparse() or x2.issparse()

    @classmethod
    def constant_cls(cls):
        return TensorBitandConstant


class TensorBitandConstant(operands.BitandConstant, TensorConstant):
    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super(TensorBitandConstant, self).__init__(_casting=casting, _err=err,
                                                   _dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def _is_sparse(cls, x1, x2):
        if (hasattr(x1, 'issparse') and x1.issparse()) or \
                (hasattr(x2, 'issparse') and x2.issparse()):
            return True
        return False


@infer_dtype(np.bitwise_and)
def bitand(x1, x2, out=None, where=None, **kwargs):
    """
    Compute the bit-wise AND of two tensors element-wise.

    Computes the bit-wise AND of the underlying binary representation of
    the integers in the input arrays. This ufunc implements the C/Python
    operator ``&``.

    Parameters
    ----------
    x1, x2 : array_like
        Only integer and boolean types are handled.
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
        Result.

    See Also
    --------
    logical_and
    bitwise_or
    bitwise_xor

    Examples
    --------
    The number 13 is represented by ``00001101``.  Likewise, 17 is
    represented by ``00010001``.  The bit-wise AND of 13 and 17 is
    therefore ``000000001``, or 1:

    >>> import mars.tensor as mt

    >>> mt.bitwise_and(13, 17).execute()
    1

    >>> mt.bitwise_and(14, 13).execute()
    12
    >>> mt.bitwise_and([14,3], 13).execute()
    array([12,  1])

    >>> mt.bitwise_and([11,7], [4,25]).execute()
    array([0, 1])
    >>> mt.bitwise_and(mt.array([2,5,255]), mt.array([3,14,16])).execute()
    array([ 2,  4, 16])
    >>> mt.bitwise_and([True, True], [False, True]).execute()
    array([False,  True])
    """
    op = TensorBitand(**kwargs)
    return op(x1, x2, out=out, where=where)


@infer_dtype(np.bitwise_and, reverse=True)
def rbitand(x1, x2, **kwargs):
    op = TensorBitand(**kwargs)
    return op.rcall(x1, x2)
