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
from .core import TensorUnaryOp


class TensorAbsolute(operands.Absolute, TensorUnaryOp):
    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super(TensorAbsolute, self).__init__(_casting=casting, _err=err,
                                             _dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def _is_sparse(cls, x):
        return x.issparse()


@infer_dtype(np.absolute)
def absolute(x, out=None, where=None, **kwargs):
    r"""
    Calculate the absolute value element-wise.

    Parameters
    ----------
    x : array_like
        Input tensor.
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
    absolute : Tensor
        An tensor containing the absolute value of
        each element in `x`.  For complex input, ``a + ib``, the
        absolute value is :math:`\sqrt{ a^2 + b^2 }`.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.array([-1.2, 1.2])
    >>> mt.absolute(x).execute()
    array([ 1.2,  1.2])
    >>> mt.absolute(1.2 + 1j).execute()
    1.5620499351813308
    """
    op = TensorAbsolute(**kwargs)
    return op(x, out=out, where=where)
