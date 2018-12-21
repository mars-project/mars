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


class TensorSign(operands.Sign, TensorUnaryOp):
    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super(TensorSign, self).__init__(_casting=casting, _err=err,
                                         _dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def _is_sparse(cls, x):
        return x.issparse()


@infer_dtype(np.sign)
def sign(x, out=None, where=None, **kwargs):
    r"""
    Returns an element-wise indication of the sign of a number.

    The `sign` function returns ``-1 if x < 0, 0 if x==0, 1 if x > 0``.  nan
    is returned for nan inputs.

    For complex inputs, the `sign` function returns
    ``sign(x.real) + 0j if x.real != 0 else sign(x.imag) + 0j``.

    complex(nan, 0) is returned for complex nan inputs.

    Parameters
    ----------
    x : array_like
      Input values.
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
      The sign of `x`.

    Notes
    -----
    There is more than one definition of sign in common use for complex
    numbers.  The definition used here is equivalent to :math:`x/\sqrt{x*x}`
    which is different from a common alternative, :math:`x/|x|`.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.sign([-5., 4.5]).execute()
    array([-1.,  1.])
    >>> mt.sign(0).execute()
    0
    >>> mt.sign(5-2j).execute()
    (1+0j)
    """
    op = TensorSign(**kwargs)
    return op(x, out=out, where=where)
