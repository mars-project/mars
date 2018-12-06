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
from ..utils import inject_dtype
from .core import TensorCompare, TensorCompareConstant


class TensorGreaterEqual(operands.GreaterEqual, TensorCompare):
    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=None, **kw):
        err = err if err is not None else np.geterr()
        super(TensorGreaterEqual, self).__init__(_casting=casting, _err=err,
                                                 _dtype=dtype, _sparse=sparse, **kw)

    @classmethod
    def constant_cls(cls):
        return TensorGeConstant


class TensorGeConstant(operands.GeConstant, TensorCompareConstant):
    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=None, **kw):
        err = err if err is not None else np.geterr()
        super(TensorGeConstant, self).__init__(_casting=casting, _err=err,
                                               _dtype=dtype, _sparse=sparse, **kw)


@inject_dtype(np.bool_)
def greater_equal(x1, x2, out=None, where=None, **kwargs):
    """
    Return the truth value of (x1 >= x2) element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input tensors.  If ``x1.shape != x2.shape``, they must be
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
    out : bool or Tensor of bool
        Array of bools, or a single bool if `x1` and `x2` are scalars.

    See Also
    --------
    greater, less, less_equal, equal, not_equal

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.greater_equal([4, 2, 1], [2, 2, 2]).execute()
    array([ True, True, False])
    """
    op = TensorGreaterEqual(**kwargs)
    return op(x1, x2, out=out, where=where)
