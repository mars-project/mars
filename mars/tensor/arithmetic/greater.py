#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

from ... import opcodes as OperandDef
from ..utils import inject_dtype
from .utils import arithmetic_operand
from .core import TensorBinOp


@arithmetic_operand(sparse_mode='binary_and')
class TensorGreaterThan(TensorBinOp):
    _op_type_ = OperandDef.GT
    _func_name = 'greater'

    def __init__(self, casting='same_kind', err=None, dtype=None, sparse=None, **kw):
        err = err if err is not None else np.geterr()
        super().__init__(_casting=casting, _err=err, _dtype=dtype, _sparse=sparse, **kw)


@inject_dtype(np.bool_)
def greater(x1, x2, out=None, where=None, **kwargs):
    """
    Return the truth value of (x1 > x2) element-wise.

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
    greater_equal, less, less_equal, equal, not_equal

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.greater([4,2],[2,2]).execute()
    array([ True, False])

    If the inputs are ndarrays, then np.greater is equivalent to '>'.

    >>> a = mt.array([4,2])
    >>> b = mt.array([2,2])
    >>> (a > b).execute()
    array([ True, False])
    """
    op = TensorGreaterThan(**kwargs)
    return op(x1, x2, out=out, where=where)
