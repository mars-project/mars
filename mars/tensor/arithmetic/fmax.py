#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2021 Alibaba Group Holding Ltd.
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
from ..utils import infer_dtype
from .core import TensorBinOp
from .utils import arithmetic_operand


@arithmetic_operand
class TensorFMax(TensorBinOp):
    _op_type_ = OperandDef.FMAX
    _func_name = 'fmax'

    @classmethod
    def _is_sparse(cls, x1, x2):
        if hasattr(x1, 'issparse') and x1.issparse() and np.isscalar(x2) and x2 <= 0:
            return True
        if hasattr(x2, 'issparse') and x2.issparse() and np.isscalar(x1) and x1 <= 0:
            return True
        return False


@infer_dtype(np.fmax)
def fmax(x1, x2, out=None, where=None, **kwargs):
    """
    Element-wise maximum of array elements.

    Compare two tensors and returns a new tensor containing the element-wise
    maxima. If one of the elements being compared is a NaN, then the
    non-nan element is returned. If both elements are NaNs then the first
    is returned.  The latter distinction is important for complex NaNs,
    which are defined as at least one of the real or imaginary parts being
    a NaN. The net effect is that NaNs are ignored when possible.

    Parameters
    ----------
    x1, x2 : array_like
        The tensors holding the elements to be compared. They must have
        the same shape.
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
    y : Tensor or scalar
        The maximum of `x1` and `x2`, element-wise.  Returns scalar if
        both  `x1` and `x2` are scalars.

    See Also
    --------
    fmin :
        Element-wise minimum of two tensors, ignores NaNs.
    maximum :
        Element-wise maximum of two tensors, propagates NaNs.
    amax :
        The maximum value of an tensor along a given axis, propagates NaNs.
    nanmax :
        The maximum value of an tensor along a given axis, ignores NaNs.

    minimum, amin, nanmin

    Notes
    -----
    The fmax is equivalent to ``mt.where(x1 >= x2, x1, x2)`` when neither
    x1 nor x2 are NaNs, but it is faster and does proper broadcasting.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.fmax([2, 3, 4], [1, 5, 2]).execute()
    array([ 2.,  5.,  4.])

    >>> mt.fmax(mt.eye(2), [0.5, 2]).execute()
    array([[ 1. ,  2. ],
           [ 0.5,  2. ]])

    >>> mt.fmax([mt.nan, 0, mt.nan],[0, mt.nan, mt.nan]).execute()
    array([  0.,   0.,  NaN])
    """
    op = TensorFMax(**kwargs)
    return op(x1, x2, out=out, where=where)
