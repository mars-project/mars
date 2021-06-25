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
class TensorMinimum(TensorBinOp):
    _op_type_ = OperandDef.MINIMUM
    _func_name = 'minimum'

    @classmethod
    def _is_sparse(cls, x1, x2):
        if hasattr(x1, 'issparse') and x1.issparse() and np.isscalar(x2) and x2 >= 0:
            return True
        if hasattr(x2, 'issparse') and x2.issparse() and np.isscalar(x1) and x1 >= 0:
            return True
        return False


@infer_dtype(np.minimum)
def minimum(x1, x2, out=None, where=None, **kwargs):
    """
    Element-wise minimum of tensor elements.

    Compare two tensors and returns a new tensor containing the element-wise
    minima. If one of the elements being compared is a NaN, then that
    element is returned. If both elements are NaNs then the first is
    returned. The latter distinction is important for complex NaNs, which
    are defined as at least one of the real or imaginary parts being a NaN.
    The net effect is that NaNs are propagated.

    Parameters
    ----------
    x1, x2 : array_like
        The tensors holding the elements to be compared. They must have
        the same shape, or shapes that can be broadcast to a single shape.
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
        The minimum of `x1` and `x2`, element-wise.  Returns scalar if
        both  `x1` and `x2` are scalars.

    See Also
    --------
    maximum :
        Element-wise maximum of two tensors, propagates NaNs.
    fmin :
        Element-wise minimum of two tensors, ignores NaNs.
    amin :
        The minimum value of a tensor along a given axis, propagates NaNs.
    nanmin :
        The minimum value of a tenosr along a given axis, ignores NaNs.

    fmax, amax, nanmax

    Notes
    -----
    The minimum is equivalent to ``mt.where(x1 <= x2, x1, x2)`` when
    neither x1 nor x2 are NaNs, but it is faster and does proper
    broadcasting.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.minimum([2, 3, 4], [1, 5, 2]).execute()
    array([1, 3, 2])

    >>> mt.minimum(mt.eye(2), [0.5, 2]).execute() # broadcasting
    array([[ 0.5,  0. ],
           [ 0. ,  1. ]])

    >>> mt.minimum([mt.nan, 0, mt.nan],[0, mt.nan, mt.nan]).execute()
    array([ NaN,  NaN,  NaN])
    >>> mt.minimum(-mt.Inf, 1).execute()
    -inf
    """
    op = TensorMinimum(**kwargs)
    return op(x1, x2, out=out, where=where)
