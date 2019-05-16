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

from .... import opcodes as OperandDef
from ..utils import infer_dtype
from .core import TensorUnaryOp
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='unary')
class TensorArctanh(TensorUnaryOp):
    _op_type_ = OperandDef.ARCTANH


@infer_dtype(np.arctanh)
def arctanh(x, out=None, where=None, **kwargs):
    """
    Inverse hyperbolic tangent element-wise.

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
    out : Tensor
        Array of the same shape as `x`.

    Notes
    -----
    `arctanh` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `tanh(z) = x`. The convention is to return
    the `z` whose imaginary part lies in `[-pi/2, pi/2]`.

    For real-valued input data types, `arctanh` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arctanh` is a complex analytical function
    that has branch cuts `[-1, -inf]` and `[1, inf]` and is continuous from
    above on the former and from below on the latter.

    The inverse hyperbolic tangent is also known as `atanh` or ``tanh^-1``.

    References
    ----------
    .. [1] M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
           10th printing, 1964, pp. 86. http://www.math.sfu.ca/~cbm/aands/
    .. [2] Wikipedia, "Inverse hyperbolic function",
           http://en.wikipedia.org/wiki/Arctanh

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.arctanh([0, -0.5]).execute()
    array([ 0.        , -0.54930614])
    """
    op = TensorArctanh(**kwargs)
    return op(x, out=out, where=where)
