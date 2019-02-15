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
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='unary')
class TensorSinh(operands.Sinh, TensorUnaryOp):
    pass


@infer_dtype(np.sinh)
def sinh(x, out=None, where=None, **kwargs):
    """
    Hyperbolic sine, element-wise.

    Equivalent to ``1/2 * (mt.exp(x) - mt.exp(-x))`` or
    ``-1j * mt.sin(1j*x)``.

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
    y : Tensor
        The corresponding hyperbolic sine values.

    Notes
    -----
    If `out` is provided, the function writes the result into it,
    and returns a reference to `out`.  (See Examples)

    References
    ----------
    M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions.
    New York, NY: Dover, 1972, pg. 83.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.sinh(0).execute()
    0.0
    >>> mt.sinh(mt.pi*1j/2).execute()
    1j
    >>> mt.sinh(mt.pi*1j).execute() # (exact value is 0)
    1.2246063538223773e-016j
    >>> # Discrepancy due to vagaries of floating point arithmetic.

    >>> # Example of providing the optional output parameter
    >>> out1 = mt.zeros(1)
    >>> out2 = mt.sinh([0.1], out1)
    >>> out2 is out1
    True

    >>> # Example of ValueError due to provision of shape mis-matched `out`
    >>> mt.sinh(mt.zeros((3,3)),mt.zeros((2,2))).execute()
    Traceback (most recent call last):
    ...
    ValueError:  operands could not be broadcast together with shapes (3,3) (2,2)
    """
    op = TensorSinh(**kwargs)
    return op(x, out=out, where=where)
