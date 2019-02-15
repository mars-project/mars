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
class TensorArcsin(operands.Arcsin, TensorUnaryOp):
    pass


@infer_dtype(np.arcsin)
def arcsin(x, out=None, where=None, **kwargs):
    """
    Inverse sine, element-wise.

    Parameters
    ----------
    x : array_like
        `y`-coordinate on the unit circle.
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
    angle : Tensor
        The inverse sine of each element in `x`, in radians and in the
        closed interval ``[-pi/2, pi/2]``.  If `x` is a scalar, a scalar
        is returned, otherwise a tensor.

    See Also
    --------
    sin, cos, arccos, tan, arctan, arctan2, emath.arcsin

    Notes
    -----
    `arcsin` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that :math:`sin(z) = x`.  The convention is to
    return the angle `z` whose real part lies in [-pi/2, pi/2].

    For real-valued input data types, *arcsin* always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arcsin` is a complex analytic function that
    has, by convention, the branch cuts [-inf, -1] and [1, inf]  and is
    continuous from above on the former and from below on the latter.

    The inverse sine is also known as `asin` or sin^{-1}.

    References
    ----------
    Abramowitz, M. and Stegun, I. A., *Handbook of Mathematical Functions*,
    10th printing, New York: Dover, 1964, pp. 79ff.
    http://www.math.sfu.ca/~cbm/aands/

    Examples
    --------
    >>> import mars.tensor as mt
    >>> mt.arcsin(1).execute()     # pi/2
    1.5707963267948966
    >>> mt.arcsin(-1).execute()    # -pi/2
    -1.5707963267948966
    >>> mt.arcsin(0).execute()
    0.0
    """
    op = TensorArcsin(**kwargs)
    return op(x, out=out, where=where)
