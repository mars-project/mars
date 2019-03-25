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
class TensorArccos(operands.Arccos, TensorUnaryOp):
    pass


@infer_dtype(np.arccos)
def arccos(x, out=None, where=None, **kwargs):
    """
    Trigonometric inverse cosine, element-wise.

    The inverse of `cos` so that, if ``y = cos(x)``, then ``x = arccos(y)``.

    Parameters
    ----------
    x : array_like
        `x`-coordinate on the unit circle.
        For real arguments, the domain is [-1, 1].
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
        The angle of the ray intersecting the unit circle at the given
        `x`-coordinate in radians [0, pi]. If `x` is a scalar then a
        scalar is returned, otherwise an array of the same shape as `x`
        is returned.

    See Also
    --------
    cos, arctan, arcsin

    Notes
    -----
    `arccos` is a multivalued function: for each `x` there are infinitely
    many numbers `z` such that `cos(z) = x`. The convention is to return
    the angle `z` whose real part lies in `[0, pi]`.

    For real-valued input data types, `arccos` always returns real output.
    For each value that cannot be expressed as a real number or infinity,
    it yields ``nan`` and sets the `invalid` floating point error flag.

    For complex-valued input, `arccos` is a complex analytic function that
    has branch cuts `[-inf, -1]` and `[1, inf]` and is continuous from
    above on the former and from below on the latter.

    The inverse `cos` is also known as `acos` or cos^-1.

    References
    ----------
    M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
    10th printing, 1964, pp. 79. http://www.math.sfu.ca/~cbm/aands/

    Examples
    --------
    We expect the arccos of 1 to be 0, and of -1 to be pi:
    >>> import mars.tensor as mt

    >>> mt.arccos([1, -1]).execute()
    array([ 0.        ,  3.14159265])

    Plot arccos:

    >>> import matplotlib.pyplot as plt
    >>> x = mt.linspace(-1, 1, num=100)
    >>> plt.plot(x.execute(), mt.arccos(x).execute())
    >>> plt.axis('tight')
    >>> plt.show()
    """
    op = TensorArccos(**kwargs)
    return op(x, out=out, where=where)
