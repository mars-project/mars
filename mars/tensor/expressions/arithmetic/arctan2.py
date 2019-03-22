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
from .core import TensorBinOp, TensorConstant
from .utils import arithmetic_operand


@arithmetic_operand
class TensorArctan2(operands.Arctan2, TensorBinOp):
    @classmethod
    def _is_sparse(cls, x1, x2):
        # x2 is sparse or not does not matter
        return x1.issparse()

    @classmethod
    def constant_cls(cls):
        return TensorArct2Constant


@arithmetic_operand
class TensorArct2Constant(operands.Arct2Constant, TensorConstant):
    @classmethod
    def _is_sparse(cls, x1, x2):
        if hasattr(x1, 'issparse') and x1.issparse() and np.isscalar(x2):
            return True
        elif x1 == 0:
            return True
        return False


@infer_dtype(np.arctan2)
def arctan2(x1, x2, out=None, where=None, **kwargs):
    """
    Element-wise arc tangent of ``x1/x2`` choosing the quadrant correctly.

    The quadrant (i.e., branch) is chosen so that ``arctan2(x1, x2)`` is
    the signed angle in radians between the ray ending at the origin and
    passing through the point (1,0), and the ray ending at the origin and
    passing through the point (`x2`, `x1`).  (Note the role reversal: the
    "`y`-coordinate" is the first function parameter, the "`x`-coordinate"
    is the second.)  By IEEE convention, this function is defined for
    `x2` = +/-0 and for either or both of `x1` and `x2` = +/-inf (see
    Notes for specific values).

    This function is not defined for complex-valued arguments; for the
    so-called argument of complex values, use `angle`.

    Parameters
    ----------
    x1 : array_like, real-valued
        `y`-coordinates.
    x2 : array_like, real-valued
        `x`-coordinates. `x2` must be broadcastable to match the shape of
        `x1` or vice versa.
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
        Array of angles in radians, in the range ``[-pi, pi]``.

    See Also
    --------
    arctan, tan, angle

    Notes
    -----
    *arctan2* is identical to the `atan2` function of the underlying
    C library.  The following special values are defined in the C
    standard: [1]_

    ====== ====== ================
    `x1`   `x2`   `arctan2(x1,x2)`
    ====== ====== ================
    +/- 0  +0     +/- 0
    +/- 0  -0     +/- pi
     > 0   +/-inf +0 / +pi
     < 0   +/-inf -0 / -pi
    +/-inf +inf   +/- (pi/4)
    +/-inf -inf   +/- (3*pi/4)
    ====== ====== ================

    Note that +0 and -0 are distinct floating point numbers, as are +inf
    and -inf.

    References
    ----------
    .. [1] ISO/IEC standard 9899:1999, "Programming language C."

    Examples
    --------
    Consider four points in different quadrants:
    >>> import mars.tensor as mt

    >>> x = mt.array([-1, +1, +1, -1])
    >>> y = mt.array([-1, -1, +1, +1])
    >>> (mt.arctan2(y, x) * 180 / mt.pi).execute()
    array([-135.,  -45.,   45.,  135.])

    Note the order of the parameters. `arctan2` is defined also when `x2` = 0
    and at several other special points, obtaining values in
    the range ``[-pi, pi]``:

    >>> mt.arctan2([1., -1.], [0., 0.]).execute()
    array([ 1.57079633, -1.57079633])
    >>> mt.arctan2([0., 0., mt.inf], [+0., -0., mt.inf]).execute()
    array([ 0.        ,  3.14159265,  0.78539816])
    """
    op = TensorArctan2(**kwargs)
    return op(x1, x2, out=out, where=where)
