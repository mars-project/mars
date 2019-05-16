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
class TensorI0(TensorUnaryOp):
    _op_type_ = OperandDef.I0


@infer_dtype(np.i0)
def i0(x, **kwargs):
    """
    Modified Bessel function of the first kind, order 0.

    Usually denoted :math:`I_0`.  This function does broadcast, but will *not*
    "up-cast" int dtype arguments unless accompanied by at least one float or
    complex dtype argument (see Raises below).

    Parameters
    ----------
    x : array_like, dtype float or complex
        Argument of the Bessel function.

    Returns
    -------
    out : Tensor, shape = x.shape, dtype = x.dtype
        The modified Bessel function evaluated at each of the elements of `x`.

    Raises
    ------
    TypeError: array cannot be safely cast to required type
        If argument consists exclusively of int dtypes.

    See Also
    --------
    scipy.special.iv, scipy.special.ive

    Notes
    -----
    We use the algorithm published by Clenshaw [1]_ and referenced by
    Abramowitz and Stegun [2]_, for which the function domain is
    partitioned into the two intervals [0,8] and (8,inf), and Chebyshev
    polynomial expansions are employed in each interval. Relative error on
    the domain [0,30] using IEEE arithmetic is documented [3]_ as having a
    peak of 5.8e-16 with an rms of 1.4e-16 (n = 30000).

    References
    ----------
    .. [1] C. W. Clenshaw, "Chebyshev series for mathematical functions", in
           *National Physical Laboratory Mathematical Tables*, vol. 5, London:
           Her Majesty's Stationery Office, 1962.
    .. [2] M. Abramowitz and I. A. Stegun, *Handbook of Mathematical
           Functions*, 10th printing, New York: Dover, 1964, pp. 379.
           http://www.math.sfu.ca/~cbm/aands/page_379.htm
    .. [3] http://kobesearch.cpan.org/htdocs/Math-Cephes/Math/Cephes.html

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.i0([0.]).execute()
    array([1.])
    >>> mt.i0([0., 1. + 2j]).execute()
    array([ 1.00000000+0.j        ,  0.18785373+0.64616944j])

    """
    op = TensorI0(**kwargs)
    return op(x)
