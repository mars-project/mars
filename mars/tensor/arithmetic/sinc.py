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
from .core import TensorUnaryOp
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='unary')
class TensorSinc(TensorUnaryOp):
    _op_type_ = OperandDef.SINC
    _func_name = 'sinc'


@infer_dtype(np.sinc)
def sinc(x, **kwargs):
    r"""
    Return the sinc function.

    The sinc function is :math:`\\sin(\\pi x)/(\\pi x)`.

    Parameters
    ----------
    x : Tensor
        Tensor (possibly multi-dimensional) of values for which to to
        calculate ``sinc(x)``.

    Returns
    -------
    out : Tensor
        ``sinc(x)``, which has the same shape as the input.

    Notes
    -----
    ``sinc(0)`` is the limit value 1.

    The name sinc is short for "sine cardinal" or "sinus cardinalis".

    The sinc function is used in various signal processing applications,
    including in anti-aliasing, in the construction of a Lanczos resampling
    filter, and in interpolation.

    For bandlimited interpolation of discrete-time signals, the ideal
    interpolation kernel is proportional to the sinc function.

    References
    ----------
    .. [1] Weisstein, Eric W. "Sinc Function." From MathWorld--A Wolfram Web
           Resource. http://mathworld.wolfram.com/SincFunction.html
    .. [2] Wikipedia, "Sinc function",
           http://en.wikipedia.org/wiki/Sinc_function

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.linspace(-4, 4, 41)
    >>> mt.sinc(x).execute()
    array([ -3.89804309e-17,  -4.92362781e-02,  -8.40918587e-02,
            -8.90384387e-02,  -5.84680802e-02,   3.89804309e-17,
             6.68206631e-02,   1.16434881e-01,   1.26137788e-01,
             8.50444803e-02,  -3.89804309e-17,  -1.03943254e-01,
            -1.89206682e-01,  -2.16236208e-01,  -1.55914881e-01,
             3.89804309e-17,   2.33872321e-01,   5.04551152e-01,
             7.56826729e-01,   9.35489284e-01,   1.00000000e+00,
             9.35489284e-01,   7.56826729e-01,   5.04551152e-01,
             2.33872321e-01,   3.89804309e-17,  -1.55914881e-01,
            -2.16236208e-01,  -1.89206682e-01,  -1.03943254e-01,
            -3.89804309e-17,   8.50444803e-02,   1.26137788e-01,
             1.16434881e-01,   6.68206631e-02,   3.89804309e-17,
            -5.84680802e-02,  -8.90384387e-02,  -8.40918587e-02,
            -4.92362781e-02,  -3.89804309e-17])

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x.execute(), np.sinc(x).execute())
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title("Sinc Function")
    <matplotlib.text.Text object at 0x...>
    >>> plt.ylabel("Amplitude")
    <matplotlib.text.Text object at 0x...>
    >>> plt.xlabel("X")
    <matplotlib.text.Text object at 0x...>
    >>> plt.show()
    """
    op = TensorSinc(**kwargs)
    return op(x)
