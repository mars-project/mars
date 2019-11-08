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
from ..utils import infer_dtype
from .core import TensorUnaryOp
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='unary')
class TensorExp(TensorUnaryOp):
    _op_type_ = OperandDef.EXP
    _func_name = 'exp'


@infer_dtype(np.exp)
def exp(x, out=None, where=None, **kwargs):
    r"""
    Calculate the exponential of all elements in the input tensor.

    Parameters
    ----------
    x : array_like
        Input values.
    out : Tensor, None, or tuple of Tensor and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or `None`,
        a freshly-allocated tensor is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        Values of True indicate to calculate the ufunc at that position, values
        of False indicate to leave the value in the output alone.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : Tensor
        Output tensor, element-wise exponential of `x`.

    See Also
    --------
    expm1 : Calculate ``exp(x) - 1`` for all elements in the array.
    exp2  : Calculate ``2**x`` for all elements in the array.

    Notes
    -----
    The irrational number ``e`` is also known as Euler's number.  It is
    approximately 2.718281, and is the base of the natural logarithm,
    ``ln`` (this means that, if :math:`x = \ln y = \log_e y`,
    then :math:`e^x = y`. For real input, ``exp(x)`` is always positive.

    For complex arguments, ``x = a + ib``, we can write
    :math:`e^x = e^a e^{ib}`.  The first term, :math:`e^a`, is already
    known (it is the real argument, described above).  The second term,
    :math:`e^{ib}`, is :math:`\cos b + i \sin b`, a function with
    magnitude 1 and a periodic phase.

    References
    ----------
    .. [1] Wikipedia, "Exponential function",
           http://en.wikipedia.org/wiki/Exponential_function
    .. [2] M. Abramovitz and I. A. Stegun, "Handbook of Mathematical Functions
           with Formulas, Graphs, and Mathematical Tables," Dover, 1964, p. 69,
           http://www.math.sfu.ca/~cbm/aands/page_69.htm

    Examples
    --------
    Plot the magnitude and phase of ``exp(x)`` in the complex plane:

    >>> import mars.tensor as mt
    >>> import matplotlib.pyplot as plt

    >>> x = mt.linspace(-2*mt.pi, 2*mt.pi, 100)
    >>> xx = x + 1j * x[:, mt.newaxis] # a + ib over complex plane
    >>> out = mt.exp(xx)

    >>> plt.subplot(121)
    >>> plt.imshow(mt.abs(out).execute(),
    ...            extent=[-2*mt.pi, 2*mt.pi, -2*mt.pi, 2*mt.pi], cmap='gray')
    >>> plt.title('Magnitude of exp(x)')

    >>> plt.subplot(122)
    >>> plt.imshow(mt.angle(out).execute(),
    ...            extent=[-2*mt.pi, 2*mt.pi, -2*mt.pi, 2*mt.pi], cmap='hsv')
    >>> plt.title('Phase (angle) of exp(x)')
    >>> plt.show()
    """
    op = TensorExp(**kwargs)
    return op(x, out=out, where=where)
