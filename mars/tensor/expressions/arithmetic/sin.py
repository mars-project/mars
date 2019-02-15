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
class TensorSin(operands.Sin, TensorUnaryOp):
    pass


@infer_dtype(np.sin)
def sin(x, out=None, where=None, **kwargs):
    """
    Trigonometric sine, element-wise.

    Parameters
    ----------
    x : array_like
        Angle, in radians (:math:`2 \pi` rad equals 360 degrees).
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
    y : array_like
        The sine of each element of x.

    See Also
    --------
    arcsin, sinh, cos

    Notes
    -----
    The sine is one of the fundamental functions of trigonometry (the
    mathematical study of triangles).  Consider a circle of radius 1
    centered on the origin.  A ray comes in from the :math:`+x` axis, makes
    an angle at the origin (measured counter-clockwise from that axis), and
    departs from the origin.  The :math:`y` coordinate of the outgoing
    ray's intersection with the unit circle is the sine of that angle.  It
    ranges from -1 for :math:`x=3\pi / 2` to +1 for :math:`\pi / 2.`  The
    function has zeroes where the angle is a multiple of :math:`\pi`.
    Sines of angles between :math:`\pi` and :math:`2\pi` are negative.
    The numerous properties of the sine and related functions are included
    in any standard trigonometry text.

    Examples
    --------
    Print sine of one angle:

    >>> import mars.tensor as mt

    >>> mt.sin(mt.pi/2.).execute()
    1.0

    Print sines of an array of angles given in degrees:

    >>> mt.sin(mt.array((0., 30., 45., 60., 90.)) * mt.pi / 180. ).execute()
    array([ 0.        ,  0.5       ,  0.70710678,  0.8660254 ,  1.        ])

    Plot the sine function:

    >>> import matplotlib.pylab as plt
    >>> x = mt.linspace(-mt.pi, mt.pi, 201)
    >>> plt.plot(x.execute(), mt.sin(x).execute())
    >>> plt.xlabel('Angle [rad]')
    >>> plt.ylabel('sin(x)')
    >>> plt.axis('tight')
    >>> plt.show()
    """
    op = TensorSin(**kwargs)
    return op(x, out=out, where=where)
