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
class TensorDeg2rad(operands.Deg2rad, TensorUnaryOp):
    pass


@infer_dtype(np.deg2rad)
def deg2rad(x, out=None, where=None, **kwargs):
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    x : array_like
        Angles in degrees.
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
        The corresponding angle in radians.

    See Also
    --------
    rad2deg : Convert angles from radians to degrees.
    unwrap : Remove large jumps in angle by wrapping.

    Notes
    -----
    ``deg2rad(x)`` is ``x * pi / 180``.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.deg2rad(180).execute()
    3.1415926535897931
    """
    op = TensorDeg2rad(**kwargs)
    return op(x, out=out, where=where)
