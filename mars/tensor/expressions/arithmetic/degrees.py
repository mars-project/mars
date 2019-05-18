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
class TensorDegrees(TensorUnaryOp):
    _op_type_ = OperandDef.DEGREES


@infer_dtype(np.degrees)
def degrees(x, out=None, where=None, **kwargs):
    """
    Convert angles from radians to degrees.

    Parameters
    ----------
    x : array_like
        Input tensor in radians.
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
    y : Tensor of floats
        The corresponding degree values; if `out` was supplied this is a
        reference to it.

    See Also
    --------
    rad2deg : equivalent function

    Examples
    --------
    Convert a radian array to degrees

    >>> import mars.tensor as mt

    >>> rad = mt.arange(12.)*mt.pi/6
    >>> mt.degrees(rad).execute()
    array([   0.,   30.,   60.,   90.,  120.,  150.,  180.,  210.,  240.,
            270.,  300.,  330.])

    >>> out = mt.zeros((rad.shape))
    >>> r = mt.degrees(out)
    >>> mt.all(r == out).execute()
    True
    """
    op = TensorDegrees(**kwargs)
    return op(x, out=out, where=where)
