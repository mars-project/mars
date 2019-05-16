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
class TensorRadians(TensorUnaryOp):
    _op_type_ = OperandDef.RADIANS


@infer_dtype(np.radians)
def radians(x, out=None, where=None, **kwargs):
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    x : array_like
        Input tensor in degrees.
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
        The corresponding radian values.

    See Also
    --------
    deg2rad : equivalent function

    Examples
    --------
    Convert a degree array to radians

    >>> import mars.tensor as mt

    >>> deg = mt.arange(12.) * 30.
    >>> mt.radians(deg).execute()
    array([ 0.        ,  0.52359878,  1.04719755,  1.57079633,  2.0943951 ,
            2.61799388,  3.14159265,  3.66519143,  4.1887902 ,  4.71238898,
            5.23598776,  5.75958653])

    >>> out = mt.zeros((deg.shape))
    >>> ret = mt.radians(deg, out)
    >>> ret is out
    True
    """
    op = TensorRadians(**kwargs)
    return op(x, out=out, where=where)
