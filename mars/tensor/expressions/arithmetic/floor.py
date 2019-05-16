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
class TensorFloor(TensorUnaryOp):
    _op_type_ = OperandDef.FLOOR


@infer_dtype(np.floor)
def floor(x, out=None, where=None, **kwargs):
    r"""
    Return the floor of the input, element-wise.

    The floor of the scalar `x` is the largest integer `i`, such that
    `i <= x`.  It is often denoted as :math:`\lfloor x \rfloor`.

    Parameters
    ----------
    x : array_like
        Input data.
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
    y : Tensor or scalar
        The floor of each element in `x`.

    See Also
    --------
    ceil, trunc, rint

    Notes
    -----
    Some spreadsheet programs calculate the "floor-towards-zero", in other
    words ``floor(-2.5) == -2``.  NumPy instead uses the definition of
    `floor` where `floor(-2.5) == -3`.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> mt.floor(a).execute()
    array([-2., -2., -1.,  0.,  1.,  1.,  2.])
    """
    op = TensorFloor(**kwargs)
    return op(x, out=out, where=where)
