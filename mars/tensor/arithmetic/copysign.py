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
from .core import TensorBinOp
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='always_false')
class TensorCopysign(TensorBinOp):
    _op_type_ = OperandDef.COPYSIGN
    _func_name = 'copysign'


@infer_dtype(np.copysign)
def copysign(x1, x2, out=None, where=None, **kwargs):
    """
    Change the sign of x1 to that of x2, element-wise.

    If both arguments are arrays or sequences, they have to be of the same
    length. If `x2` is a scalar, its sign will be copied to all elements of
    `x1`.

    Parameters
    ----------
    x1 : array_like
        Values to change the sign of.
    x2 : array_like
        The sign of `x2` is copied to `x1`.
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
    out : array_like
        The values of `x1` with the sign of `x2`.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.copysign(1.3, -1).execute()
    -1.3
    >>> (1/mt.copysign(0, 1)).execute()
    inf
    >>> (1/mt.copysign(0, -1)).execute()
    -inf

    >>> mt.copysign([-1, 0, 1], -1.1).execute()
    array([-1., -0., -1.])
    >>> mt.copysign([-1, 0, 1], mt.arange(3)-1).execute()
    array([-1.,  0.,  1.])
    """
    op = TensorCopysign(**kwargs)
    return op(x1, x2, out=out, where=where)
