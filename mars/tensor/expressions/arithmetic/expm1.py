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
class TensorExpm1(TensorUnaryOp):
    _op_type_ = OperandDef.EXPM1


@infer_dtype(np.expm1)
def expm1(x, out=None, where=None, **kwargs):
    """
    Calculate ``exp(x) - 1`` for all elements in the tensor.

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

    Returns
    -------
    out : Tensor
        Element-wise exponential minus one: ``out = exp(x) - 1``.

    See Also
    --------
    log1p : ``log(1 + x)``, the inverse of expm1.


    Notes
    -----
    This function provides greater precision than ``exp(x) - 1``
    for small values of ``x``.

    Examples
    --------
    The true value of ``exp(1e-10) - 1`` is ``1.00000000005e-10`` to
    about 32 significant digits. This example shows the superiority of
    expm1 in this case.

    >>> import mars.tensor as mt

    >>> mt.expm1(1e-10).execute()
    1.00000000005e-10
    >>> (mt.exp(1e-10) - 1).execute()
    1.000000082740371e-10
    """
    op = TensorExpm1(**kwargs)
    return op(x, out=out, where=where)
