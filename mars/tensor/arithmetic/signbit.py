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
from ..utils import inject_dtype
from .core import TensorUnaryOp
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='unary')
class TensorSignbit(TensorUnaryOp):
    _op_type_ = OperandDef.SIGNBIT
    _func_name = 'signbit'


@inject_dtype(np.bool_)
def signbit(x, out=None, where=None, **kwargs):
    """
    Returns element-wise True where signbit is set (less than zero).

    Parameters
    ----------
    x : array_like
        The input value(s).
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
    result : Tensor of bool
        Output tensor, or reference to `out` if that was supplied.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.signbit(-1.2).execute()
    True
    >>> mt.signbit(mt.array([1, -2.3, 2.1])).execute()
    array([False,  True, False])
    """
    op = TensorSignbit(**kwargs)
    return op(x, out=out, where=where)
