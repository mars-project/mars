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
class TensorIsNan(TensorUnaryOp):
    _op_type_ = OperandDef.ISNAN
    _func_name = 'isnan'


@inject_dtype(np.bool_)
def isnan(x, out=None, where=None, **kwargs):
    """
    Test element-wise for NaN and return result as a boolean tensor.

    Parameters
    ----------
    x : array_like
        Input tensor.
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
    y : Tensor or bool
        For scalar input, the result is a new boolean with value True if
        the input is NaN; otherwise the value is False.

        For array input, the result is a boolean tensor of the same
        dimensions as the input and the values are True if the
        corresponding element of the input is NaN; otherwise the values are
        False.

    See Also
    --------
    isinf, isneginf, isposinf, isfinite, isnat

    Notes
    -----
    Mars uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.isnan(mt.nan).execute()
    True
    >>> mt.isnan(mt.inf).execute()
    False
    >>> mt.isnan([mt.log(-1.).execute(),1.,mt.log(0).execute()]).execute()
    array([ True, False, False])
    """
    op = TensorIsNan(**kwargs)
    return op(x, out=out, where=where)
