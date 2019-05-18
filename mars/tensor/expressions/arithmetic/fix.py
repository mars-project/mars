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
class TensorFix(TensorUnaryOp):
    _op_type_ = OperandDef.FIX


@infer_dtype(np.fix)
def fix(x, out=None, **kwargs):
    """
    Round to nearest integer towards zero.

    Round a tensor of floats element-wise to nearest integer towards zero.
    The rounded values are returned as floats.

    Parameters
    ----------
    x   : array_like
        An tensor of floats to be rounded
    out : Tensor, optional
        Output tensor

    Returns
    -------
    out : Tensor of floats
        The array of rounded numbers

    See Also
    --------
    trunc, floor, ceil
    around : Round to given number of decimals

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.fix(3.14).execute()
    3.0
    >>> mt.fix(3).execute()
    3.0
    >>> mt.fix([2.1, 2.9, -2.1, -2.9]).execute()
    array([ 2.,  2., -2., -2.])

    """
    op = TensorFix(**kwargs)
    return op(x, out=out)
