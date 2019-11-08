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
class TensorLogAddExp2(TensorBinOp):
    _op_type_ = OperandDef.LOGADDEXP2
    _func_name = 'logaddexp2'


@infer_dtype(np.logaddexp2)
def logaddexp2(x1, x2, out=None, where=None, **kwargs):
    """
    Logarithm of the sum of exponentiations of the inputs in base-2.

    Calculates ``log2(2**x1 + 2**x2)``. This function is useful in machine
    learning when the calculated probabilities of events may be so small as
    to exceed the range of normal floating point numbers.  In such cases
    the base-2 logarithm of the calculated probability can be used instead.
    This function allows adding probabilities stored in such a fashion.

    Parameters
    ----------
    x1, x2 : array_like
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
    result : Tensor
        Base-2 logarithm of ``2**x1 + 2**x2``.

    See Also
    --------
    logaddexp: Logarithm of the sum of exponentiations of the inputs.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> prob1 = mt.log2(1e-50)
    >>> prob2 = mt.log2(2.5e-50)
    >>> prob12 = mt.logaddexp2(prob1, prob2)
    >>> prob1.execute(), prob2.execute(), prob12.execute()
    (-166.09640474436813, -164.77447664948076, -164.28904982231052)
    >>> (2**prob12).execute()
    3.4999999999999914e-50
    """
    op = TensorLogAddExp2(**kwargs)
    return op(x1, x2, out=out, where=where)
