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
from ..utils import infer_dtype
from .core import TensorBinOp
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='always_false')
class TensorLogAddExp(TensorBinOp):
    _op_type_ = OperandDef.LOGADDEXP
    _func_name = 'logaddexp'


@infer_dtype(np.logaddexp)
def logaddexp(x1, x2, out=None, where=None, **kwargs):
    """
    Logarithm of the sum of exponentiations of the inputs.

    Calculates ``log(exp(x1) + exp(x2))``. This function is useful in
    statistics where the calculated probabilities of events may be so small
    as to exceed the range of normal floating point numbers.  In such cases
    the logarithm of the calculated probability is stored. This function
    allows adding probabilities stored in such a fashion.

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
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    result : Tensor
        Logarithm of ``exp(x1) + exp(x2)``.

    See Also
    --------
    logaddexp2: Logarithm of the sum of exponentiations of inputs in base 2.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> prob1 = mt.log(1e-50)
    >>> prob2 = mt.log(2.5e-50)
    >>> prob12 = mt.logaddexp(prob1, prob2)
    >>> prob12.execute()
    -113.87649168120691
    >>> mt.exp(prob12).execute()
    3.5000000000000057e-50
    """
    op = TensorLogAddExp(**kwargs)
    return op(x1, x2, out=out, where=where)
