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
from .core import TensorUnaryOp
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='unary')
class TensorReal(TensorUnaryOp):
    _op_type_ = OperandDef.REAL
    _func_name = 'real'


@infer_dtype(np.real)
def real(val, **kwargs):
    """
    Return the real part of the complex argument.

    Parameters
    ----------
    val : array_like
        Input tensor.

    Returns
    -------
    out : Tensor or scalar
        The real component of the complex argument. If `val` is real, the type
        of `val` is used for the output.  If `val` has complex elements, the
        returned type is float.

    See Also
    --------
    real_if_close, imag, angle

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.array([1+2j, 3+4j, 5+6j])
    >>> a.real.execute()
    array([ 1.,  3.,  5.])
    >>> a.real = 9
    >>> a.execute()
    array([ 9.+2.j,  9.+4.j,  9.+6.j])
    >>> a.real = mt.array([9, 8, 7])
    >>> a.execute()
    array([ 9.+2.j,  8.+4.j,  7.+6.j])
    >>> mt.real(1 + 1j).execute()
    1.0

    """
    op = TensorReal(**kwargs)
    return op(val)
