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
from ..utils import inject_dtype
from .core import TensorUnaryOp
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='unary')
class TensorIsComplex(TensorUnaryOp):
    _op_type_ = OperandDef.ISCOMPLEX


@inject_dtype(np.bool_)
def iscomplex(x, **kwargs):
    """
    Returns a bool tensor, where True if input element is complex.

    What is tested is whether the input has a non-zero imaginary part, not if
    the input type is complex.

    Parameters
    ----------
    x : array_like
        Input tensor.

    Returns
    -------
    out : Tensor of bools
        Output tensor.

    See Also
    --------
    isreal
    iscomplexobj : Return True if x is a complex type or an array of complex
                   numbers.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.iscomplex([1+1j, 1+0j, 4.5, 3, 2, 2j]).execute()
    array([ True, False, False, False, False,  True])

    """
    op = TensorIsComplex(**kwargs)
    return op(x)
