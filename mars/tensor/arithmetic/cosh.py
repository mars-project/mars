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
class TensorCosh(TensorUnaryOp):
    _op_type_ = OperandDef.COSH
    _func_name = 'cosh'


@infer_dtype(np.cosh)
def cosh(x, out=None, where=None, **kwargs):
    """
    Hyperbolic cosine, element-wise.

    Equivalent to ``1/2 * (mt.exp(x) + mt.exp(-x))`` and ``mt.cos(1j*x)``.

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
    out : Tensor
        Output array of same shape as `x`.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.cosh(0).execute()
    1.0

    The hyperbolic cosine describes the shape of a hanging cable:

    >>> import matplotlib.pyplot as plt
    >>> x = mt.linspace(-4, 4, 1000)
    >>> plt.plot(x.execute(), mt.cosh(x).execute())
    >>> plt.show()
    """
    op = TensorCosh(**kwargs)
    return op(x, out=out, where=where)
