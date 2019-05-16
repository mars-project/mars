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
from .core import TensorBinOp, TensorConstant
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='binary_and')
class TensorNextafter(TensorBinOp):
    _op_type_ = OperandDef.NEXTAFTER

    @classmethod
    def constant_cls(cls):
        return TensorNextafterConstant


@arithmetic_operand(sparse_mode='binary_and_const')
class TensorNextafterConstant(TensorConstant):
    _op_type_ = OperandDef.NEXTAFTER_CONSTANT


@infer_dtype(np.nextafter)
def nextafter(x1, x2, out=None, where=None, **kwargs):
    """
    Return the next floating-point value after x1 towards x2, element-wise.

    Parameters
    ----------
    x1 : array_like
        Values to find the next representable value of.
    x2 : array_like
        The direction where to look for the next representable value of `x1`.
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
        The next representable values of `x1` in the direction of `x2`.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> eps = mt.finfo(mt.float64).eps
    >>> (mt.nextafter(1, 2) == eps + 1).execute()
    True
    >>> (mt.nextafter([1, 2], [2, 1]) == [eps + 1, 2 - eps]).execute()
    array([ True,  True])
    """
    op = TensorNextafter(**kwargs)
    return op(x1, x2, out=out, where=where)
