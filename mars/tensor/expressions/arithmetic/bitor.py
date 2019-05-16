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


@arithmetic_operand(sparse_mode='binary_or')
class TensorBitor(TensorBinOp):
    _op_type_ = OperandDef.BITOR

    @classmethod
    def constant_cls(cls):
        return TensorBitorConstant


@arithmetic_operand(sparse_mode='binary_or_const')
class TensorBitorConstant(TensorConstant):
    _op_type_ = OperandDef.BITOR_CONSTANT


@infer_dtype(np.bitwise_or)
def bitor(x1, x2, out=None, where=None, **kwargs):
    """
    Compute the bit-wise OR of two tensors element-wise.

    Computes the bit-wise OR of the underlying binary representation of
    the integers in the input arrays. This ufunc implements the C/Python
    operator ``|``.

    Parameters
    ----------
    x1, x2 : array_like
        Only integer and boolean types are handled.
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
        Result.

    See Also
    --------
    logical_or
    bitwise_and
    bitwise_xor
    binary_repr :
        Return the binary representation of the input number as a string.

    Examples
    --------
    The number 13 has the binaray representation ``00001101``. Likewise,
    16 is represented by ``00010000``.  The bit-wise OR of 13 and 16 is
    then ``000111011``, or 29:

    >>> import mars.tensor as mt

    >>> mt.bitwise_or(13, 16).execute()
    29

    >>> mt.bitwise_or(32, 2).execute()
    34
    >>> mt.bitwise_or([33, 4], 1).execute()
    array([33,  5])
    >>> mt.bitwise_or([33, 4], [1, 2]).execute()
    array([33,  6])

    >>> mt.bitwise_or(mt.array([2, 5, 255]), mt.array([4, 4, 4])).execute()
    array([  6,   5, 255])
    >>> (mt.array([2, 5, 255]) | mt.array([4, 4, 4])).execute()
    array([  6,   5, 255])
    >>> mt.bitwise_or(mt.array([2, 5, 255, 2147483647], dtype=mt.int32),
    ...               mt.array([4, 4, 4, 2147483647], dtype=mt.int32)).execute()
    array([         6,          5,        255, 2147483647])
    >>> mt.bitwise_or([True, True], [False, True]).execute()
    array([ True,  True])
    """
    op = TensorBitor(**kwargs)
    return op(x1, x2, out=out, where=where)


@infer_dtype(np.bitwise_or, reverse=True)
def rbitor(x1, x2, **kwargs):
    op = TensorBitor(**kwargs)
    return op.rcall(x1, x2)
