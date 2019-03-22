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

from .... import operands
from ..utils import infer_dtype
from .core import TensorBinOp, TensorConstant
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='binary_or')
class TensorBitxor(operands.Bitxor, TensorBinOp):
    @classmethod
    def constant_cls(cls):
        return TensorBitxorConstant


@arithmetic_operand(sparse_mode='binary_or_const')
class TensorBitxorConstant(operands.BitxorConstant, TensorConstant):
    pass


@infer_dtype(np.bitwise_xor)
def bitxor(x1, x2, out=None, where=None, **kwargs):
    """
    Compute the bit-wise XOR of two arrays element-wise.

    Computes the bit-wise XOR of the underlying binary representation of
    the integers in the input arrays. This ufunc implements the C/Python
    operator ``^``.

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
    logical_xor
    bitwise_and
    bitwise_or
    binary_repr :
        Return the binary representation of the input number as a string.

    Examples
    --------
    The number 13 is represented by ``00001101``. Likewise, 17 is
    represented by ``00010001``.  The bit-wise XOR of 13 and 17 is
    therefore ``00011100``, or 28:

    >>> import mars.tensor as mt

    >>> mt.bitwise_xor(13, 17).execute()
    28

    >>> mt.bitwise_xor(31, 5).execute()
    26
    >>> mt.bitwise_xor([31,3], 5).execute()
    array([26,  6])

    >>> mt.bitwise_xor([31,3], [5,6]).execute()
    array([26,  5])
    >>> mt.bitwise_xor([True, True], [False, True]).execute()
    array([ True, False])
    """
    op = TensorBitxor(**kwargs)
    return op(x1, x2, out=out, where=where)


@infer_dtype(np.bitwise_xor, reverse=True)
def rbitxor(x1, x2, **kwargs):
    op = TensorBitxor(**kwargs)
    return op.rcall(x1, x2)
