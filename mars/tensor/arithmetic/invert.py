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
from .core import TensorUnaryOp
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='unary')
class TensorInvert(TensorUnaryOp):
    _op_type_ = OperandDef.INVERT
    _func_name = 'invert'


@infer_dtype(np.invert)
def invert(x, out=None, where=None, **kwargs):
    """
    Compute bit-wise inversion, or bit-wise NOT, element-wise.

    Computes the bit-wise NOT of the underlying binary representation of
    the integers in the input tensors. This ufunc implements the C/Python
    operator ``~``.

    For signed integer inputs, the two's complement is returned.  In a
    two's-complement system negative numbers are represented by the two's
    complement of the absolute value. This is the most common method of
    representing signed integers on computers [1]_. A N-bit
    two's-complement system can represent every integer in the range
    :math:`-2^{N-1}` to :math:`+2^{N-1}-1`.

    Parameters
    ----------
    x : array_like
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
    bitwise_and, bitwise_or, bitwise_xor
    logical_not

    Notes
    -----
    `bitwise_not` is an alias for `invert`:

    >>> import mars.tensor as mt

    >>> mt.bitwise_not is mt.invert
    True

    References
    ----------
    .. [1] Wikipedia, "Two's complement",
        http://en.wikipedia.org/wiki/Two's_complement

    Examples
    --------
    We've seen that 13 is represented by ``00001101``.
    The invert or bit-wise NOT of 13 is then:

    >>> mt.invert(mt.array([13], dtype=mt.uint8)).execute()
    array([242], dtype=uint8)

    The result depends on the bit-width:

    >>> mt.invert(mt.array([13], dtype=mt.uint16)).execute()
    array([65522], dtype=uint16)

    When using signed integer types the result is the two's complement of
    the result for the unsigned type:

    >>> mt.invert(mt.array([13], dtype=mt.int8)).execute()
    array([-14], dtype=int8)

    Booleans are accepted as well:

    >>> mt.invert(mt.array([True, False])).execute()
    array([False,  True])
    """
    op = TensorInvert(**kwargs)
    return op(x, out=out, where=where)
