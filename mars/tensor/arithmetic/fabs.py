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

from ... import opcodes as OperandDef
from .core import TensorUnaryOp
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='unary')
class TensorFabs(TensorUnaryOp):
    _op_type_ = OperandDef.FABS
    _func_name = 'fabs'


def fabs(x, out=None, where=None, **kwargs):
    """
    Compute the absolute values element-wise.

    This function returns the absolute values (positive magnitude) of the
    data in `x`. Complex values are not handled, use `absolute` to find the
    absolute values of complex data.

    Parameters
    ----------
    x : array_like
        The tensor of numbers for which the absolute values are required. If
        `x` is a scalar, the result `y` will also be a scalar.
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
    y : Tensor or scalar
        The absolute values of `x`, the returned values are always floats.

    See Also
    --------
    absolute : Absolute values including `complex` types.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.fabs(-1).execute()
    1.0
    >>> mt.fabs([-1.2, 1.2]).execute()
    array([ 1.2,  1.2])
    """
    op = TensorFabs(**kwargs)
    return op(x, out=out, where=where)
