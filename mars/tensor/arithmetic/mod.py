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


@arithmetic_operand(sparse_mode='binary_or')
class TensorMod(TensorBinOp):
    _op_type_ = OperandDef.MOD
    _func_name = 'mod'


@infer_dtype(np.mod)
def mod(x1, x2, out=None, where=None, **kwargs):
    """
    Return element-wise remainder of division.

    Computes the remainder complementary to the `floor_divide` function.  It is
    equivalent to the Python modulus operator``x1 % x2`` and has the same sign
    as the divisor `x2`. The MATLAB function equivalent to ``np.remainder``
    is ``mod``.

    .. warning::

        This should not be confused with:

        * Python 3.7's `math.remainder` and C's ``remainder``, which
          computes the IEEE remainder, which are the complement to
          ``round(x1 / x2)``.
        * The MATLAB ``rem`` function and or the C ``%`` operator which is the
          complement to ``int(x1 / x2)``.

    Parameters
    ----------
    x1 : array_like
        Dividend array.
    x2 : array_like
        Divisor array.
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
    y : Tensor
        The element-wise remainder of the quotient ``floor_divide(x1, x2)``.
        Returns a scalar if both  `x1` and `x2` are scalars.

    See Also
    --------
    floor_divide : Equivalent of Python ``//`` operator.
    divmod : Simultaneous floor division and remainder.
    fmod : Equivalent of the MATLAB ``rem`` function.
    divide, floor

    Notes
    -----
    Returns 0 when `x2` is 0 and both `x1` and `x2` are (tensors of)
    integers.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.remainder([4, 7], [2, 3]).execute()
    array([0, 1])
    >>> mt.remainder(mt.arange(7), 5).execute()
    array([0, 1, 2, 3, 4, 0, 1])
    """
    op = TensorMod(**kwargs)
    return op(x1, x2, out=out, where=where)


remainder = mod


@infer_dtype(np.mod, reverse=True)
def rmod(x1, x2, **kwargs):
    op = TensorMod(**kwargs)
    return op.rcall(x1, x2)
