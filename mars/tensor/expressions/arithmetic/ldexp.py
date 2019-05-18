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
from ..datasource import tensor as astensor
from .core import TensorBinOp, TensorConstant
from .utils import arithmetic_operand


@arithmetic_operand
class TensorLdexp(TensorBinOp):
    _op_type_ = OperandDef.LDEXP

    @classmethod
    def _is_sparse(cls, x1, x2):
        return x1.issparse()

    @classmethod
    def constant_cls(cls):
        return TensorLdexpConstant


@arithmetic_operand
class TensorLdexpConstant(TensorConstant):
    _op_type_ = OperandDef.LDEXP_CONSTANT

    @classmethod
    def _is_sparse(cls, x1, x2):
        if hasattr(x1, 'issparse') and x1.issparse():
            return True
        return False


@infer_dtype(np.ldexp)
def ldexp(x1, x2, out=None, where=None, **kwargs):
    """
    Returns x1 * 2**x2, element-wise.

    The mantissas `x1` and twos exponents `x2` are used to construct
    floating point numbers ``x1 * 2**x2``.

    Parameters
    ----------
    x1 : array_like
        Tensor of multipliers.
    x2 : array_like, int
        Tensor of twos exponents.
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
        The result of ``x1 * 2**x2``.

    See Also
    --------
    frexp : Return (y1, y2) from ``x = y1 * 2**y2``, inverse to `ldexp`.

    Notes
    -----
    Complex dtypes are not supported, they will raise a TypeError.

    `ldexp` is useful as the inverse of `frexp`, if used by itself it is
    more clear to simply use the expression ``x1 * 2**x2``.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.ldexp(5, mt.arange(4)).execute()
    array([  5.,  10.,  20.,  40.], dtype=float32)

    >>> x = mt.arange(6)
    >>> mt.ldexp(*mt.frexp(x)).execute()
    array([ 0.,  1.,  2.,  3.,  4.,  5.])
    """
    x2_dtype = astensor(x2).dtype
    casting = kwargs.get('casting', 'safe')
    if not np.can_cast(x2_dtype, np.int64, casting=casting):
        raise TypeError("ufunc 'ldexp' not supported for the input types, "
                        "and the inputs could not be safely coerced to any supported types "
                        "according to the casting rule ''{0}''".format(casting))

    op = TensorLdexp(**kwargs)
    return op(x1, x2, out=out, where=where)
