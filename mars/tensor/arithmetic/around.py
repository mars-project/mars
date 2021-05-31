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
from ...serialization.serializables import Int32Field
from ..array_utils import device, as_same_device
from ..datasource import tensor as astensor
from .core import TensorUnaryOp
from .utils import arithmetic_operand


@arithmetic_operand(init=False, sparse_mode='unary')
class TensorAround(TensorUnaryOp):
    _op_type_ = OperandDef.AROUND

    _decimals = Int32Field('decimals')
    _func_name = 'around'

    @property
    def decimals(self):
        return self._decimals

    def __init__(self, decimals=None, casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super().__init__(_decimals=decimals, _casting=casting, _err=err, _dtype=dtype,
                         _sparse=sparse, **kw)

    @property
    def ufunc_extra_params(self):
        return {'decimals': self._decimals}

    @classmethod
    def execute(cls, ctx, op):
        (a,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            ctx[op.outputs[0].key] = xp.around(a, decimals=op.decimals)


def around(a, decimals=0, out=None):
    """
    Evenly round to the given number of decimals.

    Parameters
    ----------
    a : array_like
        Input data.
    decimals : int, optional
        Number of decimal places to round to (default: 0).  If
        decimals is negative, it specifies the number of positions to
        the left of the decimal point.
    out : Tensor, optional
        Alternative output tensor in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary.

    Returns
    -------
    rounded_array : Tensor
        An tensor of the same type as `a`, containing the rounded values.
        Unless `out` was specified, a new tensor is created.  A reference to
        the result is returned.

        The real and imaginary parts of complex numbers are rounded
        separately.  The result of rounding a float is a float.

    See Also
    --------
    Tensor.round : equivalent method

    ceil, fix, floor, rint, trunc


    Notes
    -----
    For values exactly halfway between rounded decimal values, NumPy
    rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0,
    -0.5 and 0.5 round to 0.0, etc. Results may also be surprising due
    to the inexact representation of decimal fractions in the IEEE
    floating point standard [1]_ and errors introduced when scaling
    by powers of ten.

    References
    ----------
    .. [1] "Lecture Notes on the Status of  IEEE 754", William Kahan,
           http://www.cs.berkeley.edu/~wkahan/ieee754status/IEEE754.PDF
    .. [2] "How Futile are Mindless Assessments of
           Roundoff in Floating-Point Computation?", William Kahan,
           http://www.cs.berkeley.edu/~wkahan/Mindless.pdf

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.around([0.37, 1.64]).execute()
    array([ 0.,  2.])
    >>> mt.around([0.37, 1.64], decimals=1).execute()
    array([ 0.4,  1.6])
    >>> mt.around([.5, 1.5, 2.5, 3.5, 4.5]).execute() # rounds to nearest even value
    array([ 0.,  2.,  2.,  4.,  4.])
    >>> mt.around([1,2,3,11], decimals=1).execute() # tensor of ints is returned
    array([ 1,  2,  3, 11])
    >>> mt.around([1,2,3,11], decimals=-1).execute()
    array([ 0,  0,  0, 10])

    """
    dtype = astensor(a).dtype
    op = TensorAround(decimals=decimals, dtype=dtype)
    return op(a, out=out)


round_ = around
