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
from ..core import Tensor
from .core import TensorUnaryOp
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='unary')
class TensorNanToNum(TensorUnaryOp):
    _op_type_ = OperandDef.NAN_TO_NUM
    _func_name = 'nan_to_num'


@infer_dtype(np.nan_to_num)
def nan_to_num(x, copy=True, **kwargs):
    """
    Replace nan with zero and inf with large finite numbers.

    If `x` is inexact, NaN is replaced by zero, and infinity and -infinity
    replaced by the respectively largest and most negative finite floating
    point values representable by ``x.dtype``.

    For complex dtypes, the above is applied to each of the real and
    imaginary components of `x` separately.

    If `x` is not inexact, then no replacements are made.

    Parameters
    ----------
    x : array_like
        Input data.
    copy : bool, optional
        Whether to create a copy of `x` (True) or to replace values
        in-place (False). The in-place operation only occurs if
        casting to an array does not require a copy.
        Default is True.

    Returns
    -------
    out : Tensor
        `x`, with the non-finite values replaced. If `copy` is False, this may
        be `x` itself.

    See Also
    --------
    isinf : Shows which elements are positive or negative infinity.
    isneginf : Shows which elements are negative infinity.
    isposinf : Shows which elements are positive infinity.
    isnan : Shows which elements are Not a Number (NaN).
    isfinite : Shows which elements are finite (not NaN, not infinity)

    Notes
    -----
    Mars uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.array([mt.inf, -mt.inf, mt.nan, -128, 128])
    >>> mt.nan_to_num(x).execute()
    array([  1.79769313e+308,  -1.79769313e+308,   0.00000000e+000,
            -1.28000000e+002,   1.28000000e+002])
    >>> y = mt.array([complex(mt.inf, mt.nan), mt.nan, complex(mt.nan, mt.inf)])
    >>> mt.nan_to_num(y).execute()
    array([  1.79769313e+308 +0.00000000e+000j,
             0.00000000e+000 +0.00000000e+000j,
             0.00000000e+000 +1.79769313e+308j])
    """
    op = TensorNanToNum(**kwargs)
    ret = op(x)

    if copy:
        return ret

    # set back, make sure x is a Tensor
    if not isinstance(x, Tensor):
        raise ValueError(f'`x` must be a Tensor, got {type(x)} instead')
    x.data = ret.data
    return x
