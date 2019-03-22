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
from .core import TensorUnaryOp
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='always_false')
class TensorSpacing(operands.Spacing, TensorUnaryOp):
    pass


@infer_dtype(np.spacing)
def spacing(x, out=None, where=None, **kwargs):
    """
    Return the distance between x and the nearest adjacent number.

    Parameters
    ----------
    x : array_like
        Values to find the spacing of.
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
        The spacing of values of `x1`.

    Notes
    -----
    It can be considered as a generalization of EPS:
    ``spacing(mt.float64(1)) == mt.finfo(mt.float64).eps``, and there
    should not be any representable number between ``x + spacing(x)`` and
    x for any finite x.

    Spacing of +- inf and NaN is NaN.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> (mt.spacing(1) == mt.finfo(mt.float64).eps).execute()
    True
    """
    op = TensorSpacing(**kwargs)
    return op(x, out=out, where=where)
