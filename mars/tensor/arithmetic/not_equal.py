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
from ..utils import inject_dtype
from .core import TensorBinOp
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='binary_and')
class TensorNotEqual(TensorBinOp):
    _op_type_ = OperandDef.NE
    _func_name = 'not_equal'


@inject_dtype(np.bool_)
def not_equal(x1, x2, out=None, where=None, **kwargs):
    """
    Return (x1 != x2) element-wise.

    Parameters
    ----------
    x1, x2 : array_like
      Input tensors.
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
    not_equal : tensor bool, scalar bool
      For each element in `x1, x2`, return True if `x1` is not equal
      to `x2` and False otherwise.


    See Also
    --------
    equal, greater, greater_equal, less, less_equal

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.not_equal([1.,2.], [1., 3.]).execute()
    array([False,  True])
    >>> mt.not_equal([1, 2], [[1, 3],[1, 4]]).execute()
    array([[False,  True],
           [False,  True]])
    """
    op = TensorNotEqual(**kwargs)
    return op(x1, x2, out=out, where=where)
