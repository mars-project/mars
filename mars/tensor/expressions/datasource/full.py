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
from ....serialize import AnyField
from .core import TensorNoInput
from .array import tensor


class TensorFull(TensorNoInput):
    _op_type_ = OperandDef.TENSOR_FULL

    _fill_value = AnyField('fill_value')

    def __init__(self, fill_value=None, dtype=None, gpu=None, **kw):
        if dtype is not None:
            dtype = np.dtype(dtype)
            if fill_value is not None:
                fill_value = dtype.type(fill_value)
        elif fill_value is not None:
            dtype = np.array(fill_value).dtype
        super(TensorFull, self).__init__(_fill_value=fill_value, _dtype=dtype, _gpu=gpu, **kw)

    @property
    def fill_value(self):
        return self._fill_value


def full(shape, fill_value, dtype=None, chunk_size=None, gpu=False):
    """
    Return a new tensor of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new tensor, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        The desired data-type for the tensor  The default, `None`, means
         `np.array(fill_value).dtype`.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default

    Returns
    -------
    out : Tensor
        Tensor of `fill_value` with the given shape, dtype, and order.

    See Also
    --------
    zeros_like : Return a tensor of zeros with shape and type of input.
    ones_like : Return a tensor of ones with shape and type of input.
    empty_like : Return an empty tensor with shape and type of input.
    full_like : Fill a tensor with shape and type of input.
    zeros : Return a new tensor setting values to zero.
    ones : Return a new tensor setting values to one.
    empty : Return a new uninitialized tensor.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.full((2, 2), mt.inf).execute()
    array([[ inf,  inf],
           [ inf,  inf]])
    >>> mt.full((2, 2), 10).execute()
    array([[10, 10],
           [10, 10]])

    """
    v = np.asarray(fill_value)
    if len(v.shape) > 0:
        from ..base import broadcast_to
        return broadcast_to(tensor(v, dtype=dtype, chunk_size=chunk_size, gpu=gpu), shape)

    op = TensorFull(fill_value, dtype=dtype, gpu=gpu)
    return op(shape, chunk_size=chunk_size)
