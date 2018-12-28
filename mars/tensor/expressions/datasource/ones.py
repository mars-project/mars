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
from ....serialize import KeyField
from .core import TensorNoInput, TensorLike
from .array import tensor


class TensorOnes(TensorNoInput):
    _op_type_ = OperandDef.TENSOR_ONES

    def __init__(self, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype or 'f8')
        super(TensorOnes, self).__init__(_dtype=dtype, _gpu=gpu, **kw)


def ones(shape, dtype=None, chunk_size=None, gpu=False):
    """
    Return a new tensor of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new tensor, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the tensor, e.g., `mt.int8`.  Default is
        `mt.float64`.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default

    Returns
    -------
    out : Tensor
        Tensor of ones with the given shape, dtype, and order.

    See Also
    --------
    zeros, ones_like

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.ones(5).execute()
    array([ 1.,  1.,  1.,  1.,  1.])

    >>> mt.ones((5,), dtype=int).execute()
    array([1, 1, 1, 1, 1])

    >>> mt.ones((2, 1)).execute()
    array([[ 1.],
           [ 1.]])

    >>> s = (2,2)
    >>> mt.ones(s).execute()
    array([[ 1.,  1.],
           [ 1.,  1.]])

    """
    op = TensorOnes(dtype=dtype, gpu=gpu)
    return op(shape, chunk_size=chunk_size)


class TensorOnesLike(TensorLike):
    _op_type_ = OperandDef.TENSOR_ONES_LIKE

    _input = KeyField('input')

    def __init__(self, dtype=None, gpu=None, sparse=False, **kw):
        dtype = np.dtype(dtype) if dtype is not None else None
        super(TensorOnesLike, self).__init__(_dtype=dtype, _gpu=gpu, _sparse=sparse, **kw)


def ones_like(a, dtype=None, gpu=None):
    """
    Return a tensor of ones with the same shape and type as a given tensor.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned tensor.
    dtype : data-type, optional
        Overrides the data type of the result.
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default

    Returns
    -------
    out : Tensor
        Tensor of ones with the same shape and type as `a`.

    See Also
    --------
    zeros_like : Return a tensor of zeros with shape and type of input.
    empty_like : Return a empty tensor with shape and type of input.
    zeros : Return a new tensor setting values to zero.
    ones : Return a new tensor setting values to one.
    empty : Return a new uninitialized tensor.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x.execute()
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> mt.ones_like(x).execute()
    array([[1, 1, 1],
           [1, 1, 1]])

    >>> y = mt.arange(3, dtype=float)
    >>> y.execute()
    array([ 0.,  1.,  2.])
    >>> mt.ones_like(y).execute()
    array([ 1.,  1.,  1.])

    """
    a = tensor(a)
    op = TensorOnesLike(dtype=dtype, gpu=gpu, sparse=a.issparse())
    return op(a)
