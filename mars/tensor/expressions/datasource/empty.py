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


class TensorEmptyBase(object):
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super(TensorEmptyBase, self).__init__(*args, **kwargs)
        self._gen_rand()

    def _gen_rand(self):
        if getattr(self, '_rand', None) is None:
            self._obj_set('_rand', np.random.random())

    def to_chunk_op(self, *args):
        op = self.copy().reset_key()
        op._rand = None
        op._gen_rand()
        return op


class TensorEmpty(TensorEmptyBase, TensorNoInput):
    __slots__ = '_rand',
    _op_type_ = OperandDef.TENSOR_EMPTY

    def __init__(self, dtype=None, gpu=None, **kw):
        dtype = np.dtype(dtype or 'f8')
        super(TensorEmpty, self).__init__(_dtype=dtype, _gpu=gpu, **kw)


def empty(shape, dtype=None, chunk_size=None, gpu=False):
    """
    Return a new tensor of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty tensor
    dtype : data-type, optional
        Desired output data-type.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default

    Returns
    -------
    out : Tensor
        Tensor of uninitialized (arbitrary) data of the given shape, dtype, and
        order.  Object arrays will be initialized to None.

    See Also
    --------
    empty_like, zeros, ones

    Notes
    -----
    `empty`, unlike `zeros`, does not set the array values to zero,
    and may therefore be marginally faster.  On the other hand, it requires
    the user to manually set all the values in the array, and should be
    used with caution.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.empty([2, 2]).execute()
    array([[ -9.74499359e+001,   6.69583040e-309],
           [  2.13182611e-314,   3.06959433e-309]])         #random

    >>> mt.empty([2, 2], dtype=int).execute()
    array([[-1073741821, -1067949133],
           [  496041986,    19249760]])                     #random
    """
    op = TensorEmpty(dtype=dtype, gpu=gpu)
    return op(shape, chunk_size=chunk_size)


class TensorEmptyLike(TensorEmptyBase, TensorLike):
    __slots__ = '_rand',
    _op_type_ = OperandDef.TENSOR_EMPTY_LIKE

    _input = KeyField('input')

    def __init__(self, dtype=None, gpu=None, sparse=False, **kw):
        dtype = np.dtype(dtype) if dtype is not None else None
        super(TensorEmptyLike, self).__init__(_dtype=dtype, _gpu=gpu,
                                              _sparse=sparse, **kw)


def empty_like(a, dtype=None, gpu=None):
    """
    Return a new tensor with the same shape and type as a given tensor.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of the
        returned tensor.
    dtype : data-type, optional
        Overrides the data type of the result.
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default

    Returns
    -------
    out : Tensor
        Array of uninitialized (arbitrary) data with the same
        shape and type as `a`.

    See Also
    --------
    ones_like : Return a tensor of ones with shape and type of input.
    zeros_like : Return a tensor of zeros with shape and type of input.
    empty : Return a new uninitialized tensor.
    ones : Return a new tensor setting values to one.
    zeros : Return a new tensor setting values to zero.

    Notes
    -----
    This function does *not* initialize the returned tensor; to do that use
    `zeros_like` or `ones_like` instead.  It may be marginally faster than
    the functions that do set the array values.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = ([1,2,3], [4,5,6])                         # a is array-like
    >>> mt.empty_like(a).execute()
    array([[-1073741821, -1073741821,           3],    #ranm
           [          0,           0, -1073741821]])
    >>> a = mt.array([[1., 2., 3.],[4.,5.,6.]])
    >>> mt.empty_like(a).execute()
    array([[ -2.00000715e+000,   1.48219694e-323,  -2.00000572e+000],#random
           [  4.38791518e-305,  -2.00000715e+000,   4.17269252e-309]])
    """
    a = tensor(a)
    op = TensorEmptyLike(dtype=dtype, gpu=gpu, sparse=a.issparse())
    return op(a)
