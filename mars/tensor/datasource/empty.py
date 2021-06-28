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
from ...serialization.serializables import KeyField, StringField
from ...lib.sparse import SparseNDArray
from ...lib.sparse.core import naked, get_array_module, get_sparse_module
from ..array_utils import create_array
from ..utils import get_order
from .core import TensorNoInput, TensorLike
from .array import tensor


class TensorEmptyBase(object):
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    _order = StringField('order')

    def __init__(self, dtype=None, order=None, **kw):
        dtype = np.dtype(dtype or 'f8')
        super().__init__(dtype=dtype, _order=order, **kw)

    @property
    def order(self):
        return self._order

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        ctx[chunk.key] = create_array(op)('empty', chunk.shape, dtype=op.dtype,
                                          order=op.order)


def empty(shape, dtype=None, chunk_size=None, gpu=False, order='C'):
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
    order : {'C', 'F'}, optional, default: 'C'
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.

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
    tensor_order = get_order(order, None, available_options='CF',
                             err_msg="only 'C' or 'F' order is permitted")
    op = TensorEmpty(dtype=dtype, gpu=gpu, order=order)
    return op(shape, chunk_size=chunk_size, order=tensor_order)


class TensorEmptyLike(TensorEmptyBase, TensorLike):
    __slots__ = '_rand',
    _op_type_ = OperandDef.TENSOR_EMPTY_LIKE

    _input = KeyField('input')
    _order = StringField('order')

    def __init__(self, dtype=None, gpu=None, sparse=False, order=None, **kw):
        dtype = np.dtype(dtype) if dtype is not None else None
        super().__init__(_dtype=dtype, _gpu=gpu, _order=order, _sparse=sparse, **kw)

    @property
    def order(self):
        return self._order

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        if op.issparse():
            in_data = naked(ctx[op.inputs[0].key])
            xps = get_sparse_module(in_data)
            xp = get_array_module(in_data)
            ctx[chunk.key] = SparseNDArray(xps.csr_matrix(
                (xp.empty_like(in_data.data, dtype=op.dtype),
                 in_data.indices, in_data.indptr), shape=in_data.shape
            ))
        else:
            ctx[chunk.key] = create_array(op)(
                'empty_like', ctx[op.inputs[0].key], dtype=op.dtype, order=op.order)


def empty_like(a, dtype=None, gpu=None, order='K'):
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
        Allocate the tensor on GPU if True, None as default
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if ``prototype`` is Fortran
        contiguous, 'C' otherwise. 'K' means match the layout of ``prototype``
        as closely as possible.

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
    tensor_order = get_order(order, a.order)
    gpu = a.op.gpu if gpu is None else gpu
    op = TensorEmptyLike(dtype=dtype, gpu=gpu, sparse=a.issparse(), order=order)
    return op(a, order=tensor_order)
