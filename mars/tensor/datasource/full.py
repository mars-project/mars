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
from ...serialization.serializables import AnyField, StringField, KeyField
from ...lib.sparse.core import get_sparse_module, get_array_module, naked
from ...lib.sparse import SparseNDArray
from ..array_utils import create_array
from ..utils import get_order
from .core import TensorNoInput, TensorLike
from .array import tensor


class TensorFull(TensorNoInput):
    _op_type_ = OperandDef.TENSOR_FULL

    _fill_value = AnyField('fill_value')
    _order = StringField('order')

    def __init__(self, fill_value=None, dtype=None, order=None, **kw):
        if dtype is not None:
            dtype = np.dtype(dtype)
            if fill_value is not None:
                fill_value = dtype.type(fill_value)
        elif fill_value is not None:
            dtype = np.array(fill_value).dtype
        super().__init__(_fill_value=fill_value, dtype=dtype, _order=order, **kw)

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def order(self):
        return self._order

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        ctx[chunk.key] = create_array(op)('full', chunk.shape,
                                          op.fill_value, dtype=op.dtype, order=op.order)


def full(shape, fill_value, dtype=None, chunk_size=None, gpu=False, order='C'):
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
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.

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
        return broadcast_to(
            tensor(v, dtype=dtype, chunk_size=chunk_size, gpu=gpu, order=order), shape)

    tensor_order = get_order(order, None, available_options='CF',
                             err_msg="only 'C' or 'F' order is permitted")
    op = TensorFull(fill_value, dtype=dtype, gpu=gpu, order=order)
    return op(shape, chunk_size=chunk_size, order=tensor_order)


class TensorFullLike(TensorLike):
    _op_type_ = OperandDef.TENSOR_FULL_LIKE

    _input = KeyField('input')
    _fill_value = AnyField('fill_value')
    _order = StringField('order')

    def __init__(self, fill_value=None, dtype=None, gpu=None, sparse=False, order=None, **kw):
        if dtype is not None:
            dtype = np.dtype(dtype)
            if fill_value is not None:
                fill_value = dtype.type(fill_value)
        elif fill_value is not None:
            dtype = np.array(fill_value).dtype
        super().__init__(_fill_value=fill_value, _dtype=dtype, _gpu=gpu, _order=order,
                         _sparse=sparse, **kw)

    @property
    def fill_value(self):
        return self._fill_value

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
                (xp.full_like(in_data.data, op.fill_value, dtype=op.dtype),
                 in_data.indices, in_data.indptr), shape=in_data.shape
            ))
        else:
            ctx[chunk.key] = create_array(op)(
                'full_like', ctx[op.inputs[0].key], op.fill_value,
                dtype=op.dtype, order=op.order)


def full_like(a, fill_value, dtype=None, gpu=None, order='K'):
    """
    Return a full tensor with the same shape and type as a given tensor.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned tensor.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        Overrides the data type of the result.
    gpu : bool, optional
        Allocate the tensor on GPU if True, None as default
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.

    Returns
    -------
    out : Tensor
        Tensor of `fill_value` with the same shape and type as `a`.

    See Also
    --------
    empty_like : Return an empty tensor with shape and type of input.
    ones_like : Return a tensor of ones with shape and type of input.
    zeros_like : Return a tensor of zeros with shape and type of input.
    full : Return a new tensor of given shape filled with value.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> x = mt.arange(6, dtype=int)
    >>> mt.full_like(x, 1).execute()
    array([1, 1, 1, 1, 1, 1])
    >>> mt.full_like(x, 0.1).execute()
    array([0, 0, 0, 0, 0, 0])
    >>> mt.full_like(x, 0.1, dtype=mt.double).execute()
    array([ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1])
    >>> mt.full_like(x, mt.nan, dtype=mt.double).execute()
    array([ nan,  nan,  nan,  nan,  nan,  nan])

    >>> y = mt.arange(6, dtype=mt.double)
    >>> mt.full_like(y, 0.1).execute()
    array([ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1])

    """
    a = tensor(a)
    tensor_order = get_order(order, a.order)
    if dtype is None:
        dtype = a.dtype
    gpu = a.op.gpu if gpu is None else gpu
    op = TensorFullLike(fill_value=fill_value, dtype=dtype,
                        gpu=gpu, sparse=a.issparse())
    return op(a, order=tensor_order)
