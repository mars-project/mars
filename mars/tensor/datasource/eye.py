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
from ...serialize import Int32Field, StringField
from ...config import options
from ..utils import decide_chunk_sizes, get_order
from .diag import TensorDiagBase
from .core import TensorNoInput
from ...lib import sparse
from ..array_utils import create_array


class TensorEye(TensorNoInput, TensorDiagBase):
    _op_type_ = OperandDef.TENSOR_EYE

    _k = Int32Field('k')
    _order = StringField('order')

    def __init__(self, k=None, dtype=None, gpu=None, sparse=False, order=None, **kw):
        dtype = np.dtype(dtype or 'f8')
        super().__init__(_k=k, _dtype=dtype, _gpu=gpu, _sparse=sparse, _order=order, **kw)

    @property
    def k(self):
        return getattr(self, '_k', 0)

    @property
    def order(self):
        return self._order

    @classmethod
    def _get_nsplits(cls, op):
        tensor = op.outputs[0]
        chunk_size = tensor.extra_params.raw_chunk_size or options.chunk_size
        return decide_chunk_sizes(tensor.shape, chunk_size, tensor.dtype.itemsize)

    @classmethod
    def _get_chunk(cls, op, chunk_k, chunk_shape, chunk_idx):
        chunk_op = TensorEye(k=chunk_k, dtype=op.dtype, gpu=op.gpu, sparse=op.sparse)
        return chunk_op.new_chunk(None, shape=chunk_shape, index=chunk_idx)

    @classmethod
    def tile(cls, op):
        return TensorDiagBase.tile(op)

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        if op.sparse:
            ctx[chunk.key] = sparse.eye(chunk.shape[0], M=chunk.shape[1], k=op.k,
                                        dtype=op.dtype, gpu=op.gpu)
        else:
            ctx[chunk.key] = create_array(op)(
                'eye', chunk.shape[0], M=chunk.shape[1], k=op.k,
                dtype=op.dtype, order=op.order)


def eye(N, M=None, k=0, dtype=None, sparse=False, gpu=False, chunk_size=None, order='C'):
    """
    Return a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
      Number of rows in the output.
    M : int, optional
      Number of columns in the output. If None, defaults to `N`.
    k : int, optional
      Index of the diagonal: 0 (the default) refers to the main diagonal,
      a positive value refers to an upper diagonal, and a negative value
      to a lower diagonal.
    dtype : data-type, optional
      Data-type of the returned tensor.
    sparse: bool, optional
        Create sparse tensor if True, False as default
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    order : {'C', 'F'}, optional
        Whether the output should be stored in row-major (C-style) or
        column-major (Fortran-style) order in memory.

    Returns
    -------
    I : Tensor of shape (N,M)
      An tensor where all elements are equal to zero, except for the `k`-th
      diagonal, whose values are equal to one.

    See Also
    --------
    identity : (almost) equivalent function
    diag : diagonal 2-D tensor from a 1-D tensor specified by the user.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.eye(2, dtype=int).execute()
    array([[1, 0],
           [0, 1]])
    >>> mt.eye(3, k=1).execute()
    array([[ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  0.,  0.]])

    """
    if M is None:
        M = N

    shape = (N, M)
    tensor_order = get_order(order, None, available_options='CF',
                             err_msg="only 'C' or 'F' order is permitted")
    op = TensorEye(k, dtype=dtype, gpu=gpu, sparse=sparse, order=order)
    return op(shape, chunk_size=chunk_size, order=tensor_order)
