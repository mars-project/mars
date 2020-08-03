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

import itertools

import numpy as np

from ... import opcodes as OperandDef
from ...serialize import KeyField, Int32Field
from ...utils import check_chunks_unknown_shape
from ...tiles import TilesError
from ..core import TENSOR_TYPE
from ...lib.sparse import diag as sparse_diag
from ...lib.sparse.core import issparse, get_array_module, get_sparse_module
from ...lib import sparse
from ..core import TensorOrder
from ..array_utils import create_array
from .core import TensorHasInput
from .zeros import TensorZeros
from .array import tensor


def _get_diag_shape(v_shape, k):
    size_0, size_1 = 0, 0
    if k > 0:
        size_1 += k
    elif k < 0:
        size_0 -= k
    size = min(v_shape[0] - size_0, v_shape[1] - size_1)
    return size,


class TensorDiagBase(object):
    __slots__ = ()

    def to_chunk_op(self, *args):
        op = self.copy().reset_key()
        k, = args
        op._k = k
        return op

    @classmethod
    def _get_nsplits(cls, op):
        raise NotImplementedError

    @classmethod
    def _get_chunk(cls, op, chunk_k, chunk_shape, chunk_idx):
        raise NotImplementedError

    @classmethod
    def tile(cls, op):
        if op.inputs:
            check_chunks_unknown_shape(op.inputs, TilesError)
        tensor = op.outputs[0]

        # op can be TensorDiag or TensorEye
        k = op.k
        nsplits = op._get_nsplits(op)

        fx = lambda x, y: x - y + k
        cum_size = [np.cumsum(s).tolist() for s in nsplits]
        out_chunks = []
        for out_idx in itertools.product(*[range(len(s)) for s in nsplits]):
            i, j = out_idx
            ld_pos = cum_size[0][i] - 1, cum_size[1][j] - nsplits[1][j]
            ru_pos = cum_size[0][i] - nsplits[0][i], cum_size[1][j] - 1

            ld_fx = fx(*ld_pos)
            ru_fx = fx(*ru_pos)

            chunk_shape = (nsplits[0][i], nsplits[1][j])
            if (ld_fx > 0 and ru_fx > 0) or (ld_fx < 0 and ru_fx < 0):
                # does not cross, fill with zeros
                chunk_op = TensorZeros(dtype=op.dtype, gpu=op.gpu, sparse=op.sparse)
                chunk = chunk_op.new_chunk(None, shape=chunk_shape, index=out_idx)
            else:
                lu_pos = ru_pos[0], ld_pos[1]
                chunk_k = fx(*lu_pos)
                chunk = op._get_chunk(op, chunk_k, chunk_shape, out_idx)

            out_chunks.append(chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, tensor.shape, chunks=out_chunks,
                                  nsplits=nsplits)


class TensorDiag(TensorDiagBase, TensorHasInput):
    _op_type_ = OperandDef.TENSOR_DIAG

    _input = KeyField('input')
    _k = Int32Field('k')

    def __init__(self, k=None, dtype=None, gpu=None, sparse=False, **kw):
        super().__init__(_k=k, _dtype=dtype, _gpu=gpu, _sparse=sparse, **kw)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if self.dtype is None:
            self._dtype = self.input.dtype

    def to_chunk_op(self, *args):
        return TensorDiagBase.to_chunk_op(self, *args)

    @classmethod
    def _get_nsplits(cls, op):
        assert op.input.ndim == 1
        k = op.k
        nsplits_1d = op.input.nsplits[0]
        nsplit_0, nsplit_1 = list(nsplits_1d), list(nsplits_1d)
        if k > 0:
            nsplit_0.append(k)
            nsplit_1.insert(0, k)
        elif k < 0:
            nsplit_0.insert(0, abs(k))
            nsplit_1.append(abs(k))
        return nsplit_0, nsplit_1

    @classmethod
    def _get_chunk(cls, op, chunk_k, chunk_shape, chunk_idx):
        assert chunk_shape[0] == chunk_shape[1]
        input_idx = chunk_idx[1] if op.k < 0 else chunk_idx[0]
        input_chunk = op.inputs[0].cix[input_idx, ]
        op = TensorDiag(k=chunk_k, dtype=op.dtype, gpu=op.gpu, sparse=op.sparse)
        return op.new_chunk([input_chunk], shape=chunk_shape, index=chunk_idx)

    def __call__(self, v, shape, chunk_size=None):
        return self.new_tensor([v], shape, raw_chunk_size=chunk_size,
                               order=TensorOrder.C_ORDER)

    @classmethod
    def tile(cls, op):
        tensor = op.outputs[0]

        v = op.input
        k = op.k
        idx = itertools.count(0)
        if v.ndim == 2:
            check_chunks_unknown_shape(op.inputs, TilesError)
            chunks = []
            nsplit = []

            fx = lambda x, y: x - y + k
            in_nsplits = v.nsplits
            cum_size = [np.cumsum(s).tolist() for s in in_nsplits]
            for c in v.chunks:
                i, j = c.index
                ld_pos = cum_size[0][i] - 1, cum_size[1][j] - in_nsplits[1][j]
                ru_pos = cum_size[0][i] - in_nsplits[0][i], cum_size[1][j] - 1

                ld_fx = fx(*ld_pos)
                ru_fx = fx(*ru_pos)

                if (ld_fx > 0 and ru_fx > 0) or (ld_fx < 0 and ru_fx < 0):
                    continue

                lu_pos = ru_pos[0], ld_pos[1]
                chunk_k = fx(*lu_pos)

                chunk_shape = _get_diag_shape(c.shape, chunk_k)
                chunk_idx = (next(idx),)
                chunk_op = op.to_chunk_op(chunk_k)
                chunk = chunk_op.new_chunk([c], shape=chunk_shape,
                                           index=chunk_idx, order=tensor.order)
                nsplit.append(chunk_shape[0])
                chunks.append(chunk)

            new_op = op.copy()
            return new_op.new_tensors(op.inputs, op.outputs[0].shape, order=tensor.order,
                                      chunks=chunks, nsplits=(tuple(nsplit),))
        else:
            return super().tile(op)

    @property
    def k(self):
        return getattr(self, '_k', 0)

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        if op.sparse:
            ctx[chunk.key] = sparse.diag(ctx[op.inputs[0].key], k=op.k, gpu=op.gpu)
        else:
            ctx[chunk.key] = create_array(op)(
                'diag', ctx[op.inputs[0].key], k=op.k)


def diag(v, k=0, sparse=None, gpu=False, chunk_size=None):
    """
    Extract a diagonal or construct a diagonal tensor.

    See the more detailed documentation for ``mt.diagonal`` if you use this
    function to extract a diagonal and wish to write to the resulting tensor

    Parameters
    ----------
    v : array_like
        If `v` is a 2-D tensor, return its `k`-th diagonal.
        If `v` is a 1-D tensor, return a 2-D tensor with `v` on the `k`-th
        diagonal.
    k : int, optional
        Diagonal in question. The default is 0. Use `k>0` for diagonals
        above the main diagonal, and `k<0` for diagonals below the main
        diagonal.
    sparse: bool, optional
        Create sparse tensor if True, False as default
    gpu : bool, optional
        Allocate the tensor on GPU if True, False as default
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension

    Returns
    -------
    out : Tensor
        The extracted diagonal or constructed diagonal tensor.

    See Also
    --------
    diagonal : Return specified diagonals.
    diagflat : Create a 2-D array with the flattened input as a diagonal.
    trace : Sum along diagonals.
    triu : Upper triangle of a tensor.
    tril : Lower triangle of a tensor.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.arange(9).reshape((3,3))
    >>> x.execute()
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    >>> mt.diag(x).execute()
    array([0, 4, 8])
    >>> mt.diag(x, k=1).execute()
    array([1, 5])
    >>> mt.diag(x, k=-1).execute()
    array([3, 7])

    >>> mt.diag(mt.diag(x)).execute()
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 8]])

    """
    if not isinstance(v, TENSOR_TYPE):
        tensor_v = tensor(v)
        if tensor_v.issparse():
            xps = get_sparse_module(tensor_v.data)
            v = xps.csr_matrix((tensor_v.op.data, tensor_v.op.indices, tensor_v.op.indptr),
                               tensor_v.shape)
            diag_v = sparse_diag(v, k=k)
        else:
            v = tensor(v).op.data
            diag_v = get_array_module(v).diag(v, k=k)
        sparse = sparse if sparse is not None else issparse(v)
        return tensor(diag_v, gpu=gpu, sparse=sparse, chunk_size=chunk_size)

    sparse = sparse if sparse is not None else v.issparse()

    if v.ndim == 1:
        shape = (v.size + abs(k),) * 2
    elif v.ndim == 2:
        shape = _get_diag_shape(v.shape, k)
    else:
        raise ValueError('Input must be 1- or 2-d.')

    op = TensorDiag(k, dtype=v.dtype, gpu=gpu, sparse=sparse)
    return op(v, shape)
