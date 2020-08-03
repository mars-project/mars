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

from ...lib import sparse
from ... import opcodes as OperandDef
from ...serialize import KeyField, Int32Field
from ...utils import check_chunks_unknown_shape
from ...tiles import TilesError
from ..array_utils import create_array
from ..core import TensorOrder
from .core import TensorHasInput
from .zeros import TensorZeros
from .array import tensor as astensor


class TensorTri(TensorHasInput):
    def __call__(self, m, order=None):
        order = TensorOrder.C_ORDER if order is None else order
        return self.new_tensor([m], shape=m.shape, order=order)

    def to_chunk_op(self, *args):
        k, = args
        op = self.copy().reset_key()
        op._k = k
        return op

    @classmethod
    def tile(cls, op):
        check_chunks_unknown_shape(op.inputs, TilesError)
        tensor = op.outputs[0]

        m = op.input
        k = op.k
        is_triu = type(op) == TensorTriu

        fx = lambda x, y: x - y + k
        nsplits = m.nsplits
        cum_size = [np.cumsum(s).tolist() for s in nsplits]

        out_chunks = []
        for out_idx in itertools.product(*[range(len(s)) for s in nsplits]):
            i, j = out_idx[-2:]
            ld_pos = cum_size[-2][i] - 1, cum_size[-1][j] - nsplits[-1][j]
            ru_pos = cum_size[-2][i] - nsplits[-2][i], cum_size[-1][j] - 1

            ld_fx = fx(*ld_pos)
            ru_fx = fx(*ru_pos)

            chunk_shape = tuple(nsplits[i][idx] for i, idx in enumerate(out_idx))
            if (is_triu and ld_fx > 0 and ru_fx > 0) or (not is_triu and ld_fx < 0 and ru_fx < 0):
                # does not cross, fill with zeros
                chunk_op = TensorZeros(dtype=op.dtype, gpu=op.gpu, sparse=op.sparse)
                out_chunk = chunk_op.new_chunk(None, shape=chunk_shape,
                                               index=out_idx, order=tensor.order)
            else:
                lu_pos = ru_pos[0], ld_pos[1]
                chunk_k = fx(*lu_pos)

                input_chunk = m.cix[out_idx]
                chunk_op = op.to_chunk_op(chunk_k)
                out_chunk = chunk_op.new_chunk([input_chunk], shape=chunk_shape,
                                               index=out_idx, order=tensor.order)

            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, tensor.shape, chunks=out_chunks, nsplits=m.nsplits)

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        f = 'triu' if isinstance(op, TensorTriu) else 'tril'
        if op.sparse:
            ctx[chunk.key] = getattr(sparse, f)(ctx[op.inputs[0].key], k=op.k)
        else:
            ctx[chunk.key] = create_array(op)(
                f, ctx[op.inputs[0].key], op.k)


class TensorTriu(TensorTri):
    _op_type_ = OperandDef.TENSOR_TRIU

    _input = KeyField('input')
    _k = Int32Field('k')

    def __init__(self, k=None, dtype=None, sparse=False, gpu=None, **kw):
        super().__init__(_k=k, _dtype=dtype, _sparse=sparse, _gpu=gpu, **kw)

    @property
    def k(self):
        return self._k


def triu(m, k=0, gpu=None):
    """
    Upper triangle of a tensor.

    Return a copy of a matrix with the elements below the `k`-th diagonal
    zeroed.

    Please refer to the documentation for `tril` for further details.

    See Also
    --------
    tril : lower triangle of a tensor

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1).execute()
    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 0,  8,  9],
           [ 0,  0, 12]])

    """
    m = astensor(m)
    gpu = m.op.gpu if gpu is None else gpu
    op = TensorTriu(k, dtype=m.dtype, sparse=m.issparse(), gpu=gpu)
    return op(m)


class TensorTril(TensorTri):
    _op_type_ = OperandDef.TENSOR_TRIL

    _input = KeyField('input')
    _k = Int32Field('k')

    def __init__(self, k=None, dtype=None, sparse=False, gpu=None, **kw):
        super().__init__(_k=k, _dtype=dtype, _sparse=sparse, _gpu=gpu, **kw)

    @property
    def k(self):
        return self._k


def tril(m, k=0, gpu=None):
    """
    Lower triangle of a tensor.

    Return a copy of a tensor with elements above the `k`-th diagonal zeroed.

    Parameters
    ----------
    m : array_like, shape (M, N)
        Input tensor.
    k : int, optional
        Diagonal above which to zero elements.  `k = 0` (the default) is the
        main diagonal, `k < 0` is below it and `k > 0` is above.
    gpu : bool, optional
        Allocate the tensor on GPU if True, None as default

    Returns
    -------
    tril : Tensor, shape (M, N)
        Lower triangle of `m`, of same shape and data-type as `m`.

    See Also
    --------
    triu : same thing, only for the upper triangle

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1).execute()
    array([[ 0,  0,  0],
           [ 4,  0,  0],
           [ 7,  8,  0],
           [10, 11, 12]])

    """
    m = astensor(m)
    gpu = m.op.gpu if gpu is None else gpu
    op = TensorTril(k, dtype=m.dtype, sparse=m.issparse(), gpu=gpu)
    return op(m)
