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
from numpy.linalg import LinAlgError

from .... import opcodes as OperandDef
from ....serialize import KeyField
from ....core import ExecutableTuple
from ..utils import recursive_tile
from ..core import TensorHasInput, TensorOperandMixin
from ..datasource import tensor as astensor


class TensorLU(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.LU

    _input = KeyField('input')

    def __init__(self, dtype=None, sparse=False, **kw):
        super(TensorLU, self).__init__(_dtype=dtype, _sparse=sparse, **kw)

    @property
    def output_limit(self):
        return 3

    def __call__(self, a):
        import scipy.linalg

        a = astensor(a)
        if a.ndim != 2:
            raise LinAlgError('{0}-dimensional array given. '
                              'Tensor must be two-dimensional'.format(a.ndim))

        if a.shape[0] > a.shape[1]:
            p_shape = (a.shape[0],) * 2
            l_shape = a.shape
            u_shape = (a.shape[1],) * 2
        elif a.shape[0] < a.shape[1]:
            p_shape = (a.shape[0],) * 2
            l_shape = (a.shape[0],) * 2
            u_shape = a.shape
        else:
            p_shape, l_shape, u_shape = (a.shape,) * 3

        tiny_p, tiny_l, tiny_u = scipy.linalg.lu(np.array([[1, 2], [2, 5]], dtype=a.dtype))

        p, l, u = self.new_tensors([a],
                                   kws=[
                                       {'side': 'p', 'dtype': tiny_p.dtype, 'shape': p_shape},
                                       {'side': 'l', 'dtype': tiny_l.dtype, 'shape': l_shape},
                                       {'side': 'u', 'dtype': tiny_u.dtype, 'shape': u_shape},
                                   ])
        return ExecutableTuple([p, l, u])

    @classmethod
    def tile(cls, op):
        from ..arithmetic.subtract import TensorSubtract
        from ..arithmetic.utils import tree_add
        from ..base.transpose import TensorTranspose
        from ..merge.vstack import vstack
        from ..merge.hstack import hstack
        from ..datasource.zeros import TensorZeros, zeros
        from .dot import TensorDot
        from .solve_triangular import TensorSolveTriangular

        P, L, U = op.outputs
        raw_in_tensor = in_tensor = op.input

        if in_tensor.shape[0] > in_tensor.shape[1]:
            zero_tensor = zeros((in_tensor.shape[0], in_tensor.shape[0] - in_tensor.shape[1]),
                                dtype=in_tensor.dtype, sparse=in_tensor.issparse(),
                                gpu=in_tensor.op.gpu,
                                chunk_size=(in_tensor.nsplits[0], max(in_tensor.nsplits[1])))
            in_tensor = hstack([in_tensor, zero_tensor])
            recursive_tile(in_tensor)
        elif in_tensor.shape[0] < in_tensor.shape[1]:
            zero_tensor = zeros((in_tensor.shape[1] - in_tensor.shape[0], in_tensor.shape[1]),
                                dtype=in_tensor.dtype, sparse=in_tensor.issparse(),
                                gpu=in_tensor.op.gpu,
                                chunk_size=(max(in_tensor.nsplits[0]), in_tensor.nsplits[1]))
            in_tensor = vstack([in_tensor, zero_tensor])
            recursive_tile(in_tensor)

        if in_tensor.nsplits[0] != in_tensor.nsplits[1]:
            # all chunks on diagonal should be square
            nsplits = in_tensor.nsplits[0]
            in_tensor = in_tensor.rechunk([nsplits, nsplits]).single_tiles()

        p_chunks, p_invert_chunks, lower_chunks, l_permuted_chunks, upper_chunks = {}, {}, {}, {}, {}
        for i in range(in_tensor.chunk_shape[0]):
            for j in range(in_tensor.chunk_shape[1]):
                if i < j:
                    chunk_shape = (in_tensor.nsplits[0][i], in_tensor.nsplits[1][j])
                    p_chunk = TensorZeros(sparse=op.sparse).new_chunk(None, shape=chunk_shape, index=(i, j))
                    lower_chunk = TensorZeros(sparse=op.sparse).new_chunk(None, shape=chunk_shape, index=(i, j))
                    p_chunks[p_chunk.index] = p_chunk
                    lower_chunks[lower_chunk.index] = lower_chunk

                    target_u = in_tensor.cix[i, j]
                    p_invert = p_invert_chunks[i, i]
                    target = TensorDot(dtype=U.dtype, sparse=U.op.sparse).new_chunk(
                        [p_invert, target_u], shape=(p_invert.shape[0], target_u.shape[1]))
                    if i > 0:
                        prev_chunks_u = []
                        for p in range(i):
                            a, b = lower_chunks[i, p], upper_chunks[p, j]
                            prev_chunk = TensorDot(dtype=U.dtype, sparse=U.op.sparse).new_chunk(
                                [a, b], shape=(a.shape[0], b.shape[1]))
                            prev_chunks_u.append(prev_chunk)
                        if len(prev_chunks_u) == 1:
                            s = prev_chunks_u[0]
                        else:
                            s = tree_add(prev_chunks_u[0].dtype, prev_chunks_u,
                                         None, prev_chunks_u[0].shape, sparse=op.sparse)
                        target = TensorSubtract(dtype=U.dtype, lhs=target, rhs=s).new_chunk(
                            [target, s], shape=target.shape)
                    upper_chunk = TensorSolveTriangular(lower=True, dtype=U.dtype, strict=False,
                                                        sparse=lower_chunks[i, i].op.sparse).new_chunk(
                        [lower_chunks[i, i], target], shape=target.shape, index=(i, j))
                    upper_chunks[upper_chunk.index] = upper_chunk
                elif i == j:
                    target = in_tensor.cix[i, j]
                    if i > 0:
                        prev_chunks = []
                        for p in range(i):
                            a, b = l_permuted_chunks[i, p], upper_chunks[p, j]
                            prev_chunk = TensorDot(dtype=a.dtype, sparse=op.sparse).new_chunk(
                                [a, b], shape=(a.shape[0], b.shape[1]))
                            prev_chunks.append(prev_chunk)
                        if len(prev_chunks) == 1:
                            s = prev_chunks[0]
                        else:
                            s = tree_add(prev_chunks[0].dtype, prev_chunks,
                                         None, prev_chunks[0].shape, sparse=op.sparse)
                        target = TensorSubtract(dtype=L.dtype, lhs=target, rhs=s).new_chunk(
                            [target, s], shape=target.shape)
                    new_op = TensorLU(dtype=op.dtype, sparse=target.op.sparse)
                    lu_chunks = new_op.new_chunks([target],
                                                  index=(i, j),
                                                  kws=[
                                                      {'side': 'p', 'dtype': P.dtype, 'shape': target.shape},
                                                      {'side': 'l', 'dtype': L.dtype, 'shape': target.shape},
                                                      {'side': 'u', 'dtype': U.dtype, 'shape': target.shape},
                                                  ])
                    p_chunk, lower_chunk, upper_chunk = lu_chunks
                    # transposed p equals to inverted p
                    p_chunk_invert = TensorTranspose(dtype=p_chunk.dtype, sparse=op.sparse).new_chunk(
                        [p_chunk], shape=p_chunk.shape, index=p_chunk.index)
                    p_chunks[p_chunk.index] = p_chunk
                    p_invert_chunks[p_chunk_invert.index] = p_chunk_invert
                    lower_chunks[lower_chunk.index] = lower_chunk
                    upper_chunks[upper_chunk.index] = upper_chunk

                    # l_permuted should be transferred to the final lower triangular
                    for p in range(i):
                        l_permuted_chunk = l_permuted_chunks[i, p]
                        l_chunk = TensorDot(dtype=L.dtype, sparse=L.op.sparse).new_chunk(
                            [p_chunk_invert, l_permuted_chunk],
                            shape=(p_chunk_invert.shape[0], l_permuted_chunk.shape[1]),
                            index=l_permuted_chunk.index
                        )
                        lower_chunks[l_permuted_chunk.index] = l_chunk
                else:
                    chunk_shape = (in_tensor.nsplits[0][i], in_tensor.nsplits[1][j])
                    p_chunk = TensorZeros(sparse=op.sparse).new_chunk(None, shape=chunk_shape, index=(i, j))
                    upper_chunk = TensorZeros(sparse=op.sparse).new_chunk(None, shape=chunk_shape, index=(i, j))
                    p_chunks[p_chunk.index] = p_chunk
                    upper_chunks[upper_chunk.index] = upper_chunk
                    target_l = in_tensor.cix[i, j]
                    if j > 0:
                        prev_chunks_l = []
                        for p in range(j):
                            a, b = l_permuted_chunks[i, p], upper_chunks[p, j]
                            prev_chunk = TensorDot(dtype=L.dtype, sparse=L.op.sparse).new_chunk(
                                [a, b], shape=(a.shape[0], b.shape[1]))
                            prev_chunks_l.append(prev_chunk)
                        if len(prev_chunks_l) == 1:
                            s = prev_chunks_l[0]
                        else:
                            s = tree_add(prev_chunks_l[0].dtype, prev_chunks_l,
                                         None, prev_chunks_l[0].shape, sparse=op.sparse)
                        target_l = TensorSubtract(dtype=L.dtype, lhs=target_l, rhs=s).new_chunk(
                            [target_l, s], shape=target_l.shape)
                    u = upper_chunks[j, j]
                    a_transpose = TensorTranspose(dtype=u.dtype, sparse=op.sparse).new_chunk([u], shape=u.shape)
                    target_transpose = TensorTranspose(dtype=target_l.dtype, sparse=op.sparse).new_chunk(
                        [target_l], shape=target_l.shape)
                    lower_permuted_chunk = TensorSolveTriangular(
                        lower=True, dtype=L.dtype, strict=False, sparse=op.sparse).new_chunk(
                        [a_transpose, target_transpose], shape=target_l.shape, index=(i, j))
                    lower_transpose = TensorTranspose(dtype=lower_permuted_chunk.dtype, sparse=op.sparse).new_chunk(
                        [lower_permuted_chunk], shape=lower_permuted_chunk.shape, index=lower_permuted_chunk.index)
                    l_permuted_chunks[lower_permuted_chunk.index] = lower_transpose

        new_op = op.copy()
        kws = [
            {'chunks': list(p_chunks.values()), 'nsplits': in_tensor.nsplits,
             'dtype': P.dtype, 'shape': P.shape},
            {'chunks': list(lower_chunks.values()), 'nsplits': in_tensor.nsplits,
             'dtype': L.dtype, 'shape': L.shape},
            {'chunks': list(upper_chunks.values()), 'nsplits': in_tensor.nsplits,
             'dtype': U.dtype, 'shape': U.shape}
        ]
        if raw_in_tensor.shape[0] == raw_in_tensor.shape[1]:
            return new_op.new_tensors(op.inputs, kws=kws)

        p, l, u = new_op.new_tensors(op.inputs, kws=kws)
        if raw_in_tensor.shape[0] > raw_in_tensor.shape[1]:
            l = l[:, :raw_in_tensor.shape[1]].single_tiles()
            u = u[:raw_in_tensor.shape[1], :raw_in_tensor.shape[1]].single_tiles()
        else:
            p = p[:raw_in_tensor.shape[0], :raw_in_tensor.shape[0]].single_tiles()
            l = l[:raw_in_tensor.shape[0], :raw_in_tensor.shape[0]].single_tiles()
            u = u[:raw_in_tensor.shape[0], :].single_tiles()
        kws = [
            {'chunks': p.chunks, 'nsplits': p.nsplits, 'dtype': P.dtype, 'shape': p.shape},
            {'chunks': l.chunks, 'nsplits': l.nsplits, 'dtype': l.dtype, 'shape': l.shape},
            {'chunks': u.chunks, 'nsplits': u.nsplits, 'dtype': u.dtype, 'shape': u.shape}
        ]
        return new_op.new_tensors(op.inputs, kws=kws)


def lu(a):
    """
    LU decomposition

    The decomposition is::
        A = P L U
    where P is a permutation matrix, L lower triangular with unit diagonal elements,
    and U upper triangular.

    Parameters
    ----------
    a : (M, N) array_like
        Array to decompose

    Returns
    -------
    p : (M, M) ndarray
        Permutation matrix
    l : (M, K) ndarray
        Lower triangular or trapezoidal matrix with unit diagonal.
        K = min(M, N)
    u : (K, N) ndarray
        Upper triangular or trapezoidal matrix

    Examples
    --------
    >>> import mars.tensor as mt

    >>> A = mt.array([[1,2],[2,3]])
    >>> A.execute()
    array([[ 1,  2],
           [ 2,  3]])
    >>> P, L, U = mt.linalg.lu(A)
    >>> P.execute()
    array([[ 0,  1],
           [ 1,  0]])
    >>> L.execute()
    array([[ 1,  0],
           [ 0.5,  1]])
    >>> U.execute()
    array([[ 2,  3],
           [ 0,  0.5]])
    >>> mt.dot(P.dot(L), U).execute() # verify that PL * U = A
    array([[ 1,  2],
           [ 2,  3]])

    """
    op = TensorLU(sparse=a.issparse())
    return op(a)
