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

from .... import operands
from ..core import TensorOperandMixin
from ..datasource import tensor as astensor


def _H(chunk):
    from ..base.transpose import TensorTranspose
    from ..arithmetic.conj import TensorConj

    trans_op = TensorTranspose(dtype=chunk.dtype)
    c = trans_op.new_chunk([chunk], chunk.shape[::-1], index=chunk.index[::-1])
    conj_op = TensorConj(dtype=c.dtype)
    return conj_op.new_chunk([c], c.shape, index=c.index)


class TensorCholesky(operands.Cholesky, TensorOperandMixin):
    def __init__(self, lower=None, dtype=None, **kw):
        super(TensorCholesky, self).__init__(_lower=lower, _dtype=dtype, **kw)

    def _set_inputs(self, inputs):
        super(TensorCholesky, self)._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, a):
        return self.new_tensor([a], a.shape)

    @classmethod
    def tile(cls, op):
        from ..datasource.zeros import TensorZeros
        from ..arithmetic.subtract import TensorSubtract
        from ..arithmetic.utils import tree_add
        from .dot import TensorDot
        from .solve_triangular import TensorSolveTriangular

        tensor = op.outputs[0]
        in_tensor = op.input
        if in_tensor.nsplits[0] != in_tensor.nsplits[1]:
            # all chunks on diagonal should be square
            nsplits = in_tensor.nsplits[0]
            in_tensor = in_tensor.rechunk([nsplits, nsplits]).single_tiles()

        lower_chunks, upper_chunks = {}, {}
        for i in range(in_tensor.chunk_shape[0]):
            for j in range(in_tensor.chunk_shape[1]):
                if i < j:
                    lower_chunk = TensorZeros(dtype=tensor.dtype).new_chunk(
                        None, (in_tensor.nsplits[0][i], in_tensor.nsplits[1][j]), index=(i, j))
                    upper_chunk = TensorZeros(dtype=tensor.dtype).new_chunk(
                        None, (in_tensor.nsplits[1][j], in_tensor.nsplits[0][i]), index=(j, i))
                    lower_chunks[lower_chunk.index] = lower_chunk
                    upper_chunks[upper_chunk.index] = upper_chunk
                elif i == j:
                    target = in_tensor.cix[i, j]
                    if i > 0:
                        prev_chunks = []
                        for p in range(i):
                            a, b = lower_chunks[i, p], upper_chunks[p, j]
                            prev_chunk = TensorDot(dtype=tensor.dtype).new_chunk(
                                [a, b], (a.shape[0], b.shape[1]))
                            prev_chunks.append(prev_chunk)
                        if len(prev_chunks) == 1:
                            s = prev_chunks[0]
                        else:
                            s = tree_add(prev_chunks[0].dtype, prev_chunks,
                                         None, prev_chunks[0].shape)
                        target = TensorSubtract(dtype=tensor.dtype, lhs=target, rhs=s).new_chunk(
                            [target, s], shape=target.shape)
                    lower_chunk = TensorCholesky(lower=True, dtype=tensor.dtype).new_chunk(
                        [target], target.shape, index=(i, j))
                    upper_chunk = _H(lower_chunk)
                    lower_chunks[lower_chunk.index] = lower_chunk
                    upper_chunks[upper_chunk.index] = upper_chunk
                else:
                    target = in_tensor.cix[j, i]
                    if j > 0:
                        prev_chunks = []
                        for p in range(j):
                            a, b = lower_chunks[j, p], upper_chunks[p, i]
                            prev_chunk = TensorDot(dtype=tensor.dtype).new_chunk(
                                [a, b], (a.shape[0], b.shape[1]))
                            prev_chunks.append(prev_chunk)
                        if len(prev_chunks) == 1:
                            s = prev_chunks[0]
                        else:
                            s = tree_add(prev_chunks[0].dtype, prev_chunks,
                                         None, prev_chunks[0].shape)
                        target = TensorSubtract(dtype=tensor.dtype, lhs=target, rhs=s).new_chunk(
                            [target, s], shape=target.shape)
                    upper_chunk = TensorSolveTriangular(lower=True, dtype=tensor.dtype).new_chunk(
                        [lower_chunks[j, j], target], target.shape, index=(j, i))
                    lower_chunk = _H(upper_chunk)
                    lower_chunks[lower_chunk.index] = lower_chunk
                    upper_chunks[upper_chunk.index] = upper_chunk

        new_op = op.copy()
        if op.lower:
            return new_op.new_tensors(op.inputs, tensor.shape,
                                      chunks=list(lower_chunks.values()), nsplits=in_tensor.nsplits)
        else:
            return new_op.new_tensors(op.inputs, tensor.shape,
                                      chunks=list(upper_chunks.values()), nsplits=in_tensor.nsplits)


def cholesky(a, lower=False):
    """
    Cholesky decomposition.

    Return the Cholesky decomposition, `L * L.H`, of the square matrix `a`,
    where `L` is lower-triangular and .H is the conjugate transpose operator
    (which is the ordinary transpose if `a` is real-valued).  `a` must be
    Hermitian (symmetric if real-valued) and positive-definite.  Only `L` is
    actually returned.

    Parameters
    ----------
    a : (..., M, M) array_like
        Hermitian (symmetric if all elements are real), positive-definite
        input matrix.
    lower : bool
        Whether to compute the upper or lower triangular Cholesky
        factorization.  Default is upper-triangular.

    Returns
    -------
    L : (..., M, M) array_like
        Upper or lower-triangular Cholesky factor of `a`.

    Raises
    ------
    LinAlgError
       If the decomposition fails, for example, if `a` is not
       positive-definite.

    Notes
    -----

    Broadcasting rules apply, see the `mt.linalg` documentation for
    details.

    The Cholesky decomposition is often used as a fast way of solving

    .. math:: A \\mathbf{x} = \\mathbf{b}

    (when `A` is both Hermitian/symmetric and positive-definite).

    First, we solve for :math:`\\mathbf{y}` in

    .. math:: L \\mathbf{y} = \\mathbf{b},

    and then for :math:`\\mathbf{x}` in

    .. math:: L.H \\mathbf{x} = \\mathbf{y}.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> A = mt.array([[1,-2j],[2j,5]])
    >>> A.execute()
    array([[ 1.+0.j,  0.-2.j],
           [ 0.+2.j,  5.+0.j]])
    >>> L = mt.linalg.cholesky(A, lower=True)
    >>> L.execute()
    array([[ 1.+0.j,  0.+0.j],
           [ 0.+2.j,  1.+0.j]])
    >>> mt.dot(L, L.T.conj()).execute() # verify that L * L.H = A
    array([[ 1.+0.j,  0.-2.j],
           [ 0.+2.j,  5.+0.j]])
    >>> A = [[1,-2j],[2j,5]] # what happens if A is only array_like?
    >>> mt.linalg.cholesky(A, lower=True).execute()
    array([[ 1.+0.j,  0.+0.j],
           [ 0.+2.j,  1.+0.j]])

    """
    a = astensor(a)

    if a.ndim != 2:
        raise LinAlgError('{0}-dimensional array given. '
                          'Tensor must be two-dimensional'.format(a.ndim))
    if a.shape[0] != a.shape[1]:
        raise LinAlgError('Input must be square')

    cho = np.linalg.cholesky(np.array([[1, 2], [2, 5]], dtype=a.dtype))

    op = TensorCholesky(lower=lower, dtype=cho.dtype)
    return op(a)
