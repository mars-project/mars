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
from numpy.linalg import LinAlgError

from ...serialize import KeyField, BoolField
from ... import opcodes as OperandDef
from ...utils import check_chunks_unknown_shape
from ...tiles import TilesError
from ..operands import TensorHasInput, TensorOperand, TensorOperandMixin
from ..datasource import tensor as astensor
from ..core import TensorOrder
from ..array_utils import as_same_device, device


class TensorCholesky(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.CHOLESKY

    _input = KeyField('input')
    _lower = BoolField('lower')

    def __init__(self, lower=None, dtype=None, **kw):
        super().__init__(_lower=lower, _dtype=dtype, **kw)

    @property
    def lower(self):
        return self._lower

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, a):
        return self.new_tensor([a], a.shape, order=TensorOrder.C_ORDER)

    @classmethod
    def tile(cls, op):
        from ..datasource.zeros import TensorZeros
        from ..base import TensorTranspose
        from ..utils import reverse_order
        from .dot import TensorDot
        from .solve_triangular import TensorSolveTriangular

        tensor = op.outputs[0]
        in_tensor = op.input
        check_chunks_unknown_shape([in_tensor], TilesError)
        if in_tensor.nsplits[0] != in_tensor.nsplits[1]:
            # all chunks on diagonal should be square
            nsplits = in_tensor.nsplits[0]
            in_tensor = in_tensor.rechunk([nsplits, nsplits])._inplace_tile()

        lower_chunks, upper_chunks = {}, {}
        for i in range(in_tensor.chunk_shape[0]):
            for j in range(in_tensor.chunk_shape[1]):
                if i < j:
                    lower_chunk = TensorZeros(dtype=tensor.dtype).new_chunk(
                        None, shape=(in_tensor.nsplits[0][i], in_tensor.nsplits[1][j]),
                        index=(i, j), order=tensor.order)
                    upper_chunk = TensorZeros(dtype=tensor.dtype).new_chunk(
                        None, shape=(in_tensor.nsplits[1][j], in_tensor.nsplits[0][i]),
                        index=(j, i), order=tensor.order)
                    lower_chunks[lower_chunk.index] = lower_chunk
                    upper_chunks[upper_chunk.index] = upper_chunk
                elif i == j:
                    target = in_tensor.cix[i, j]
                    if i > 0:
                        prev_chunks = []
                        for p in range(i):
                            a, b = lower_chunks[i, p], upper_chunks[p, j]
                            prev_chunk = TensorDot(dtype=tensor.dtype).new_chunk(
                                [a, b], shape=(a.shape[0], b.shape[1]), order=tensor.order)
                            prev_chunks.append(prev_chunk)

                        cholesky_fuse_op = TensorCholeskyFuse()
                        lower_chunk = cholesky_fuse_op.new_chunk([target] + prev_chunks,
                                                                 shape=target.shape, index=(i, j),
                                                                 order=tensor.order)
                    else:
                        lower_chunk = TensorCholesky(lower=True, dtype=tensor.dtype).new_chunk(
                            [target], shape=target.shape, index=(i, j), order=tensor.order)

                    upper_chunk = TensorTranspose(dtype=lower_chunk.dtype).new_chunk(
                        [lower_chunk], shape=lower_chunk.shape[::-1],
                        index=lower_chunk.index[::-1], order=reverse_order(lower_chunk.order))
                    lower_chunks[lower_chunk.index] = lower_chunk
                    upper_chunks[upper_chunk.index] = upper_chunk
                else:
                    target = in_tensor.cix[j, i]
                    if j > 0:
                        prev_chunks = []
                        for p in range(j):
                            a, b = lower_chunks[j, p], upper_chunks[p, i]
                            prev_chunk = TensorDot(dtype=tensor.dtype).new_chunk(
                                [a, b], shape=(a.shape[0], b.shape[1]), order=tensor.order)
                            prev_chunks.append(prev_chunk)
                        cholesky_fuse_op = TensorCholeskyFuse(by_solve_triangular=True)
                        upper_chunk = cholesky_fuse_op.new_chunk([target] + [lower_chunks[j, j]] + prev_chunks,
                                                                 shape=target.shape, index=(j, i),
                                                                 order=tensor.order)
                    else:
                        upper_chunk = TensorSolveTriangular(lower=True, dtype=tensor.dtype).new_chunk(
                            [lower_chunks[j, j], target], shape=target.shape,
                            index=(j, i), order=tensor.order)
                    lower_chunk = TensorTranspose(dtype=upper_chunk.dtype).new_chunk(
                        [upper_chunk], shape=upper_chunk.shape[::-1],
                        index=upper_chunk.index[::-1], order=reverse_order(upper_chunk.order))
                    lower_chunks[lower_chunk.index] = lower_chunk
                    upper_chunks[upper_chunk.index] = upper_chunk

        new_op = op.copy()
        if op.lower:
            return new_op.new_tensors(op.inputs, tensor.shape, order=tensor.order,
                                      chunks=list(lower_chunks.values()), nsplits=in_tensor.nsplits)
        else:
            return new_op.new_tensors(op.inputs, tensor.shape, order=tensor.order,
                                      chunks=list(upper_chunks.values()), nsplits=in_tensor.nsplits)

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        (a,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            if xp is np:
                try:
                    import scipy.linalg

                    ctx[chunk.key] = scipy.linalg.cholesky(a, lower=op.lower)
                    return
                except ImportError:  # pragma: no cover
                    pass

            r = xp.linalg.cholesky(a)
            if not chunk.op.lower:
                r = r.T.conj()

            ctx[chunk.key] = r


class TensorCholeskyFuse(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.CHOLESKY_FUSE

    _by_solve_triangular = BoolField('by_solve_triangular')

    def __init__(self, by_solve_triangular=None, **kw):
        super().__init__(_by_solve_triangular=by_solve_triangular, **kw)

    @property
    def by_solve_triangular(self):
        return self._by_solve_triangular

    @classmethod
    def _execute_by_cholesky(cls, inputs):
        import scipy.linalg

        target = inputs[0]
        return scipy.linalg.cholesky((target - sum(inputs[1:])), lower=True)

    @classmethod
    def _execute_by_solve_striangular(cls, inputs):
        import scipy.linalg

        target = inputs[0]
        lower = inputs[1]
        return scipy.linalg.solve_triangular(lower, (target - sum(inputs[2:])), lower=True)

    @classmethod
    def execute(cls, ctx, op):
        inputs = [ctx[c.key] for c in op.inputs]
        if op.by_solve_triangular:
            ret = cls._execute_by_solve_striangular(inputs)
        else:
            ret = cls._execute_by_cholesky(inputs)
        ctx[op.outputs[0].key] = ret


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
