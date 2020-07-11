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

from ... import opcodes as OperandDef
from ...serialize import KeyField
from ..array_utils import as_same_device, device
from ..datasource import tensor as astensor
from ..operands import TensorHasInput, TensorOperandMixin
from ..core import TensorOrder


class TensorInv(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.INV

    _input = KeyField('input')

    def __init__(self, dtype=None, sparse=False, **kw):
        super().__init__(_dtype=dtype, _sparse=sparse, **kw)

    def __call__(self, a):
        a = astensor(a)
        return self.new_tensor([a], a.shape, order=TensorOrder.C_ORDER)

    @classmethod
    def _tile_one_chunk(cls, op):
        out = op.outputs[0]
        chunk_op = op.copy().reset_key()
        chunk_params = out.params
        chunk_params['index'] = (0,) * out.ndim
        out_chunk = chunk_op.new_chunk(op.inputs[0].chunks, kws=[chunk_params])

        new_op = op.copy()
        params = out.params
        params['nsplits'] = tuple((s,) for s in out.shape)
        params['chunks'] = [out_chunk]
        return new_op.new_tensors(op.inputs, kws=[params])

    @classmethod
    def tile(cls, op):
        """
        Use LU decomposition to compute inverse of matrix.
        Given a square matrix A:
        P, L, U = lu(A)
        b_eye is an identity matrix with the same shape as matrix A, then,
        (P * L * U) * A_inv = b_eye
        L * (U * A_inv) = P.T * b_eye
        use `solve_triangular` twice to compute the inverse of matrix A.
        """
        from .lu import lu
        from ..datasource import eye
        from ..base.transpose import TensorTranspose
        from .tensordot import tensordot
        from .solve_triangular import solve_triangular
        in_tensor = op.input
        is_sparse = in_tensor.is_sparse()

        if len(in_tensor.chunks) == 1:
            return cls._tile_one_chunk(op)

        b_eye = eye(in_tensor.shape[0], chunk_size=in_tensor.nsplits, sparse=is_sparse)
        b_eye._inplace_tile()

        p, l, u = lu(in_tensor)
        p._inplace_tile()

        # transposed p equals to inverse of p
        p_transpose = TensorTranspose(
            dtype=p.dtype, sparse=p.op.sparse, axes=list(range(in_tensor.ndim))[::-1]).new_tensor([p], p.shape)
        p_transpose._inplace_tile()

        b = tensordot(p_transpose, b_eye, axes=((p_transpose.ndim - 1,), (b_eye.ndim - 2,)))
        b._inplace_tile()

        # as `l` is a lower matrix, `lower=True` should be specified.
        uy = solve_triangular(l, b, lower=True, sparse=op.sparse)
        uy._inplace_tile()

        a_inv = solve_triangular(u, uy, sparse=op.sparse)
        a_inv._inplace_tile()
        return [a_inv]

    def execute(cls, ctx, op):
        (inp,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            ctx[op.outputs[0].key] = xp.linalg.inv(inp)


def inv(a, sparse=None):
    """
    Compute the (multiplicative) inverse of a matrix.
    Given a square matrix `a`, return the matrix `ainv` satisfying
    ``dot(a, ainv) = dot(ainv, a) = eye(a.shape[0])``.

    Parameters
    ----------
    a : (..., M, M) array_like
        Matrix to be inverted.
    sparse: bool, optional
        Return sparse value or not.

    Returns
    -------
    ainv : (..., M, M) ndarray or matrix
        (Multiplicative) inverse of the matrix `a`.

    Raises
    ------
    LinAlgError
        If `a` is not square or inversion fails.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> a = np.array([[1., 2.], [3., 4.]])
    >>> ainv = mt.linalg.inv(a)
    >>> mt.allclose(mt.dot(a, ainv), mt.eye(2)).execute()
    True

    >>> mt.allclose(mt.dot(ainv, a), mt.eye(2)).execute()
    True

    >>> ainv.execute()
    array([[ -2. ,  1. ],
           [ 1.5, -0.5]])
    """

    # TODO: using some parallel algorithm for matrix inversion.
    a = astensor(a)
    if a.ndim != 2:
        raise LinAlgError('{0}-dimensional array given. '
                          'Tensor must be two-dimensional'.format(a.ndim))
    if a.shape[0] != a.shape[1]:
        raise LinAlgError('Input must be square')

    tiny_inv = np.linalg.inv(np.array([[1, 2], [2, 5]], dtype=a.dtype))
    sparse = sparse if sparse is not None else a.issparse()
    op = TensorInv(dtype=tiny_inv.dtype, sparse=sparse)
    return op(a)
