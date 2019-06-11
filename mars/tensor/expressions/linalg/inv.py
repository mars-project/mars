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
from ..datasource import tensor as astensor
from ..core import TensorHasInput, TensorOperandMixin


class TensorInv(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.INV

    _input = KeyField('input')

    def __init__(self, dtype=None, sparse=False, **kw):
        super(TensorInv, self).__init__(_dtype=dtype, _sparse=sparse, **kw)

    def __call__(self, a):
        a = astensor(a)
        return self.new_tensor([a], a.shape)

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

        b_eye = eye(in_tensor.shape[0], chunk_size=in_tensor.nsplits, sparse=is_sparse)
        b_eye.single_tiles()

        p, l, u = lu(in_tensor)
        p.single_tiles()

        # transposed p equals to inverse of p
        p_transpose = TensorTranspose(
            dtype=p.dtype, sparse=p.op.sparse, axes=list(range(in_tensor.ndim))[::-1]).new_tensor([p], p.shape)
        p_transpose.single_tiles()

        b = tensordot(p_transpose, b_eye, axes=((p_transpose.ndim - 1,), (b_eye.ndim - 2,)))
        b.single_tiles()

        # as `l` is a lower matrix, `lower=True` should be specified.
        uy = solve_triangular(l, b, lower=True, sparse=op.sparse)
        uy.single_tiles()

        a_inv = solve_triangular(u, uy, sparse=op.sparse)
        a_inv.single_tiles()
        return [a_inv]


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
