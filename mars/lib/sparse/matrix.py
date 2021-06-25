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
from collections.abc import Iterable

from .core import issparse, get_array_module, cp, cps, \
    get_sparse_module, naked, sps, splinalg
from .array import SparseNDArray, SparseArray


def zeros_sparse_matrix(shape, dtype=float, gpu=False):
    m = sps if not gpu else cps
    return SparseMatrix(m.csr_matrix(shape, dtype=np.dtype(dtype)))


def diag_sparse_matrix(v, k=0, gpu=False):
    v = naked(v)
    if gpu and get_array_module(v) is not cp:
        v = cp.asarray(v)
    if not gpu and get_array_module(v) is not np:
        v = v.get()

    if v.ndim == 1:
        sparse_m = sps if not gpu else cps
        m = n = v.size + k
        mat = sparse_m.spdiags(v[None], [k], m, n, format='csr')
        return SparseMatrix(mat)
    else:
        assert v.ndim == 2
        sparse_m = sps if not gpu else cps
        sparse_eye = sparse_m.eye(v.shape[0], v.shape[1], k=k)
        mat = sparse_eye.multiply(v).tocoo()
        size = sparse_eye.nnz
        col = mat.col - max(k, 0)
        row = get_array_module(col).zeros((len(col),))
        return SparseNDArray(sparse_m.csr_matrix((mat.data, (row, col)), shape=(1, size)),
                             shape=(size,))


def eye_sparse_matrix(N, M=None, k=0, dtype=float, gpu=False):
    m = sps if not gpu else cps
    return SparseMatrix(m.eye(N, n=M, k=k, dtype=dtype, format='csr'))


def triu_sparse_matrix(m, k=0, gpu=False):
    m = naked(m)
    if gpu and get_array_module(m) is not cp:
        m = cp.asarray(m)
    if not gpu and get_array_module(m) is not np:
        m = m.get()

    sparse_m = sps if not gpu else cps
    mat = sparse_m.triu(m, k=k)
    return SparseMatrix(mat)


def tril_sparse_matrix(m, k=0, gpu=False):
    m = naked(m)
    if gpu and get_array_module(m) is not cp:
        m = cp.asarray(m)
    if not gpu and get_array_module(m) is not np:
        m = m.get()

    sparse_m = sps if not gpu else cps
    mat = sparse_m.tril(m, k=k)
    return SparseMatrix(mat)


def where(cond, x, y):
    cond, x, y = [SparseMatrix(i) if issparse(i) else i
                  for i in (cond, x, y)]
    return cond * x + (cond * (-y) + y)


def lu_sparse_matrix(a):
    a = naked(a)
    a = a.tocsc()
    super_lu = splinalg.splu(a, permc_spec="NATURAL", diag_pivot_thresh=0, options={"SymmetricMode": True})
    l_ = super_lu.L
    u = super_lu.U
    p = sps.lil_matrix(a.shape)
    p[super_lu.perm_r.copy(), np.arange(a.shape[1])] = 1
    return SparseMatrix(p), SparseMatrix(l_), SparseMatrix(u),


def solve_triangular_sparse_matrix(a, b, lower=False, sparse=True):
    a = naked(a)
    b = b.toarray() if issparse(b) else b

    x = splinalg.spsolve_triangular(a, b, lower=lower)
    if sparse:
        spx = sps.csr_matrix(x).reshape(x.shape[0], 1) if len(x.shape) == 1 else sps.csr_matrix(x)
        return SparseNDArray(spx, shape=x.shape)
    else:
        return x


class SparseMatrix(SparseArray):
    __slots__ = 'spmatrix',

    def __init__(self, spmatrix, shape=()):
        if shape and len(shape) != 2:
            raise ValueError('Only accept 2-d array')
        if isinstance(spmatrix, SparseMatrix):
            self.spmatrix = spmatrix.spmatrix
        else:
            self.spmatrix = spmatrix.tocsr()

    @property
    def shape(self):
        return self.spmatrix.shape

    @property
    def size(self):
        return int(np.prod(self.shape))

    def transpose(self, axes=None):
        assert axes is None or tuple(axes) == (1, 0)
        return SparseMatrix(self.spmatrix.transpose())

    @property
    def T(self):
        return SparseMatrix(self.spmatrix.T)

    def dot(self, other, sparse=True):
        other_shape = other.shape
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if sparse:
            if len(other_shape) == 1:
                x = self.spmatrix.dot(other.T)
            else:
                x = self.spmatrix.dot(other)
        else:
            a = self.spmatrix.toarray()
            if issparse(other):
                other = other.toarray().reshape(other_shape)
            x = a.dot(other)
        if issparse(x):
            shape = (x.shape[0],) if len(other_shape) == 1 else x.shape
            return SparseNDArray(x, shape=shape)
        return get_array_module(x).asarray(x)

    def concatenate(self, other, axis=0):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if issparse(other):
            xps = get_sparse_module(self.spmatrix)
            if axis not in (0, 1):
                raise ValueError('axis can only be 0 or 1')
            method = xps.vstack if axis == 0 else xps.hstack
            x = method((self.spmatrix, other))
        else:
            xp = get_array_module(self.spmatrix)
            x = xp.concatenate((self.spmatrix.toarray(), other), axis=axis)

        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def _reduction(self, method_name, axis=None, dtype=None, keepdims=None, todense=False, **kw):
        # TODO: support keepdims
        if isinstance(axis, tuple):
            if sorted(axis) != [0, 1]:
                assert len(axis) == 1
                axis = axis[0]
            else:
                axis = None

        if todense:
            x = self.spmatrix.toarray()
            x = getattr(get_array_module(x), method_name)(x, axis=axis, **kw)
        else:
            x = getattr(self.spmatrix, method_name)(axis=axis, **kw)
        if not isinstance(axis, Iterable):
            axis = (axis,)
        axis = list(range(len(self.shape))) if axis is None else axis
        shape = tuple(s if i not in axis else 1 for i, s in enumerate(self.shape)
                      if keepdims or i not in axis)
        m = get_array_module(x)
        if issparse(x):
            return SparseNDArray(x, shape=shape)
        if m.isscalar(x):
            if keepdims:
                return m.array([x])[0].reshape((1,) * self.ndim)
            else:
                return m.array([x])[0]
        else:
            return m.asarray(x).reshape(shape)
