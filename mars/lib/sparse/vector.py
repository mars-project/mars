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

from .array import SparseNDArray
from .core import get_array_module, get_sparse_module, cp, cps, naked, issparse


class SparseVector(SparseNDArray):
    __slots__ = 'spmatrix',

    def __init__(self, spvector, shape=()):
        if shape and len(shape) != 1:
            raise ValueError('Only accept 1-d array')
        if isinstance(spvector, SparseVector):
            self.spmatrix = spvector.spmatrix
        else:
            self.spmatrix = spvector.tocsr()

    def __add__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented
        if issparse(other):
            x = self.spmatrix + other.reshape(self.spmatrix.shape)
        else:
            x = self.toarray() + other
        if issparse(x):
            return SparseVector(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __radd__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        x = other + self.toarray()
        if issparse(x):
            return SparseVector(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __sub__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented
        if issparse(other):
            x = self.spmatrix - other.reshape(self.spmatrix.shape)
        else:
            x = self.toarray() - other
        if issparse(x):
            return SparseVector(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __rsub__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        x = other - self.toarray()
        if issparse(x):
            return SparseVector(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    @property
    def ndim(self):
        return 1

    def toarray(self):
        return self.spmatrix.toarray().reshape(self.shape)

    def todense(self):
        return self.spmatrix.toarray().reshape(self.shape)

    def tocsr(self):
        return self

    def ascupy(self):
        is_cp = get_array_module(self.spmatrix) is cp
        if is_cp:
            return self
        mat_tuple = (cp.asarray(self.data), cp.asarray(self.indices), cp.asarray(self.indptr))
        return SparseVector(cps.csr_matrix(mat_tuple, shape=self.spmatrix.shape))

    def asscipy(self):
        is_cp = get_array_module(self.spmatrix) is cp
        if not is_cp:
            return self
        return SparseVector(self.spmatrix.get())

    def __array__(self, dtype=None):
        x = self.toarray()
        if dtype and x.dtype != dtype:
            return x.astype(dtype)
        return x

    @property
    def nbytes(self):
        return self.spmatrix.data.nbytes + self.spmatrix.indptr.nbytes \
               + self.spmatrix.indices.nbytes

    @property
    def raw(self):
        return self.spmatrix

    @property
    def data(self):
        return self.spmatrix.data

    @property
    def indptr(self):
        return self.spmatrix.indptr

    @property
    def indices(self):
        return self.spmatrix.indices

    @property
    def nnz(self):
        return self.spmatrix.nnz

    @property
    def shape(self):
        v_shape = self.spmatrix.shape
        if v_shape[0] != 1:
            return v_shape[0],
        else:
            return v_shape[1],

    @property
    def dtype(self):
        return self.spmatrix.dtype

    def dot(self, other, sparse=True):
        other_shape = other.shape
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if not sparse:
            a = self.toarray()
            if issparse(other):
                other = other.toarray().reshape(other_shape)

            x = a.dot(other)
        else:
            v = self.spmatrix.T if self.spmatrix.shape[1] == 1 else self.spmatrix
            if len(other_shape) == 1 and other.shape[0] == 1:
                x = v.dot(other.T)
            else:
                x = v.dot(other)
        if issparse(x):
            shape = (other.shape[1],)
            return SparseNDArray(x, shape=shape)
        return get_array_module(x).asarray(x)

    def concatenate(self, other, axis=0):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if issparse(other):
            xps = get_sparse_module(self.spmatrix)
            if axis != 0:
                raise ValueError('axis can only be 0')
            x = xps.hstack((self.spmatrix, other))
        else:
            xp = get_array_module(self.spmatrix)
            x = xp.concatenate((self.spmatrix.toarray().reshape(self.shape), other), axis=axis)

        if issparse(x):
            return SparseNDArray(x, shape=(x.shape[1],))
        return get_array_module(x).asarray(x)
