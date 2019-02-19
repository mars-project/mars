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
from .core import get_array_module, cp, cps, naked, issparse


class SparseVector(SparseNDArray):
    __slots__ = 'spvector',

    def __init__(self, spvector):
        if isinstance(spvector, SparseVector):
            self.spvector = spvector.spvector
        else:
            self.spvector = spvector.tocsr()
            
    @property
    def ndim(self):
        return 1

    def toarray(self):
        return self.spvector.toarray()

    def todense(self):
        return self.spvector.toarray()

    def ascupy(self):
        is_cp = get_array_module(self.spvector) is cp
        if is_cp:
            return self
        mat_tuple = (cp.asarray(self.data), cp.asarray(self.indices), cp.asarray(self.indptr))
        return SparseVector(cps.csr_matrix(mat_tuple, shape=self.shape))

    def asscipy(self):
        is_cp = get_array_module(self.spvector) is cp
        if not is_cp:
            return self
        return SparseVector(self.spvector.get())

    def __array__(self, dtype=None):
        x = self.toarray()
        if dtype and x.dtype != dtype:
            return x.astype(dtype)
        return x

    @property
    def nbytes(self):
        return self.spvector.data.nbytes + self.spvector.indptr.nbytes \
               + self.spvector.indices.nbytes

    @property
    def raw(self):
        return self.spvector

    @property
    def data(self):
        return self.spvector.data

    @property
    def indptr(self):
        return self.spvector.indptr

    @property
    def indices(self):
        return self.spvector.indices

    @property
    def nnz(self):
        return self.spvector.nnz

    @property
    def shape(self):
        return self.spvector.shape[1],

    @property
    def dtype(self):
        return self.spvector.dtype

    def dot(self, other, sparse=True):
        other_ndim = other.ndim
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if not sparse:
            a = self.spvector.toarray()
            if issparse(other):
                other = other.toarray()
            x = a.dot(other)
        else:
            if other_ndim == 1 and other.shape[0] == 1:
                x = self.spvector.dot(other.T)
            else:
                x = self.spvector.dot(other)
        if issparse(x):
            return SparseNDArray(x)
        return get_array_module(x).asarray(x)
