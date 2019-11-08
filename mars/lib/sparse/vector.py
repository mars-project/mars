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

from .array import SparseArray, SparseNDArray
from .core import get_array_module, get_sparse_module, naked, issparse, np, is_cupy


class SparseVector(SparseArray):
    __slots__ = 'spmatrix',

    def __init__(self, spvector, shape=()):
        if shape and len(shape) != 1:
            raise ValueError('Only accept 1-d array')
        if isinstance(spvector, SparseVector):
            self.spmatrix = spvector.spmatrix
        else:
            spvector = spvector.reshape(1, shape[0])
            self.spmatrix = spvector.tocsr()

    @property
    def shape(self):
        return self.spmatrix.shape[1],

    def transpose(self, axes=None):
        assert axes is None or tuple(axes) == (0,)
        return self

    @property
    def T(self):
        return self

    def __truediv__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented
        x = self.spmatrix / other
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        if x.shape != self.shape:
            x = np.asarray(x).reshape(self.shape)
        return get_array_module(x).asarray(x)

    def __rtruediv__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented
        try:
            x = other / self.spmatrix
        except TypeError:
            x = other / self.spmatrix.toarray()
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        if x.shape != self.shape:
            x = np.asarray(x).reshape(self.shape)
        return get_array_module(x).asarray(x)

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
            if len(other_shape) == 1:
                x = self.spmatrix.dot(other.T)
            else:
                x = self.spmatrix.dot(other)
        if issparse(x):
            if x.shape == (1, 1):
                # return scalar
                return x.toarray()[0, 0]
            shape = (x.shape[1],)
            return SparseNDArray(x, shape=shape)
        return get_array_module(x).asarray(x)

    def concatenate(self, other, axis=0):
        if other.ndim != 1:
            raise ValueError('all the input arrays must have same number of dimensions')

        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if issparse(other):
            xps = get_sparse_module(self.spmatrix)
            if axis != 0:
                raise ValueError('axis can only be 0')
            other = other.reshape(1, other.shape[0]) if other.shape[0] != 1 else other
            x = xps.hstack((self.spmatrix.reshape(1, self.shape[0]), other))
        else:
            xp = get_array_module(self.spmatrix)
            x = xp.concatenate((self.spmatrix.toarray().reshape(self.shape), other), axis=axis)

        if issparse(x):
            return SparseNDArray(x, shape=(x.shape[1],))
        return get_array_module(x).asarray(x)

    def _reduction(self, method_name, axis=None, dtype=None, keepdims=None, todense=False, **kw):
        if not todense:
            assert keepdims is None or keepdims is False

        if isinstance(axis, tuple):
            assert axis == (0, )
            axis = None

        if todense:
            x = self.spmatrix.toarray()
            x = getattr(get_array_module(x), method_name)(x, axis=axis, **kw)
        else:
            x = getattr(self.spmatrix, method_name)(axis=axis, **kw)

        m = get_array_module(x)
        return m.array([x])[0]

    def __setitem__(self, key, value):
        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            x = self.spmatrix.tolil()
            key = (0,) + (key, )
            x[key] = value
            x = x.tocsr()
        self.spmatrix = x
