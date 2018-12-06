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

from .core import issparse, get_array_module, is_cupy, cp, cps, \
    get_sparse_module, naked, sps
from .array import SparseNDArray


def zeros_sparse_matrix(shape, dtype=float, gpu=False):
    m = sps if not gpu else cps
    return SparseMatrix(m.csr_matrix(shape, dtype=np.dtype(dtype)))


def diag_sparse_matrix(v, k=0, gpu=False):
    v = naked(v)
    if gpu and get_array_module(v) is not cp:
        v = cp.asarray(v)
    if not gpu and get_array_module(v) is not np:
        v = v.get()

    sparse_m = sps if not gpu else cps
    m = n = v.size + k
    mat = sparse_m.spdiags(v[None], [k], m, n, format='csr')
    return SparseMatrix(mat)


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
    assparsematrix = lambda i: SparseMatrix(i) if issparse(i) else i

    cond, x, y = [assparsematrix(i) for i in (cond, x, y)]
    return cond * x + (cond * (-y) + y)


class SparseMatrix(SparseNDArray):
    __slots__ = 'spmatrix',

    def __init__(self, spmatrix):
        if isinstance(spmatrix, SparseMatrix):
            self.spmatrix = spmatrix.spmatrix
        else:
            self.spmatrix = spmatrix.tocsr()

    @property
    def ndim(self):
        return self.spmatrix.ndim

    def toarray(self):
        return self.spmatrix.toarray()

    def todense(self):
        return self.spmatrix.toarray()

    def ascupy(self):
        is_cp = get_array_module(self.spmatrix) is cp
        if is_cp:
            return self
        mat_tuple = (cp.asarray(self.data), cp.asarray(self.indices), cp.asarray(self.indptr))
        return SparseMatrix(cps.csr_matrix(mat_tuple, shape=self.shape))

    def asscipy(self):
        is_cp = get_array_module(self.spmatrix) is cp
        if not is_cp:
            return self
        return SparseMatrix(self.spmatrix.get())

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
        return self.spmatrix.shape

    @property
    def dtype(self):
        return self.spmatrix.dtype

    def copy(self):
        return SparseMatrix(self.spmatrix.copy())

    @property
    def real(self):
        xps = get_sparse_module(self.spmatrix)
        return SparseMatrix(xps.csr_matrix(
            (self.spmatrix.data.real, self.spmatrix.indices, self.spmatrix.indptr),
            self.spmatrix.shape
        ))

    @real.setter
    def real(self, r):
        xps = get_sparse_module(self.spmatrix)
        x = self.spmatrix.toarray()
        if issparse(r):
            r = r.toarray()
        x.real = r
        self.spmatrix = xps.csr_matrix(x)

    @property
    def imag(self):
        xps = get_sparse_module(self.spmatrix)
        return SparseMatrix(xps.csr_matrix(
            (self.spmatrix.data.imag, self.spmatrix.indices, self.spmatrix.indptr),
            self.spmatrix.shape
        ))

    @imag.setter
    def imag(self, imag):
        xps = get_sparse_module(self.spmatrix)
        x = self.spmatrix.toarray()
        if issparse(imag):
            imag = imag.toarray()
        x.imag = imag
        self.spmatrix = xps.csr_matrix(x)

    def __getattr__(self, attr):
        is_cp = get_array_module(self.spmatrix) is cp
        if attr == 'device' and is_cp:
            try:
                return self.spmatrix.device
            except NotImplementedError:
                return cp.cuda.Device(0)
        if attr == 'get' and is_cp:
            return lambda: SparseMatrix(self.spmatrix.get())

        return super(SparseMatrix, self).__getattribute__(attr)

    def astype(self, dtype):
        dtype = np.dtype(dtype)
        if self.dtype == dtype:
            return self
        return SparseMatrix(self.spmatrix.astype(dtype))

    def transpose(self, axes=None):
        assert axes is None or tuple(axes) == (1, 0)
        return SparseMatrix(self.spmatrix.transpose())

    def swapaxes(self, axis1, axis2):
        if axis1 == 0 and axis2 == 1:
            return self

        assert axis1 == 1 and axis2 == 0
        return self.transpose()

    def reshape(self, shape):
        return SparseMatrix(self.spmatrix.tolil().reshape(shape))

    def broadcast_to(self, shape):
        # TODO(jisheng): implement broadcast_to
        raise NotImplementedError

    def squeeze(self, axis=None):
        # TODO(jisheng): implement squeeze
        raise NotImplementedError

    @property
    def T(self):
        return SparseMatrix(self.spmatrix.T)

    # ---------------- arithmetic ----------------------

    def __add__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented
        xp = get_array_module(self.spmatrix)
        if xp is cp and issparse(other) and \
                xp.all(self.spmatrix.indices == other.indices) and \
                xp.all(self.spmatrix.indptr == other.indptr):
            x = cps.csr_matrix(
                (self.spmatrix.data + other.data, self.spmatrix.indices, self.spmatrix.indptr),
                self.spmatrix.shape)
        else:
            try:
                x = self.spmatrix + other
            except NotImplementedError:
                x = self.spmatrix.toarray() + other
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def __radd__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented
        xp = get_array_module(self.spmatrix)
        if xp is cp and issparse(other) and \
                xp.all(self.spmatrix.indices == other.indices) and \
                xp.all(self.spmatrix.indptr == other.indptr):
            x = cps.csr_matrix(
                (other.data + self.spmatrix.data, self.spmatrix.indices, self.spmatrix.indptr),
                self.spmatrix.shape)
        else:
            try:
                x = other + self.spmatrix
            except NotImplementedError:
                x = other + self.spmatrix.toarray()
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def __sub__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented
        try:
            x = self.spmatrix - other
        except NotImplementedError:
            x = self.spmatrix.toarray() - other
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def __rsub__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented
        try:
            x = other - self.spmatrix
        except NotImplementedError:
            x = other - self.spmatrix.toarray()
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def __mul__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented
        if is_cupy(self.spmatrix):
            if not cp.isscalar(other):
                # TODO(jisheng): cupy does not implement multiply method
                is_other_sparse = issparse(other)
                if is_other_sparse and self.spmatrix.nnz == other.nnz and \
                        cp.all(self.spmatrix.indptr == other.indptr) and \
                        cp.all(self.spmatrix.indices == other.indices):
                    x = cps.csr_matrix((self.spmatrix.data * other.data,
                                        self.spmatrix.indices,
                                        self.spmatrix.indptr), self.spmatrix.shape)
                else:
                    if is_other_sparse:
                        other = other.toarray()
                    dense = self.spmatrix.toarray()
                    res = cp.multiply(dense, other, out=dense)
                    x = cps.csr_matrix(res)
            else:
                x = self.spmatrix * other
        else:
            x = self.spmatrix.multiply(other)
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def __rmul__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented
        if is_cupy(self.spmatrix):
            if not cp.isscalar(other):
                # TODO(jisheng): cupy does not implement multiply method
                is_other_sparse = issparse(other)
                if is_other_sparse and self.spmatrix.nnz == other.nnz and \
                        cp.all(self.spmatrix.indptr == other.indptr) and \
                        cp.all(self.spmatrix.indices == other.indices):
                    x = cps.csr_matrix((other.data * self.spmatrix.data,
                                        self.spmatrix.indices,
                                        self.spmatrix.indptr), self.spmatrix.shape)
                else:
                    if is_other_sparse:
                        other = other.toarray()
                    dense = self.spmatrix.toarray()
                    res = cp.multiply(other, dense, out=dense)
                    x = cps.csr_matrix(res)
            else:
                x = other * self.spmatrix
        else:
            x = self.spmatrix.multiply(other)
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def __div__(self, other):
        return self.__truediv__(other)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __truediv__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented
        x = self.spmatrix / other
        if issparse(x):
            return SparseMatrix(x)
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
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def __floordiv__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented
        if get_array_module(other).isscalar(other):
            m = get_sparse_module(self.spmatrix)
            data = self.spmatrix.data // other
            x = m.csr_matrix((data, self.spmatrix.indices, self.spmatrix.indptr),
                             self.spmatrix.shape)
        else:
            if issparse(other):
                other = other.toarray()
            x = get_sparse_module(self.spmatrix).csr_matrix(
                self.spmatrix.toarray() // other)
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def __rfloordiv__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented
        is_sparse = issparse(other)
        if is_sparse:
            other = other.toarray()
        x = other // self.spmatrix.toarray()
        if is_sparse:
            x = get_sparse_module(x).csr_matrix(x)
        return get_array_module(x).asarray(x)

    def __pow__(self, other, modulo=None):
        if modulo is not None:
            return NotImplemented

        try:
            other = naked(other)
        except TypeError:
            return NotImplemented
        if get_array_module(other).isscalar(other):
            if other >= 0:
                x = self.spmatrix.power(other)
            else:
                data = 1 / (self.spmatrix.data ** -other)
                x = get_sparse_module(self.spmatrix).csr_matrix(
                    (data, self.spmatrix.indices, self.spmatrix.indptr),
                    self.spmatrix.shape)
        else:
            if issparse(other):
                other = other.toarray()
            x = self.spmatrix.toarray() ** other
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def __rpow__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented
        if issparse(other):
            other = other.toarray()
        x = other ** self.spmatrix.toarray()
        return get_array_module(x).asarray(x)

    def float_power(self, other):
        ret = self.__pow__(other)
        ret = naked(ret).astype(float)
        if issparse(ret):
            return SparseMatrix(ret)
        return ret

    def __mod__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented
        if get_array_module(other).isscalar(other):
            data = self.spmatrix.data % other
            x = get_sparse_module(self.spmatrix).csr_matrix(
                (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape)
        else:
            if issparse(other):
                other = other.toarray()
            x = get_sparse_module(self.spmatrix).csr_matrix(
                self.spmatrix.toarray() % other)
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def __rmod__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented
        is_sparse = issparse(other)
        if issparse(other):
            other = other.toarray()
        if get_array_module(other).isscalar(other):
            data = other % self.spmatrix.data
            x = get_sparse_module(self.spmatrix).csr_matrix(
                (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape)
        else:
            x = other % self.spmatrix.toarray()
            if is_sparse:
                x = get_sparse_module(self.spmatrix).csr_matrix(x)
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def fmod(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        xp = get_array_module(self.spmatrix)
        if get_array_module(other).isscalar(other):
            data = xp.fmod(self.spmatrix.data, other)
            x = get_sparse_module(self.spmatrix).csr_matrix(
                (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape)
        else:
            if issparse(other):
                other = other.toarray()
            x = get_sparse_module(self.spmatrix).csr_matrix(
                xp.fmod(self.spmatrix.toarray(), other))
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def logaddexp(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        xp = get_array_module(self.spmatrix)
        if issparse(other):
            other = other.toarray()
        return xp.logaddexp(self.spmatrix.toarray(), other)

    def logaddexpr2(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        xp = get_array_module(self.spmatrix)
        if issparse(other):
            other = other.toarray()
        return xp.logaddexp2(self.spmatrix.toarray(), other)

    def __neg__(self):
        return SparseMatrix(-self.spmatrix)

    def __pos__(self):
        return SparseMatrix(self.spmatrix.copy())

    def __abs__(self):
        return SparseMatrix(abs(self.spmatrix))

    def fabs(self):
        xp = get_array_module(self.spmatrix)
        return SparseMatrix(get_sparse_module(self.spmatrix).csr_matrix(
            xp.abs(self.spmatrix), dtype='f8'))

    def rint(self):
        return SparseMatrix(self.spmatrix.rint())

    def sign(self):
        return SparseMatrix(self.spmatrix.sign())

    def conj(self):
        return SparseMatrix(self.spmatrix.conj())

    def exp(self):
        xp = get_array_module(self.spmatrix)
        return xp.exp(self.spmatrix.toarray())

    def exp2(self):
        xp = get_array_module(self.spmatrix)
        return xp.exp2(self.spmatrix.toarray())

    def log(self):
        xp = get_array_module(self.spmatrix)
        return xp.log(self.spmatrix.toarray())

    def log2(self):
        xp = get_array_module(self.spmatrix)
        return xp.log2(self.spmatrix.toarray())

    def log10(self):
        xp = get_array_module(self.spmatrix)
        return xp.log10(self.spmatrix.toarray())

    def expm1(self):
        return SparseMatrix(self.spmatrix.expm1())

    def log1p(self):
        return SparseMatrix(self.spmatrix.log1p())

    def sqrt(self):
        return SparseMatrix(self.spmatrix.sqrt())

    def square(self):
        xp = get_array_module(self.spmatrix)
        data = xp.square(self.spmatrix.data)
        x = get_sparse_module(self.spmatrix).csr_matrix(
            (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape
        )
        return SparseMatrix(x)

    def cbrt(self):
        xp = get_array_module(self.spmatrix)
        if hasattr(xp, 'cbrt'):
            data = xp.cbrt(self.spmatrix.data)
        else:
            data = self.spmatrix.data ** (1 / 3)
        x = get_sparse_module(self.spmatrix).csr_matrix(
            (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape
        )
        return SparseMatrix(x)

    def reciprocal(self):
        xp = get_array_module(self.spmatrix)
        return xp.reciprocal(self.spmatrix.toarray())

    def __eq__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            x = self.spmatrix == other
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def __ne__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            x = self.spmatrix != other
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def __lt__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            x = self.spmatrix < other
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def __le__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            x = self.spmatrix <= other
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def __gt__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            x = self.spmatrix > other
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def __ge__(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            try:
                x = self.spmatrix >= other
            except NotImplementedError:
                x = self.spmatrix.toarray() >= other
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def logical_and(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            other_xp = get_array_module(other)
            if other_xp.isscalar(other):
                other = other_xp.array(other).astype(bool)
            else:
                other = other.astype(bool)
            x = self.spmatrix.astype(bool).multiply(other)
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def logical_or(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            other_xp = get_array_module(other)
            if other_xp.isscalar(other):
                if other != 0:
                    x = np.logical_and(self.spmatrix.toarray(), other)
                else:
                    x = self.spmatrix.astype(bool)
            else:
                other = other.astype(bool)
                x = (self.spmatrix.astype(bool) + other).astype(bool)
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def logical_xor(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            other_xp = get_array_module(other)
            if other_xp.isscalar(other):
                other = other_xp.array(other).astype(bool)
            else:
                other = other.astype(bool)
            x = self.spmatrix.astype(bool) != other
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def logical_not(self):
        xp = get_array_module(self.spmatrix)
        return xp.logical_not(self.spmatrix.toarray())

    @staticmethod
    def _bitwise(this, other, method_name):
        try:
            this = naked(this)
        except TypeError:
            return NotImplemented
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if not issparse(this):
            return SparseMatrix._bitwise(other, this, method_name)

        if issparse(other):
            other = other.toarray()

        xp = get_array_module(this)
        xps = get_sparse_module(this)
        return SparseMatrix(xps.csr_matrix(getattr(xp, method_name)(this.toarray(), other)))

    def __and__(self, other):
        return self._bitwise(self.spmatrix, other, 'bitwise_and')

    def __rand__(self, other):
        return self._bitwise(other, self.spmatrix, 'bitwise_and')

    def __or__(self, other):
        return self._bitwise(self.spmatrix, other, 'bitwise_or')

    def __ror__(self, other):
        return self._bitwise(other, self.spmatrix, 'bitwise_or')

    def __xor__(self, other):
        return self._bitwise(self.spmatrix, other, 'bitwise_xor')

    def __rxor__(self, other):
        return self._bitwise(other, self.spmatrix, 'bitwise_xor')

    def isclose(self, other, **kw):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        xp = get_array_module(other)
        if issparse(other):
            other = other.toarray()
        return xp.isclose(self.spmatrix.toarray(), other, **kw)

    def __invert__(self):
        xp = get_array_module(self.spmatrix)
        return xp.invert(self.spmatrix.toarray())

    @staticmethod
    def _shift(this, other, method_name):
        try:
            this = naked(this)
        except TypeError:
            return NotImplemented
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        xps = get_sparse_module(this)
        xp = get_array_module(this)

        if xp.isscalar(this):
            other_xp = get_array_module(other)
            data = getattr(other_xp, method_name)(this, other.data)
            indices, indptr, shape = other.indices, other.indptr, other.shape
        elif isinstance(this, xp.ndarray):
            # dense
            return getattr(xp, method_name)(this, other.toarray())
        else:
            tp = np.int32 if is_cupy(this) else np.bool_  # cupy.sparse does not support bool
            mask = xps.csr_matrix(
                ((this.data > 0).astype(tp), this.indices, this.indptr),
                this.shape)
            other = mask.multiply(other)
            indices, indptr, shape = this.indices, this.indptr, this.shape
            data = getattr(xp, method_name)(this.data, other.data)

        return SparseMatrix(xps.csr_matrix((data, indices, indptr), shape))

    def __lshift__(self, other):
        return self._shift(self.spmatrix, other, 'left_shift')

    def __rlshift__(self, other):
        return self._shift(other, self.spmatrix, 'left_shift')

    def __rshift__(self, other):
        return self._shift(self.spmatrix, other, 'right_shift')

    def __rrshift__(self, other):
        return self._shift(other, self.spmatrix, 'right_shift')

    def sin(self):
        return SparseMatrix(self.spmatrix.sin())

    def cos(self):
        xp = get_array_module(self.spmatrix)
        return xp.cos(self.spmatrix.toarray())

    def tan(self):
        return SparseMatrix(self.spmatrix.tan())

    def arcsin(self):
        return SparseMatrix(self.spmatrix.arcsin())

    def arccos(self):
        xp = get_array_module(self.spmatrix)
        return xp.arccos(self.spmatrix.toarray())

    def arctan(self):
        return SparseMatrix(self.spmatrix.arctan())

    def arctan2(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        xp = get_array_module(self.spmatrix)
        if issparse(other):
            other = other.toarray()
        x = xp.arctan2(self.spmatrix.toarray(), other)
        return SparseMatrix(get_sparse_module(x).csr_matrix(x))

    def hypot(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        xp = get_array_module(self.spmatrix)
        if issparse(other):
            other = other.toarray()
        x = xp.hypot(self.spmatrix.toarray(), other)
        return SparseMatrix(get_sparse_module(x).csr_matrix(x))

    def sinh(self):
        return SparseMatrix(self.spmatrix.sinh())

    def cosh(self):
        xp = get_array_module(self.spmatrix)
        return xp.cosh(self.spmatrix.toarray())

    def tanh(self):
        return SparseMatrix(self.spmatrix.tanh())

    def arcsinh(self):
        return SparseMatrix(self.spmatrix.arcsinh())

    def arccosh(self):
        xp = get_array_module(self.spmatrix)
        return xp.arccosh(self.spmatrix.toarray())

    def arctanh(self):
        return SparseMatrix(self.spmatrix.arctanh())

    def around(self, decimals=0):
        xp = get_array_module(self.spmatrix)
        data = xp.around(self.spmatrix.data, decimals=decimals)
        x = get_sparse_module(self.spmatrix).csr_matrix(
            (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape
        )
        return SparseMatrix(x)

    def deg2rad(self):
        return SparseMatrix(self.spmatrix.deg2rad())

    def rad2deg(self):
        return SparseMatrix(self.spmatrix.rad2deg())

    def angle(self, deg=0):
        xp = get_array_module(self.spmatrix)
        data = xp.angle(self.spmatrix.data, deg=deg)
        x = get_sparse_module(self.spmatrix).csr_matrix(
            (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape
        )
        return SparseMatrix(x)

    def dot(self, other, sparse=True):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if sparse:
            x = self.spmatrix.dot(other)
        else:
            a = self.spmatrix.toarray()
            if issparse(other):
                other = other.toarray()
            x = a.dot(other)
        if issparse(x):
            return SparseMatrix(x)
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
        if not todense:
            assert keepdims is None or keepdims is False

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
        if issparse(x):
            return SparseMatrix(x)
        shape = (set(self.spmatrix.shape) - set([axis])).pop()
        m = get_array_module(x)
        if m.isscalar(x):
            return m.array([x])[0]
        else:
            return m.asarray(x).reshape(shape)

    def sum(self, axis=None, dtype=None, keepdims=None):
        return self._reduction('sum', axis=axis, dtype=dtype, keepdims=keepdims)

    def prod(self, axis=None, dtype=None, keepdims=None):
        return self._reduction('sum', axis=axis, dtype=dtype, keepdims=keepdims, todense=True)

    def amax(self, axis=None, dtype=None, keepdims=None):
        return self._reduction('max', axis=axis, dtype=dtype, keepdims=keepdims)

    def amin(self, axis=None, dtype=None, keepdims=None):
        return self._reduction('min', axis=axis, dtype=dtype, keepdims=keepdims)

    def all(self, axis=None, dtype=None, keepdims=None):
        ret = self._reduction('all', axis=axis, dtype=dtype, keepdims=keepdims, todense=True)
        if not issparse(ret):
            xps = get_sparse_module(self.spmatrix)
            ret = SparseMatrix(xps.csr_matrix(ret))
            return ret
        return ret

    def any(self, axis=None, dtype=None, keepdims=None):
        ret = self._reduction('any', axis=axis, dtype=dtype, keepdims=keepdims, todense=True)
        if not issparse(ret):
            xps = get_sparse_module(self.spmatrix)
            ret = SparseMatrix(xps.csr_matrix(ret))
            return ret
        return ret

    def mean(self, axis=None, dtype=None, keepdims=None):
        return self._reduction('mean', axis=axis, dtype=dtype, keepdims=keepdims)

    def nansum(self, axis=None, dtype=None, keepdims=None):
        return self._reduction('nansum', axis=axis, dtype=dtype, keepdims=keepdims, todense=True)

    def nanprod(self, axis=None, dtype=None, keepdims=None):
        return self._reduction('nanprod', axis=axis, dtype=dtype, keepdims=keepdims, todense=True)

    def nanmax(self, axis=None, dtype=None, keepdims=None):
        ret = self._reduction('nanmax', axis=axis, dtype=dtype, keepdims=keepdims, todense=True)
        if not issparse(ret):
            xps = get_sparse_module(self.spmatrix)
            ret = SparseMatrix(xps.csr_matrix(ret))
            return ret
        return ret

    def nanmin(self, axis=None, dtype=None, keepdims=None):
        ret = self._reduction('nanmin', axis=axis, dtype=dtype, keepdims=keepdims, todense=True)
        if not issparse(ret):
            xps = get_sparse_module(self.spmatrix)
            ret = SparseMatrix(xps.csr_matrix(ret))
            return ret
        return ret

    def nanmean(self, axis=None, dtype=None, keepdims=None):
        return self._reduction('nanmean', axis=axis, dtype=dtype, keepdims=keepdims, todense=True)

    def argmax(self, axis=None, dtype=None, keepdims=None):
        return self._reduction('argmax', axis=axis, dtype=dtype, keepdims=keepdims)

    def nanargmax(self, axis=None, dtype=None, keepdims=None):
        return self._reduction('nanargmax', axis=axis, dtype=dtype, keepdims=keepdims, todense=True)

    def argmin(self, axis=None, dtype=None, keepdims=None):
        return self._reduction('argmin', axis=axis, dtype=dtype, keepdims=keepdims)

    def nanargmin(self, axis=None, dtype=None, keepdims=None):
        return self._reduction('nanargmin', axis=axis, dtype=dtype, keepdims=keepdims, todense=True)

    def var(self, axis=None, dtype=None, ddof=0, keepdims=None):
        return self._reduction('var', axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, todense=True)

    def cumsum(self, axis=None, dtype=None):
        return self.spmatrix.toarray().cumsum(axis=axis)

    def cumprod(self, axis=None, dtype=None):
        return self.spmatrix.toarray().cumprod(axis=axis)

    def nancumsum(self, axis=None, dtype=None):
        xp = get_array_module(self.spmatrix)
        return xp.nancumsum(self.spmatrix.toarray(), axis=axis)

    def nancumprod(self, axis=None, dtype=None):
        xp = get_array_module(self.spmatrix)
        return xp.nancumprod(self.spmatrix.toarray(), axis=axis)

    def count_nonzero(self, axis=None, dtype=None, keepdims=None):
        if axis is None:
            return get_array_module(self.spmatrix).array([self.spmatrix.count_nonzero()])[0]
        else:
            return get_array_module(self.spmatrix).count_nonzero(self.spmatrix.toarray(), axis=axis)

    def __getitem__(self, item):
        get = lambda x: x.spmatrix if isinstance(x, SparseMatrix) else x
        item = get(item)
        if isinstance(item, list):
            item = tuple(item)

        if not all(isinstance(i, slice) for i in item):
            raise NotImplementedError('sparse matrix only support slice for indexing')

        x = self.spmatrix[item]
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def __setitem__(self, key, value):
        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            x = self.spmatrix.tolil()
            x[key] = value
            x = x.tocsr()
        self.spmatrix = x

    def _maximum_minimum(self, other, method_name):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if is_cupy(self.spmatrix):
            # TODO(jisheng): cupy does not implement sparse maximum and minimum
            return NotImplemented

        xps = get_sparse_module(self.spmatrix)
        xp = get_array_module(self.spmatrix)
        has_nan = xps.csr_matrix(
            (xp.isnan(self.spmatrix.data), self.spmatrix.indices, self.spmatrix.indptr),
            self.spmatrix.shape)
        if issparse(other):
            has_nan += xps.csr_matrix(
                (xp.isnan(other.data), other.indices, other.indptr), other.shape)

        x = getattr(self.spmatrix, method_name)(other)
        if has_nan.sum() > 0:
            x = x + (has_nan * np.nan)

        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def maximum(self, other):
        return self._maximum_minimum(other, 'maximum')

    def minimum(self, other):
        return self._maximum_minimum(other, 'minimum')

    def fmax(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        x = self.spmatrix.maximum(other)
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def fmin(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        x = self.spmatrix.minimum(other)
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def isinf(self):
        xp = get_array_module(self.spmatrix)
        data = xp.isinf(self.spmatrix.data)
        x = get_sparse_module(self.spmatrix).csr_matrix(
            (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape
        )
        return SparseMatrix(x)

    def isnan(self):
        xp = get_array_module(self.spmatrix)
        data = xp.isnan(self.spmatrix.data)
        x = get_sparse_module(self.spmatrix).csr_matrix(
            (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape
        )
        return SparseMatrix(x)

    def signbit(self):
        xp = get_array_module(self.spmatrix)
        data = xp.signbit(self.spmatrix.data)
        x = get_sparse_module(self.spmatrix).csr_matrix(
            (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape
        )
        return SparseMatrix(x)

    def floor(self):
        return SparseMatrix(self.spmatrix.floor())

    def ceil(self):
        return SparseMatrix(self.spmatrix.ceil())

    def trunc(self):
        return SparseMatrix(self.spmatrix.trunc())

    def degrees(self):
        xp = get_array_module(self.spmatrix)
        data = xp.degrees(self.spmatrix.data)
        x = get_sparse_module(self.spmatrix).csr_matrix(
            (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape
        )
        return SparseMatrix(x)

    def radians(self):
        xp = get_array_module(self.spmatrix)
        data = xp.radians(self.spmatrix.data)
        x = get_sparse_module(self.spmatrix).csr_matrix(
            (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape
        )
        return SparseMatrix(x)

    def clip(self, a_min, a_max):
        try:
            a_min = naked(a_min)
        except TypeError:
            return NotImplemented

        try:
            a_max = naked(a_max)
        except TypeError:
            return NotImplemented

        x = self.spmatrix.maximum(a_min)
        if issparse(x):
            x = x.minimum(a_max)
        elif issparse(a_max):
            x = a_max.minimum(x)
        else:
            xp = get_array_module(x)
            x = xp.minimum(x, a_max)
        if issparse(x):
            return SparseMatrix(x)
        return get_array_module(x).asarray(x)

    def iscomplex(self):
        xp = get_array_module(self.spmatrix)
        data = xp.iscomplex(self.spmatrix.data)
        x = get_sparse_module(self.spmatrix).csr_matrix(
            (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape
        )
        return SparseMatrix(x)

    def fix(self):
        xp = get_array_module(self.spmatrix)
        data = xp.fix(self.spmatrix.data)
        x = get_sparse_module(self.spmatrix).csr_matrix(
            (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape
        )
        return SparseMatrix(x)

    def i0(self):
        xp = get_array_module(self.spmatrix)
        data = xp.i0(self.spmatrix.data).reshape(self.spmatrix.data.shape)
        x = get_sparse_module(self.spmatrix).csr_matrix(
            (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape
        )
        return SparseMatrix(x)

    def nan_to_num(self):
        xp = get_array_module(self.spmatrix)
        data = xp.nan_to_num(self.spmatrix.data)
        x = get_sparse_module(self.spmatrix).csr_matrix(
            (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape
        )
        return SparseMatrix(x)

    def copysign(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if issparse(other):
            other = other.toarray()

        xp = get_array_module(self.spmatrix)
        return xp.copysign(self.spmatrix.toarray(), other)

    def nextafter(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        ret_sparse = False
        if issparse(other):
            ret_sparse = True
            other = other.toarray()

        xp = get_array_module(self.spmatrix)
        xps = get_sparse_module(self.spmatrix)

        x = xp.nextafter(self.spmatrix.toarray(), other)
        if ret_sparse:
            return SparseMatrix(xps.csr_matrix(x))
        return x

    def spacing(self):
        return get_array_module(self.spmatrix).spacing(self.spmatrix.toarray())

    def ldexp(self, other):
        try:
            other = naked(other)
        except TypeError:
            return NotImplemented

        if issparse(other):
            other = other.toarray()

        return SparseMatrix(self.spmatrix.multiply(2 ** other))

    def frexp(self, **kw):
        xp = get_array_module(self.spmatrix)
        xps = get_sparse_module(self.spmatrix)
        x, y = xp.frexp(self.spmatrix.toarray(), **kw)
        return SparseMatrix(xps.csr_matrix(x)), SparseMatrix(xps.csr_matrix(y))

    def modf(self, **kw):
        xp = get_array_module(self.spmatrix)
        xps = get_sparse_module(self.spmatrix)
        x, y = xp.modf(self.spmatrix.toarray(), **kw)
        return SparseMatrix(xps.csr_matrix(x)), SparseMatrix(xps.csr_matrix(y))

    def sinc(self):
        xp = get_array_module(self.spmatrix)
        return xp.sinc(self.spmatrix.toarray())

    def isfinite(self):
        xp = get_array_module(self.spmatrix)
        return xp.isfinite(self.spmatrix.toarray())

    def isreal(self):
        xp = get_array_module(self.spmatrix)
        return xp.isreal(self.spmatrix.toarray())

    def digitize(self, bins, right=False):
        xp = get_array_module(self.spmatrix)
        data = xp.digitize(self.spmatrix.data, bins, right)
        x = get_sparse_module(self.spmatrix).csr_matrix(
            (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape
        )
        return SparseMatrix(x)

    def repeat(self, repeats, axis=None):
        if axis is None:
            raise NotImplementedError

        xp = get_array_module(self.spmatrix)
        xps = get_sparse_module(self.spmatrix)
        x = xps.csr_matrix(xp.repeat(self.spmatrix.toarray(), repeats, axis=axis))
        return SparseMatrix(x)
