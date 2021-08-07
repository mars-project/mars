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

from functools import partialmethod

from ...utils import ceildiv
from .core import issparse, get_array_module, \
    get_sparse_module, cp, cps, np, naked, is_cupy


class SparseNDArray:
    __slots__ = '__weakref__',
    __array_priority__ = 21

    def __new__(cls, *args, **kwargs):

        shape = kwargs.get('shape', None)
        if shape is not None and len(shape) == 1:
            from .vector import SparseVector

            return object.__new__(SparseVector)
        if len(args) == 1 and issparse(args[0]) and args[0].ndim == 2:
            from .matrix import SparseMatrix

            return object.__new__(SparseMatrix)

        else:
            from .coo import COONDArray
            return object.__new__(COONDArray)

    @property
    def raw(self):
        raise NotImplementedError


def call_sparse(method, *args, **kwargs):
    new_args = []
    make_dense = False
    matrix = None
    for arg in args:
        if hasattr(arg, 'spmatrix'):
            # todo add support for multiple sparse arrays
            if make_dense or matrix is not None:
                make_dense = True
            matrix = arg
            new_args.append(matrix.spmatrix.data)
        else:
            if isinstance(arg, np.ndarray):
                make_dense = True
            new_args.append(arg)

    spmatrix = matrix.spmatrix
    if make_dense:
        new_args = [arg.toarray() if hasattr(arg, 'spmatrix') else arg
                    for arg in args]

    xp = get_array_module(spmatrix)
    try:
        new_data = getattr(xp, method)(*new_args, **kwargs)
    except AttributeError:
        if xp is np:
            from scipy import special
        else:
            from cupyx.scipy import special
        new_data = getattr(special, method)(*new_args, **kwargs)

    if not make_dense:
        new_spmatrix = get_sparse_module(spmatrix).csr_matrix(
            (new_data, spmatrix.indices, spmatrix.indptr), spmatrix.shape)
    else:
        new_spmatrix = get_sparse_module(spmatrix).csr_matrix(new_data)
    return SparseNDArray(new_spmatrix, shape=matrix.shape)


class SparseArray(SparseNDArray):
    __slots__ = 'spmatrix',

    @property
    def ndim(self):
        return len(self.shape)

    def tocsr(self):
        return self

    def toarray(self):
        if self.shape != self.spmatrix.shape:
            return self.spmatrix.toarray().reshape(self.shape)
        else:
            return self.spmatrix.toarray()

    def todense(self):
        return self.toarray()

    def ascupy(self):
        is_cp = get_array_module(self.spmatrix) is cp
        if is_cp:
            return self
        mat_tuple = (cp.asarray(self.data), cp.asarray(self.indices), cp.asarray(self.indptr))
        return SparseNDArray(cps.csr_matrix(mat_tuple, shape=self.spmatrix.shape), shape=self.shape)

    def asscipy(self):
        is_cp = get_array_module(self.spmatrix) is cp
        if not is_cp:
            return self
        return SparseNDArray(self.spmatrix.get(), shape=self.shape)

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
        raise self.spmatrix.shape

    @property
    def dtype(self):
        return self.spmatrix.dtype

    def copy(self):
        return SparseNDArray(self.spmatrix.copy(), shape=self.shape)

    @property
    def real(self):
        xps = get_sparse_module(self.spmatrix)
        return SparseNDArray(xps.csr_matrix(
            (self.spmatrix.data.real, self.spmatrix.indices, self.spmatrix.indptr),
            self.spmatrix.shape
        ), shape=self.shape)

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
        return SparseNDArray(xps.csr_matrix(
            (self.spmatrix.data.imag, self.spmatrix.indices, self.spmatrix.indptr),
            self.spmatrix.shape), shape=self.shape)

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
            return lambda: SparseNDArray(self.spmatrix.get(), shape=self.shape)

        return super().__getattribute__(attr)

    def astype(self, dtype, **_):
        dtype = np.dtype(dtype)
        if self.dtype == dtype:
            return self
        return SparseNDArray(self.spmatrix.astype(dtype), shape=self.shape)

    def transpose(self, axes=None):
        raise NotImplementedError

    def swapaxes(self, axis1, axis2):
        if axis1 == 0 and axis2 == 1:
            return self

        assert axis1 == 1 and axis2 == 0
        return self.transpose()

    def reshape(self, shape, **_):
        sp_shape = shape if len(shape) == 2 else (1, shape[0])
        spmatrix = self.spmatrix.tolil().reshape(sp_shape)
        return SparseNDArray(spmatrix, shape=shape)

    def broadcast_to(self, shape):
        # TODO(jisheng): implement broadcast_to
        raise NotImplementedError

    def squeeze(self, axis=None):
        # TODO(jisheng): implement squeeze
        raise NotImplementedError

    @property
    def T(self):
        raise NotImplementedError

    # ---------------- arithmetic ----------------------

    def __add__(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented
        other_xp = get_array_module(naked_other)
        if other_xp.isscalar(naked_other):
            return call_sparse('add', self, naked_other)
        if issparse(naked_other):
            x = self.spmatrix + naked_other
        else:
            x = self.toarray() + naked_other
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __radd__(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented
        other_xp = get_array_module(naked_other)
        if other_xp.isscalar(naked_other):
            return call_sparse('add', naked_other, self)
        if issparse(naked_other):
            x = self.spmatrix + naked_other
        else:
            x = self.toarray() + naked_other
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __sub__(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented
        other_xp = get_array_module(naked_other)
        if other_xp.isscalar(naked_other):
            return call_sparse('subtract', self, naked_other)
        if issparse(naked_other):
            x = self.spmatrix - naked_other
        else:
            x = self.toarray() - naked_other
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __rsub__(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented
        other_xp = get_array_module(naked_other)
        if other_xp.isscalar(naked_other):
            return call_sparse('subtract', naked_other, self)
        if issparse(naked_other):
            x = naked_other - self.spmatrix
        else:
            x = naked_other - self.toarray()
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __mul__(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented
        if is_cupy(self.spmatrix):
            if not cp.isscalar(naked_other):
                # TODO(jisheng): cupy does not implement multiply method
                is_other_sparse = issparse(naked_other)
                if is_other_sparse and self.spmatrix.nnz == naked_other.nnz and \
                        cp.all(self.spmatrix.indptr == naked_other.indptr) and \
                        cp.all(self.spmatrix.indices == naked_other.indices):
                    x = cps.csr_matrix((self.spmatrix.data * naked_other.data,
                                        self.spmatrix.indices,
                                        self.spmatrix.indptr), self.spmatrix.shape)
                else:
                    if is_other_sparse:
                        naked_other = other.toarray()
                    dense = self.spmatrix.toarray()
                    res = cp.multiply(dense, naked_other, out=dense)
                    x = cps.csr_matrix(res)
            else:
                x = self.spmatrix * naked_other
        else:
            x = self.spmatrix.multiply(naked_other)
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __rmul__(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented
        if is_cupy(self.spmatrix):
            if not cp.isscalar(naked_other):
                # TODO(jisheng): cupy does not implement multiply method
                is_other_sparse = issparse(naked_other)
                if is_other_sparse and self.spmatrix.nnz == naked_other.nnz and \
                        cp.all(self.spmatrix.indptr == naked_other.indptr) and \
                        cp.all(self.spmatrix.indices == naked_other.indices):
                    x = cps.csr_matrix((naked_other.data * self.spmatrix.data,
                                        self.spmatrix.indices,
                                        self.spmatrix.indptr), self.spmatrix.shape)
                else:
                    if is_other_sparse:
                        naked_other = other.toarray()
                    dense = self.spmatrix.toarray()
                    res = cp.multiply(naked_other, dense, out=dense)
                    x = cps.csr_matrix(res)
            else:
                x = naked_other * self.spmatrix
        else:
            x = self.spmatrix.multiply(naked_other)
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __div__(self, other):
        return self.__truediv__(other)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __truediv__(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented
        x = self.spmatrix / naked_other
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __rtruediv__(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented
        try:
            x = naked_other / self.spmatrix
        except TypeError:
            x = naked_other / self.spmatrix.toarray()
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __floordiv__(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented
        other_xp = get_array_module(naked_other)
        if other_xp.isscalar(naked_other):
            return call_sparse('floor_divide', self, naked_other)
        else:
            if issparse(naked_other):
                naked_other = other.toarray()
                x = get_sparse_module(self.spmatrix).csr_matrix(
                    self.toarray() // naked_other)
            else:
                x = self.toarray() // naked_other
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __rfloordiv__(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented
        other_xp = get_array_module(naked_other)
        if other_xp.isscalar(naked_other):
            return call_sparse('floor_divide', naked_other, self)
        else:
            if issparse(naked_other):
                naked_other = other.toarray()
                x = get_sparse_module(self.spmatrix).csr_matrix(
                    naked_other // self.toarray())
            else:
                x = naked_other // self.toarray()
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __pow__(self, other, modulo=None):
        if modulo is not None:
            return NotImplemented

        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented
        if get_array_module(naked_other).isscalar(naked_other):
            x = self.spmatrix.power(naked_other)
        else:
            if issparse(naked_other):
                naked_other = other.toarray()
            x = self.toarray() ** naked_other
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __rpow__(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented
        if issparse(naked_other):
            naked_other = other.toarray()
        x = naked_other ** self.toarray()
        return get_array_module(x).asarray(x)

    def float_power(self, other):
        ret = self.__pow__(other)
        ret = naked(ret).astype(float)
        if issparse(ret):
            return SparseNDArray(ret, shape=self.shape)
        return ret

    def __mod__(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented
        if get_array_module(naked_other).isscalar(naked_other):
            data = self.spmatrix.data % naked_other
            x = get_sparse_module(self.spmatrix).csr_matrix(
                (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape)
        else:
            if issparse(naked_other):
                naked_other = other.toarray()
            x = get_sparse_module(self.spmatrix).csr_matrix(
                self.toarray() % naked_other)
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __rmod__(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented
        is_sparse = issparse(naked_other)
        if issparse(naked_other):
            naked_other = other.toarray()
        if get_array_module(naked_other).isscalar(naked_other):
            data = naked_other % self.spmatrix.data
            x = get_sparse_module(self.spmatrix).csr_matrix(
                (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape)
        else:
            x = naked_other % self.toarray()
            if is_sparse:
                x = get_sparse_module(self.spmatrix).csr_matrix(x)
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def fmod(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        xp = get_array_module(self.spmatrix)
        if get_array_module(naked_other).isscalar(naked_other):
            return call_sparse('fmod', self, naked_other)
        else:
            if issparse(naked_other):
                naked_other = other.toarray()
            x = get_sparse_module(self.spmatrix).csr_matrix(
                xp.fmod(self.toarray(), naked_other))
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def logaddexp(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        xp = get_array_module(self.spmatrix)
        if get_array_module(naked_other).isscalar(naked_other):
            return call_sparse('logaddexp', self, naked_other)
        if issparse(naked_other):
            naked_other = other.toarray()
        return xp.logaddexp(self.toarray(), naked_other)

    def logaddexp2(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        xp = get_array_module(self.spmatrix)
        if get_array_module(naked_other).isscalar(naked_other):
            return call_sparse('logaddexp2', self, naked_other)
        if issparse(naked_other):
            naked_other = other.toarray()
        return xp.logaddexp2(self.toarray(), naked_other)

    def __neg__(self):
        return SparseNDArray(-self.spmatrix, shape=self.shape)

    def __pos__(self):
        return SparseNDArray(self.spmatrix.copy(), shape=self.shape)

    def __abs__(self):
        return SparseNDArray(abs(self.spmatrix), shape=self.shape)

    def fabs(self):
        xp = get_array_module(self.spmatrix)
        return SparseNDArray(get_sparse_module(self.spmatrix).csr_matrix(
            xp.abs(self.spmatrix), dtype='f8'), shape=self.shape)

    def rint(self):
        return SparseNDArray(self.spmatrix.rint(), shape=self.shape)

    def sign(self):
        return SparseNDArray(self.spmatrix.sign(), shape=self.shape)

    def conj(self):
        return SparseNDArray(self.spmatrix.conj(), shape=self.shape)

    def exp(self):
        return call_sparse('exp', self)

    def exp2(self):
        return call_sparse('exp2', self)

    def log(self):
        return call_sparse('log', self)

    def log2(self):
        return call_sparse('log2', self)

    def log10(self):
        return call_sparse('log10', self)

    def expm1(self):
        return SparseNDArray(self.spmatrix.expm1(), shape=self.shape)

    def log1p(self):
        return SparseNDArray(self.spmatrix.log1p(), shape=self.shape)

    def sqrt(self):
        return SparseNDArray(self.spmatrix.sqrt(), shape=self.shape)

    def square(self):
        return call_sparse('square', self)

    def cbrt(self):
        return call_sparse('cbrt', self)

    def reciprocal(self):
        return call_sparse('reciprocal', self)

    def _scipy_unary(self, func_name):
        spmatrix = self.spmatrix
        xp = get_array_module(spmatrix)
        if xp is np:
            from scipy import special
        else:
            from cupyx.scipy import special

        new_data = getattr(special, func_name)(spmatrix.data)
        new_spmatrix = get_sparse_module(spmatrix).csr_matrix(
            (new_data, spmatrix.indices, spmatrix.indptr), spmatrix.shape)
        return SparseNDArray(new_spmatrix, shape=self.shape)

    def _scipy_binary(self, func_name, other):
        try:
            naked_other = naked(other)
        except TypeError:  # pragma: no cover
            return NotImplemented

        xp = get_array_module(self.spmatrix)

        if xp is np:
            from scipy import special
        else:  # pragma: no cover
            from cupyx.scipy import special

        func = getattr(special, func_name)

        if get_array_module(naked_other).isscalar(naked_other):  # pragma: no cover
            return call_sparse(func, self, naked_other)
        else:
            if issparse(naked_other):  # pragma: no cover
                naked_other = other.toarray()
            x = get_sparse_module(self.spmatrix).csr_matrix(
                func(self.toarray(), naked_other))
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    gamma = partialmethod(_scipy_unary, 'gamma')
    gammaln = partialmethod(_scipy_unary, 'gammaln')
    loggamma = partialmethod(_scipy_unary, 'loggamma')
    gammasgn = partialmethod(_scipy_unary, 'gammasgn')
    gammainc = partialmethod(_scipy_binary, 'gammainc')
    gammaincinv = partialmethod(_scipy_binary, 'gammaincinv')
    gammaincc = partialmethod(_scipy_binary, 'gammaincc')
    gammainccinv = partialmethod(_scipy_binary, 'gammainccinv')
    beta = partialmethod(_scipy_binary, 'beta')
    betaln = partialmethod(_scipy_binary, 'betaln')
    psi = partialmethod(_scipy_unary, 'psi')
    rgamma = partialmethod(_scipy_unary, 'rgamma')
    polygamma = partialmethod(_scipy_binary, 'polygamma')
    multigammaln = partialmethod(_scipy_binary, 'multigammaln')
    digamma = partialmethod(_scipy_unary, 'digamma')
    poch = partialmethod(_scipy_binary, 'poch')

    erf = partialmethod(_scipy_unary, 'erf')
    entr = partialmethod(_scipy_unary, 'entr')

    rel_entr = partialmethod(_scipy_binary, 'rel_entr')
    kl_div = partialmethod(_scipy_binary, 'kl_div')
    xlogy = partialmethod(_scipy_binary, 'xlogy')

    jv = partialmethod(_scipy_binary, 'jv')
    jve = partialmethod(_scipy_binary, 'jve')
    yn = partialmethod(_scipy_binary, 'yn')
    yv = partialmethod(_scipy_binary, 'yv')
    yve = partialmethod(_scipy_binary, 'yve')
    kn = partialmethod(_scipy_binary, 'kn')
    kv = partialmethod(_scipy_binary, 'kv')
    kve = partialmethod(_scipy_binary, 'kve')
    iv = partialmethod(_scipy_binary, 'iv')
    ive = partialmethod(_scipy_binary, 'ive')
    hankel1 = partialmethod(_scipy_binary, 'hankel1')
    hankel1e = partialmethod(_scipy_binary, 'hankel1e')
    hankel2 = partialmethod(_scipy_binary, 'hankel2')
    hankel2e = partialmethod(_scipy_binary, 'hankel2e')

    def __eq__(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        if get_array_module(naked_other).isscalar(naked_other):
            return call_sparse('equal', self, naked_other)
        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            if issparse(naked_other):
                x = self.spmatrix == naked_other
            else:
                x = self.toarray() == other
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __ne__(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        if get_array_module(naked_other).isscalar(naked_other):
            return call_sparse('not_equal', self, naked_other)
        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            if issparse(naked_other):
                x = self.spmatrix != naked_other
            else:
                x = self.toarray() != other
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __lt__(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        if get_array_module(naked_other).isscalar(naked_other):
            return call_sparse('less', self, naked_other)
        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            if issparse(naked_other):
                x = self.spmatrix < naked_other
            else:
                x = self.toarray() < other
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __le__(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        if get_array_module(naked_other).isscalar(naked_other):
            return call_sparse('less_equal', self, naked_other)
        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            if issparse(naked_other):
                x = self.spmatrix <= naked_other
            else:
                x = self.toarray() <= other
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __gt__(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        if get_array_module(naked_other).isscalar(naked_other):
            return call_sparse('greater', self, naked_other)
        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            if issparse(naked_other):
                x = self.spmatrix > naked_other
            else:
                x = self.toarray() > other
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def __ge__(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        if get_array_module(naked_other).isscalar(naked_other):
            return call_sparse('greater_equal', self, naked_other)
        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            if issparse(naked_other):
                x = self.spmatrix >= naked_other
            else:
                x = self.toarray() >= other
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def logical_and(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            other_xp = get_array_module(naked_other)
            if other_xp.isscalar(naked_other):
                naked_other = other_xp.array(naked_other).astype(bool)
            else:
                naked_other = naked_other.astype(bool)
            x = self.spmatrix.astype(bool).multiply(naked_other)
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def logical_or(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            other_xp = get_array_module(naked_other)
            if other_xp.isscalar(naked_other):
                if naked_other != 0:
                    x = np.logical_and(self.toarray(), naked_other)
                else:
                    x = self.spmatrix.astype(bool)
            else:
                naked_other = naked_other.astype(bool)
                x = (self.spmatrix.astype(bool) + naked_other).astype(bool)
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def logical_xor(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        if is_cupy(self.spmatrix):
            return NotImplemented
        else:
            other_xp = get_array_module(naked_other)
            if other_xp.isscalar(naked_other):
                naked_other = other_xp.array(naked_other).astype(bool)
            else:
                naked_other = naked_other.astype(bool)
            x = self.spmatrix.astype(bool) != naked_other
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def logical_not(self):
        return call_sparse('logical_not', self)

    @staticmethod
    def _bitwise(this, other, method_name):
        try:
            naked_this = naked(this)
        except TypeError:
            return NotImplemented
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        if not issparse(naked_this):
            return SparseArray._bitwise(naked_other, naked_this, method_name)

        if issparse(naked_other):
            naked_other = other.toarray()

        xp = get_array_module(naked_this)
        xps = get_sparse_module(naked_this)
        return SparseNDArray(xps.csr_matrix(getattr(xp, method_name)(this.toarray(), naked_other)),
                             shape=naked_this.shape)

    def __and__(self, other):
        if get_array_module(other).isscalar(other):
            return call_sparse('bitwise_and', self, other)
        return self._bitwise(self.spmatrix, other, 'bitwise_and')

    def __rand__(self, other):
        if get_array_module(other).isscalar(other):
            return call_sparse('bitwise_and', other, self)
        return self._bitwise(other, self.spmatrix, 'bitwise_and')

    def __or__(self, other):
        if get_array_module(other).isscalar(other):
            return call_sparse('bitwise_or', self, other)
        return self._bitwise(self.spmatrix, other, 'bitwise_or')

    def __ror__(self, other):
        if get_array_module(other).isscalar(other):
            return call_sparse('bitwise_or', other, self)
        return self._bitwise(other, self.spmatrix, 'bitwise_or')

    def __xor__(self, other):
        if get_array_module(other).isscalar(other):
            return call_sparse('bitwise_xor', self, other)
        return self._bitwise(self.spmatrix, other, 'bitwise_xor')

    def __rxor__(self, other):
        if get_array_module(other).isscalar(other):
            return call_sparse('bitwise_xor', other, self)
        return self._bitwise(other, self.spmatrix, 'bitwise_xor')

    def isclose(self, other, **kw):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        xp = get_array_module(naked_other)
        if issparse(naked_other):
            naked_other = other.toarray()
        return xp.isclose(self.toarray(), naked_other, **kw)

    def __invert__(self):
        return call_sparse('invert', self)

    @staticmethod
    def _shift(this, other, method_name):
        try:
            naked_this = naked(this)
        except TypeError:
            return NotImplemented
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        xps = get_sparse_module(naked_this)
        xp = get_array_module(naked_this)

        if xp.isscalar(naked_this):
            other_xp = get_array_module(naked_other)
            data = getattr(other_xp, method_name)(naked_this, naked_other.data)
            indices, indptr, shape = naked_other.indices, naked_other.indptr, naked_other.shape
        elif isinstance(naked_this, xp.ndarray):
            # dense
            return getattr(xp, method_name)(naked_this, other.toarray())
        else:
            tp = np.int32 if is_cupy(naked_this) else np.bool_  # cupy.sparse does not support bool
            mask = xps.csr_matrix(
                ((naked_this.data > 0).astype(tp), naked_this.indices, naked_this.indptr),
                naked_this.shape)
            naked_other = mask.multiply(naked_other)
            indices, indptr, shape = naked_this.indices, naked_this.indptr, naked_this.shape
            data = getattr(xp, method_name)(naked_this.data, naked_other.data)

        return SparseNDArray(xps.csr_matrix((data, indices, indptr), shape), shape=shape)

    def __lshift__(self, other):
        return self._shift(self.spmatrix, other, 'left_shift')

    def __rlshift__(self, other):
        return self._shift(other, self.spmatrix, 'left_shift')

    def __rshift__(self, other):
        return self._shift(self.spmatrix, other, 'right_shift')

    def __rrshift__(self, other):
        return self._shift(other, self.spmatrix, 'right_shift')

    def sin(self):
        return SparseNDArray(self.spmatrix.sin(), shape=self.shape)

    def cos(self):
        return call_sparse('cos', self)

    def tan(self):
        return SparseNDArray(self.spmatrix.tan(), shape=self.shape)

    def arcsin(self):
        return SparseNDArray(self.spmatrix.arcsin(), shape=self.shape)

    def arccos(self):
        return call_sparse('arccos', self)

    def arctan(self):
        return SparseNDArray(self.spmatrix.arctan(), shape=self.shape)

    def arctan2(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        xp = get_array_module(self.spmatrix)
        other_xp = get_array_module(naked_other)
        if other_xp.isscalar(naked_other):
            return call_sparse('arctan2', self, naked_other)
        if issparse(naked_other):
            naked_other = other.toarray()
        x = xp.arctan2(self.toarray(), naked_other)
        return SparseNDArray(get_sparse_module(x).csr_matrix(x), shape=self.shape)

    def hypot(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        xp = get_array_module(self.spmatrix)
        other_xp = get_array_module(naked_other)
        if other_xp.isscalar(naked_other):
            return call_sparse('hypot', self, naked_other)
        if issparse(naked_other):
            naked_other = other.toarray()
        x = xp.hypot(self.toarray(), naked_other)
        return SparseNDArray(get_sparse_module(x).csr_matrix(x), shape=self.shape)

    def sinh(self):
        return SparseNDArray(self.spmatrix.sinh(), shape=self.shape)

    def cosh(self):
        xp = get_array_module(self.spmatrix)
        return xp.cosh(self.toarray())

    def tanh(self):
        return SparseNDArray(self.spmatrix.tanh(), shape=self.shape)

    def arcsinh(self):
        return SparseNDArray(self.spmatrix.arcsinh(), shape=self.shape)

    def arccosh(self):
        return call_sparse('arccosh', self)

    def arctanh(self):
        return SparseNDArray(self.spmatrix.arctanh(), shape=self.shape)

    def around(self, decimals=0):
        return call_sparse('around', self, decimals=decimals)

    def deg2rad(self):
        return SparseNDArray(self.spmatrix.deg2rad(), shape=self.shape)

    def rad2deg(self):
        return SparseNDArray(self.spmatrix.rad2deg(), shape=self.shape)

    def angle(self, deg=0):
        return call_sparse('angle', self, deg=deg)

    def dot(self, other, sparse=True):
        raise NotImplementedError

    def concatenate(self, other, axis=0):
        raise NotImplementedError

    def _reduction(self, method_name, axis=None, dtype=None, keepdims=None, todense=False, **kw):
        raise NotImplementedError

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
            if get_array_module(ret).isscalar(ret):
                return ret
            xps = get_sparse_module(self.spmatrix)
            ret = SparseNDArray(xps.csr_matrix(ret))
            return ret
        return ret

    def any(self, axis=None, dtype=None, keepdims=None):
        ret = self._reduction('any', axis=axis, dtype=dtype, keepdims=keepdims, todense=True)
        if not issparse(ret):
            if get_array_module(ret).isscalar(ret):
                return ret
            xps = get_sparse_module(self.spmatrix)
            ret = SparseNDArray(xps.csr_matrix(ret))
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
            if get_array_module(ret).isscalar(ret):
                return ret
            xps = get_sparse_module(self.spmatrix)
            ret = SparseNDArray(xps.csr_matrix(ret))
            return ret
        return ret

    def nanmin(self, axis=None, dtype=None, keepdims=None):
        ret = self._reduction('nanmin', axis=axis, dtype=dtype, keepdims=keepdims, todense=True)
        if not issparse(ret):
            if get_array_module(ret).isscalar(ret):
                return ret
            xps = get_sparse_module(self.spmatrix)
            ret = SparseNDArray(xps.csr_matrix(ret))
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
        return self.toarray().cumsum(axis=axis)

    def cumprod(self, axis=None, dtype=None):
        return self.toarray().cumprod(axis=axis)

    def nancumsum(self, axis=None, dtype=None):
        xp = get_array_module(self.spmatrix)
        return xp.nancumsum(self.toarray(), axis=axis)

    def nancumprod(self, axis=None, dtype=None):
        xp = get_array_module(self.spmatrix)
        return xp.nancumprod(self.toarray(), axis=axis)

    def count_nonzero(self, axis=None, dtype=None, keepdims=None):
        if axis is None:
            return get_array_module(self.spmatrix).array([self.spmatrix.count_nonzero()])[0]
        else:
            return get_array_module(self.spmatrix).count_nonzero(self.toarray(), axis=axis)

    def __getitem__(self, item):
        if isinstance(item, SparseArray):
            item = item.spmatrix
        if isinstance(item, list):
            item = tuple(item)

        x = self.spmatrix[item]
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
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
            naked_other = naked(other)
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
        if issparse(naked_other):
            has_nan += xps.csr_matrix(
                (xp.isnan(naked_other.data), naked_other.indices, naked_other.indptr), naked_other.shape)

        if issparse(naked_other):
            x = getattr(self.spmatrix, method_name)(naked_other)
        else:
            x = getattr(xp, method_name)(self.toarray(), naked_other)

        if has_nan.sum() > 0:
            x = x + (has_nan * np.nan)

        if issparse(x):
            return SparseNDArray(x, shape=self.shape)

        return get_array_module(x).asarray(x)

    def maximum(self, other):
        return self._maximum_minimum(other, 'maximum')

    def minimum(self, other):
        return self._maximum_minimum(other, 'minimum')

    def fmax(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        x = self.spmatrix.maximum(naked_other)
        if issparse(x):
            return SparseArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def fmin(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        x = self.spmatrix.minimum(naked_other)
        if issparse(x):
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def isinf(self):
        return call_sparse('isinf', self)

    def isnan(self):
        return call_sparse('isnan', self)

    def signbit(self):
        return call_sparse('signbit', self)

    def floor(self):
        return SparseNDArray(self.spmatrix.floor(), shape=self.shape)

    def ceil(self):
        return SparseNDArray(self.spmatrix.ceil(), shape=self.shape)

    def trunc(self):
        return SparseNDArray(self.spmatrix.trunc(), shape=self.shape)

    def degrees(self):
        return call_sparse('degrees', self)

    def radians(self):
        return call_sparse('radians', self)

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
            return SparseNDArray(x, shape=self.shape)
        return get_array_module(x).asarray(x)

    def iscomplex(self):
        return call_sparse('iscomplex', self)

    def fix(self):
        return call_sparse('fix', self)

    def i0(self):
        xp = get_array_module(self.spmatrix)
        data = xp.i0(self.spmatrix.data).reshape(self.spmatrix.data.shape)
        x = get_sparse_module(self.spmatrix).csr_matrix(
            (data, self.spmatrix.indices, self.spmatrix.indptr), self.spmatrix.shape
        )
        return SparseNDArray(x, shape=self.shape)

    def nan_to_num(self):
        return call_sparse('nan_to_num', self)

    def copysign(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        if get_array_module(naked_other).isscalar(naked_other):
            return call_sparse('copysign', self, naked_other)

        if issparse(naked_other):
            naked_other = other.toarray()

        xp = get_array_module(self.spmatrix)
        return xp.copysign(self.toarray(), naked_other)

    def nextafter(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        ret_sparse = False
        if issparse(naked_other):
            ret_sparse = True
            naked_other = other.toarray()

        xp = get_array_module(self.spmatrix)
        xps = get_sparse_module(self.spmatrix)

        x = xp.nextafter(self.toarray(), naked_other)
        if ret_sparse:
            return SparseNDArray(xps.csr_matrix(x), shape=self.shape)
        return x

    def spacing(self):
        if is_cupy(self.spmatrix):
            raise NotImplementedError
        return call_sparse('spacing', self)

    def ldexp(self, other):
        try:
            naked_other = naked(other)
        except TypeError:
            return NotImplemented

        if get_array_module(naked_other).isscalar(naked_other):
            return call_sparse('ldexp', self, naked_other)

        if issparse(naked_other):
            naked_other = other.toarray()

        return SparseNDArray(self.spmatrix.multiply(2 ** naked_other))

    def frexp(self, **kw):
        xp = get_array_module(self.spmatrix)
        xps = get_sparse_module(self.spmatrix)
        x, y = xp.frexp(self.toarray(), **kw)
        return (SparseNDArray(xps.csr_matrix(x), shape=self.shape),
                SparseNDArray(xps.csr_matrix(y), shape=self.shape))

    def modf(self, **kw):
        xp = get_array_module(self.spmatrix)
        xps = get_sparse_module(self.spmatrix)
        x, y = xp.modf(self.toarray(), **kw)
        return (SparseNDArray(xps.csr_matrix(x), shape=self.shape),
                SparseNDArray(xps.csr_matrix(y), shape=self.shape))

    def sinc(self):
        return call_sparse('sinc', self)

    def isfinite(self):
        return call_sparse('isfinite', self)

    def isreal(self):
        return call_sparse('isreal', self)

    def digitize(self, bins, right=False):
        return call_sparse('digitize', self, bins=bins, right=right)

    def repeat(self, repeats, axis=None):
        if axis is None:
            raise NotImplementedError

        xp = get_array_module(self.spmatrix)
        xps = get_sparse_module(self.spmatrix)
        r = xp.repeat(self.toarray(), repeats, axis=axis)
        x = xps.csr_matrix(r)
        return SparseNDArray(x, shape=r.shape)

    @staticmethod
    def _expand_val(val, expect_val_size, xp):
        if val.size > expect_val_size:
            val = val[:expect_val_size]
        elif val.size < expect_val_size:
            n_repeat = ceildiv(expect_val_size, val.size)
            val = xp.tile(val, n_repeat)[:expect_val_size]
        return val

    def fill_diagonal(self, val, wrap=False):
        lil_matrix = self.spmatrix.tolil()

        xp = get_array_module(self.spmatrix)
        val = xp.asarray(val)
        if val.ndim > 1:
            val = val.ravel()
        is_tall_matrix = lil_matrix.shape[0] > lil_matrix.shape[1] + 1
        n_rows, n_cols = lil_matrix.shape

        if not wrap or not is_tall_matrix:
            if val.ndim > 0:
                # check if val is long enough
                expect_val_size = min(n_rows, n_cols)
                val = self._expand_val(val, expect_val_size, xp)
            lil_matrix.setdiag(val)
            matrix = lil_matrix
        else:
            block_size = n_cols + 1

            n_block = n_rows // block_size
            n_vals = n_cols * n_block
            if n_rows % block_size > 0:
                # 1 chunk left
                n_block += 1
                n_vals += min(n_rows % block_size, n_cols)

            if val.ndim > 0:
                val = self._expand_val(val, n_vals, xp)

            sub_matrices = []
            for i in range(n_block):
                sub_lil_matrix = lil_matrix[i * block_size: (i + 1) * block_size]
                if val.ndim > 0:
                    sub_val = val[i * n_cols: (i + 1) * n_cols]
                else:
                    sub_val = val
                sub_lil_matrix.setdiag(sub_val)
                sub_matrices.append(sub_lil_matrix)

            xps = get_sparse_module(self.spmatrix)
            matrix = SparseArray(xps.vstack(sub_matrices, format='csr'))

        self.spmatrix = matrix.tocsr()

    def unique(self, return_index=False, return_inverse=False,
               return_counts=False, axis=None):
        if return_inverse or return_index:  # pragma: no cover
            raise NotImplementedError
        if self.ndim == 2 and axis is not None:  # pragma: no cover
            raise NotImplementedError

        xp = get_array_module(self.spmatrix)
        return xp.unique(self.spmatrix.data, return_counts=return_counts)
