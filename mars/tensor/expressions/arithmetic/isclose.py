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

from .... import opcodes as OperandDef
from ....serialize import Float64Field, BoolField
from ..utils import inject_dtype
from .core import TensorBinOp, TensorConstant


class TensorIsclose(TensorBinOp):
    _op_type_ = OperandDef.ISCLOSE

    _rtol = Float64Field('rtol')
    _atol = Float64Field('atol')
    _equal_nan = BoolField('equal_nan')

    def __init__(self, rtol=None, atol=None, equal_nan=None,
                 casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super(TensorIsclose, self).__init__(_rtol=rtol, _atol=atol, _equal_nan=equal_nan,
                                            _casting=casting, _err=err,
                                            _dtype=dtype, _sparse=sparse, **kw)

    @property
    def rtol(self):
        return self._rtol

    @property
    def atol(self):
        return self._atol

    @property
    def equal_nan(self):
        return self._equal_nan

    @classmethod
    def _is_sparse(cls, x1, x2):
        return False

    @classmethod
    def constant_cls(cls):
        return TensorIscloseConstant

    def to_constant(self, x1, x2):
        constant_op = TensorIscloseConstant(getattr(self, '_rtol'), getattr(self, '_atol'),
                                            getattr(self, '_equal_nan'), getattr(self, '_casting'),
                                            getattr(self, '_err'), getattr(self, '_dtype'),
                                            getattr(self, '_sparse'))
        return constant_op(x1, x2)


class TensorIscloseConstant(TensorConstant):
    _op_type_ = OperandDef.ISCLOSE_CONSTANT

    _rtol = Float64Field('rtol')
    _atol = Float64Field('atol')
    _equal_nan = BoolField('equal_nan')

    def __init__(self, rtol=None, atol=None, equal_nan=None,
                 casting='same_kind', err=None, dtype=None, sparse=False, **kw):
        err = err if err is not None else np.geterr()
        super(TensorIscloseConstant, self).__init__(_rtol=rtol, _atol=atol, _equal_nan=equal_nan,
                                                    _casting=casting, _err=err,
                                                    _dtype=dtype, _sparse=sparse, **kw)

    @property
    def rtol(self):
        return self._rtol

    @property
    def atol(self):
        return self._atol

    @property
    def equal_nan(self):
        return self._equal_nan

    @classmethod
    def _is_sparse(cls, x1, x2):
        if hasattr(x1, 'issparse') and x1.issparse() and \
                np.isscalar(x2) and not np.isclose(x2, 0):
            return True
        if hasattr(x2, 'issparse') and x2.issparse() and \
                np.isscalar(x1) and not np.isclose(x1, 0):
            return True
        return False


@inject_dtype(np.bool_)
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False, **kwargs):
    """
    Returns a boolean tensor where two tensors are element-wise equal within a
    tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    Parameters
    ----------
    a, b : array_like
        Input tensors to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output tensor.

    Returns
    -------
    y : array_like
        Returns a boolean tensor of where `a` and `b` are equal within the
        given tolerance. If both `a` and `b` are scalars, returns a single
        boolean value.

    See Also
    --------
    allclose

    Notes
    -----

    For finite values, isclose uses the following equation to test whether
    two floating point values are equivalent.

     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    The above equation is not symmetric in `a` and `b`, so that
    `isclose(a, b)` might be different from `isclose(b, a)` in
    some rare cases.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.isclose([1e10,1e-7], [1.00001e10,1e-8]).execute()
    array([True, False])
    >>> mt.isclose([1e10,1e-8], [1.00001e10,1e-9]).execute()
    array([True, True])
    >>> mt.isclose([1e10,1e-8], [1.0001e10,1e-9]).execute()
    array([False, True])
    >>> mt.isclose([1.0, mt.nan], [1.0, mt.nan]).execute()
    array([True, False])
    >>> mt.isclose([1.0, mt.nan], [1.0, mt.nan], equal_nan=True).execute()
    array([True, True])
    """
    op = TensorIsclose(rtol=rtol, atol=atol, equal_nan=equal_nan, **kwargs)
    return op(a, b)
