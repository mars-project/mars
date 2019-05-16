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
from ..datasource import tensor as astensor
from .core import TensorReduction, TensorReductionMixin


class TensorNanProd(TensorReduction, TensorReductionMixin):
    _op_type_ = OperandDef.NANPROD

    def __init__(self, axis=None, dtype=None, keepdims=None, combine_size=None, **kw):
        super(TensorNanProd, self).__init__(_axis=axis, _dtype=dtype, _keepdims=keepdims,
                                            _combine_size=combine_size, **kw)

    @staticmethod
    def _get_op_types():
        return TensorNanProd, TensorNanProd, None


def nanprod(a, axis=None, dtype=None, out=None, keepdims=None, combine_size=None):
    """
    Return the product of array elements over a given axis treating Not a
    Numbers (NaNs) as ones.

    One is returned for slices that are all-NaN or empty.

    Parameters
    ----------
    a : array_like
        Tensor containing numbers whose product is desired. If `a` is not an
        tensor, a conversion is attempted.
    axis : int, optional
        Axis along which the product is computed. The default is to compute
        the product of the flattened tensor.
    dtype : data-type, optional
        The type of the returned tensor and of the accumulator in which the
        elements are summed.  By default, the dtype of `a` is used.  An
        exception is when `a` has an integer type with less precision than
        the platform (u)intp. In that case, the default will be either
        (u)int32 or (u)int64 depending on whether the platform is 32 or 64
        bits. For inexact inputs, dtype must be inexact.
    out : Tensor, optional
        Alternate output tensor in which to place the result.  The default
        is ``None``. If provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.  See
        `doc.ufuncs` for details. The casting of NaN to integer can yield
        unexpected results.
    keepdims : bool, optional
        If True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will
        broadcast correctly against the original `arr`.
    combine_size: int, optional
        The number of chunks to combine.

    Returns
    -------
    nanprod : Tensor
        A new tensor holding the result is returned unless `out` is
        specified, in which case it is returned.

    See Also
    --------
    mt.prod : Product across array propagating NaNs.
    isnan : Show which elements are NaN.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.nanprod(1).execute()
    1
    >>> mt.nanprod([1]).execute()
    1
    >>> mt.nanprod([1, mt.nan]).execute()
    1.0
    >>> a = mt.array([[1, 2], [3, mt.nan]])
    >>> mt.nanprod(a).execute()
    6.0
    >>> mt.nanprod(a, axis=0).execute()
    array([ 3.,  2.])

    """
    a = astensor(a)
    if dtype is None:
        dtype = np.nanprod(np.empty((1,), dtype=a.dtype)).dtype
    op = TensorNanProd(axis=axis, dtype=dtype, keepdims=keepdims, combine_size=combine_size)
    return op(a, out=out)
