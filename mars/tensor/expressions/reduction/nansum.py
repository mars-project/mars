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

from .... import operands
from ..datasource import tensor as astensor
from .core import TensorReduction


class TensorNanSum(operands.NanSum, TensorReduction):
    def __init__(self, axis=None, dtype=None, keepdims=None, combine_size=None, **kw):
        super(TensorNanSum, self).__init__(_axis=axis, _dtype=dtype, _keepdims=keepdims,
                                           _combine_size=combine_size, **kw)

    @staticmethod
    def _get_op_types():
        return TensorNanSum, TensorNanSum, None


def nansum(a, axis=None, dtype=None, out=None, keepdims=None, combine_size=None):
    """
    Return the sum of array elements over a given axis treating Not a
    Numbers (NaNs) as zero.

    Zero is returned for slices that are all-NaN or
    empty.

    Parameters
    ----------
    a : array_like
        Tensor containing numbers whose sum is desired. If `a` is not an
        tensor, a conversion is attempted.
    axis : int, optional
        Axis along which the sum is computed. The default is to compute the
        sum of the flattened array.
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
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.


        If the value is anything but the default, then
        `keepdims` will be passed through to the `mean` or `sum` methods
        of sub-classes of `Tensor`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.
    combine_size: int, optional
        The number of chunks to combine.

    Returns
    -------
    nansum : Tensor.
        A new tensor holding the result is returned unless `out` is
        specified, in which it is returned. The result has the same
        size as `a`, and the same shape as `a` if `axis` is not None
        or `a` is a 1-d array.

    See Also
    --------
    mt.sum : Sum across tensor propagating NaNs.
    isnan : Show which elements are NaN.
    isfinite: Show which elements are not NaN or +/-inf.

    Notes
    -----
    If both positive and negative infinity are present, the sum will be Not
    A Number (NaN).

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.nansum(1).execute()
    1
    >>> mt.nansum([1]).execute()
    1
    >>> mt.nansum([1, mt.nan]).execute()
    1.0
    >>> a = mt.array([[1, 1], [1, mt.nan]])
    >>> mt.nansum(a).execute()
    3.0
    >>> mt.nansum(a, axis=0).execute()
    array([ 2.,  1.])
    >>> mt.nansum([1, mt.nan, mt.inf]).execute()
    inf
    >>> mt.nansum([1, mt.nan, mt.NINF]).execute()
    -inf
    >>> mt.nansum([1, mt.nan, mt.inf, -mt.inf]).execute() # both +/- infinity present
    nan

    """
    a = astensor(a)
    if dtype is None:
        dtype = np.nansum(np.empty((1,), dtype=a.dtype)).dtype
    op = TensorNanSum(axis=axis, dtype=dtype, keepdims=keepdims, combine_size=combine_size)
    return op(a, out=out)
