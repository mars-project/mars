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
from ..arithmetic.add import TensorTreeAdd
from .core import TensorCumReduction


class TensorNanCumsum(operands.NanCumsum, TensorCumReduction):
    def __init__(self, axis=None, dtype=None, **kw):
        super(TensorNanCumsum, self).__init__(_axis=axis, _dtype=dtype, **kw)

    @staticmethod
    def _get_op_types():
        return TensorNanCumsum, TensorTreeAdd


def nancumsum(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative sum of tensor elements over a given axis treating Not a
    Numbers (NaNs) as zero.  The cumulative sum does not change when NaNs are
    encountered and leading NaNs are replaced by zeros.

    Zeros are returned for slices that are all-NaN or empty.

    Parameters
    ----------
    a : array_like
        Input tensor.
    axis : int, optional
        Axis along which the cumulative sum is computed. The default
        (None) is to compute the cumsum over the flattened tensor.
    dtype : dtype, optional
        Type of the returned tensor and of the accumulator in which the
        elements are summed.  If `dtype` is not specified, it defaults
        to the dtype of `a`, unless `a` has an integer dtype with a
        precision less than that of the default platform integer.  In
        that case, the default platform integer is used.
    out : Tensor, optional
        Alternative output tensor in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary. See `doc.ufuncs`
        (Section "Output arguments") for more details.

    Returns
    -------
    nancumsum : Tensor.
        A new tensor holding the result is returned unless `out` is
        specified, in which it is returned. The result has the same
        size as `a`, and the same shape as `a` if `axis` is not None
        or `a` is a 1-d tensor.

    See Also
    --------
    numpy.cumsum : Cumulative sum across tensor propagating NaNs.
    isnan : Show which elements are NaN.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.nancumsum(1).execute()
    array([1])
    >>> mt.nancumsum([1]).execute()
    array([1])
    >>> mt.nancumsum([1, mt.nan]).execute()
    array([ 1.,  1.])
    >>> a = mt.array([[1, 2], [3, mt.nan]])
    >>> mt.nancumsum(a).execute()
    array([ 1.,  3.,  6.,  6.])
    >>> mt.nancumsum(a, axis=0).execute()
    array([[ 1.,  2.],
           [ 4.,  2.]])
    >>> mt.nancumsum(a, axis=1).execute()
    array([[ 1.,  3.],
           [ 3.,  3.]])

    """
    a = astensor(a)
    if dtype is None:
        dtype = np.nancumsum(np.empty((1,), dtype=a.dtype)).dtype
    op = TensorNanCumsum(axis=axis, dtype=dtype)
    return op(a, out=out)
