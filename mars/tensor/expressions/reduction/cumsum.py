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


class TensorCumsum(operands.Cumsum, TensorCumReduction):
    def __init__(self, axis=None, dtype=None, **kw):
        super(TensorCumsum, self).__init__(_axis=axis, _dtype=dtype, **kw)

    @staticmethod
    def _get_op_types():
        return TensorCumsum, TensorTreeAdd


def cumsum(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative sum of the elements along a given axis.

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
    cumsum_along_axis : Tensor.
        A new tensor holding the result is returned unless `out` is
        specified, in which case a reference to `out` is returned. The
        result has the same size as `a`, and the same shape as `a` if
        `axis` is not None or `a` is a 1-d tensor.


    See Also
    --------
    sum : Sum tensor elements.

    trapz : Integration of tensor values using the composite trapezoidal rule.

    diff :  Calculate the n-th discrete difference along given axis.

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.array([[1,2,3], [4,5,6]])
    >>> a.execute()
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> mt.cumsum(a).execute()
    array([ 1,  3,  6, 10, 15, 21])
    >>> mt.cumsum(a, dtype=float).execute()     # specifies type of output value(s)
    array([  1.,   3.,   6.,  10.,  15.,  21.])

    >>> mt.cumsum(a,axis=0).execute()      # sum over rows for each of the 3 columns
    array([[1, 2, 3],
           [5, 7, 9]])
    >>> mt.cumsum(a,axis=1).execute()      # sum over columns for each of the 2 rows
    array([[ 1,  3,  6],
           [ 4,  9, 15]])

    """
    a = astensor(a)
    if dtype is None:
        dtype = np.empty((1,), dtype=a.dtype).cumsum().dtype
    op = TensorCumsum(axis=axis, dtype=dtype)
    return op(a, out=out)
