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
from ..arithmetic.multiply import TensorTreeMultiply
from .core import TensorCumReduction


class TensorCumprod(operands.Cumprod, TensorCumReduction):
    def __init__(self, axis=None, dtype=None, **kw):
        super(TensorCumprod, self).__init__(_axis=axis, _dtype=dtype, **kw)

    @staticmethod
    def _get_op_types():
        return TensorCumprod, TensorTreeMultiply


def cumprod(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative product of elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input tensor.
    axis : int, optional
        Axis along which the cumulative product is computed.  By default
        the input is flattened.
    dtype : dtype, optional
        Type of the returned tensor, as well as of the accumulator in which
        the elements are multiplied.  If *dtype* is not specified, it
        defaults to the dtype of `a`, unless `a` has an integer dtype with
        a precision less than that of the default platform integer.  In
        that case, the default platform integer is used instead.
    out : Tensor, optional
        Alternative output tensor in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type of the resulting values will be cast if necessary.

    Returns
    -------
    cumprod : Tensor
        A new tensor holding the result is returned unless `out` is
        specified, in which case a reference to out is returned.

    See Also
    --------
    numpy.doc.ufuncs : Section "Output arguments"

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.array([1,2,3])
    >>> mt.cumprod(a).execute() # intermediate results 1, 1*2
    ...                         # total product 1*2*3 = 6
    array([1, 2, 6])
    >>> a = mt.array([[1, 2, 3], [4, 5, 6]])
    >>> mt.cumprod(a, dtype=float).execute() # specify type of output
    array([   1.,    2.,    6.,   24.,  120.,  720.])

    The cumulative product for each column (i.e., over the rows) of `a`:

    >>> mt.cumprod(a, axis=0).execute()
    array([[ 1,  2,  3],
           [ 4, 10, 18]])

    The cumulative product for each row (i.e. over the columns) of `a`:

    >>> mt.cumprod(a,axis=1).execute()
    array([[  1,   2,   6],
           [  4,  20, 120]])

    """
    a = astensor(a)
    if dtype is None:
        dtype = np.empty((1,), dtype=a.dtype).cumprod().dtype
    op = TensorCumprod(axis=axis, dtype=dtype)
    return op(a, out=out)
