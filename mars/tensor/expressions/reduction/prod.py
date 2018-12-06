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


class TensorProd(operands.Prod, TensorReduction):
    def __init__(self, axis=None, dtype=None, keepdims=None, combine_size=None, **kw):
        super(TensorProd, self).__init__(_axis=axis, _dtype=dtype, _keepdims=keepdims,
                                         _combine_size=combine_size, **kw)

    @staticmethod
    def _get_op_types():
        return TensorProd, TensorProd, None


def prod(a, axis=None, dtype=None, out=None, keepdims=None, combine_size=None):
    """
    Return the product of tensor elements over a given axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a product is performed.  The default,
        axis=None, will calculate the product of all the elements in the
        input tensor. If axis is negative it counts from the last to the
        first axis.

        If axis is a tuple of ints, a product is performed on all of the
        axes specified in the tuple instead of a single axis or all the
        axes as before.
    dtype : dtype, optional
        The type of the returned tensor, as well as of the accumulator in
        which the elements are multiplied.  The dtype of `a` is used by
        default unless `a` has an integer dtype of less precision than the
        default platform integer.  In that case, if `a` is signed then the
        platform integer is used while if `a` is unsigned then an unsigned
        integer of the same precision as the platform integer is used.
    out : Tensor, optional
        Alternative output tensor in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `prod` method of sub-classes of
        `Tensor`, however any non-default value will be.  If the
        sub-classes `sum` method does not implement `keepdims` any
        exceptions will be raised.
    combine_size: int, optional
        The number of chunks to combine.

    Returns
    -------
    product_along_axis : Tensor, see `dtype` parameter above.
        An tensor shaped as `a` but with the specified axis removed.
        Returns a reference to `out` if specified.

    See Also
    --------
    Tensor.prod : equivalent method

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.  That means that, on a 32-bit platform:

    >>> import mars.tensor as mt

    >>> x = mt.array([536870910, 536870910, 536870910, 536870910])
    >>> mt.prod(x).execute()  # random
    16

    The product of an empty array is the neutral element 1:

    >>> mt.prod([]).execute()
    1.0

    Examples
    --------
    By default, calculate the product of all elements:

    >>> mt.prod([1.,2.]).execute()
    2.0

    Even when the input array is two-dimensional:

    >>> mt.prod([[1.,2.],[3.,4.]]).execute()
    24.0

    But we can also specify the axis over which to multiply:

    >>> mt.prod([[1.,2.],[3.,4.]], axis=1).execute()
    array([  2.,  12.])

    If the type of `x` is unsigned, then the output type is
    the unsigned platform integer:

    >>> x = mt.array([1, 2, 3], dtype=mt.uint8)
    >>> mt.prod(x).dtype == mt.uint
    True

    If `x` is of a signed integer type, then the output type
    is the default platform integer:

    >>> x = mt.array([1, 2, 3], dtype=mt.int8)
    >>> mt.prod(x).dtype == int
    True

    """
    a = astensor(a)
    if dtype is None:
        dtype = np.empty((1,), dtype=a.dtype).prod().dtype
    op = TensorProd(axis=axis, dtype=dtype, keepdims=keepdims, combine_size=combine_size)
    return op(a, out=out)
