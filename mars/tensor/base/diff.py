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

from ..utils import validate_axis
from ..datasource import tensor as astensor


def diff(a, n=1, axis=-1):
    """
    Calculate the n-th discrete difference along the given axis.

    The first difference is given by ``out[n] = a[n+1] - a[n]`` along
    the given axis, higher differences are calculated by using `diff`
    recursively.

    Parameters
    ----------
    a : array_like
        Input tensor
    n : int, optional
        The number of times values are differenced. If zero, the input
        is returned as-is.
    axis : int, optional
        The axis along which the difference is taken, default is the
        last axis.

    Returns
    -------
    diff : Tensor
        The n-th differences. The shape of the output is the same as `a`
        except along `axis` where the dimension is smaller by `n`. The
        type of the output is the same as the type of the difference
        between any two elements of `a`. This is the same as the type of
        `a` in most cases. A notable exception is `datetime64`, which
        results in a `timedelta64` output tensor.

    See Also
    --------
    gradient, ediff1d, cumsum

    Notes
    -----
    Type is preserved for boolean tensors, so the result will contain
    `False` when consecutive elements are the same and `True` when they
    differ.

    For unsigned integer tensors, the results will also be unsigned. This
    should not be surprising, as the result is consistent with
    calculating the difference directly:

    >>> import mars.tensor as mt

    >>> u8_arr = mt.array([1, 0], dtype=mt.uint8)
    >>> mt.diff(u8_arr).execute()
    array([255], dtype=uint8)
    >>> (u8_arr[1,...] - u8_arr[0,...]).execute()
    255

    If this is not desirable, then the array should be cast to a larger
    integer type first:

    >>> i16_arr = u8_arr.astype(mt.int16)
    >>> mt.diff(i16_arr).execute()
    array([-1], dtype=int16)

    Examples
    --------
    >>> x = mt.array([1, 2, 4, 7, 0])
    >>> mt.diff(x).execute()
    array([ 1,  2,  3, -7])
    >>> mt.diff(x, n=2).execute()
    array([  1,   1, -10])

    >>> x = mt.array([[1, 3, 6, 10], [0, 5, 6, 8]])
    >>> mt.diff(x).execute()
    array([[2, 3, 4],
           [5, 1, 2]])
    >>> mt.diff(x, axis=0).execute()
    array([[-1,  2,  0, -2]])

    >>> x = mt.arange('1066-10-13', '1066-10-16', dtype=mt.datetime64)
    >>> mt.diff(x).execute()
    array([1, 1], dtype='timedelta64[D]')

    """
    a = astensor(a)
    n = int(n)

    axis = validate_axis(a.ndim, axis)
    slc1 = (slice(None),) * axis + (slice(1, None),)
    slc2 = (slice(None),) * axis + (slice(-1),)

    for _ in range(n):
        a = a[slc1] - a[slc2]

    return a
