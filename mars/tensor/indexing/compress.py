#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

from ..utils import validate_axis
from ..datasource import tensor as astensor
from ..core import Tensor


def compress(condition, a, axis=None, out=None):
    """
    Return selected slices of a tensor along given axis.

    When working along a given axis, a slice along that axis is returned in
    `output` for each index where `condition` evaluates to True. When
    working on a 1-D array, `compress` is equivalent to `extract`.

    Parameters
    ----------
    condition : 1-D tensor of bools
        Tensor that selects which entries to return. If len(condition)
        is less than the size of `a` along the given axis, then output is
        truncated to the length of the condition tensor.
    a : array_like
        Tensor from which to extract a part.
    axis : int, optional
        Axis along which to take slices. If None (default), work on the
        flattened tensor.
    out : Tensor, optional
        Output tensor.  Its type is preserved and it must be of the right
        shape to hold the output.

    Returns
    -------
    compressed_array : Tensor
        A copy of `a` without the slices along axis for which `condition`
        is false.

    See Also
    --------
    take, choose, diag, diagonal, select
    Tensor.compress : Equivalent method in ndarray
    mt.extract: Equivalent method when working on 1-D arrays

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.array([[1, 2], [3, 4], [5, 6]])
    >>> a.execute()
    array([[1, 2],
           [3, 4],
           [5, 6]])
    >>> mt.compress([0, 1], a, axis=0).execute()
    array([[3, 4]])
    >>> mt.compress([False, True, True], a, axis=0).execute()
    array([[3, 4],
           [5, 6]])
    >>> mt.compress([False, True], a, axis=1).execute()
    array([[2],
           [4],
           [6]])

    Working on the flattened tensor does not return slices along an axis but
    selects elements.

    >>> mt.compress([False, True], a).execute()
    array([2])

    """
    a = astensor(a)
    condition = astensor(condition, dtype=bool)

    if condition.ndim != 1:
        raise ValueError('condition must be an 1-d tensor')

    if axis is None:
        a = a.ravel()
        if len(condition) < a.size:
            a = a[:len(condition)]
        return a[condition]

    try:
        axis = validate_axis(a.ndim, axis)
    except ValueError:
        raise np.AxisError('axis {0} is out of bounds '
                           'for tensor of dimension {1}'.format(axis, a.ndim))

    try:
        if len(condition) < a.shape[axis]:
            a = a[(slice(None),) * axis + (slice(len(condition)),)]
        t = a[(slice(None),) * axis + (condition,)]
        if out is None:
            return t

        if out is not None and not isinstance(out, Tensor):
            raise TypeError('out should be Tensor object, got {0} instead'.format(type(out)))
        if not np.can_cast(out.dtype, t.dtype, 'safe'):
            raise TypeError('Cannot cast array data from dtype(\'{0}\') to dtype(\'{1}\') '
                            'according to the rule \'safe\''.format(out.dtype, t.dtype))
        # skip shape check because out shape is unknown
        out.data = t.astype(out.dtype, order=out.order.value).data
        return out
    except IndexError:
        raise np.AxisError('axis {0} is out of bounds '
                           'for tensor of dimension 1'.format(len(condition)))
