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


def rollaxis(tensor, axis, start=0):
    """
    Roll the specified axis backwards, until it lies in a given position.

    This function continues to be supported for backward compatibility, but you
    should prefer `moveaxis`.

    Parameters
    ----------
    a : Tensor
        Input tensor.
    axis : int
        The axis to roll backwards.  The positions of the other axes do not
        change relative to one another.
    start : int, optional
        The axis is rolled until it lies before this position.  The default,
        0, results in a "complete" roll.

    Returns
    -------
    res : Tensor
        a view of `a` is always returned.

    See Also
    --------
    moveaxis : Move array axes to new positions.
    roll : Roll the elements of an array by a number of positions along a
        given axis.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.ones((3,4,5,6))
    >>> mt.rollaxis(a, 3, 1).shape
    (3, 6, 4, 5)
    >>> mt.rollaxis(a, 2).shape
    (5, 3, 4, 6)
    >>> mt.rollaxis(a, 1, 4).shape
    (3, 5, 6, 4)

    """
    n = tensor.ndim
    axis = validate_axis(n, axis)
    if start < 0:
        start += n
    msg = "'%s' arg requires %d <= %s < %d, but %d was passed in"
    if not (0 <= start < n + 1):
        raise np.AxisError(msg % ('start', -n, 'start', n + 1, start))
    if axis < start:
        # it's been removed
        start -= 1
    if axis == start:
        return tensor
    axes = list(range(0, n))
    axes.remove(axis)
    axes.insert(start, axis)
    return tensor.transpose(axes)
