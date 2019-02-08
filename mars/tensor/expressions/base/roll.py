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
# limitations under the License.'

from collections import Iterable

import numpy as np

from ....compat import six
from ..utils import validate_axis
from ..datasource import tensor as astensor
from .ravel import ravel


def roll(a, shift, axis=None):
    """
    Roll tensor elements along a given axis.

    Elements that roll beyond the last position are re-introduced at
    the first.

    Parameters
    ----------
    a : array_like
        Input tensor.
    shift : int or tuple of ints
        The number of places by which elements are shifted.  If a tuple,
        then `axis` must be a tuple of the same size, and each of the
        given axes is shifted by the corresponding number.  If an int
        while `axis` is a tuple of ints, then the same value is used for
        all given axes.
    axis : int or tuple of ints, optional
        Axis or axes along which elements are shifted.  By default, the
        tensor is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : Tensor
        Output tensor, with the same shape as `a`.

    See Also
    --------
    rollaxis : Roll the specified axis backwards, until it lies in a
               given position.

    Notes
    -----

    Supports rolling over multiple dimensions simultaneously.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.arange(10)
    >>> mt.roll(x, 2).execute()
    array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])

    >>> x2 = mt.reshape(x, (2,5))
    >>> x2.execute()
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> mt.roll(x2, 1).execute()
    array([[9, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])
    >>> mt.roll(x2, 1, axis=0).execute()
    array([[5, 6, 7, 8, 9],
           [0, 1, 2, 3, 4]])
    >>> mt.roll(x2, 1, axis=1).execute()
    array([[4, 0, 1, 2, 3],
           [9, 5, 6, 7, 8]])

    """
    from ..merge import concatenate

    a = astensor(a)
    raw = a

    if axis is None:
        a = ravel(a)
        axis = 0

    if not isinstance(shift, Iterable):
        shift = (shift,)
    else:
        shift = tuple(shift)
    if not isinstance(axis, Iterable):
        axis = (axis,)
    else:
        axis = tuple(axis)

    for ax in axis:
        validate_axis(a.ndim, ax)
    broadcasted = np.broadcast(shift, axis)
    if broadcasted.ndim > 1:
        raise ValueError(
            "'shift' and 'axis' should be scalars or 1D sequences")

    shifts = {ax: 0 for ax in range(a.ndim)}
    for s, ax in broadcasted:
        shifts[ax] += s

    for ax, s in six.iteritems(shifts):
        if s == 0:
            continue

        s = -s
        s %= a.shape[ax]

        slc1 = (slice(None),) * ax + (slice(s, None),)
        slc2 = (slice(None),) * ax + (slice(s),)

        a = concatenate([a[slc1], a[slc2]], axis=ax)

    return a.reshape(raw.shape)
