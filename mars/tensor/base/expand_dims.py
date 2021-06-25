#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2021 Alibaba Group Holding Ltd.
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

from ..datasource import tensor as astensor


def expand_dims(a, axis):
    """
    Expand the shape of a tensor.

    Insert a new axis that will appear at the `axis` position in the expanded
    array shape.

    Parameters
    ----------
    a : array_like
        Input tensor.
    axis : int
        Position in the expanded axes where the new axis is placed.

    Returns
    -------
    res : Tensor
        Output tensor. The number of dimensions is one greater than that of
        the input tensor.

    See Also
    --------
    squeeze : The inverse operation, removing singleton dimensions
    reshape : Insert, remove, and combine dimensions, and resize existing ones
    doc.indexing, atleast_1d, atleast_2d, atleast_3d

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.array([1,2])
    >>> x.shape
    (2,)

    The following is equivalent to ``x[mt.newaxis,:]`` or ``x[mt.newaxis]``:

    >>> y = mt.expand_dims(x, axis=0)
    >>> y.execute()
    array([[1, 2]])
    >>> y.shape
    (1, 2)

    >>> y = mt.expand_dims(x, axis=1)  # Equivalent to x[:,mt.newaxis]
    >>> y.execute()
    array([[1],
           [2]])
    >>> y.shape
    (2, 1)

    Note that some examples may use ``None`` instead of ``np.newaxis``.  These
    are the same objects:

    >>> mt.newaxis is None
    True

    """
    a = astensor(a)

    if axis > a.ndim or axis < -a.ndim - 1:
        raise np.AxisError(f'Axis must be between -{a.ndim + 1} and {a.ndim}, got {axis}')

    axis = axis if axis >= 0 else axis + a.ndim + 1
    indexes = (slice(None),) * axis + (np.newaxis,) + (slice(None),) * (a.ndim - axis)
    return a[indexes]
