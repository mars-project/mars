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

from numpy.core.numeric import normalize_axis_tuple

from ..datasource import tensor as astensor
from .transpose import transpose


def moveaxis(a, source, destination):
    """
    Move axes of a tensor to new positions.

    Other axes remain in their original order.

    Parameters
    ----------
    a : Tensor
        The tensor whose axes should be reordered.
    source : int or sequence of int
        Original positions of the axes to move. These must be unique.
    destination : int or sequence of int
        Destination positions for each of the original axes. These must also be
        unique.

    Returns
    -------
    result : Tensor
        Array with moved axes. This tensor is a view of the input tensor.

    See Also
    --------
    transpose: Permute the dimensions of an array.
    swapaxes: Interchange two axes of an array.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.zeros((3, 4, 5))
    >>> mt.moveaxis(x, 0, -1).shape
    (4, 5, 3)
    >>> mt.moveaxis(x, -1, 0).shape
    (5, 3, 4),

    These all achieve the same result:

    >>> mt.transpose(x).shape
    (5, 4, 3)
    >>> mt.swapaxes(x, 0, -1).shape
    (5, 4, 3)
    >>> mt.moveaxis(x, [0, 1], [-1, -2]).shape
    (5, 4, 3)
    >>> mt.moveaxis(x, [0, 1, 2], [-1, -2, -3]).shape
    (5, 4, 3)

    """
    a = astensor(a)

    source = normalize_axis_tuple(source, a.ndim, 'source')
    destination = normalize_axis_tuple(destination, a.ndim, 'destination')
    if len(source) != len(destination):
        raise ValueError('`source` and `destination` arguments must have '
                         'the same number of elements')

    order = [n for n in range(a.ndim) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    return transpose(a, order)
