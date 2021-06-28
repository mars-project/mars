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

from ..datasource import tensor as astensor
from .split import split


def hsplit(a, indices_or_sections):
    """
    Split a tensor into multiple sub-tensors horizontally (column-wise).

    Please refer to the `split` documentation.  `hsplit` is equivalent
    to `split` with ``axis=1``, the tensor is always split along the second
    axis regardless of the tensor dimension.

    See Also
    --------
    split : Split an array into multiple sub-arrays of equal size.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.arange(16.0).reshape(4, 4)
    >>> x.execute()
    array([[  0.,   1.,   2.,   3.],
           [  4.,   5.,   6.,   7.],
           [  8.,   9.,  10.,  11.],
           [ 12.,  13.,  14.,  15.]])
    >>> mt.hsplit(x, 2).execute()
    [array([[  0.,   1.],
           [  4.,   5.],
           [  8.,   9.],
           [ 12.,  13.]]),
     array([[  2.,   3.],
           [  6.,   7.],
           [ 10.,  11.],
           [ 14.,  15.]])]
    >>> mt.hsplit(x, mt.array([3, 6])).execute()
    [array([[  0.,   1.,   2.],
           [  4.,   5.,   6.],
           [  8.,   9.,  10.],
           [ 12.,  13.,  14.]]),
     array([[  3.],
           [  7.],
           [ 11.],
           [ 15.]]),
     array([], dtype=float64)]

    With a higher dimensional array the split is still along the second axis.

    >>> x = mt.arange(8.0).reshape(2, 2, 2)
    >>> x.execute()
    array([[[ 0.,  1.],
            [ 2.,  3.]],
           [[ 4.,  5.],
            [ 6.,  7.]]])
    >>> mt.hsplit(x, 2)
    [array([[[ 0.,  1.]],
           [[ 4.,  5.]]]),
     array([[[ 2.,  3.]],
           [[ 6.,  7.]]])]

    """
    ary = a
    a = astensor(a)

    if a.ndim == 0:
        raise ValueError('hsplit only works on tensors of 1 or more dimensions')
    if a.ndim > 1:
        return split(ary, indices_or_sections, 1)
    else:
        return split(ary, indices_or_sections, 0)
