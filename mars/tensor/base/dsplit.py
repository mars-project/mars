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


def dsplit(a, indices_or_sections):
    """
    Split tensor into multiple sub-tensors along the 3rd axis (depth).

    Please refer to the `split` documentation.  `dsplit` is equivalent
    to `split` with ``axis=2``, the array is always split along the third
    axis provided the tensor dimension is greater than or equal to 3.

    See Also
    --------
    split : Split a tensor into multiple sub-arrays of equal size.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.arange(16.0).reshape(2, 2, 4)
    >>> x.execute()
    array([[[  0.,   1.,   2.,   3.],
            [  4.,   5.,   6.,   7.]],
           [[  8.,   9.,  10.,  11.],
            [ 12.,  13.,  14.,  15.]]])
    >>> mt.dsplit(x, 2).execute()
    [array([[[  0.,   1.],
            [  4.,   5.]],
           [[  8.,   9.],
            [ 12.,  13.]]]),
     array([[[  2.,   3.],
            [  6.,   7.]],
           [[ 10.,  11.],
            [ 14.,  15.]]])]
    >>> mt.dsplit(x, mt.array([3, 6])).execute()
    [array([[[  0.,   1.,   2.],
            [  4.,   5.,   6.]],
           [[  8.,   9.,  10.],
            [ 12.,  13.,  14.]]]),
     array([[[  3.],
            [  7.]],
           [[ 11.],
            [ 15.]]]),
     array([], dtype=float64)]

    """
    ary = a
    a = astensor(a)

    if a.ndim < 3:
        raise ValueError('dsplit only works on tensors of 3 or more dimensions')
    return split(ary, indices_or_sections, 2)
