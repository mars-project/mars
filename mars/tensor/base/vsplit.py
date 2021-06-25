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


def vsplit(a, indices_or_sections):
    """
    Split a tensor into multiple sub-tensors vertically (row-wise).

    Please refer to the ``split`` documentation.  ``vsplit`` is equivalent
    to ``split`` with `axis=0` (default), the tensor is always split along the
    first axis regardless of the tensor dimension.

    See Also
    --------
    split : Split a tensor into multiple sub-tensors of equal size.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.arange(16.0).reshape(4, 4)
    >>> x.execute()
    array([[  0.,   1.,   2.,   3.],
           [  4.,   5.,   6.,   7.],
           [  8.,   9.,  10.,  11.],
           [ 12.,  13.,  14.,  15.]])
    >>> mt.vsplit(x, 2).execute()
    [array([[ 0.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.]]),
     array([[  8.,   9.,  10.,  11.],
           [ 12.,  13.,  14.,  15.]])]
    >>> mt.vsplit(x, mt.array([3, 6])).execute()
    [array([[  0.,   1.,   2.,   3.],
           [  4.,   5.,   6.,   7.],
           [  8.,   9.,  10.,  11.]]),
     array([[ 12.,  13.,  14.,  15.]]),
     array([], dtype=float64)]

    With a higher dimensional tensor the split is still along the first axis.

    >>> x = mt.arange(8.0).reshape(2, 2, 2)
    >>> x.execute()
    array([[[ 0.,  1.],
            [ 2.,  3.]],
           [[ 4.,  5.],
            [ 6.,  7.]]])
    >>> mt.vsplit(x, 2).execute()
    [array([[[ 0.,  1.],
            [ 2.,  3.]]]),
     array([[[ 4.,  5.],
            [ 6.,  7.]]])]

    """
    ary = a
    a = astensor(a)

    if a.ndim < 2:
        raise ValueError('vsplit only works on tensors of 2 or more dimensions')
    return split(ary, indices_or_sections, 0)
