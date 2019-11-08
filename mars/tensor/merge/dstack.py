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

from ..base import atleast_3d
from .concatenate import concatenate


def dstack(tup):
    """
    Stack tensors in sequence depth wise (along third axis).

    This is equivalent to concatenation along the third axis after 2-D tensors
    of shape `(M,N)` have been reshaped to `(M,N,1)` and 1-D arrays of shape
    `(N,)` have been reshaped to `(1,N,1)`. Rebuilds arrays divided by
    `dsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of tensors
        The tensors must have the same shape along all but the third axis.
        1-D or 2-D arrays must have the same shape.

    Returns
    -------
    stacked : Tensor
        The array formed by stacking the given tensors, will be at least 3-D.

    See Also
    --------
    stack : Join a sequence of tensors along a new axis.
    vstack : Stack along first axis.
    hstack : Stack along second axis.
    concatenate : Join a sequence of arrays along an existing axis.
    dsplit : Split tensor along third axis.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.array((1,2,3))
    >>> b = mt.array((2,3,4))
    >>> mt.dstack((a,b)).execute()
    array([[[1, 2],
            [2, 3],
            [3, 4]]])

    >>> a = mt.array([[1],[2],[3]])
    >>> b = mt.array([[2],[3],[4]])
    >>> mt.dstack((a,b)).execute()
    array([[[1, 2]],
           [[2, 3]],
           [[3, 4]]])

    """
    return concatenate([atleast_3d(t) for t in tup], axis=2)
