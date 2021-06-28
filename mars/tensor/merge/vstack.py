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

from ..base import atleast_2d
from .concatenate import concatenate


def vstack(tup):
    """
    Stack tensors in sequence vertically (row wise).

    This is equivalent to concatenation along the first axis after 1-D tensors
    of shape `(N,)` have been reshaped to `(1,N)`. Rebuilds tensors divided by
    `vsplit`.

    This function makes most sense for tensors with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of tensors
        The tensors must have the same shape along all but the first axis.
        1-D tensors must have the same length.

    Returns
    -------
    stacked : Tensor
        The tensor formed by stacking the given tensors, will be at least 2-D.

    See Also
    --------
    stack : Join a sequence of tensors along a new axis.
    hstack : Stack tensors in sequence horizontally (column wise).
    dstack : Stack tensors in sequence depth wise (along third dimension).
    concatenate : Join a sequence of tensors along an existing axis.
    vsplit : Split tensor into a list of multiple sub-arrays vertically.
    block : Assemble tensors from blocks.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.array([1, 2, 3])
    >>> b = mt.array([2, 3, 4])
    >>> mt.vstack((a,b)).execute()
    array([[1, 2, 3],
           [2, 3, 4]])

    >>> a = mt.array([[1], [2], [3]])
    >>> b = mt.array([[2], [3], [4]])
    >>> mt.vstack((a,b)).execute()
    array([[1],
           [2],
           [3],
           [2],
           [3],
           [4]])

    """
    return concatenate([atleast_2d(t) for t in tup], axis=0)
