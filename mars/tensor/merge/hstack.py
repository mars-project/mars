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
# limitations under the License.

from .concatenate import concatenate


def hstack(tup):
    """
    Stack tensors in sequence horizontally (column wise).

    This is equivalent to concatenation along the second axis, except for 1-D
    tensors where it concatenates along the first axis. Rebuilds tensors divided
    by `hsplit`.

    This function makes most sense for tensors with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of tensors
        The tensors must have the same shape along all but the second axis,
        except 1-D tensors which can be any length.

    Returns
    -------
    stacked : Tensor
        The tensor formed by stacking the given tensors.

    See Also
    --------
    stack : Join a sequence of tensors along a new axis.
    vstack : Stack tensors in sequence vertically (row wise).
    dstack : Stack tensors in sequence depth wise (along third axis).
    concatenate : Join a sequence of tensors along an existing axis.
    hsplit : Split tensor along second axis.
    block : Assemble tensors from blocks.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.array((1,2,3))
    >>> b = mt.array((2,3,4))
    >>> mt.hstack((a,b)).execute()
    array([1, 2, 3, 2, 3, 4])
    >>> a = mt.array([[1],[2],[3]])
    >>> b = mt.array([[2],[3],[4]])
    >>> mt.hstack((a,b)).execute()
    array([[1, 2],
           [2, 3],
           [3, 4]])

    """
    if all(x.ndim == 1 for x in tup):
        return concatenate(tup, axis=0)
    else:
        return concatenate(tup, axis=1)
