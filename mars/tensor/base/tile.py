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


def tile(A, reps):
    """
    Construct a tensor by repeating A the number of times given by reps.

    If `reps` has length ``d``, the result will have dimension of
    ``max(d, A.ndim)``.

    If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
    axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
    or shape (1, 1, 3) for 3-D replication. If this is not the desired
    behavior, promote `A` to d-dimensions manually before calling this
    function.

    If ``A.ndim > d``, `reps` is promoted to `A`.ndim by pre-pending 1's to it.
    Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
    (1, 1, 2, 2).

    Note : Although tile may be used for broadcasting, it is strongly
    recommended to use Mars' broadcasting operations and functions.

    Parameters
    ----------
    A : array_like
        The input tensor.
    reps : array_like
        The number of repetitions of `A` along each axis.

    Returns
    -------
    c : Tensor
        The tiled output tensor.

    See Also
    --------
    repeat : Repeat elements of a tensor.
    broadcast_to : Broadcast a tensor to a new shape

    Examples
    --------
    >>> import mars.tensor as mt

    >>> a = mt.array([0, 1, 2])
    >>> mt.tile(a, 2).execute()
    array([0, 1, 2, 0, 1, 2])
    >>> mt.tile(a, (2, 2)).execute()
    array([[0, 1, 2, 0, 1, 2],
           [0, 1, 2, 0, 1, 2]])
    >>> mt.tile(a, (2, 1, 2)).execute()
    array([[[0, 1, 2, 0, 1, 2]],
           [[0, 1, 2, 0, 1, 2]]])

    >>> b = mt.array([[1, 2], [3, 4]])
    >>> mt.tile(b, 2).execute()
    array([[1, 2, 1, 2],
           [3, 4, 3, 4]])
    >>> mt.tile(b, (2, 1)).execute()
    array([[1, 2],
           [3, 4],
           [1, 2],
           [3, 4]])

    >>> c = mt.array([1,2,3,4])
    >>> mt.tile(c,(4,1)).execute()
    array([[1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4]])
    """
    from ..merge import concatenate

    try:
        tup = tuple(reps)
    except TypeError:
        tup = (reps,)

    d = len(tup)
    if A.ndim < d:
        A = A[tuple(np.newaxis for _ in range(d - A.ndim))]
    elif A.ndim > d:
        tup = (1,) * (A.ndim - d) + tup

    a = A
    for axis, rep in enumerate(tup):
        if rep == 0:
            slc = (slice(None),) * axis + (slice(0),)
            a = a[slc]
        elif rep < 0:
            raise ValueError('negative dimensions are not allowed')
        elif rep > 1:
            a = concatenate([a] * rep, axis=axis)

    return a
