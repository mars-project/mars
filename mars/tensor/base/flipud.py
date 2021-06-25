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

from .flip import flip


def flipud(m):
    """
    Flip tensor in the up/down direction.

    Flip the entries in each column in the up/down direction.
    Rows are preserved, but appear in a different order than before.

    Parameters
    ----------
    m : array_like
        Input tensor.

    Returns
    -------
    out : array_like
        A view of `m` with the rows reversed.  Since a view is
        returned, this operation is :math:`\\mathcal O(1)`.

    See Also
    --------
    fliplr : Flip tensor in the left/right direction.
    rot90 : Rotate tensor counterclockwise.

    Notes
    -----
    Equivalent to ``m[::-1,...]``.
    Does not require the tensor to be two-dimensional.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> A = mt.diag([1.0, 2, 3])
    >>> A.execute()
    array([[ 1.,  0.,  0.],
           [ 0.,  2.,  0.],
           [ 0.,  0.,  3.]])
    >>> mt.flipud(A).execute()
    array([[ 0.,  0.,  3.],
           [ 0.,  2.,  0.],
           [ 1.,  0.,  0.]])

    >>> A = mt.random.randn(2,3,5)
    >>> mt.all(mt.flipud(A) == A[::-1,...]).execute()
    True

    >>> mt.flipud([1,2]).execute()
    array([2, 1])

    """
    return flip(m, 0)
