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


def flip(m, axis):
    """
    Reverse the order of elements in a tensor along the given axis.

    The shape of the array is preserved, but the elements are reordered.

    Parameters
    ----------
    m : array_like
        Input tensor.
    axis : integer
        Axis in tensor, which entries are reversed.


    Returns
    -------
    out : array_like
        A view of `m` with the entries of axis reversed.  Since a view is
        returned, this operation is done in constant time.

    See Also
    --------
    flipud : Flip a tensor vertically (axis=0).
    fliplr : Flip a tensor horizontally (axis=1).

    Notes
    -----
    flip(m, 0) is equivalent to flipud(m).
    flip(m, 1) is equivalent to fliplr(m).
    flip(m, n) corresponds to ``m[...,::-1,...]`` with ``::-1`` at position n.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> A = mt.arange(8).reshape((2,2,2))
    >>> A.execute()
    array([[[0, 1],
            [2, 3]],

           [[4, 5],
            [6, 7]]])

    >>> mt.flip(A, 0).execute()
    array([[[4, 5],
            [6, 7]],

           [[0, 1],
            [2, 3]]])

    >>> mt.flip(A, 1).execute()
    array([[[2, 3],
            [0, 1]],

           [[6, 7],
            [4, 5]]])

    >>> A = mt.random.randn(3,4,5)
    >>> mt.all(mt.flip(A,2) == A[:,:,::-1,...]).execute()
    True
    """
    m = astensor(m)

    sl = [slice(None)] * m.ndim
    try:
        sl[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input tensor"
                         % (axis, m.ndim))

    return m[tuple(sl)]
