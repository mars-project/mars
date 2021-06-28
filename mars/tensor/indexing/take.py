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

from ..utils import validate_axis, check_out_param
from ..datasource import tensor as astensor


def take(a, indices, axis=None, out=None):
    """
    Take elements from a tensor along an axis.

    When axis is not None, this function does the same thing as "fancy"
    indexing (indexing arrays using tensors); however, it can be easier to use
    if you need elements along a given axis. A call such as
    ``mt.take(arr, indices, axis=3)`` is equivalent to
    ``arr[:,:,:,indices,...]``.

    Explained without fancy indexing, this is equivalent to the following use
    of `ndindex`, which sets each of ``ii``, ``jj``, and ``kk`` to a tuple of
    indices::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        Nj = indices.shape
        for ii in ndindex(Ni):
            for jj in ndindex(Nj):
                for kk in ndindex(Nk):
                    out[ii + jj + kk] = a[ii + (indices[jj],) + kk]

    Parameters
    ----------
    a : array_like (Ni..., M, Nk...)
        The source tensor.
    indices : array_like (Nj...)
        The indices of the values to extract.

        Also allow scalars for indices.
    axis : int, optional
        The axis over which to select values. By default, the flattened
        input tensor is used.
    out : Tensor, optional (Ni..., Nj..., Nk...)
        If provided, the result will be placed in this tensor. It should
        be of the appropriate shape and dtype.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave.

        * 'raise' -- raise an error (default)
        * 'wrap' -- wrap around
        * 'clip' -- clip to the range

        'clip' mode means that all indices that are too large are replaced
        by the index that addresses the last element along that axis. Note
        that this disables indexing with negative numbers.

    Returns
    -------
    out : Tensor (Ni..., Nj..., Nk...)
        The returned tensor has the same type as `a`.

    See Also
    --------
    compress : Take elements using a boolean mask
    Tensor.take : equivalent method

    Notes
    -----

    By eliminating the inner loop in the description above, and using `s_` to
    build simple slice objects, `take` can be expressed  in terms of applying
    fancy indexing to each 1-d slice::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        for ii in ndindex(Ni):
            for kk in ndindex(Nj):
                out[ii + s_[...,] + kk] = a[ii + s_[:,] + kk][indices]

    For this reason, it is equivalent to (but faster than) the following use
    of `apply_along_axis`::

        out = mt.apply_along_axis(lambda a_1d: a_1d[indices], axis, a)

    Examples
    --------
    >>> import mars.tensor as mt
    >>> a = [4, 3, 5, 7, 6, 8]
    >>> indices = [0, 1, 4]
    >>> mt.take(a, indices).execute()
    array([4, 3, 6])

    In this example if `a` is a tensor, "fancy" indexing can be used.

    >>> a = mt.array(a)
    >>> a[indices].execute()
    array([4, 3, 6])

    If `indices` is not one dimensional, the output also has these dimensions.

    >>> mt.take(a, [[0, 1], [2, 3]]).execute()
    array([[4, 3],
           [5, 7]])
    """
    a = astensor(a)
    if axis is None:
        t = a.ravel()[indices]
    else:
        axis = validate_axis(a.ndim, axis)
        t = a[(slice(None),) * axis + (indices,)]

    if out is None:
        return t

    if out.shape != t.shape:
        raise ValueError('output tensor has wrong shape, '
                         f'expect: {t.shape}, got: {out.shape}')
    check_out_param(out, t, 'unsafe')
    out.data = t.data
    return out
