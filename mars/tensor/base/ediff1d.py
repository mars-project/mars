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

from ..datasource import tensor as astensor
from .ravel import ravel


def ediff1d(a, to_end=None, to_begin=None):
    """
    The differences between consecutive elements of a tensor.

    Parameters
    ----------
    a : array_like
        If necessary, will be flattened before the differences are taken.
    to_end : array_like, optional
        Number(s) to append at the end of the returned differences.
    to_begin : array_like, optional
        Number(s) to prepend at the beginning of the returned differences.

    Returns
    -------
    ediff1d : Tensor
        The differences. Loosely, this is ``a.flat[1:] - a.flat[:-1]``.

    See Also
    --------
    diff, gradient

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.array([1, 2, 4, 7, 0])
    >>> mt.ediff1d(x).execute()
    array([ 1,  2,  3, -7])

    >>> mt.ediff1d(x, to_begin=-99, to_end=mt.array([88, 99])).execute()
    array([-99,   1,   2,   3,  -7,  88,  99])

    The returned tensor is always 1D.

    >>> y = [[1, 2, 4], [1, 6, 24]]
    >>> mt.ediff1d(y).execute()
    array([ 1,  2, -3,  5, 18])

    """
    from ..merge import concatenate

    a = astensor(a)
    a = ravel(a)

    t = a[1:] - a[:-1]
    if to_begin is None and to_end is None:
        return t

    to_concat = [t]
    if to_begin is not None:
        to_concat.insert(0, ravel(astensor(to_begin)))
    if to_end is not None:
        to_concat.append(ravel(astensor(to_end)))

    return concatenate(to_concat)
