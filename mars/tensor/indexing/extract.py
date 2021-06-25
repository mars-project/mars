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


def extract(condition, a):
    """
    Return the elements of a tensor that satisfy some condition.

    This is equivalent to ``mt.compress(ravel(condition), ravel(arr))``.  If
    `condition` is boolean ``mt.extract`` is equivalent to ``arr[condition]``.

    Note that `place` does the exact opposite of `extract`.

    Parameters
    ----------
    condition : array_like
        An array whose nonzero or True entries indicate the elements of `arr`
        to extract.
    a : array_like
        Input tensor of the same size as `condition`.

    Returns
    -------
    extract : Tensor
        Rank 1 tensor of values from `arr` where `condition` is True.

    See Also
    --------
    take, put, copyto, compress, place

    Examples
    --------
    >>> import mars.tensor as mt

    >>> arr = mt.arange(12).reshape((3, 4))
    >>> arr.execute()
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
    >>> condition = mt.mod(arr, 3)==0
    >>> condition.execute()
    array([[ True, False, False,  True],
           [False, False,  True, False],
           [False,  True, False, False]])
    >>> mt.extract(condition, arr).execute()
    array([0, 3, 6, 9])


    If `condition` is boolean:

    >>> arr[condition].execute()
    array([0, 3, 6, 9])

    """
    condition = astensor(condition, dtype=bool)
    return a[condition]
