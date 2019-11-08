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

import numpy as np

from ..datasource import tensor as astensor


def ravel(a, order='C'):
    """Return a contiguous flattened tensor.

    A 1-D tensor, containing the elements of the input, is returned.  A copy is
    made only if needed.

    Parameters
    ----------
    a : array_like
        Input tensor.  The elements in `a` are packed as a 1-D tensor.
    order : {'C','F', 'A', 'K'}, optional

        The elements of `a` are read using this index order. 'C' means
        to index the elements in row-major, C-style order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest.  'F' means to index the elements
        in column-major, Fortran-style order, with the
        first index changing fastest, and the last index changing
        slowest. Note that the 'C' and 'F' options take no account of
        the memory layout of the underlying array, and only refer to
        the order of axis indexing.  'A' means to read the elements in
        Fortran-like index order if `a` is Fortran *contiguous* in
        memory, C-like order otherwise.  'K' means to read the
        elements in the order they occur in memory, except for
        reversing the data when strides are negative.  By default, 'C'
        index order is used.

    Returns
    -------
    y : array_like
        If `a` is a matrix, y is a 1-D tensor, otherwise y is a tensor of
        the same subtype as `a`. The shape of the returned array is
        ``(a.size,)``. Matrices are special cased for backward
        compatibility.

    See Also
    --------
    Tensor.flat : 1-D iterator over an array.
    Tensor.flatten : 1-D array copy of the elements of an array
                      in row-major order.
    Tensor.reshape : Change the shape of an array without changing its data.

    Examples
    --------
    It is equivalent to ``reshape(-1)``.

    >>> import mars.tensor as mt

    >>> x = mt.array([[1, 2, 3], [4, 5, 6]])
    >>> print(mt.ravel(x).execute())
    [1 2 3 4 5 6]

    >>> print(x.reshape(-1).execute())
    [1 2 3 4 5 6]

    >>> print(mt.ravel(x.T).execute())
    [1 4 2 5 3 6]

    >>> a = mt.arange(12).reshape(2,3,2).swapaxes(1,2); a.execute()
    array([[[ 0,  2,  4],
            [ 1,  3,  5]],
           [[ 6,  8, 10],
            [ 7,  9, 11]]])
    >>> a.ravel().execute()
    array([ 0,  2,  4,  1,  3,  5,  6,  8, 10,  7,  9, 11])

    """
    a = astensor(a)
    if a.ndim == 0:
        return a[np.newaxis]
    return a.reshape(-1, order=order)
