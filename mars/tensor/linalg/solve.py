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
from .cholesky import cholesky
from .lu import lu
from .solve_triangular import solve_triangular


def solve(a, b, sym_pos=False, sparse=None):
    """
    Solve the equation ``a x = b`` for ``x``.

    Parameters
    ----------
    a : (M, M) array_like
        A square matrix.
    b : (M,) or (M, N) array_like
        Right-hand side matrix in ``a x = b``.
    sym_pos : bool
        Assume `a` is symmetric and positive definite. If ``True``, use Cholesky
        decomposition.
    sparse: bool, optional
        Return sparse value or not.

    Returns
    -------
    x : (M,) or (M, N) ndarray
    Solution to the system ``a x = b``.  Shape of the return matches the
    shape of `b`.

    Raises
    ------
    LinAlgError
    If `a` is singular.

    Examples
    --------
    Given `a` and `b`, solve for `x`:

    >>> import mars.tensor as mt
    >>> a = mt.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
    >>> b = mt.array([2, 4, -1])
    >>> x = mt.linalg.solve(a, b)
    >>> x.execute()
    array([ 2., -2.,  9.])

    >>> mt.dot(a, x).execute()  # Check the result
    array([ 2., 4., -1.])
    """
    a = astensor(a)
    b = astensor(b)
    if sym_pos:
        l_ = cholesky(a, lower=True)
        u = l_.T
    else:
        p, l_, u = lu(a)
        b = p.T.dot(b)
    sparse = sparse if sparse is not None else a.issparse()
    uy = solve_triangular(l_, b, lower=True, sparse=sparse)
    return solve_triangular(u, uy, sparse=sparse)
