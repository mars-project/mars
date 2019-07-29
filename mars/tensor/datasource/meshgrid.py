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

from .array import tensor


def meshgrid(*xi, **kwargs):
    """
    Return coordinate matrices from coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate tensors x1, x2,..., xn.

    Parameters
    ----------
    x1, x2,..., xn : array_like
        1-D arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of output.
        See Notes for more details.
    sparse : bool, optional
        If True a sparse grid is returned in order to conserve memory.
        Default is False.

    Returns
    -------
    X1, X2,..., XN : Tensor
        For vectors `x1`, `x2`,..., 'xn' with lengths ``Ni=len(xi)`` ,
        return ``(N1, N2, N3,...Nn)`` shaped tensors if indexing='ij'
        or ``(N2, N1, N3,...Nn)`` shaped tensors if indexing='xy'
        with the elements of `xi` repeated to fill the matrix along
        the first dimension for `x1`, the second for `x2` and so on.

    Notes
    -----
    This function supports both indexing conventions through the indexing
    keyword argument.  Giving the string 'ij' returns a meshgrid with
    matrix indexing, while 'xy' returns a meshgrid with Cartesian indexing.
    In the 2-D case with inputs of length M and N, the outputs are of shape
    (N, M) for 'xy' indexing and (M, N) for 'ij' indexing.  In the 3-D case
    with inputs of length M, N and P, outputs are of shape (N, M, P) for
    'xy' indexing and (M, N, P) for 'ij' indexing.  The difference is
    illustrated by the following code snippet::

        xv, yv = mt.meshgrid(x, y, sparse=False, indexing='ij')
        for i in range(nx):
            for j in range(ny):
                # treat xv[i,j], yv[i,j]

        xv, yv = mt.meshgrid(x, y, sparse=False, indexing='xy')
        for i in range(nx):
            for j in range(ny):
                # treat xv[j,i], yv[j,i]

    In the 1-D and 0-D case, the indexing and sparse keywords have no effect.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> nx, ny = (3, 2)
    >>> x = mt.linspace(0, 1, nx)
    >>> y = mt.linspace(0, 1, ny)
    >>> xv, yv = mt.meshgrid(x, y)
    >>> xv.execute()
    array([[ 0. ,  0.5,  1. ],
           [ 0. ,  0.5,  1. ]])
    >>> yv.execute()
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]])
    >>> xv, yv = mt.meshgrid(x, y, sparse=True)  # make sparse output arrays
    >>> xv.execute()
    array([[ 0. ,  0.5,  1. ]])
    >>> yv.execute()
    array([[ 0.],
           [ 1.]])

    `meshgrid` is very useful to evaluate functions on a grid.

    >>> import matplotlib.pyplot as plt
    >>> x = mt.arange(-5, 5, 0.1)
    >>> y = mt.arange(-5, 5, 0.1)
    >>> xx, yy = mt.meshgrid(x, y, sparse=True)
    >>> z = mt.sin(xx**2 + yy**2) / (xx**2 + yy**2)
    >>> h = plt.contourf(x,y,z)

    """
    from ..base import broadcast_to

    indexing = kwargs.pop('indexing', 'xy')
    sparse = kwargs.pop('sparse', False)

    if kwargs:
        raise TypeError(
            "meshgrid() got an unexpected keyword argument '{0}'".format(list(kwargs)[0]))
    if indexing not in ('xy', 'ij'):
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")

    xi = [tensor(x) for x in xi]
    xi = [a.ravel() for a in xi]
    shape = [x.size for x in xi]

    if indexing == 'xy' and len(xi) > 1:
        xi[0], xi[1] = xi[1], xi[0]
        shape[0], shape[1] = shape[1], shape[0]

    grid = []
    for i, x in enumerate(xi):
        slc = [None] * len(shape)
        slc[i] = slice(None)

        r = x[tuple(slc)]

        if not sparse:
            r = broadcast_to(r, shape)

        grid.append(r)

    if indexing == 'xy' and len(xi) > 1:
        grid[0], grid[1] = grid[1], grid[0]

    return grid
