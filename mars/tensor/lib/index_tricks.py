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

import math

import numpy as np
from numpy.lib.index_tricks import ndindex

from .. import datasource as _nx


class nd_grid(object):
    """
    Construct a multi-dimensional "meshgrid".

    ``grid = nd_grid()`` creates an instance which will return a mesh-grid
    when indexed.  The dimension and number of the output arrays are equal
    to the number of indexing dimensions.  If the step length is not a
    complex number, then the stop is not inclusive.

    However, if the step length is a **complex number** (e.g. 5j), then the
    integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    If instantiated with an argument of ``sparse=True``, the mesh-grid is
    open (or not fleshed out) so that only one-dimension of each returned
    argument is greater than 1.

    Parameters
    ----------
    sparse : bool, optional
        Whether the grid is sparse or not. Default is False.

    Notes
    -----
    Two instances of `nd_grid` are made available in the Mars.tensor namespace,
    `mgrid` and `ogrid`::

        mgrid = nd_grid(sparse=False)
        ogrid = nd_grid(sparse=True)

    Users should use these pre-defined instances instead of using `nd_grid`
    directly.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mgrid = mt.lib.index_tricks.nd_grid()
    >>> mgrid[0:5,0:5]
    array([[[0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4]],
           [[0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]]])
    >>> mgrid[-1:1:5j]
    array([-1. , -0.5,  0. ,  0.5,  1. ])

    >>> ogrid = mt.lib.index_tricks.nd_grid(sparse=True)
    >>> ogrid[0:5,0:5]
    [array([[0],
            [1],
            [2],
            [3],
            [4]]), array([[0, 1, 2, 3, 4]])]

    """

    def __init__(self, sparse=False):
        self.sparse = sparse

    def __getitem__(self, key):
        try:
            size = []
            typ = int
            for k in key:
                step = k.step
                start = k.start
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if isinstance(step, complex):
                    size.append(int(abs(step)))
                    typ = float
                else:
                    size.append(
                        int(math.ceil((k.stop - start)/(step*1.0))))
                if (isinstance(step, float) or
                        isinstance(start, float) or
                        isinstance(k.stop, float)):
                    typ = float
            if self.sparse:
                nn = [_nx.arange(_x, dtype=_t)
                      for _x, _t in zip(size, (typ,)*len(size))]
            else:
                nn = _nx.indices(size, typ)
            for k in range(len(size)):
                step = key[k].step
                start = key[k].start
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if isinstance(step, complex):
                    step = int(abs(step))
                    if step != 1:
                        step = (key[k].stop - start)/float(step-1)
                nn[k] = (nn[k]*step+start)
            if self.sparse:
                slobj = [np.newaxis]*len(size)
                for k in range(len(size)):
                    slobj[k] = slice(None, None)
                    nn[k] = nn[k][slobj]
                    slobj[k] = np.newaxis
            return nn
        except (IndexError, TypeError):
            step = key.step
            stop = key.stop
            start = key.start
            if start is None:
                start = 0
            if isinstance(step, complex):
                step = abs(step)
                length = int(step)
                if step != 1:
                    step = (key.stop-start)/float(step-1)
                stop = key.stop + step
                return _nx.arange(0, length, 1, float)*step + start
            else:
                return _nx.arange(start, stop, step)

    def __len__(self):
        return 0


mgrid = nd_grid(sparse=False)
ogrid = nd_grid(sparse=True)

__all__ = ['ndindex', 'mgrid', 'ogrid']
