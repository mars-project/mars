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

import math

import numpy as np
from numpy.core.numerictypes import find_common_type
from numpy.core.numeric import ScalarType
from numpy.lib.index_tricks import ndindex

from .. import datasource as _nx
from ..core import Tensor
from ..base import ndim
from ..merge import concatenate


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
        except (IndexError, TypeError):  # pragma: no cover
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


class AxisConcatenator:
    def __init__(self, axis=0, matrix=False, ndmin=1, trans1d=-1):
        self.axis = axis
        self.matrix = matrix
        self.trans1d = trans1d
        self.ndmin = ndmin

    def __getitem__(self, key):
        # handle matrix builder syntax
        if isinstance(key, str):  # pragma: no cover
            raise NotImplementedError('Does not support operation on matrix')

        if not isinstance(key, tuple):
            key = (key,)

        # copy attributes, since they can be overridden in the first argument
        trans1d = self.trans1d
        ndmin = self.ndmin
        matrix = self.matrix
        axis = self.axis

        objs = []
        scalars = []
        arraytypes = []
        scalartypes = []

        for k, item in enumerate(key):
            scalar = False
            if isinstance(item, slice):
                step = item.step
                start = item.start
                stop = item.stop
                if start is None:
                    start = 0
                if step is None:
                    step = 1
                if isinstance(step, complex):
                    size = int(abs(step))
                    newobj = _nx.linspace(start, stop, num=size)
                else:
                    newobj = _nx.arange(start, stop, step)
                if ndmin > 1:
                    newobj = _nx.array(newobj, copy=False, ndmin=ndmin)
                    if trans1d != -1:
                        newobj = newobj.swapaxes(-1, trans1d)
            elif isinstance(item, str):
                if k != 0:
                    raise ValueError("special directives must be the "
                                     "first entry.")
                if item in ('r', 'c'):  # pragma: no cover
                    raise NotImplementedError('Does not support operation on matrix')
                if ',' in item:
                    vec = item.split(',')
                    try:
                        axis, ndmin = [int(x) for x in vec[:2]]
                        if len(vec) == 3:
                            trans1d = int(vec[2])
                        continue
                    except Exception:  # pragma: no cover
                        raise ValueError("unknown special directive")
                try:
                    axis = int(item)
                    continue
                except (ValueError, TypeError):  # pragma: no cover# pragma: no cover
                    raise ValueError("unknown special directive")
            elif type(item) in ScalarType:
                newobj = np.array(item, ndmin=ndmin)
                scalars.append(len(objs))
                scalar = True
                scalartypes.append(newobj.dtype)
            else:
                item_ndim = ndim(item)
                newobj = _nx.array(item, copy=False, ndmin=ndmin)
                if trans1d != -1 and item_ndim < ndmin:
                    k2 = ndmin - item_ndim
                    k1 = trans1d
                    if k1 < 0:
                        k1 += k2 + 1
                    defaxes = list(range(ndmin))
                    axes = defaxes[:k1] + defaxes[k2:] + defaxes[k1:k2]
                    newobj = newobj.transpose(axes)
            objs.append(newobj)
            if not scalar and isinstance(newobj, Tensor):
                arraytypes.append(newobj.dtype)

        # Ensure that scalars won't up-cast unless warranted
        final_dtype = find_common_type(arraytypes, scalartypes)
        if final_dtype is not None:
            for k in scalars:
                objs[k] = objs[k].astype(final_dtype)

        res = concatenate(tuple(objs), axis=axis)

        if matrix:  # pragma: no cover
            raise NotImplementedError('Does not support operation on matrix')
        return res

    def __len__(self):
        return 0


# separate classes are used here instead of just making r_ = concatentor(0),
# etc. because otherwise we couldn't get the doc string to come out right
# in help(r_)

class RClass(AxisConcatenator):
    """
    Translates slice objects to concatenation along the first axis.

    This is a simple way to build up tensor quickly. There are two use cases.

    1. If the index expression contains comma separated tensors, then stack
       them along their first axis.
    2. If the index expression contains slice notation or scalars then create
       a 1-D tensor with a range indicated by the slice notation.

    If slice notation is used, the syntax ``start:stop:step`` is equivalent
    to ``mt.arange(start, stop, step)`` inside of the brackets. However, if
    ``step`` is an imaginary number (i.e. 100j) then its integer portion is
    interpreted as a number-of-points desired and the start and stop are
    inclusive. In other words ``start:stop:stepj`` is interpreted as
    ``mt.linspace(start, stop, step, endpoint=1)`` inside of the brackets.
    After expansion of slice notation, all comma separated sequences are
    concatenated together.

    Optional character strings placed as the first element of the index
    expression can be used to change the output. The strings 'r' or 'c' result
    in matrix output. If the result is 1-D and 'r' is specified a 1 x N (row)
    matrix is produced. If the result is 1-D and 'c' is specified, then a N x 1
    (column) matrix is produced. If the result is 2-D then both provide the
    same matrix result.

    A string integer specifies which axis to stack multiple comma separated
    tensors along. A string of two comma-separated integers allows indication
    of the minimum number of dimensions to force each entry into as the
    second integer (the axis to concatenate along is still the first integer).

    A string with three comma-separated integers allows specification of the
    axis to concatenate along, the minimum number of dimensions to force the
    entries to, and which axis should contain the start of the tensors which
    are less than the specified number of dimensions. In other words the third
    integer allows you to specify where the 1's should be placed in the shape
    of the tensors that have their shapes upgraded. By default, they are placed
    in the front of the shape tuple. The third argument allows you to specify
    where the start of the tensor should be instead. Thus, a third argument of
    '0' would place the 1's at the end of the tensor shape. Negative integers
    specify where in the new shape tuple the last dimension of upgraded tensors
    should be placed, so the default is '-1'.

    Parameters
    ----------
    Not a function, so takes no parameters


    Returns
    -------
    A concatenated tensor or matrix.

    See Also
    --------
    concatenate : Join a sequence of tensors along an existing axis.
    c_ : Translates slice objects to concatenation along the second axis.

    Examples
    --------
    >>> import mars.tensor as mt
    >>> mt.r_[mt.array([1,2,3]), 0, 0, mt.array([4,5,6])].execute()
    array([1, 2, 3, ..., 4, 5, 6])
    >>> mt.r_[-1:1:6j, [0]*3, 5, 6].execute()
    array([-1. , -0.6, -0.2,  0.2,  0.6,  1. ,  0. ,  0. ,  0. ,  5. ,  6. ])

    String integers specify the axis to concatenate along or the minimum
    number of dimensions to force entries into.

    >>> a = mt.array([[0, 1, 2], [3, 4, 5]])
    >>> mt.r_['-1', a, a].execute() # concatenate along last axis
    array([[0, 1, 2, 0, 1, 2],
           [3, 4, 5, 3, 4, 5]])
    >>> mt.r_['0,2', [1,2,3], [4,5,6]].execute() # concatenate along first axis, dim>=2
    array([[1, 2, 3],
           [4, 5, 6]])

    >>> mt.r_['0,2,0', [1,2,3], [4,5,6]].execute()
    array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]])
    >>> mt.r_['1,2,0', [1,2,3], [4,5,6]].execute()
    array([[1, 4],
           [2, 5],
           [3, 6]])
    """

    def __init__(self):
        AxisConcatenator.__init__(self, 0)


r_ = RClass()


class CClass(AxisConcatenator):
    """
    Translates slice objects to concatenation along the second axis.

    This is short-hand for ``mt.r_['-1,2,0', index expression]``, which is
    useful because of its common occurrence. In particular, tensors will be
    stacked along their last axis after being upgraded to at least 2-D with
    1's post-pended to the shape (column vectors made out of 1-D tensors).

    See Also
    --------
    column_stack : Stack 1-D tensors as columns into a 2-D tensor.
    r_ : For more detailed documentation.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.c_[mt.array([1,2,3]), mt.array([4,5,6])].execute()
    array([[1, 4],
           [2, 5],
           [3, 6]])
    >>> mt.c_[mt.array([[1,2,3]]), 0, 0, mt.array([[4,5,6]])].execute()
    array([[1, 2, 3, ..., 4, 5, 6]])

    """

    def __init__(self):
        AxisConcatenator.__init__(self, -1, ndmin=2, trans1d=0)


c_ = CClass()


__all__ = ['ndindex', 'mgrid', 'ogrid', 'r_', 'c_']
