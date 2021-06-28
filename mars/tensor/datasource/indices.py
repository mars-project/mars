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

from collections.abc import Iterable

import numpy as np

from ... import opcodes as OperandDef
from ...serialization.serializables import FieldTypes, ListField
from .core import TensorNoInput
from .arange import arange
from .empty import empty
from .meshgrid import meshgrid


class TensorIndices(TensorNoInput):
    _op_type_ = OperandDef.TENSOR_INDICES

    _dimensions = ListField('dimensions', FieldTypes.uint64)

    def __init__(self, dimensions=None, **kw):
        super().__init__(_dimensions=dimensions, **kw)

    @property
    def dimensions(self):
        return self._dimensions


def indices(dimensions, dtype=int, chunk_size=None):
    """
    Return a tensor representing the indices of a grid.

    Compute a tensor where the subtensors contain index values 0,1,...
    varying only along the corresponding axis.

    Parameters
    ----------
    dimensions : sequence of ints
        The shape of the grid.
    dtype : dtype, optional
        Data type of the result.
    chunk_size : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension

    Returns
    -------
    grid : Tensor
        The tensor of grid indices,
        ``grid.shape = (len(dimensions),) + tuple(dimensions)``.

    See Also
    --------
    mgrid, meshgrid

    Notes
    -----
    The output shape is obtained by prepending the number of dimensions
    in front of the tuple of dimensions, i.e. if `dimensions` is a tuple
    ``(r0, ..., rN-1)`` of length ``N``, the output shape is
    ``(N,r0,...,rN-1)``.

    The subtensors ``grid[k]`` contains the N-D array of indices along the
    ``k-th`` axis. Explicitly::

        grid[k,i0,i1,...,iN-1] = ik

    Examples
    --------
    >>> import mars.tensor as mt

    >>> grid = mt.indices((2, 3))
    >>> grid.shape
    (2, 2, 3)
    >>> grid[0].execute()        # row indices
    array([[0, 0, 0],
           [1, 1, 1]])
    >>> grid[1].execute()        # column indices
    array([[0, 1, 2],
           [0, 1, 2]])

    The indices can be used as an index into a tensor.

    >>> x = mt.arange(20).reshape(5, 4)
    >>> row, col = mt.indices((2, 3))
    >>> # x[row, col]  # TODO(jisheng): accomplish this if multiple fancy indexing is supported

    Note that it would be more straightforward in the above example to
    extract the required elements directly with ``x[:2, :3]``.

    """
    from ..merge import stack

    dimensions = tuple(dimensions)
    dtype = np.dtype(dtype)
    raw_chunk_size = chunk_size
    if chunk_size is not None and isinstance(chunk_size, Iterable):
        chunk_size = tuple(chunk_size)
    else:
        chunk_size = (chunk_size,) * len(dimensions)

    xi = []
    for ch, dim in zip(chunk_size, dimensions):
        xi.append(arange(dim, dtype=dtype, chunk_size=ch))

    grid = None
    if np.prod(dimensions):
        grid = meshgrid(*xi, indexing='ij')

    if grid:
        grid = stack(grid)
    else:
        if raw_chunk_size is None:
            empty_chunk_size = None
        else:
            empty_chunk_size = (1,) + chunk_size
        grid = empty((len(dimensions),) + dimensions, dtype=dtype, chunk_size=empty_chunk_size)

    return grid
