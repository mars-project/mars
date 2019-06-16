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

import itertools

import numpy as np

from ....operands import Argwhere
from ..utils import recursive_tile
from ..core import TensorOperandMixin
from ..datasource import tensor as astensor
from .ravel import ravel


class TensorArgwhere(Argwhere, TensorOperandMixin):
    def __init__(self, dtype=None, **kw):
        super(TensorArgwhere, self).__init__(_dtype=dtype, **kw)

    def _set_inputs(self, inputs):
        super(TensorArgwhere, self)._set_inputs(inputs)
        self._input = self._inputs[0]

    def __call__(self, a):
        shape = (np.nan, a.ndim)
        return self.new_tensor([a], shape)

    @classmethod
    def tile(cls, op):
        from ..datasource import arange
        from ..indexing import unravel_index
        from ..reshape.reshape import TensorReshape

        in_tensor = op.input

        flattened = ravel(in_tensor).single_tiles()
        indices = arange(flattened.size, dtype=np.intp, chunks=flattened.nsplits)
        indices = indices[flattened]
        dim_indices = unravel_index(indices, in_tensor.shape)
        [recursive_tile(ind) for ind in dim_indices]

        out_chunk_shape = dim_indices[0].chunk_shape + (in_tensor.ndim,)
        nsplits = dim_indices[0].nsplits + ((1,) * in_tensor.ndim,)
        out_chunks = []
        for out_index in itertools.product(*(map(range, out_chunk_shape))):
            dim_ind_chunk = dim_indices[out_index[1]].chunks[out_index[0]]
            chunk_shape = dim_ind_chunk.shape + (1,)
            chunk_op = TensorReshape(newshape=(-1, 1), dtype=dim_ind_chunk.dtype)
            out_chunk = chunk_op.new_chunk([dim_ind_chunk], chunk_shape, index=out_index)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, op.outputs[0].shape,
                                  chunks=out_chunks, nsplits=nsplits)


def argwhere(a):
    """
    Find the indices of tensor elements that are non-zero, grouped by element.

    Parameters
    ----------
    a : array_like
        Input data.

    Returns
    -------
    index_tensor : Tensor
        Indices of elements that are non-zero. Indices are grouped by element.

    See Also
    --------
    where, nonzero

    Notes
    -----
    ``mt.argwhere(a)`` is the same as ``mt.transpose(mt.nonzero(a))``.

    The output of ``argwhere`` is not suitable for indexing tensors.
    For this purpose use ``nonzero(a)`` instead.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.arange(6).reshape(2,3)
    >>> x.execute()
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> mt.argwhere(x>1).execute()
    array([[0, 2],
           [1, 0],
           [1, 1],
           [1, 2]])

    """
    a = astensor(a).astype(bool)
    op = TensorArgwhere(np.dtype(np.intp))
    return op(a)
