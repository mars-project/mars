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

import itertools

import numpy as np

from ... import opcodes as OperandDef
from ...core import recursive_tile
from ...serialization.serializables import KeyField
from ...utils import has_unknown_shape
from ..operands import TensorHasInput, TensorOperandMixin
from ..datasource import tensor as astensor
from .ravel import ravel


class TensorArgwhere(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.ARGWHERE

    _input = KeyField('input')

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
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
        out_tensor = op.outputs[0]

        if has_unknown_shape(in_tensor):
            yield

        flattened = yield from recursive_tile(ravel(in_tensor))
        indices = arange(flattened.size, dtype=np.intp, chunks=flattened.nsplits)
        indices = indices[flattened]
        dim_indices = unravel_index(indices, in_tensor.shape)
        dim_indices = yield from recursive_tile(*dim_indices)

        out_chunk_shape = dim_indices[0].chunk_shape + (in_tensor.ndim,)
        nsplits = dim_indices[0].nsplits + ((1,) * in_tensor.ndim,)
        out_chunks = []
        for out_index in itertools.product(*(map(range, out_chunk_shape))):
            dim_ind_chunk = dim_indices[out_index[1]].chunks[out_index[0]]
            chunk_shape = dim_ind_chunk.shape + (1,)
            chunk_op = TensorReshape(newshape=(-1, 1), dtype=dim_ind_chunk.dtype)
            out_chunk = chunk_op.new_chunk([dim_ind_chunk], shape=chunk_shape, index=out_index,
                                           order=out_tensor.order)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, out_tensor.shape, order=out_tensor.order,
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
    a = astensor(a).astype(bool, order='A')
    op = TensorArgwhere(dtype=np.dtype(np.intp))
    return op(a)
