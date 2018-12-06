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

import numpy as np

from ....operands import Transpose
from ..core import TensorOperandMixin
from ..datasource import tensor as astensor


def _reorder(x, axes):
    if x is None:
        return
    return type(x)(np.array(x)[list(axes)].tolist())


class TensorTranspose(Transpose, TensorOperandMixin):
    def __init__(self, axes=None, dtype=None, sparse=False, **kw):
        super(TensorTranspose, self).__init__(_axes=axes, _dtype=dtype,
                                              _sparse=sparse, **kw)

    def __call__(self, a):
        shape = _reorder(a.shape, self._axes)
        return self.new_tensor([a], shape)

    def _set_inputs(self, inputs):
        super(TensorTranspose, self)._set_inputs(inputs)
        self._input = self._inputs[0]

    @classmethod
    def tile(cls, op):
        out_chunks = []
        for c in op.inputs[0].chunks:
            chunk_op = op.copy().reset_key()
            chunk_shape = _reorder(c.shape, op.axes)
            chunk_idx = _reorder(c.index, op.axes)
            out_chunk = chunk_op.new_chunk([c], chunk_shape, index=chunk_idx)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        nsplits = _reorder(op.inputs[0].nsplits, op.axes)
        return new_op.new_tensors(op.inputs, op.outputs[0].shape,
                                  chunks=out_chunks, nsplits=nsplits)


def transpose(a, axes=None):
    """
    Permute the dimensions of a tensor.

    Parameters
    ----------
    a : array_like
        Input tensor.
    axes : list of ints, optional
        By default, reverse the dimensions, otherwise permute the axes
        according to the values given.

    Returns
    -------
    p : Tensor
        `a` with its axes permuted.  A view is returned whenever
        possible.

    See Also
    --------
    moveaxis
    argsort

    Notes
    -----
    Use `transpose(a, argsort(axes))` to invert the transposition of tensors
    when using the `axes` keyword argument.

    Transposing a 1-D array returns an unchanged view of the original tensor.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.arange(4).reshape((2,2))
    >>> x.execute()
    array([[0, 1],
           [2, 3]])

    >>> mt.transpose(x).execute()
    array([[0, 2],
           [1, 3]])

    >>> x = mt.ones((1, 2, 3))
    >>> mt.transpose(x, (1, 0, 2)).shape
    (2, 1, 3)

    """
    a = astensor(a)
    if axes:
        if len(axes) != a.ndim:
            raise ValueError("axes don't match tensor")

    axes = axes or list(range(a.ndim))[::-1]
    op = TensorTranspose(axes, dtype=a.dtype, sparse=a.issparse())
    return op(a)
