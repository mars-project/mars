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
from ...serialization.serializables import KeyField
from ...utils import has_unknown_shape
from ..utils import broadcast_shape, unify_chunks
from ..array_utils import as_same_device, device
from ..core import TENSOR_TYPE
from ..operands import TensorOperand, TensorOperandMixin
from ..datasource import tensor as astensor
from .broadcast_to import broadcast_to


class TensorWhere(TensorOperand, TensorOperandMixin):
    _op_type_ = OperandDef.WHERE

    _condition = KeyField('condition')
    _x = KeyField('x')
    _y = KeyField('y')

    @property
    def condition(self):
        return self._condition

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._condition = self._inputs[0]
        self._x = self._inputs[1]
        self._y = self._inputs[2]

    def __call__(self, condition, x, y, shape=None):
        shape = shape or broadcast_shape(condition.shape, x.shape, y.shape)
        return self.new_tensor([condition, x, y], shape)

    @classmethod
    def tile(cls, op):
        if has_unknown_shape(*op.inputs):
            yield
        inputs = yield from unify_chunks(
            *[(input, list(range(input.ndim))[::-1]) for input in op.inputs])
        chunk_shapes = [t.chunk_shape if isinstance(t, TENSOR_TYPE) else t
                        for t in inputs]
        out_chunk_shape = broadcast_shape(*chunk_shapes)
        output = op.outputs[0]

        out_chunks = []
        nsplits = [[np.nan] * shape for shape in out_chunk_shape]
        get_index = lambda idx, t: tuple(0 if t.nsplits[i] == (1,) else ix for i, ix in enumerate(idx))
        for out_index in itertools.product(*(map(range, out_chunk_shape))):
            in_chunks = [t.cix[get_index(out_index[-t.ndim:], t)] if t.ndim != 0 else t.chunks[0]
                         for t in inputs]
            chunk_shape = broadcast_shape(*(c.shape for c in in_chunks))
            out_chunk = op.copy().reset_key().new_chunk(in_chunks, shape=chunk_shape,
                                                        index=out_index, order=output.order)
            out_chunks.append(out_chunk)
            for i, idx, s in zip(itertools.count(0), out_index, out_chunk.shape):
                nsplits[i][idx] = s

        new_op = op.copy()
        return new_op.new_tensors(inputs, output.shape, order=output.order,
                                  chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        (cond, x, y), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            ctx[op.outputs[0].key] = xp.where(cond, x, y)


def where(condition, x=None, y=None):
    """
    Return elements, either from `x` or `y`, depending on `condition`.

    If only `condition` is given, return ``condition.nonzero()``.

    Parameters
    ----------
    condition : array_like, bool
        When True, yield `x`, otherwise yield `y`.
    x, y : array_like, optional
        Values from which to choose. `x`, `y` and `condition` need to be
        broadcastable to some shape.

    Returns
    -------
    out : Tensor or tuple of Tensors
        If both `x` and `y` are specified, the output tensor contains
        elements of `x` where `condition` is True, and elements from
        `y` elsewhere.

        If only `condition` is given, return the tuple
        ``condition.nonzero()``, the indices where `condition` is True.

    See Also
    --------
    nonzero, choose

    Notes
    -----
    If `x` and `y` are given and input arrays are 1-D, `where` is
    equivalent to::

        [xv if c else yv for (c,xv,yv) in zip(condition,x,y)]

    Examples
    --------
    >>> import mars.tensor as mt

    >>> mt.where([[True, False], [True, True]],
    ...          [[1, 2], [3, 4]],
    ...          [[9, 8], [7, 6]]).execute()
    array([[1, 8],
           [3, 4]])

    >>> mt.where([[0, 1], [1, 0]]).execute()
    (array([0, 1]), array([1, 0]))

    >>> x = mt.arange(9.).reshape(3, 3)
    >>> mt.where( x > 5 ).execute()
    (array([2, 2, 2]), array([0, 1, 2]))
    >>> mt.where(x < 5, x, -1).execute()               # Note: broadcasting.
    array([[ 0.,  1.,  2.],
           [ 3.,  4., -1.],
           [-1., -1., -1.]])

    Find the indices of elements of `x` that are in `goodvalues`.

    >>> goodvalues = [3, 4, 7]
    >>> ix = mt.isin(x, goodvalues)
    >>> ix.execute()
    array([[False, False, False],
           [ True,  True, False],
           [False,  True, False]])
    >>> mt.where(ix).execute()
    (array([1, 1, 2]), array([0, 1, 1]))
    """
    if (x is None) != (y is None):
        raise ValueError('either both or neither of x and y should be given')

    if x is None and y is None:
        return astensor(condition).nonzero()

    x, y = astensor(x), astensor(y)
    dtype = np.result_type(x.dtype, y.dtype)
    shape = broadcast_shape(x.shape, y.shape)

    if np.isscalar(condition):
        return broadcast_to(x if condition else y, shape).astype(dtype)
    else:
        condition = astensor(condition)
        op = TensorWhere(dtype=dtype)
        return op(condition, x, y, shape=shape)
