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

from ... import opcodes as OperandDef
from ...serialize import KeyField, Int32Field
from ..utils import validate_axis, reverse_order
from ..operands import TensorHasInput, TensorOperandMixin
from ..array_utils import as_same_device, device
from ..core import TensorOrder


def _swap(it, axis1, axis2):
    new_it = list(it)
    new_it[axis1], new_it[axis2] = it[axis2], it[axis1]

    return tuple(new_it)


class TensorSwapAxes(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.SWAPAXES

    _input = KeyField('input')
    _axis1 = Int32Field('axis1')
    _axis2 = Int32Field('axis2')

    def __init__(self, axis1=None, axis2=None, dtype=None, sparse=False, **kw):
        super().__init__(_axis1=axis1, _axis2=axis2, _dtype=dtype, _sparse=sparse,
                         _create_view=True, **kw)

    @property
    def axis1(self):
        return self._axis1

    @property
    def axis2(self):
        return self._axis2

    def __call__(self, a):
        axis1, axis2 = self._axis1, self._axis2
        if (axis1 == 0 and axis2 == a.ndim - 1) or (axis1 == a.ndim - 1 and axis2 == 0):
            tensor_order = reverse_order(a.order)
        else:
            tensor_order = TensorOrder.C_ORDER
        shape = _swap(a.shape, self.axis1, self.axis2)
        return self.new_tensor([a], shape, order=tensor_order)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    def on_output_modify(self, new_output):
        op = TensorSwapAxes(axis1=self._axis2, axis2=self._axis1, dtype=new_output.dtype,
                            sparse=new_output.issparse())
        return op(new_output)

    def on_input_modify(self, new_input):
        op = self.copy().reset_key()
        return op(new_input)

    @classmethod
    def tile(cls, op):
        axis1, axis2 = op.axis1, op.axis2
        in_tensor = op.inputs[0]
        out_tensor = op.outputs[0]

        out_chunks = []
        for c in in_tensor.chunks:
            chunk_shape = _swap(c.shape, axis1, axis2)
            chunk_idx = _swap(c.index, axis1, axis2)
            chunk_op = op.copy().reset_key()
            out_chunk = chunk_op.new_chunk([c], shape=chunk_shape,
                                           index=chunk_idx, order=out_tensor.order)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        nsplits = _swap(in_tensor.nsplits, axis1, axis2)
        return new_op.new_tensors([in_tensor], out_tensor.shape, order=out_tensor.order,
                                  chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        (x,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        axis1, axis2 = op.axis1, op.axis2
        with device(device_id):
            ctx[op.outputs[0].key] = xp.swapaxes(x, axis1, axis2)


def swapaxes(a, axis1, axis2):
    """
    Interchange two axes of a tensor.

    Parameters
    ----------
    a : array_like
        Input tensor.
    axis1 : int
        First axis.
    axis2 : int
        Second axis.

    Returns
    -------
    a_swapped : Tensor
        If `a` is a Tensor, then a view of `a` is
        returned; otherwise a new tensor is created.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.array([[1,2,3]])
    >>> mt.swapaxes(x,0,1).execute()
    array([[1],
           [2],
           [3]])

    >>> x = mt.array([[[0,1],[2,3]],[[4,5],[6,7]]])
    >>> x.execute()
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])

    >>> mt.swapaxes(x,0,2).execute()
    array([[[0, 4],
            [2, 6]],
           [[1, 5],
            [3, 7]]])

    """
    axis1 = validate_axis(a.ndim, axis1)
    axis2 = validate_axis(a.ndim, axis2)

    if axis1 == axis2:
        return a

    op = TensorSwapAxes(axis1, axis2, dtype=a.dtype, sparse=a.issparse())
    return op(a)
