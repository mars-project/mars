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

from collections.abc import Iterable

from ...serialize import ValueType, KeyField, TupleField
from ... import opcodes as OperandDef
from ..operands import TensorHasInput, TensorOperandMixin
from ..array_utils import as_same_device, device


def _get_squeeze_shape(shape, axis):
    if axis is not None:
        if isinstance(axis, Iterable):
            axis = tuple(axis)
        else:
            axis = (axis,)

        for ax in axis:
            if shape[ax] != 1:
                raise ValueError('cannot select an axis to squeeze out '
                                 'which has size not equal to one')
        shape = tuple(s for i, s in enumerate(shape) if i not in axis)
    else:
        axis = tuple(i for i, s in enumerate(shape) if s == 1)
        shape = tuple(s for s in shape if s != 1)

    return shape, axis


class TensorSqueeze(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.SQUEEZE

    _input = KeyField('input')
    _axis = TupleField('axis', ValueType.int32)

    def __init__(self, axis=None, dtype=None, sparse=False, **kw):
        super().__init__(_axis=axis, _dtype=dtype, _sparse=sparse, _create_view=True, **kw)

    def on_output_modify(self, new_output):
        slcs = [slice(None)] * new_output.ndim
        for axis in self._axis:
            slcs.insert(axis, None)
        return new_output[slcs]

    def on_input_modify(self, new_input):
        op = self.copy().reset_key()
        return op(new_input, self.outputs[0].shape)

    @property
    def axis(self):
        return self._axis

    def __call__(self, a, shape):
        return self.new_tensor([a], shape, order=a.order)

    @classmethod
    def tile(cls, op):
        in_tensor = op.input
        out_tensor = op.outputs[0]
        axis_set = set(op.axis)

        out_chunks = []
        for c in in_tensor.chunks:
            chunk_op = op.copy().reset_key()
            chunk_shape = _get_squeeze_shape(c.shape, op.axis)[0]
            chunk_idx = tuple(idx for i, idx in enumerate(c.index) if i not in axis_set)
            out_chunk = chunk_op.new_chunk([c], shape=chunk_shape, index=chunk_idx,
                                           order=out_tensor.order)
            out_chunks.append(out_chunk)
        nsplits = [nsplit for i, nsplit in enumerate(in_tensor.nsplits) if i not in axis_set]

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, op.outputs[0].shape, order=out_tensor.order,
                                  chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        (a,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            ctx[op.outputs[0].key] = xp.squeeze(a, axis=op.axis)


def squeeze(a, axis=None):
    """
    Remove single-dimensional entries from the shape of a tensor.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Selects a subset of the single-dimensional entries in the
        shape. If an axis is selected with shape entry greater than
        one, an error is raised.

    Returns
    -------
    squeezed : Tensor
        The input tensor, but with all or a subset of the
        dimensions of length 1 removed. This is always `a` itself
        or a view into `a`.

    Raises
    ------
    ValueError
        If `axis` is not `None`, and an axis being squeezed is not of length 1

    See Also
    --------
    expand_dims : The inverse operation, adding singleton dimensions
    reshape : Insert, remove, and combine dimensions, and resize existing ones

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.array([[[0], [1], [2]]])
    >>> x.shape
    (1, 3, 1)
    >>> mt.squeeze(x).shape
    (3,)
    >>> mt.squeeze(x, axis=0).shape
    (3, 1)
    >>> mt.squeeze(x, axis=1).shape
    Traceback (most recent call last):
    ...
    ValueError: cannot select an axis to squeeze out which has size not equal to one
    >>> mt.squeeze(x, axis=2).shape
    (1, 3)

    """
    shape, axis = _get_squeeze_shape(a.shape, axis)

    if 1 not in a.shape:
        return a

    op = TensorSqueeze(axis=axis, dtype=a.dtype, sparse=a.issparse())
    return op(a, shape)
