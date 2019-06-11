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

from ....compat import izip
from .... import opcodes as OperandDef
from ....serialize import KeyField, TupleField
from ..core import TensorHasInput, TensorOperandMixin
from ..datasource import tensor as astensor


class TensorBroadcastTo(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.BROADCAST_TO

    _input = KeyField('input')
    _shape = TupleField('shape')

    def __init__(self, shape=None, dtype=None, sparse=False, **kw):
        super(TensorBroadcastTo, self).__init__(_shape=shape, _dtype=dtype, _sparse=sparse, **kw)

    @property
    def shape(self):
        return self._shape

    def __call__(self, tensor, shape):
        return self.new_tensor([tensor], shape)

    @classmethod
    def tile(cls, op):
        tensor = op.outputs[0]
        in_tensor = op.inputs[0]
        shape = op.shape
        new_dim = tensor.ndim - in_tensor.ndim

        out_chunks = []
        for c in in_tensor.chunks:
            chunk_shape = shape[:new_dim] + tuple(s if in_tensor.shape[idx] != 1 else shape[new_dim+idx]
                                                  for idx, s in enumerate(c.shape))
            chunk_idx = (0,) * new_dim + c.index
            chunk_op = op.copy().reset_key()
            chunk_op._shape = chunk_shape
            out_chunk = chunk_op.new_chunk([c], shape=chunk_shape, index=chunk_idx)
            out_chunks.append(out_chunk)

        nsplits = [tuple(c.shape[i] for c in out_chunks if all(idx == 0 for j, idx in enumerate(c.index) if j != i))
                   for i in range(len(out_chunks[0].shape))]
        new_op = op.copy()
        return new_op.new_tensors([in_tensor], tensor.shape, chunks=out_chunks, nsplits=nsplits)


def broadcast_to(tensor, shape):
    """Broadcast an tensor to a new shape.

    Parameters
    ----------
    tensor : array_like
        The tensor to broadcast.
    shape : tuple
        The shape of the desired array.

    Returns
    -------
    broadcast : Tensor

    Raises
    ------
    ValueError
        If the tensor is not compatible with the new shape according to Mars's
        broadcasting rules.

    Examples
    --------
    >>> import mars.tensor as mt

    >>> x = mt.array([1, 2, 3])
    >>> mt.broadcast_to(x, (3, 3)).execute()
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])
    """
    from ...core import Tensor

    tensor = tensor if isinstance(tensor, Tensor) else astensor(tensor)
    shape = tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)

    if tensor.shape == shape:
        return tensor

    new_ndim = len(shape) - tensor.ndim
    if new_ndim < 0:
        raise ValueError('input operand has more dimensions than allowed by the axis remapping')
    if any(o != n for o, n in izip(tensor.shape, shape[new_ndim:]) if o != 1):
        raise ValueError('operands could not be broadcast together '
                         'with remapped shapes [original->remapped]: {0} '
                         'and requested shape {1}'.format(tensor.shape, shape))

    op = TensorBroadcastTo(shape, dtype=tensor.dtype, sparse=tensor.issparse())
    return op(tensor, shape)
