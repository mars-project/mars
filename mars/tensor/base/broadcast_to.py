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

import numpy as np

from ... import opcodes as OperandDef
from ...serialize import KeyField, TupleField
from ..operands import TensorHasInput, TensorOperandMixin
from ..datasource import tensor as astensor
from ..array_utils import get_array_module, device


class TensorBroadcastTo(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.BROADCAST_TO

    _input = KeyField('input')
    _shape = TupleField('shape')

    def __init__(self, shape=None, dtype=None, sparse=False, **kw):
        super().__init__(_shape=shape, _dtype=dtype, _sparse=sparse, **kw)

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
            out_chunk = chunk_op.new_chunk([c], shape=chunk_shape, index=chunk_idx, order=tensor.order)
            out_chunks.append(out_chunk)

        nsplits = [tuple(c.shape[i] for c in out_chunks if all(idx == 0 for j, idx in enumerate(c.index) if j != i))
                   for i in range(len(out_chunks[0].shape))]
        new_op = op.copy()
        return new_op.new_tensors([in_tensor], tensor.shape, order=tensor.order,
                                  chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        xp = get_array_module(ctx[op.input.key])
        input_data = ctx[op.input.key]
        device_id = input_data.device.id if hasattr(input_data, 'device') else -1

        with device(device_id):
            shape = op.shape
            if any(np.isnan(s) for s in shape):
                shape = list(shape)
                new_dim = len(shape) - input_data.ndim
                for i in range(input_data.ndim):
                    if np.isnan(shape[i + new_dim]):
                        shape[i + new_dim] = input_data.shape[i]
            ctx[op.outputs[0].key] = xp.broadcast_to(input_data, shape)


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
    from ..core import Tensor

    tensor = tensor if isinstance(tensor, Tensor) else astensor(tensor)
    shape = tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)

    if any(np.isnan(s) for s in tensor.shape):
        raise ValueError('input tensor has unknown shape, '
                         'need to call `.execute()` first')

    if tensor.shape == shape:
        return tensor

    new_ndim = len(shape) - tensor.ndim
    if new_ndim < 0:
        raise ValueError('input operand has more dimensions than allowed by the axis remapping')
    if any(o != n for o, n in zip(tensor.shape, shape[new_ndim:]) if o != 1):
        raise ValueError('operands could not be broadcast together '
                         'with remapped shapes [original->remapped]: {0} '
                         'and requested shape {1}'.format(tensor.shape, shape))

    op = TensorBroadcastTo(shape, dtype=tensor.dtype, sparse=tensor.issparse())
    return op(tensor, shape)
