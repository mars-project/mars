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

from numbers import Integral

import numpy as np

from ... import opcodes as OperandDef
from ...serialize import KeyField, ListField, AnyField
from ...core import Base, Entity
from ...utils import check_chunks_unknown_shape
from ...tiles import TilesError
from ..core import TENSOR_TYPE
from ..operands import TensorHasInput, TensorOperandMixin
from ..utils import filter_inputs
from .core import process_index, calc_shape
from .getitem import TensorIndex


class TensorIndexSetValue(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.INDEXSETVALUE

    _input = KeyField('input')
    _indexes = ListField('indexes')
    _value = AnyField('value')

    def __init__(self, dtype=None, sparse=False, indexes=None, value=None, **kw):
        super(TensorIndexSetValue, self).__init__(_dtype=dtype, _sparse=sparse,
                                                  _indexes=indexes, _value=value, **kw)

    @property
    def indexes(self):
        return self._indexes

    @property
    def value(self):
        return self._value

    def _set_inputs(self, inputs):
        super(TensorIndexSetValue, self)._set_inputs(inputs)
        inputs_iter = iter(self._inputs[1:])
        new_indexes = [next(inputs_iter) if isinstance(index, (Base, Entity)) else index
                       for index in self._indexes]
        self._indexes = new_indexes
        self._value = next(inputs_iter) if isinstance(self._value, (Base, Entity)) else self._value

    def __call__(self, a, index, value):
        inputs = filter_inputs([a] + list(index) + [value])
        self._indexes = index
        self._value = value
        return self.new_tensor(inputs, a.shape, order=a.order)

    def on_output_modify(self, new_output):
        return new_output

    def on_input_modify(self, new_input):
        new_op = self.copy().reset_key()
        new_inputs = [new_input] + self.inputs[1:]
        return new_op.new_tensor(new_inputs, shape=self.outputs[0].shape)

    @classmethod
    def tile(cls, op):
        tensor = op.outputs[0]
        value = op.value
        is_value_tensor = isinstance(value, TENSOR_TYPE)

        index_tensor_op = TensorIndex(dtype=tensor.dtype, sparse=op.sparse, indexes=op.indexes)
        index_tensor_inputs = filter_inputs([op.input] + op.indexes)
        index_tensor = index_tensor_op.new_tensor(index_tensor_inputs, tensor.shape)._inplace_tile()

        to_check_tensors = [index_tensor]
        if is_value_tensor:
            to_check_tensors.append(value)
        check_chunks_unknown_shape(to_check_tensors, TilesError)

        nsplits = index_tensor.nsplits
        if is_value_tensor:
            value = value.rechunk(nsplits)._inplace_tile()

        chunk_mapping = {c.op.input.index: c for c in index_tensor.chunks}
        out_chunks = []
        for chunk in op.input.chunks:
            index_chunk = chunk_mapping.get(chunk.index)
            if index_chunk is None:
                out_chunks.append(chunk)
                continue

            value_chunk = value.cix[index_chunk.index] if is_value_tensor else value
            chunk_op = TensorIndexSetValue(dtype=op.dtype, sparse=op.sparse,
                                           indexes=index_chunk.op.indexes, value=value_chunk)
            chunk_inputs = filter_inputs([chunk] + index_chunk.op.indexes + [value_chunk])
            out_chunk = chunk_op.new_chunk(chunk_inputs, shape=chunk.shape,
                                           index=chunk.index, order=tensor.order)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, tensor.shape, order=tensor.order,
                                  chunks=out_chunks, nsplits=op.input.nsplits)

    @classmethod
    def execute(cls, ctx, op):
        indexes = [ctx[index.key] if hasattr(index, 'key') else index
                   for index in op.indexes]
        input_ = ctx[op.inputs[0].key].copy()
        value = ctx[op.value.key] if hasattr(op.value, 'key') else op.value
        if hasattr(input_, 'flags') and not input_.flags.writeable:
            input_.setflags(write=True)
        input_[tuple(indexes)] = value
        ctx[op.outputs[0].key] = input_


def _setitem(a, item, value):
    from ..base import broadcast_to

    index = process_index(a.ndim, item)
    shape = calc_shape(a.shape, index)

    for ix in index:
        if not isinstance(ix, (slice, Integral)):
            raise NotImplementedError('Only slice or int supported by now, got {0}'.format(type(ix)))

    # don't broadcast for tuple when dtype is record type.
    if not (np.isscalar(value) or (isinstance(value, tuple) and a.dtype.fields)):
        value = broadcast_to(value, shape).astype(a.dtype)

    # __setitem__ on a view should be still a view, see GH #732.
    op = TensorIndexSetValue(dtype=a.dtype, sparse=a.issparse(), indexes=index, value=value, _create_view=a.op.create_view)
    ret = op(a, index, value)
    a.data = ret.data
