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

from numbers import Integral

import numpy as np

from ... import opcodes as OperandDef
from ...core import ENTITY_TYPE, recursive_tile
from ...serialization.serializables import KeyField, TupleField, AnyField
from ...tensor import tensor as astensor
from ...utils import has_unknown_shape
from ..core import TENSOR_TYPE
from ..operands import TensorHasInput, TensorOperandMixin
from ..utils import filter_inputs
from .core import process_index


class TensorIndexSetValue(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.INDEXSETVALUE

    _input = KeyField('input')
    _indexes = TupleField('indexes')
    _value = AnyField('value')

    def __init__(self, indexes=None, value=None, **kw):
        super().__init__(_indexes=indexes, _value=value, **kw)

    @property
    def indexes(self):
        return self._indexes

    @property
    def value(self):
        return self._value

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs[1:])
        new_indexes = [next(inputs_iter) if isinstance(index, ENTITY_TYPE) else index
                       for index in self._indexes]
        self._indexes = tuple(new_indexes)
        if isinstance(self._value, ENTITY_TYPE):
            self._value = next(inputs_iter)

    def __call__(self, a, index, value):
        inputs = filter_inputs([a] + list(index) + [value])
        self._indexes = tuple(index)
        self._value = value
        return self.new_tensor(inputs, a.shape, order=a.order)

    def on_output_modify(self, new_output):
        return new_output

    def on_input_modify(self, new_input):
        new_op = self.copy().reset_key()
        new_inputs = [new_input] + self.inputs[1:]
        return new_op.new_tensor(new_inputs, shape=self.outputs[0].shape)

    @classmethod
    def tile(cls, op: "TensorIndexSetValue"):
        from ..base import broadcast_to
        from .getitem import _getitem_nocheck

        tensor = op.outputs[0]
        value = op.value
        indexed = yield from recursive_tile(
            _getitem_nocheck(op.input, op.indexes, convert_bool_to_fancy=False))
        is_value_tensor = isinstance(value, TENSOR_TYPE)

        if is_value_tensor and value.ndim > 0:
            if has_unknown_shape(indexed, value):
                yield indexed.chunks + [indexed]

            value = yield from recursive_tile(
                broadcast_to(value, indexed.shape).astype(op.input.dtype, copy=False))
            nsplits = indexed.nsplits
            value = yield from recursive_tile(value.rechunk(nsplits))

        chunk_mapping = {c.op.input.index: c for c in indexed.chunks}
        out_chunks = []
        for chunk in indexed.op.input.chunks:
            index_chunk = chunk_mapping.get(chunk.index)
            if index_chunk is None:
                out_chunks.append(chunk)
                continue

            if is_value_tensor:
                if value.ndim > 0:
                    value_chunk = value.cix[index_chunk.index]
                else:
                    value_chunk = value.chunks[0]
            else:
                # non tensor
                value_chunk = value
            chunk_op = TensorIndexSetValue(dtype=op.dtype, sparse=op.sparse,
                                           indexes=tuple(index_chunk.op.indexes),
                                           value=value_chunk)
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


def _check_support(index):
    if isinstance(index, (slice, Integral)):
        pass
    elif isinstance(index, (np.ndarray, TENSOR_TYPE)) and index.dtype == np.bool_:
        pass
    else:  # pragma: no cover
        raise NotImplementedError('Only slice, int, or bool indexing '
                                  f'supported by now, got {type(index)}')


def _setitem(a, item, value):
    index = process_index(a.ndim, item, convert_bool_to_fancy=False)
    if not (np.isscalar(value) or (isinstance(value, tuple) and a.dtype.fields)):
        # do not convert for tuple when dtype is record type.
        value = astensor(value)

    for ix in index:
        _check_support(ix)

    # __setitem__ on a view should be still a view, see GH #732.
    op = TensorIndexSetValue(dtype=a.dtype, sparse=a.issparse(),
                             indexes=tuple(index), value=value,
                             create_view=a.op.create_view)
    ret = op(a, index, value)
    a.data = ret.data
