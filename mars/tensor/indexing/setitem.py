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

import functools
import operator
from numbers import Integral

import numpy as np

from ... import opcodes as OperandDef
from ...core import ENTITY_TYPE, recursive_tile
from ...serialization.serializables import KeyField, TupleField, AnyField, BoolField
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
    _is_fancy_index = BoolField('is_fancy_index')
    _index_offset = TupleField('index_offset')

    def __init__(self, indexes=None, value=None,
                 is_fancy_index=None, index_offset=None, **kw):
        super().__init__(_indexes=indexes, _value=value,
                         _is_fancy_index=is_fancy_index,
                         _index_offset=index_offset,
                         **kw)

    @property
    def indexes(self):
        return self._indexes

    @property
    def value(self):
        return self._value

    @property
    def is_fancy_index(self):
        return self._is_fancy_index

    @property
    def index_offset(self):
        return self._index_offset

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
    def _tile_fancy_index(cls, op: "TensorIndexSetValue"):
        from ..merge import column_stack, TensorConcatenate

        tensor = op.outputs[0]
        inp = op.inputs[0]
        value = op.value
        indexes = op.indexes

        if has_unknown_shape(inp):
            yield
        axis_to_tensor_index = dict((axis, ind) for axis, ind
                                    in enumerate(indexes) if isinstance(ind, ENTITY_TYPE))
        offsets_on_axis = [np.cumsum([0] + list(split)) for split in inp.nsplits]

        out_chunks = []
        for c in inp.chunks:
            chunk_filters = []
            chunk_index_offset = []
            for axis in range(len(c.shape)):
                offset = offsets_on_axis[axis][c.index[axis]]
                chunk_index_offset.append(offset)
                if axis in axis_to_tensor_index:
                    index_on_axis = axis_to_tensor_index[axis]
                    filtered = (index_on_axis >= offset) & \
                               (index_on_axis < offset + c.shape[axis])
                    chunk_filters.append(filtered)
            combined_filter = functools.reduce(operator.and_, chunk_filters)
            if isinstance(value, ENTITY_TYPE):
                concat_tensor = column_stack(list(axis_to_tensor_index.values()) + [value])
            else:
                concat_tensor = column_stack(list(axis_to_tensor_index.values()))
            tiled_tensor = yield from recursive_tile(
                concat_tensor[combined_filter])

            chunk_indexes = []
            tensor_index_order = 0
            for axis in range(len(c.shape)):
                if axis in axis_to_tensor_index:
                    index_chunks = [tiled_tensor.cix[i, tensor_index_order]
                                    for i in range(tiled_tensor.chunk_shape[0])]
                    concat_op = TensorConcatenate(axis=0, dtype=index_chunks[0].dtype)
                    chunk_indexes.append(concat_op.new_chunk(
                        index_chunks, shape=(tiled_tensor.shape[0],), index=(0,)))
                else:
                    chunk_indexes.append(slice(None))
                tensor_index_order += 1

            if isinstance(value, ENTITY_TYPE):
                value_chunks = [tiled_tensor.cix[i, -1]
                                for i in range(tiled_tensor.chunk_shape[0])]
                concat_op = TensorConcatenate(axis=0, dtype=value_chunks[0].dtype)
                chunk_value = concat_op.new_chunk(
                    value_chunks, shape=(tiled_tensor.shape[0],), index=(0,))
            else:
                chunk_value = value
            chunk_op = TensorIndexSetValue(
                dtype=op.dtype, sparse=op.sparse,
                indexes=tuple(chunk_indexes),
                index_offset=tuple(chunk_index_offset),
                value=chunk_value)
            input_chunks = filter_inputs([c] + chunk_indexes + [chunk_value])
            out_chunk = chunk_op.new_chunk(input_chunks, shape=c.shape,
                                           index=c.index, order=tensor.order)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(op.inputs, tensor.shape, order=tensor.order,
                                  chunks=out_chunks, nsplits=op.input.nsplits)

    @classmethod
    def _tile(cls, op: "TensorIndexSetValue"):
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
    def tile(cls, op: "TensorIndexSetValue"):
        if op.is_fancy_index:
            return (yield from cls._tile_fancy_index(op))
        else:
            return (yield from cls._tile(op))

    @classmethod
    def execute(cls, ctx, op):
        indexes = [ctx[index.key] if hasattr(index, 'key') else index
                   for index in op.indexes]
        if getattr(op, 'index_offset', None) is not None:
            new_indexs = []
            index_iter = iter(indexes)
            for ind, offset in zip(indexes, op.index_offset):
                if isinstance(ind, np.ndarray):
                    new_indexs.append(next(index_iter) - offset)
                else:
                    new_indexs.append(ind)
            indexes = new_indexs
        input_ = ctx[op.inputs[0].key].copy()
        value = ctx[op.value.key] if hasattr(op.value, 'key') else op.value
        if hasattr(input_, 'flags') and not input_.flags.writeable:
            input_.setflags(write=True)

        input_[tuple(indexes)] = value
        ctx[op.outputs[0].key] = input_


def _check_support(indexes):
    if all((isinstance(ix, (TENSOR_TYPE, np.ndarray)) and ix.dtype != np.bool_
           or isinstance(ix, slice) and ix == slice(None)) for ix in indexes):
        if any(isinstance(ix, (TENSOR_TYPE, np.ndarray)) for ix in indexes):
            return True
    for index in indexes:
        if isinstance(index, (slice, Integral)):
            pass
        elif isinstance(index, (np.ndarray, TENSOR_TYPE)) and index.dtype == np.bool_:
            pass
        else:  # pragma: no cover
            raise NotImplementedError('Only slice, int, or bool indexing '
                                      f'supported by now, got {type(index)}')
    return False


def _setitem(a, item, value):
    index = process_index(a.ndim, item, convert_bool_to_fancy=False)
    if not (np.isscalar(value) or (isinstance(value, tuple) and a.dtype.fields)):
        # do not convert for tuple when dtype is record type.
        value = astensor(value)

    is_fancy_index = _check_support(index)
    if is_fancy_index:
        index = [astensor(ind) if isinstance(ind, np.ndarray) else ind
                 for ind in index]

    # __setitem__ on a view should be still a view, see GH #732.
    op = TensorIndexSetValue(dtype=a.dtype, sparse=a.issparse(),
                             is_fancy_index=is_fancy_index,
                             indexes=tuple(index), value=value,
                             create_view=a.op.create_view)
    ret = op(a, index, value)
    a.data = ret.data
