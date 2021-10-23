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
import itertools
import operator
from numbers import Integral
from typing import Union

import numpy as np

from ... import opcodes as OperandDef
from ...core import ENTITY_TYPE, recursive_tile
from ...core.context import Context
from ...core.operand import OperandStage
from ...serialization.serializables import KeyField, TupleField, AnyField, BoolField
from ...tensor import tensor as astensor
from ...utils import has_unknown_shape
from ..base import broadcast_to
from ..core import TENSOR_TYPE, TensorOrder
from ..operands import (
    TensorMapReduceOperand,
    TensorOperandMixin,
    TensorShuffleProxy,
)
from ..utils import broadcast_shape, filter_inputs
from .core import process_index


class TensorIndexSetValue(TensorMapReduceOperand, TensorOperandMixin):
    _op_type_ = OperandDef.INDEXSETVALUE

    input = KeyField("input")
    indexes = TupleField("indexes")
    value = AnyField("value")
    is_fancy_index = BoolField("is_fancy_index")
    input_nsplits = TupleField("input_nsplits")
    chunk_offsets = TupleField("chunk_offsets")
    shuffle_axes = TupleField("shuffle_axes")

    def __init__(
        self,
        indexes=None,
        value=None,
        is_fancy_index=None,
        input_nsplits=None,
        chunk_offsets=None,
        shuffle_axes=None,
        **kw,
    ):
        super().__init__(
            indexes=indexes,
            value=value,
            is_fancy_index=is_fancy_index,
            input_nsplits=input_nsplits,
            chunk_offsets=chunk_offsets,
            shuffle_axes=shuffle_axes,
            **kw,
        )

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        if self.stage == OperandStage.reduce:
            self.input = self._inputs[0]
            return
        elif self.stage == OperandStage.map:
            inputs_iter = iter(self._inputs)
        else:
            self.input = self._inputs[0]
            inputs_iter = iter(self._inputs[1:])
        new_indexes = [
            next(inputs_iter) if isinstance(index, ENTITY_TYPE) else index
            for index in self.indexes
        ]
        self.indexes = tuple(new_indexes)
        if isinstance(self.value, ENTITY_TYPE):
            self.value = next(inputs_iter)

    def __call__(self, a, index, value):
        inputs = filter_inputs([a] + list(index) + [value])
        self.indexes = tuple(index)
        self.value = value
        return self.new_tensor(inputs, a.shape, order=a.order)

    def on_output_modify(self, new_output):
        return new_output

    def on_input_modify(self, new_input):
        new_op = self.copy().reset_key()
        new_inputs = [new_input] + self.inputs[1:]
        return new_op.new_tensor(new_inputs, shape=self.outputs[0].shape)

    @classmethod
    def _tile_fancy_index(cls, op: "TensorIndexSetValue"):
        from ..utils import unify_chunks

        tensor = op.outputs[0]
        inp = op.inputs[0]
        value = op.value
        indexes = op.indexes

        if has_unknown_shape(inp):
            yield

        fancy_indexes = [index for index in indexes if isinstance(index, ENTITY_TYPE)]
        shape = broadcast_shape(*[ind.shape for ind in fancy_indexes])
        fancy_indexes = [broadcast_to(ind, shape) for ind in fancy_indexes]
        if isinstance(value, ENTITY_TYPE):
            value = broadcast_to(value, shape)
            value, *fancy_indexes = yield from unify_chunks(value, *fancy_indexes)
            value = value.chunks
        else:
            fancy_indexes = yield from unify_chunks(*fancy_indexes)
            value = [value] * len(fancy_indexes[0].chunks)
        input_nsplits = inp.nsplits
        shuffle_axes = tuple(
            axis for axis, ind in enumerate(indexes) if isinstance(ind, ENTITY_TYPE)
        )

        map_chunks = []
        for value_chunk, *index_chunks in zip(
            value, *[index.chunks for index in fancy_indexes]
        ):
            map_op = TensorIndexSetValue(
                stage=OperandStage.map,
                input_nsplits=input_nsplits,
                value=value_chunk,
                indexes=tuple(index_chunks),
                shuffle_axes=shuffle_axes,
                dtype=tensor.dtype,
            )
            inputs = filter_inputs([value_chunk] + list(index_chunks))
            map_chunk = map_op.new_chunk(
                inputs,
                shape=(np.nan,),
                index=index_chunks[0].index,
                order=TensorOrder.C_ORDER,
            )
            map_chunks.append(map_chunk)

        proxy_chunk = TensorShuffleProxy(dtype=tensor.dtype).new_chunk(
            map_chunks, shape=(), order=TensorOrder.C_ORDER
        )

        reducer_chunks = []
        offsets_on_axis = [np.cumsum([0] + list(split)) for split in input_nsplits]
        for input_chunk in inp.chunks:
            chunk_offsets = tuple(
                offsets_on_axis[axis][input_chunk.index[axis]]
                for axis in range(len(inp.shape))
            )
            reducer_op = TensorIndexSetValue(
                stage=OperandStage.reduce,
                dtype=input_chunk.dtype,
                shuffle_axes=shuffle_axes,
                chunk_offsets=chunk_offsets,
            )
            reducer_chunk = reducer_op.new_chunk(
                [input_chunk, proxy_chunk],
                index=input_chunk.index,
                shape=input_chunk.shape,
                order=input_chunk.order,
            )
            reducer_chunks.append(reducer_chunk)

        new_op = op.copy()
        return new_op.new_tensors(
            op.inputs,
            tensor.shape,
            order=tensor.order,
            chunks=reducer_chunks,
            nsplits=op.input.nsplits,
        )

    @classmethod
    def _tile(cls, op: "TensorIndexSetValue"):
        from ..base import broadcast_to
        from .getitem import _getitem_nocheck

        tensor = op.outputs[0]
        value = op.value
        indexed = yield from recursive_tile(
            _getitem_nocheck(op.input, op.indexes, convert_bool_to_fancy=False)
        )
        is_value_tensor = isinstance(value, TENSOR_TYPE)

        if is_value_tensor and value.ndim > 0:
            if has_unknown_shape(indexed, value):
                yield indexed.chunks + [indexed]

            value = yield from recursive_tile(
                broadcast_to(value, indexed.shape).astype(op.input.dtype, copy=False)
            )
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
            chunk_op = TensorIndexSetValue(
                dtype=op.dtype,
                sparse=op.sparse,
                indexes=tuple(index_chunk.op.indexes),
                value=value_chunk,
            )
            chunk_inputs = filter_inputs(
                [chunk] + index_chunk.op.indexes + [value_chunk]
            )
            out_chunk = chunk_op.new_chunk(
                chunk_inputs, shape=chunk.shape, index=chunk.index, order=tensor.order
            )
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_tensors(
            op.inputs,
            tensor.shape,
            order=tensor.order,
            chunks=out_chunks,
            nsplits=op.input.nsplits,
        )

    @classmethod
    def tile(cls, op: "TensorIndexSetValue"):
        if op.is_fancy_index:
            return (yield from cls._tile_fancy_index(op))
        else:
            return (yield from cls._tile(op))

    @classmethod
    def execute(cls, ctx: Union[dict, Context], op: "TensorIndexSetValue"):
        if op.stage == OperandStage.map:
            return cls._execute_map(ctx, op)
        elif op.stage == OperandStage.reduce:
            return cls._execute_reduce(ctx, op)
        else:
            return cls._execute(ctx, op)

    @classmethod
    def _execute(cls, ctx, op):
        indexes = [
            ctx[index.key] if hasattr(index, "key") else index for index in op.indexes
        ]
        input_ = ctx[op.inputs[0].key].copy()
        value = ctx[op.value.key] if hasattr(op.value, "key") else op.value
        if hasattr(input_, "flags") and not input_.flags.writeable:
            input_.setflags(write=True)
        input_[tuple(indexes)] = value
        ctx[op.outputs[0].key] = input_

    @classmethod
    def _execute_map(cls, ctx, op):
        nsplits = op.input_nsplits
        shuffle_axes = op.shuffle_axes
        all_inputs = [ctx[inp.key] for inp in op.inputs]
        if hasattr(op.value, "key"):
            value = ctx[op.value.key]
            indexes = all_inputs[1:]
        else:
            value = op.value
            indexes = all_inputs

        offsets_on_axis = [np.cumsum([0] + list(split)) for split in nsplits]
        for reducer_index in itertools.product(
            *(map(range, [len(s) for s in nsplits]))
        ):
            chunk_filters = []
            indexes_iter = iter(indexes)
            for axis, _ in enumerate(reducer_index):
                start = offsets_on_axis[axis][reducer_index[axis]]
                end = offsets_on_axis[axis][reducer_index[axis] + 1]
                if axis in shuffle_axes:
                    index_on_axis = next(indexes_iter)
                    filtered = (index_on_axis >= start) & (index_on_axis < end)
                    chunk_filters.append(filtered)
            combined_filter = functools.reduce(operator.and_, chunk_filters)
            if hasattr(op.value, "key"):
                ctx[op.outputs[0].key, reducer_index] = tuple(
                    inp[combined_filter] for inp in all_inputs
                )
            else:
                ctx[op.outputs[0].key, reducer_index] = tuple(
                    [value] + [inp[combined_filter] for inp in all_inputs]
                )

    @classmethod
    def _execute_reduce(cls, ctx, op):
        input_data = ctx[op.inputs[0].key].copy()
        for index_value in op.iter_mapper_data(ctx, input_id=1):
            value = index_value[0]
            indexes_with_offset = index_value[1:]
            indexes = []
            index_iter = iter(indexes_with_offset)
            for axis in range(input_data.ndim):
                if axis in op.shuffle_axes:
                    indexes.append(next(index_iter) - op.chunk_offsets[axis])
            input_data[indexes] = value

        ctx[op.outputs[0].key] = input_data


def _check_support(indexes):
    if all(
        (
            isinstance(ix, (TENSOR_TYPE, np.ndarray))
            and ix.dtype != np.bool_
            or isinstance(ix, slice)
            and ix == slice(None)
        )
        for ix in indexes
    ):
        if any(isinstance(ix, (TENSOR_TYPE, np.ndarray)) for ix in indexes):
            return True
    for index in indexes:
        if isinstance(index, (slice, Integral)):
            pass
        elif isinstance(index, (np.ndarray, TENSOR_TYPE)) and index.dtype == np.bool_:
            pass
        else:  # pragma: no cover
            raise NotImplementedError(
                "Only slice, int, or bool indexing "
                f"supported by now, got {type(index)}"
            )
    return False


def _setitem(a, item, value):
    index = process_index(a.ndim, item, convert_bool_to_fancy=False)
    if not (np.isscalar(value) or (isinstance(value, tuple) and a.dtype.fields)):
        # do not convert for tuple when dtype is record type.
        value = astensor(value)

    is_fancy_index = _check_support(index)
    if is_fancy_index:
        index = [astensor(ind) if isinstance(ind, np.ndarray) else ind for ind in index]

    # __setitem__ on a view should be still a view, see GH #732.
    op = TensorIndexSetValue(
        dtype=a.dtype,
        sparse=a.issparse(),
        is_fancy_index=is_fancy_index,
        indexes=tuple(index),
        value=value,
        create_view=a.op.create_view,
    )
    ret = op(a, index, value)
    a.data = ret.data
