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
from ...serialize import ValueType, KeyField, ListField, TupleField, Int32Field
from ...core import Base, Entity
from ...operands import OperandStage
from ...utils import get_shuffle_input_keys_idxes
from ..core import TENSOR_TYPE, TensorOrder
from ..utils import split_indexes_into_chunks, calc_pos, filter_inputs
from ..operands import TensorHasInput, TensorOperandMixin, TensorMapReduceOperand
from ..array_utils import get_array_module
from .core import process_index, calc_shape
from .index_lib import TensorIndexesHandler

FANCY_INDEX_TYPES = TENSOR_TYPE + (np.ndarray,)


class TensorIndex(TensorHasInput, TensorOperandMixin):
    _op_type_ = OperandDef.INDEX

    _input = KeyField('input')
    _indexes = ListField('indexes')

    def __init__(self, dtype=None, sparse=False, indexes=None, create_view=False, **kw):
        super().__init__(_dtype=dtype, _sparse=sparse, _indexes=indexes,
                         _create_view=create_view, **kw)

    @property
    def indexes(self):
        return self._indexes

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        inputs_iter = iter(self._inputs[1:])
        new_indexes = [next(inputs_iter) if isinstance(index, (Base, Entity)) else index
                       for index in self._indexes]
        self._indexes = new_indexes

    def on_output_modify(self, new_output):
        from .setitem import TensorIndexSetValue

        if self._create_view:
            a = self.input
            op = TensorIndexSetValue(dtype=a.dtype, sparse=a.issparse(),
                                     indexes=self._indexes, value=new_output)
            return op(a, self._indexes, new_output)

    def on_input_modify(self, new_input):
        if self._create_view:
            new_op = self.copy().reset_key()
            new_inputs = [new_input] + self.inputs[1:]
            return new_op.new_tensor(new_inputs, shape=self.outputs[0].shape)

    def __call__(self, a, index, shape, order):
        self._indexes = index
        return self.new_tensor(filter_inputs([a] + list(index)), shape, order=order)

    @classmethod
    def tile(cls, op):
        handler = TensorIndexesHandler()
        return [handler.handle(op)]

    @classmethod
    def execute(cls, ctx, op):
        indexes = tuple(ctx[index.key] if hasattr(index, 'key') else index
                        for index in op.indexes)
        input_ = ctx[op.inputs[0].key]
        xp = get_array_module(input_)
        ret = xp.asarray(input_)[indexes]
        if hasattr(ret, 'astype'):
            ret = ret.astype(
                ret.dtype, order=op.outputs[0].order.value, copy=False)
        ctx[op.outputs[0].key] = ret

    @classmethod
    def estimate_size(cls, ctx, op):
        from mars.core import Base, Entity
        chunk = op.outputs[0]
        shape = chunk.shape
        new_indexes = [index for index in op._indexes if index is not None]

        new_shape = []
        first_fancy_index = False
        for index in new_indexes:
            if isinstance(index, (Base, Entity)):
                if index.dtype != np.bool_:
                    if not first_fancy_index:
                        first_fancy_index = True
                    else:
                        continue
                new_shape.append(ctx[index.key][0] // index.dtype.itemsize)

        rough_shape = []
        idx = 0
        for s in shape:
            if np.isnan(s):
                rough_shape.append(new_shape[idx])
                idx += 1
            else:
                rough_shape.append(s)
        result = int(np.prod(rough_shape) * chunk.dtype.itemsize)
        ctx[chunk.key] = (result, result)


class FancyIndexingDistribute(TensorMapReduceOperand, TensorOperandMixin):
    _op_type_ = OperandDef.FANCY_INDEX_DISTRIBUTE

    _input = KeyField('input')
    _dest_nsplits = TupleField('dest_nsplits', ValueType.tuple(ValueType.uint64))
    _axes = TupleField('axes', ValueType.int32)

    def __init__(self, stage=None, dest_nsplits=None, axes=None, dtype=None, sparse=None,
                 shuffle_key=None, **kw):
        super().__init__(_stage=stage, _dest_nsplits=dest_nsplits, _axes=axes,
                         _dtype=dtype, _sparse=sparse, _shuffle_key=shuffle_key, **kw)

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self._input = self._inputs[0]

    @property
    def output_limit(self):
        if self.stage == OperandStage.map:
            return 1
        # return fancy indexes on each axis as well as original position
        return len(self._axes) + 1

    @property
    def dest_nsplits(self):
        return self._dest_nsplits

    @property
    def axes(self):
        return self._axes

    @classmethod
    def _execute_map(cls, ctx, op):
        nsplits = op.dest_nsplits
        axes = op.axes
        fancy_index_nsplits = [nsplits[ax] for ax in axes]
        indexes = ctx[op.inputs[0].key]
        flatten_indexes = indexes.reshape(indexes.shape[0], -1)
        idx_to_fancy_indexes, idx_to_poses = \
            split_indexes_into_chunks(fancy_index_nsplits, flatten_indexes, False)
        for idx in idx_to_fancy_indexes:
            group_key = ','.join(str(i) for i in idx)
            ctx[(op.outputs[0].key, group_key)] = (idx_to_fancy_indexes[idx], idx_to_poses[idx])

    @classmethod
    def _execute_reduce(cls, ctx, op):
        in_chunk = op.inputs[0]

        input_keys, _ = get_shuffle_input_keys_idxes(in_chunk)

        fancy_indexes = []
        poses = []
        shuffle_key = op.shuffle_key
        xp = None
        for input_key in input_keys:
            key = (input_key, shuffle_key)
            fancy_index, pos = ctx[key]
            if xp is None:
                xp = get_array_module(fancy_index)
            if fancy_index.size == 0:
                fancy_index = fancy_index.reshape(len(op.axes), 0)
            fancy_indexes.append(fancy_index)
            poses.append(pos)

        fancy_index = np.hstack(fancy_indexes)
        pos = np.hstack(poses)

        assert len(op.outputs) - 1 == len(fancy_index)
        for out_chunk, axis_fancy_index in zip(op.outputs[:-1], fancy_index):
            ctx[out_chunk.key] = axis_fancy_index
        ctx[op.outputs[-1].key] = np.asarray([len(p) for p in poses]), pos

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls._execute_map(ctx, op)
        else:
            cls._execute_reduce(ctx, op)

    @classmethod
    def estimate_size(cls, ctx, op):
        if op.stage == OperandStage.map:
            fancy_index_size = len(op.axes)
            inp_size = ctx[op.inputs[0].key][0]
            factor = 1 / float(fancy_index_size) + fancy_index_size  # 1/#fancy_index is the poses
            ctx[op.outputs[0].key] = (inp_size * factor,) * 2
        else:
            sum_size = 0
            for shuffle_input in op.inputs[0].inputs or ():
                sum_size += ctx[shuffle_input.key]
            for out_chunk in op.outputs:
                ctx[out_chunk.key] = sum_size, sum_size


class FancyIndexingConcat(TensorMapReduceOperand, TensorOperandMixin):
    _op_type_ = OperandDef.FANCY_INDEX_CONCAT

    _fancy_index_axis = Int32Field('fancy_index_axis')
    _fancy_index_shape = TupleField('fancy_index_shape', ValueType.int64)

    def __init__(self, stage=None, fancy_index_axis=None, fancy_index_shape=None,
                 shuffle_key=None, dtype=None, sparse=None, **kw):
        super().__init__(_stage=stage, _fancy_index_axis=fancy_index_axis,
                         _fancy_index_shape=fancy_index_shape,
                         _shuffle_key=shuffle_key, _dtype=dtype, _sparse=sparse, **kw)

    @property
    def input(self):
        return self._input

    @property
    def fancy_index_axis(self):
        return self._fancy_index_axis

    @property
    def fancy_index_shape(self):
        return self._fancy_index_shape

    @classmethod
    def _execute_map(cls, ctx, op):
        indexed_array = ctx[op.inputs[0].key]
        sizes, pos = ctx[op.inputs[1].key]
        acc_sizes = np.cumsum(sizes)
        fancy_index_axis = op.fancy_index_axis

        for i in range(len(sizes)):
            start = 0 if i == 0 else acc_sizes[i - 1]
            end = acc_sizes[i]
            select = (slice(None),) * fancy_index_axis + (slice(start, end),)
            ctx[(op.outputs[0].key, str(i))] = (indexed_array[select], pos[start: end])

    @classmethod
    def _execute_reduce(cls, ctx, op):
        in_chunk = op.inputs[0]
        input_keys, _ = get_shuffle_input_keys_idxes(in_chunk)
        fancy_index_axis = op.fancy_index_axis
        fancy_index_shape = op.fancy_index_shape

        indexed_arrays = []
        poses = []
        shuffle_key = op.shuffle_key
        for input_key in input_keys:
            index_array, pos = ctx[(input_key, shuffle_key)]
            indexed_arrays.append(index_array)
            poses.append(pos)

        concat_array = get_array_module(indexed_arrays[0]).concatenate(
            indexed_arrays, axis=fancy_index_axis)
        concat_pos = get_array_module(poses[0]).hstack(poses)
        select_pos = calc_pos(fancy_index_shape, concat_pos,
                              xp=get_array_module(poses[0]))
        select = (slice(None),) * fancy_index_axis + (select_pos,)
        ctx[op.outputs[0].key] = concat_array[select]

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls._execute_map(ctx, op)
        else:
            cls._execute_reduce(ctx, op)

    @classmethod
    def estimate_size(cls, ctx, op):
        if op.stage == OperandStage.map:
            input_size = ctx[op.inputs[0].key][0]
            pos_size = ctx[op.inputs[0].key][0]
            ctx[op.outputs[0].key] = input_size + pos_size, input_size + pos_size * 2
        else:
            chunk = op.outputs[0]
            input_sizes = [ctx[c.key][0] for c in op.inputs[0].inputs or ()]
            ctx[chunk.key] = chunk.nbytes, chunk.nbytes + sum(input_sizes)


def _is_bool_index(index_obj):
    return isinstance(index_obj, TENSOR_TYPE) and index_obj.dtype == np.bool_


def _is_fancy_index(index_obj):
    return isinstance(index_obj, FANCY_INDEX_TYPES) and index_obj.dtype != np.bool_


def _is_create_view(index):
    # is view if all of index is slice, int or newaxis
    return all(isinstance(ind, (slice, Integral)) or ind is None for ind in index)


def _calc_order(a, index):
    if a.order == TensorOrder.C_ORDER:
        return TensorOrder.C_ORDER

    in_axis = 0
    for ind in index:
        if _is_bool_index(ind):
            in_axis += ind.ndim
            return TensorOrder.C_ORDER
        elif _is_fancy_index(ind):
            in_axis += 1
            return TensorOrder.C_ORDER
        elif ind is None:
            continue
        elif isinstance(ind, slice):
            shape = a.shape[in_axis]
            slc = ind.indices(shape)
            if slc[0] == 0 and slc[1] == shape and slc[2] == 1:
                continue
            else:
                return TensorOrder.C_ORDER
        else:
            assert isinstance(ind, Integral)
            in_axis += 1
            return TensorOrder.C_ORDER

    return TensorOrder.F_ORDER


def _getitem_nocheck(a, item, convert_bool_to_fancy=None):
    index = process_index(a.ndim, item,
                          convert_bool_to_fancy=convert_bool_to_fancy)
    if convert_bool_to_fancy is False:
        # come from __setitem__, the bool index is not converted to fancy index
        # if multiple bool indexes or bool + fancy indexes exist,
        # thus the shape will be wrong,
        # here we just convert when calculating shape,
        # refer to issue #1282.
        shape = calc_shape(a.shape, process_index(a.ndim, index))
    else:
        shape = calc_shape(a.shape, index)
    tensor_order = _calc_order(a, index)
    op = TensorIndex(dtype=a.dtype, sparse=a.issparse(), indexes=index,
                     create_view=_is_create_view(index))
    return op(a, index, tuple(shape), order=tensor_order)


def _getitem(a, item):
    if isinstance(item, (list, tuple)) and \
            all(isinstance(it, slice) and it == slice(None) for it in item):
        # nothing to do
        return a

    # TODO(jisheng): field access, e.g. t['a'], t[['a', 'b']]
    return _getitem_nocheck(a, item)
