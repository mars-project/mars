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

import numpy as np

from .array import as_same_device, device, get_array_module
from ..expressions.utils import split_indexes_into_chunks, calc_pos
from ...compat import izip
from ...utils import get_shuffle_input_keys_idxes


def _slice(ctx, chunk):
    ctx[chunk.key] = ctx[chunk.inputs[0].key][tuple(chunk.op.slices)]


def _index(ctx, chunk):
    indexes = tuple(ctx[index.key] if hasattr(index, 'key') else index
                    for index in chunk.op.indexes)
    input_ = ctx[chunk.inputs[0].key]
    ctx[chunk.key] = input_[indexes]


def _index_estimate_size(ctx, chunk):
    from mars.core import Base, Entity

    op = chunk.op
    shape = op.outputs[0].shape
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


def _fancy_index_distribute_map(ctx, chunk):
    nsplits = chunk.op.dest_nsplits
    axes = chunk.op.axes
    fancy_index_nsplits = [nsplits[ax] for ax in axes]
    indexes = ctx[chunk.inputs[0].key]
    flatten_indexes = indexes.reshape(indexes.shape[0], -1)
    idx_to_fancy_indexes, idx_to_poses = \
        split_indexes_into_chunks(fancy_index_nsplits, flatten_indexes, False)
    for idx in idx_to_fancy_indexes:
        group_key = ','.join(str(i) for i in idx)
        ctx[(chunk.key, group_key)] = (idx_to_fancy_indexes[idx], idx_to_poses[idx])


def _fancy_index_distribute_map_estimate_size(ctx, chunk):
    fancy_index_size = len(chunk.op.axes)
    inp_size = ctx[chunk.inputs[0].key][0]
    factor = 1 / float(fancy_index_size) + fancy_index_size  # 1/#fancy_index is the poses
    ctx[chunk.key] = (inp_size * factor,) * 2


def _fancy_index_distribute_reduce(ctx, chunk):
    in_chunk = chunk.inputs[0]
    input_keys, _ = get_shuffle_input_keys_idxes(in_chunk)

    fancy_indexes = []
    poses = []
    shuffle_key = chunk.op.shuffle_key
    xp = None
    for input_key in input_keys:
        key = (input_key, shuffle_key)
        fancy_index, pos = ctx[key]
        if xp is None:
            xp = get_array_module(fancy_index)
        if fancy_index.size == 0:
            fancy_index = fancy_index.reshape(len(chunk.op.axes), 0)
        fancy_indexes.append(fancy_index)
        poses.append(pos)

    fancy_index = np.hstack(fancy_indexes)
    pos = np.hstack(poses)

    assert len(chunk.op.outputs) - 1 == len(fancy_index)
    for out_chunk, axis_fancy_index in zip(chunk.op.outputs[:-1], fancy_index):
        ctx[out_chunk.key] = axis_fancy_index
    ctx[chunk.op.outputs[-1].key] = np.asarray([len(p) for p in poses]), pos


def _fancy_index_distribute_reduce_estimate_size(ctx, chunk):
    sum_size = 0
    for shuffle_input in chunk.inputs[0].inputs or ():
        sum_size += ctx[shuffle_input.key]
    for out_chunk in chunk.op.outputs:
        ctx[out_chunk.key] = sum_size, sum_size


def _fancy_index_concat_map(ctx, chunk):
    indexed_array = ctx[chunk.inputs[0].key]
    sizes, pos = ctx[chunk.inputs[1].key]
    acc_sizes = np.cumsum(sizes)
    fancy_index_axis = chunk.op.fancy_index_axis

    for i in range(len(sizes)):
        start = 0 if i == 0 else acc_sizes[i - 1]
        end = acc_sizes[i]
        select = (slice(None),) * fancy_index_axis + (slice(start, end),)
        ctx[(chunk.key, str(i))] = (indexed_array[select], pos[start: end])


def _fancy_index_concat_map_estimate_size(ctx, chunk):
    input_size = ctx[chunk.inputs[0].key][0]
    pos_size = ctx[chunk.inputs[0].key][0]
    ctx[chunk.key] = input_size + pos_size, input_size + pos_size * 2


def _fancy_index_concat_reduce(ctx, chunk):
    in_chunk = chunk.inputs[0]
    input_keys, _ = get_shuffle_input_keys_idxes(in_chunk)
    fancy_index_axis = chunk.op.fancy_index_axis
    fancy_index_shape = chunk.op.fancy_index_shape

    indexed_arrays = []
    poses = []
    shuffle_key = chunk.op.shuffle_key
    for input_key in input_keys:
        index_array, pos = ctx[(input_key, shuffle_key)]
        indexed_arrays.append(index_array)
        poses.append(pos)

    concat_array = np.concatenate(indexed_arrays, axis=fancy_index_axis)
    concat_pos = np.hstack(poses)
    select_pos = calc_pos(fancy_index_shape, concat_pos)
    select = (slice(None),) * fancy_index_axis + (select_pos,)
    ctx[chunk.key] = concat_array[select]


def _fancy_index_concat_reduce_estimate_size(ctx, chunk):
    input_sizes = [ctx[c.key][0] for c in chunk.inputs[0].inputs or ()]
    ctx[chunk.key] = chunk.nbytes, chunk.nbytes + sum(input_sizes)


def _index_set_value(ctx, chunk):
    indexes = [ctx[index.key] if hasattr(index, 'key') else index
               for index in chunk.op.indexes]
    input_ = ctx[chunk.inputs[0].key].copy()
    value = ctx[chunk.op.value.key] if hasattr(chunk.op.value, 'key') else chunk.op.value
    if hasattr(input_, 'flags') and not input_.flags.writeable:
        input_.setflags(write=True)
    input_[tuple(indexes)] = value
    ctx[chunk.key] = input_


def _choose(ctx, chunk):
    inputs, device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)
    a, choices = inputs[0], inputs[1:]

    with device(device_id):
        ctx[chunk.key] = xp.choose(a, choices, mode=chunk.op.mode)


def _unravel_index(ctx, chunk):
    inputs, device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)
    indices = inputs[0]

    with device(device_id):
        outputs = xp.unravel_index(indices, chunk.op.dims)
        for o, output in izip(chunk.op.outputs, outputs):
            ctx[o.key] = output


def register_indexing_handler():
    from ..expressions import indexing
    from ...executor import register

    register(indexing.TensorSlice, _slice)
    register(indexing.TensorIndex, _index, _index_estimate_size)
    register(indexing.FancyIndexingDistributeMap, _fancy_index_distribute_map,
             _fancy_index_distribute_map_estimate_size)
    register(indexing.FancyIndexingDistributeReduce, _fancy_index_distribute_reduce,
             _fancy_index_distribute_reduce_estimate_size)
    register(indexing.FancyIndexingConcatMap, _fancy_index_concat_map,
             _fancy_index_concat_map_estimate_size)
    register(indexing.FancyIndexingConcatReduce, _fancy_index_concat_reduce,
             _fancy_index_concat_reduce_estimate_size)
    register(indexing.TensorIndexSetValue, _index_set_value)
    register(indexing.TensorChoose, _choose)
    register(indexing.TensorUnravelIndex, _unravel_index)
