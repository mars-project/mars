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

import logging
import numpy as np

from .array import as_same_device, device
from ...utils import get_shuffle_input_keys_idxes

logger = logging.getLogger(__name__)


def _reshape(ctx, chunk):
    (x,), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)

    with device(device_id):
        ctx[chunk.key] = x.reshape(chunk.op.newshape)


def _reshape_map(ctx, chunk):
    # todo this function is an experimental one making shuffle runnable.
    # try elevate performance when needed.
    old_shape = chunk.op.oldshape
    new_shape = chunk.op.newshape
    new_chunk_size = chunk.op.new_chunk_size
    axis_offset = chunk.op.axis_offsets

    logger.debug('Reshape mapper: Start mapping step for %s', chunk.key)

    data = ctx[chunk.inputs[0].key]
    indices = list(np.nonzero(data))
    nz_data = data[indices]

    for idx in range(len(old_shape)):
        indices[idx] = np.add(indices[idx], axis_offset[idx], out=indices[idx])
    rest_indices = indices[0]
    indices[0] = None
    for idx in range(1, len(old_shape)):
        rest_indices = np.multiply(rest_indices, old_shape[idx], out=rest_indices)
        rest_indices = np.add(rest_indices, indices[idx], out=rest_indices)
        indices[idx] = None
    del indices

    new_indices = []
    for dim_size in reversed(new_shape[1:]):
        new_index = rest_indices % dim_size
        new_indices.append(new_index)
        rest_indices = np.floor_divide(rest_indices, dim_size, out=rest_indices)
    new_indices.append(rest_indices)
    new_indices.reverse()
    del rest_indices

    logger.debug('Reshape mapper: remapping to new locations for %s', chunk.key)

    dim_chunk_counts = [int(np.ceil(dim_size * 1.0 / chunk_size))
                        for dim_size, chunk_size in zip(new_shape, new_chunk_size)]
    target = new_indices[0] // new_chunk_size[0]
    for new_index, chunk_size, dim_chunk_count in zip(new_indices[1:], new_chunk_size[1:], dim_chunk_counts[1:]):
        target = np.multiply(target, dim_chunk_count, out=target)
        target = np.add(target, new_index // chunk_size, out=target)

    for idx, chunk_size in enumerate(new_chunk_size):
        new_indices[idx] = np.mod(new_indices[idx], chunk_size, out=new_indices[idx])

    logger.debug('Reshape mapper: sorting for %s', chunk.key)

    sort_idx = np.argsort(target)
    target = target[sort_idx]
    nz_data = nz_data[sort_idx]
    for idx in range(len(new_indices)):
        new_indices[idx] = new_indices[idx][sort_idx]
    del sort_idx

    logger.debug('Reshape mapper: splitting for %s', chunk.key)

    for t in np.unique(target):
        data_slice = slice(np.searchsorted(target, t), np.searchsorted(target, t, 'right'))
        group_indices = tuple(new_indices[idx][data_slice] for idx in range(len(new_shape)))
        group_data = nz_data[data_slice]

        target_chunk_idx = [None] * len(dim_chunk_counts)
        for idx, dim_chunk_count in enumerate(reversed(dim_chunk_counts)):
            t, target_chunk_idx[idx] = divmod(t, dim_chunk_count)
        target_chunk_idx.reverse()
        group_key = ','.join(str(v) for v in target_chunk_idx)

        ctx[(chunk.key, group_key)] = group_indices + (group_data,)


def _reshape_map_estimate_size(ctx, chunk):
    inp_chunk = chunk.inputs[0]
    inp_size, inp_calc = ctx[inp_chunk.key]
    store_overhead = np.int64().itemsize * inp_chunk.ndim
    calc_overhead = np.int64().itemsize * (inp_chunk.ndim + 2)
    ctx[chunk.key] = (store_overhead + inp_size, calc_overhead + inp_calc)


def _reshape_reduce(ctx, chunk):
    try:
        result_array = ctx[chunk.key]
    except KeyError:
        result_array = np.zeros(chunk.shape, dtype=chunk.dtype)

    in_chunk = chunk.inputs[0]
    input_keys, _ = get_shuffle_input_keys_idxes(in_chunk)

    shuffle_key = chunk.op.shuffle_key
    for input_key in input_keys:
        key = (input_key, shuffle_key)
        if ctx.get(key) is not None:
            data_tuple = ctx[key]
            result_array[data_tuple[:-1]] = data_tuple[-1]
        else:
            ctx[key] = None
    ctx[chunk.key] = result_array


def _reshape_reduce_estimate_size(ctx, chunk):
    sum_size = 0
    for shuffle_input in chunk.inputs[0].inputs or ():
        key = (shuffle_input.key, chunk.op.shuffle_key)
        if ctx.get(key) is not None:
            sum_size += ctx[key][0]
        else:
            ctx[key] = None
    ctx[chunk.key] = (chunk.nbytes, max(sum_size, chunk.nbytes))


def register_reshape_handler():
    from ...executor import register
    from ..expressions.reshape.reshape import TensorReshape, TensorReshapeMap, TensorReshapeReduce

    register(TensorReshape, _reshape)
    register(TensorReshapeMap, _reshape_map, _reshape_map_estimate_size)
    register(TensorReshapeReduce, _reshape_reduce, _reshape_reduce_estimate_size)
