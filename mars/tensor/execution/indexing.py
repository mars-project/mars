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

from .array import as_same_device, device
from ...compat import izip


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
    for index in new_indexes:
        if isinstance(index, (Base, Entity)) and index.dtype == np.bool_:
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
    register(indexing.TensorIndexSetValue, _index_set_value)
    register(indexing.TensorChoose, _choose)
    register(indexing.TensorUnravelIndex, _unravel_index)
