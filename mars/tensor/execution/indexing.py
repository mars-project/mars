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

from .array import as_same_device, device
from ...compat import izip


def _slice(ctx, chunk):
    ctx[chunk.key] = ctx[chunk.inputs[0].key][chunk.op.slices]


def _index(ctx, chunk):
    indexes = tuple(ctx[index.key] if hasattr(index, 'key') else index
                    for index in chunk.op.indexes)
    input = ctx[chunk.inputs[0].key]
    ctx[chunk.key] = input[indexes]


def _index_set_value(ctx, chunk):
    indexes = [ctx[index.key] if hasattr(index, 'key') else index
               for index in chunk.op.indexes]
    input = ctx[chunk.inputs[0].key]
    value = ctx[chunk.op.value.key] if hasattr(chunk.op.value, 'key') else chunk.op.value
    if hasattr(input, 'flags') and not input.flags.writeable:
        input.setflags(write=True)
    input[tuple(indexes)] = value
    ctx[chunk.key] = input


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
    from ...operands import Slice, Index, IndexSetValue, Choose, UnravelIndex
    from .core import register

    register(Slice, _slice)
    register(Index, _index)
    register(IndexSetValue, _index_set_value)
    register(Choose, _choose)
    register(UnravelIndex, _unravel_index)
