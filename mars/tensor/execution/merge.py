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

import itertools
import operator
from collections import Iterable

from ...compat import six, lrange, lmap
from ..expressions.indexing import TensorSlice
from .array import as_same_device, device


def _get_index(chunk):
    try:
        return chunk.index
    except AttributeError:
        if isinstance(chunk.op, TensorSlice):
            return chunk.inputs[0].index
        raise


def _norm_axis(axis):
    if isinstance(axis, six.integer_types):
        return axis, True
    if isinstance(axis, Iterable):
        axis = sorted(tuple(axis))
        if len(axis) == 1:
            return axis[0], True
        return axis, False

    assert axis is None
    return None, False


def _concatenate(ctx, chunk):
    inputs = [ctx[input.key] for input in chunk.inputs]

    if isinstance(inputs[0], tuple):
        ctx[chunk.key] = tuple(_base_concatenate(chunk, [input[i] for input in inputs])
                               for i in range(len(inputs[0])))
    else:
        ctx[chunk.key] = _base_concatenate(chunk, inputs)


def _base_concatenate(chunk, inputs):
    inputs, device_id, xp = as_same_device(inputs, device=chunk.op.device, ret_extra=True)

    axis, single_axis = _norm_axis(chunk.op.axis)
    if single_axis:
        with device(device_id):
            res = xp.concatenate(tuple(inputs), axis=axis)
    else:
        axes = axis or lrange(chunk.ndim)
        chunks = [(_get_index(input), data) for input, data in zip(chunk.inputs, inputs)]
        with device(device_id):
            for i in range(len(axes) - 1):
                new_chunks = []
                for idx, cs in itertools.groupby(chunks, key=lambda t: t[0][:-1]):
                    cs = lmap(operator.itemgetter(1), cs)
                    new_chunks.append((idx, xp.concatenate(cs, axis=len(axes)-i-1)))
                chunks = new_chunks
            res = xp.concatenate(lmap(operator.itemgetter(1), chunks), axis=axes[0])
    return res


def _stack(ctx, chunk):
    inputs, device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    axis = chunk.op.axis
    with device(device_id):
        ctx[chunk.key] = xp.stack(inputs, axis=axis)


def register_merge_handler():
    from ..expressions.merge import TensorConcatenate, TensorStack
    from ...executor import register

    register(TensorConcatenate, _concatenate)
    register(TensorStack, _stack)
