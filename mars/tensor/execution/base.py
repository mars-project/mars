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

from ...compat import izip
from ...compat.numpy_compat import broadcast_to
from .array import as_same_device, device


def _virtual(ctx, chunk):
    ctx[chunk.key] = None


def _copyto(ctx, chunk):
    inputs, device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)

    with device(device_id):
        dst = inputs[1].copy()
        src = inputs[0]
        where = inputs[2] if len(inputs) > 2 else None

        xp.copyto(dst, src, casting=chunk.op.casting, where=where)
        ctx[chunk.key] = dst


def _astype(ctx, chunk):
    (x,), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)

    with device(device_id):
        if chunk.op.sparse:
            ctx[chunk.key] = x.astype(chunk.op.dtype)
        else:
            ctx[chunk.key] = x.astype(chunk.op.dtype, casting=chunk.op.casting)


def _transpose(ctx, chunk):
    (x,), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)

    axes = chunk.op.axes
    with device(device_id):
        ctx[chunk.key] = xp.transpose(x, axes or None)


def _swapaxes(ctx, chunk):
    (x,), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)

    axis1, axis2 = chunk.op.axis1, chunk.op.axis2
    with device(device_id):
        ctx[chunk.key] = xp.swapaxes(x, axis1, axis2)


def _broadcast_to(ctx, chunk):
    (x,), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)

    if xp == np:
        func = broadcast_to
    else:
        func = xp.broadcast_to

    with device(device_id):
        ctx[chunk.key] = func(x, chunk.op.shape)


def _where(ctx, chunk):
    (cond, x, y), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)

    with device(device_id):
        ctx[chunk.key] = xp.where(cond, x, y)


def _split(ctx, chunk):
    inputs, device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)

    a = inputs[0]
    ind = inputs[1] if len(inputs) > 1 else chunk.op.indices_or_sections
    with device(device_id):
        ret = xp.array_split(a, ind)
        assert len(chunk.op.outputs) == len(ret)
        for o, r in izip(chunk.op.outputs, ret):
            ctx[o.key] = r


def _squeeze(ctx, chunk):
    (a,), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)

    with device(device_id):
        ctx[chunk.key] = xp.squeeze(a, axis=chunk.op.axis)


def _digitize(ctx, chunk):
    inputs, device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)

    x = inputs[0]
    if len(inputs) > 1:
        bins = inputs[1]
    else:
        bins = chunk.op.bins

    with device(device_id):
        ctx[chunk.key] = xp.digitize(x, bins=bins, right=chunk.op.right)


def _repeat(ctx, chunk):
    inputs, device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)

    a = inputs[0]
    if len(inputs) > 1:
        repeats = inputs[1]
    else:
        repeats = chunk.op.repeats

    with device(device_id):
        ctx[chunk.key] = xp.repeat(a, repeats=repeats, axis=chunk.op.axis)


def _isin(ctx, chunk):
    (element, test_elements), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)

    with device(device_id):
        ctx[chunk.key] = xp.isin(element, test_elements,
                                 assume_unique=chunk.op.assume_unique,
                                 invert=chunk.op.invert)


def register_basic_handler():
    from ... import operands
    from ..expressions import base
    from ...executor import register

    register(operands.VirtualOperand, _virtual, _virtual)

    register(base.TensorCopyTo, _copyto)
    register(base.TensorAstype, _astype)
    register(base.TensorTranspose, _transpose)
    register(base.TensorSwapAxes, _swapaxes)
    register(base.TensorBroadcastTo, _broadcast_to)
    register(base.TensorWhere, _where)
    register(base.TensorSplit, _split)
    register(base.TensorSqueeze, _squeeze)
    register(base.TensorDigitize, _digitize)
    register(base.TensorRepeat, _repeat)
    register(base.TensorIsIn, _isin)
