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

import operator
from math import factorial
from functools import wraps
try:
    from collections.abc import Container, Iterable, Sequence
except ImportError:  # pragma: no cover
    from collections import Container, Iterable, Sequence

from ...compat import reduce
from ..array_utils import get_array_module, as_same_device, device


def get_axis(axis):
    return tuple(axis) if axis is not None else axis


def get_arg_axis(axis, ndim):
    return None if len(axis) == ndim or ndim == 1 else axis[0]


def keepdims_wrapper(a_callable):
    @wraps(a_callable)
    def keepdims_wrapped_callable(x, axis=None, keepdims=None, *args, **kwargs):
        xp = get_array_module(x)
        if xp == np:
            func = a_callable
        else:
            func = getattr(xp, a_callable.__name__)

        r = func(x, axis=axis, *args, **kwargs)

        if not keepdims:
            return xp.asarray(r)

        axes = axis

        if axes is None:
            axes = range(x.ndim)

        if not isinstance(axes, (Container, Iterable, Sequence)):
            axes = [axes]

        if r.ndim != x.ndim:
            r_slice = tuple()
            for each_axis in range(x.ndim):
                if each_axis in axes:
                    r_slice += (None,)
                else:
                    r_slice += (slice(None),)

            r = r[r_slice]

        return r

    return keepdims_wrapped_callable


sum_ = keepdims_wrapper(np.sum)
nansum_ = keepdims_wrapper(np.nansum)


def numel(x, **kwargs):
    xp = get_array_module(x)
    return sum_(xp.ones_like(x), **kwargs)


def nannumel(x, **kwargs):
    x_size = reduce(operator.mul, x.shape)
    xp = get_array_module(x)
    return x_size - sum_(xp.isnan(x), **kwargs)


def mean_chunk_execute(ctx, op, count_func, sum_func):
    (in_chunk,), device_id, _ = as_same_device(
        [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

    axis = get_axis(op.axis)

    with device(device_id):
        chunk_count = count_func(in_chunk, axis=axis, dtype=np.int64,
                                 keepdims=bool(op.keepdims))
        chunk_sum = sum_func(in_chunk, axis=axis, dtype=op.dtype,
                             keepdims=bool(op.keepdims))
        ctx[op.outputs[0].key] = (chunk_sum, chunk_count)


def mean_combine_execute(ctx, op):
    axis = get_axis(op.axis)
    (_data, _count), device_id, _ = as_same_device(
        ctx[op.inputs[0].key], device=op.device, ret_extra=True)

    with device(device_id):
        chunk_count = sum_(_count, axis=axis, dtype=np.int64,
                           keepdims=bool(op.keepdims))
        chunk_sum = sum_(_data, axis=axis, dtype=op.dtype,
                         keepdims=bool(op.keepdims))
        ctx[op.outputs[0].key] = (chunk_sum, chunk_count)


def mean_execute(ctx, op, mean_func):
    axis = get_axis(op.axis)

    a = ctx[op.inputs[0].key]
    if not isinstance(a, (list, tuple)):
        (inp,), device_id, xp = as_same_device(
            [a], device=op.device, ret_extra=True)

        with device(device_id):
            ctx[op.outputs[0].key] = mean_func(inp, axis=axis, dtype=op.dtype,
                                               keepdims=bool(op.keepdims))
    else:
        (_data, _count), device_id, xp = as_same_device(
            a, device=op.device, ret_extra=True)

        with device(device_id):
            chunk_count = sum_(_count, axis=axis, dtype=op.dtype,
                               keepdims=bool(op.keepdims))
            chunk_sum = sum_(_data, axis=axis, dtype=op.dtype,
                             keepdims=bool(op.keepdims))
            ctx[op.outputs[0].key] = xp.true_divide(chunk_sum, chunk_count,
                                                    dtype=op.dtype)


def moment_chunk_execute(ctx, op, count_func, sum_func):
    (in_chunk,), device_id, xp = as_same_device(
        [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

    axis = get_axis(op.axis)
    moment = op.moment
    dtype = op.dtype
    empty = get_array_module(in_chunk, nosparse=True).empty

    with device(device_id):
        chunk_count = count_func(in_chunk, axis=axis, dtype=np.int64,
                                 keepdims=bool(op.keepdims))
        chunk_sum = sum_func(in_chunk, axis=axis, dtype=dtype,
                             keepdims=bool(op.keepdims))
        avg = xp.true_divide(chunk_sum, chunk_count)
        var_square = empty(chunk_count.shape + (moment - 1,), dtype=dtype)
        for i in range(2, moment + 1):
            var_square[..., i - 2] = sum_func((in_chunk - avg) ** i, axis=axis, dtype=dtype,
                                              keepdims=bool(op.keepdims))
        ctx[op.outputs[0].key] = (chunk_sum, chunk_count, var_square)


def _reduce_var_square(var_square, avg_diff, count, op, axis, sum_func):
    moment = op.moment
    dtype = op.dtype
    kw = dict(axis=axis, dtype=dtype, keepdims=bool(op.keepdims))

    reduced_var_square = var_square[..., moment - 2].sum(**kw) + \
                         sum_func(count * avg_diff ** moment, **kw)
    for i in range(1, moment - 1):
        coeff = factorial(moment) / float(factorial(i) * factorial(moment - i))
        reduced_var_square += coeff * sum_func(var_square[..., moment - i - 2] * avg_diff ** moment, **kw)
    return reduced_var_square


def moment_combine_execcute(ctx, op, sum_func):
    axis = get_axis(op.axis)
    moment = op.moment
    dtype = op.dtype

    (_data, _count, _var_square), device_id, xp = as_same_device(
        ctx[op.inputs[0].key], device=op.device, ret_extra=True)
    empty = get_array_module(_data, nosparse=True).empty

    with device(device_id):
        chunk_count = sum_func(_count, axis=axis, dtype=np.int64,
                               keepdims=bool(op.keepdims))
        chunk_sum = sum_func(_data, axis=axis, dtype=dtype,
                             keepdims=bool(op.keepdims))
        avg = xp.true_divide(chunk_sum, chunk_count, dtype=dtype)
        avg_diff = xp.true_divide(_data, _count, dtype=dtype) - avg
        var_square = empty(chunk_count.shape + (moment - 1,), dtype=dtype)

        for m in range(2, moment + 1):
            var_square[..., m - 2] = \
                _reduce_var_square(_var_square, avg_diff, _count, op, axis, sum_func)

        ctx[op.outputs[0].key] = (chunk_sum, chunk_count, var_square)


def moment_execute(ctx, op, sum_func):
    axis = get_axis(op.axis)
    dtype = op.dtype

    (_data, _count, _var_square), device_id, xp = as_same_device(
        ctx[op.inputs[0].key], device=op.device, ret_extra=True)

    with device(device_id):
        chunk_count = sum_func(_count, axis=axis, dtype=np.int64,
                               keepdims=True)
        chunk_sum = sum_func(_data, axis=axis, dtype=dtype, keepdims=True)
        avg = xp.true_divide(chunk_sum, chunk_count, dtype=dtype)
        avg_diff = xp.true_divide(_data, _count, dtype=dtype) - avg
        var_square = _reduce_var_square(_var_square, avg_diff, _count, op, axis, sum_func)

        ctx[op.outputs[0].key] = xp.true_divide(
            var_square,
            sum_(chunk_count, axis=axis, dtype=dtype, keepdims=bool(op.keepdims)) - op.ddof,
            dtype=dtype)


def var_execute(ctx, op):
    from .var import TensorVar

    axis = get_axis(op.axis)
    (in_chunk,), device_id, xp = as_same_device(
        [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

    with device(device_id):
        func = xp.var if type(op) is TensorVar else xp.nanvar
        ctx[op.outputs[0].key] = func(in_chunk, axis=axis, dtype=op.dtype, ddof=op.ddof,
                                      keepdims=bool(op.keepdims))

