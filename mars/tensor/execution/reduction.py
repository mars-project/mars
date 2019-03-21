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

from collections import Container, Iterable, Sequence
from functools import wraps
from math import factorial
import operator

import numpy as np

from ...compat import getargspec, numpy_compat, reduce
from ...operands import Mean
from .array import get_array_module, as_same_device, device, cp


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

sum = keepdims_wrapper(np.sum)
prod = keepdims_wrapper(np.prod)
min = keepdims_wrapper(np.min)
max = keepdims_wrapper(np.max)
argmin = keepdims_wrapper(np.argmin)
nanargmin = keepdims_wrapper(np.nanargmin)
argmax = keepdims_wrapper(np.argmax)
nanargmax = keepdims_wrapper(np.nanargmax)
any = keepdims_wrapper(np.any)
all = keepdims_wrapper(np.all)
nansum = keepdims_wrapper(np.nansum)

try:
    from numpy import nanprod, nancumprod, nancumsum
except ImportError:  # pragma: no cover
    nanprod = numpy_compat.nanprod
    nancumprod = numpy_compat.nancumprod
    nancumsum = numpy_compat.nancumsum

nanprod = keepdims_wrapper(nanprod)
nancumprod = keepdims_wrapper(nancumprod)
nancumsum = keepdims_wrapper(nancumsum)

nanmin = keepdims_wrapper(np.nanmin)
nanmax = keepdims_wrapper(np.nanmax)
mean = keepdims_wrapper(np.mean)


def _handle_reduction(op):
    def _handle(ctx, chunk):
        (input_chunk,), device_id, _ = as_same_device(
            [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)
        axis = _get_axis(chunk.op.axis)
        with device(device_id):
            if "dtype" in getargspec(op).args:
                ctx[chunk.key] = op(input_chunk, axis=axis,
                                    dtype=chunk.op.dtype,
                                    keepdims=bool(chunk.op.keepdims))
            else:
                ctx[chunk.key] = op(input_chunk, axis=axis,
                                    keepdims=bool(chunk.op.keepdims))
    return _handle


def _mean_chunk(ctx, chunk):
    from ..expressions.reduction import TensorNanMeanChunk

    (in_chunk,), device_id, _ = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)

    axis = _get_axis(chunk.op.axis)
    _count_func = _nannumel if (type(chunk.op) is TensorNanMeanChunk) else _numel
    _sum_func = nansum if (type(chunk.op) is TensorNanMeanChunk) else sum

    with device(device_id):
        chunk_count = _count_func(in_chunk, axis=axis, dtype=np.int64,
                                  keepdims=bool(chunk.op.keepdims))
        chunk_sum = _sum_func(in_chunk, axis=axis, dtype=chunk.op.dtype,
                              keepdims=bool(chunk.op.keepdims))
        ctx[chunk.key] = (chunk_sum, chunk_count)


def _mean_combine(ctx, chunk):
    axis = _get_axis(chunk.op.axis)
    (_data, _count), device_id, _ = as_same_device(
        ctx[chunk.inputs[0].key], device=chunk.op.device, ret_extra=True)

    with device(device_id):
        chunk_count = sum(_count, axis=axis, dtype=np.int64,
                          keepdims=bool(chunk.op.keepdims))
        chunk_sum = sum(_data, axis=axis, dtype=chunk.op.dtype,
                        keepdims=bool(chunk.op.keepdims))
        ctx[chunk.key] = (chunk_sum, chunk_count)


def _mean(ctx, chunk):
    axis = _get_axis(chunk.op.axis)

    a = ctx[chunk.inputs[0].key]
    if not isinstance(a, (list, tuple)):
        (inp,), device_id, xp = as_same_device(
            [a], device=chunk.op.device, ret_extra=True)

        with device(device_id):
            func = xp.mean if isinstance(chunk.op, Mean) else xp.nanmean
            ctx[chunk.key] = func(inp, axis=axis, dtype=chunk.op.dtype,
                                  keepdims=bool(chunk.op.keepdims))
    else:
        (_data, _count), device_id, xp = as_same_device(
            a, device=chunk.op.device, ret_extra=True)

        with device(device_id):
            chunk_count = sum(_count, axis=axis, dtype=chunk.op.dtype,
                              keepdims=bool(chunk.op.keepdims))
            chunk_sum = sum(_data, axis=axis, dtype=chunk.op.dtype,
                            keepdims=bool(chunk.op.keepdims))
            ctx[chunk.key] = xp.true_divide(chunk_sum, chunk_count,
                                            dtype=chunk.op.dtype)


def _handle_arg_chunk(op, arg_op):
    def _handle(ctx, chunk):
        arg_axis = _get_arg_axis(chunk.op.axis, chunk.inputs[0].ndim)
        (in_chunk,), device_id, xp = as_same_device(
            [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)
        offset = chunk.op.offset

        with device(device_id):
            vals = op(in_chunk, axis=arg_axis, keepdims=True)
            arg = arg_op(in_chunk, axis=arg_axis, keepdims=True)

            if arg_axis is None:
                if xp == cp:
                    # we need to copy to do cpu computation, then copy back to gpu
                    # cuz unravel_index and ravel_multi_index are not implemented in cupy
                    in_chunk = in_chunk.get()

                total_shape = chunk.op.total_shape
                ind = np.unravel_index(arg.ravel()[0], in_chunk.shape)
                total_ind = tuple(o + i for (o, i) in zip(offset, ind))
                res = np.ravel_multi_index(total_ind, total_shape)

                if xp == cp:
                    # copy back
                    with xp.cuda.Device(in_chunk.device.id):
                        arg[:] = xp.asarray(res)
                else:
                    arg[:] = res
            else:
                arg += offset
            ctx[chunk.key] = (vals, arg)
    return _handle


def _handle_arg_combine(arg_op):
    def _handle(ctx, chunk):
        axis = _get_arg_axis(chunk.op.axis, chunk.inputs[0].ndim)
        (vals, arg), device_id, xp = as_same_device(
            ctx[chunk.inputs[0].key], device=chunk.op.device, ret_extra=True)
        keepdims = bool(chunk.op.keepdims)

        with device(device_id):
            if axis is None:
                local_args = arg_op(vals, axis=axis, keepdims=keepdims)
                vals = vals.ravel()[local_args]
                arg = arg.ravel()[local_args]
            else:
                local_args = arg_op(vals, axis=axis)
                inds = np.ogrid[tuple(map(slice, local_args.shape))]
                if xp != np:
                    inds = [xp.asarray(it) for it in inds]
                inds.insert(axis, local_args)
                inds_tuple = tuple(inds)
                vals = vals[inds_tuple]
                arg = arg[inds_tuple]
                if keepdims:
                    vals = xp.expand_dims(vals, axis)
                    arg = xp.expand_dims(arg, axis)
            ctx[chunk.key] = (vals, arg)
    return _handle


def _handle_arg(arg_op):
    def _handle(ctx, chunk):
        axis = _get_arg_axis(chunk.op.axis, chunk.inputs[0].ndim)
        (vals, arg), device_id, xp = as_same_device(
            ctx[chunk.inputs[0].key], device=chunk.op.device, ret_extra=True)

        with device(device_id):
            if xp.any(xp.isnan(vals)) and arg_op in [_nanargmin, _nanargmax]:
                raise ValueError("All NaN slice encountered")
            keepdims = bool(chunk.op.keepdims)
            if axis is None:
                local_args = arg_op(vals, axis=axis, keepdims=keepdims)
                arg = arg.ravel()[local_args]
            else:
                local_args = arg_op(vals, axis=axis)
                inds = np.ogrid[tuple(map(slice, local_args.shape))]
                if xp != np:
                    inds = [xp.asarray(it) for it in inds]
                inds.insert(axis, local_args)
                arg = arg[tuple(inds)]
                if keepdims:
                    arg = xp.expand_dims(arg, axis)
            ctx[chunk.key] = arg
    return _handle


def _handle_cum(cum_op):
    def _handle(ctx, chunk):
        (x,), device_id, xp = as_same_device(
            [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)

        if xp != np:
            func = getattr(xp, cum_op.__name__)
        else:
            func = cum_op

        with device(device_id):
            ctx[chunk.key] = func(x, axis=chunk.op.axis,
                                  dtype=chunk.op.dtype)
    return _handle


def _get_axis(axis):
    return tuple(axis) if axis is not None else axis


def _get_arg_axis(axis, ndim):
    return None if len(axis) == ndim or ndim == 1 else axis[0]


def _numel(x, **kwargs):
    xp = get_array_module(x)
    return sum(xp.ones_like(x), **kwargs)


def _nannumel(x, **kwargs):
    x_size = reduce(operator.mul, x.shape)
    xp = get_array_module(x)
    return x_size - sum(xp.isnan(x), **kwargs)


def _nanargmin(x, **kwargs):
    try:
        return nanargmin(x, **kwargs)
    except ValueError:
        xp = get_array_module(x)
        return nanargmin(xp.where(xp.isnan(x), np.inf, x), **kwargs)


def _nanargmax(x, axis, **kwargs):
    try:
        return nanargmax(x, axis, **kwargs)
    except ValueError:
        xp = get_array_module(x)
        return nanargmax(xp.where(xp.isnan(x), -np.inf, x), **kwargs)


def _count_nonzero(ctx, chunk):
    (x,), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], chunk.op.device, ret_extra=True)

    axis = _get_arg_axis(chunk.op.axis, chunk.inputs[0].ndim)
    keepdims = chunk.op.keepdims
    with device(device_id):
        nz = xp.count_nonzero(x, axis=axis)
        if keepdims:
            slcs = [slice(None)] * chunk.inputs[0].ndim
            for ax in chunk.op.axis:
                slcs[ax] = np.newaxis
            nz = xp.asarray(nz)[tuple(slcs)]

        ctx[chunk.key] = nz


def _moment_chunk(ctx, chunk):
    from ..expressions.reduction import TensorNanMomentChunk

    (in_chunk,), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)

    axis = _get_axis(chunk.op.axis)
    moment = chunk.op.moment
    dtype = chunk.op.dtype
    _count_func = _nannumel if type(chunk.op) is TensorNanMomentChunk else _numel
    _sum_func = nansum if type(chunk.op) is TensorNanMomentChunk else sum
    empty = get_array_module(in_chunk, nosparse=True).empty

    with device(device_id):
        chunk_count = _count_func(in_chunk, axis=axis, dtype=np.int64,
                                  keepdims=bool(chunk.op.keepdims))
        chunk_sum = _sum_func(in_chunk, axis=axis, dtype=dtype,
                              keepdims=bool(chunk.op.keepdims))
        avg = xp.true_divide(chunk_sum, chunk_count)
        var_square = empty(chunk_count.shape + (moment - 1,), dtype=dtype)
        for i in range(2, moment + 1):
            var_square[..., i - 2] = _sum_func((in_chunk - avg) ** i, axis=axis, dtype=dtype,
                                               keepdims=bool(chunk.op.keepdims))
        ctx[chunk.key] = (chunk_sum, chunk_count, var_square)


def _reduce_var_square(var_square, avg_diff, count, chunk, axis, sum_func):
    moment = chunk.op.moment
    dtype = chunk.op.dtype
    kw = dict(axis=axis, dtype=dtype, keepdims=bool(chunk.op.keepdims))

    reduced_var_square = var_square[..., moment - 2].sum(**kw) + \
                         sum_func(count * avg_diff ** moment, **kw)
    for i in range(1, moment - 1):
        coeff = factorial(moment) / float(factorial(i) * factorial(moment - i))
        reduced_var_square += coeff * sum_func(var_square[..., moment - i - 2] * avg_diff ** moment, **kw)
    return reduced_var_square


def _moment_combine(ctx, chunk):
    from ..expressions.reduction import TensorNanMomentCombine

    axis = _get_axis(chunk.op.axis)
    moment = chunk.op.moment
    dtype = chunk.op.dtype
    _sum_func = nansum if type(chunk.op) is TensorNanMomentCombine else sum

    (_data, _count, _var_square), device_id, xp = as_same_device(
        ctx[chunk.inputs[0].key], device=chunk.op.device, ret_extra=True)
    empty = get_array_module(_data, nosparse=True).empty

    with device(device_id):
        chunk_count = _sum_func(_count, axis=axis, dtype=np.int64,
                                keepdims=bool(chunk.op.keepdims))
        chunk_sum = _sum_func(_data, axis=axis, dtype=dtype,
                              keepdims=bool(chunk.op.keepdims))
        avg = xp.true_divide(chunk_sum, chunk_count, dtype=dtype)
        avg_diff = xp.true_divide(_data, _count, dtype=dtype) - avg
        var_square = empty(chunk_count.shape + (moment - 1,), dtype=dtype)

        for m in range(2, moment + 1):
            var_square[..., m - 2] = \
                _reduce_var_square(_var_square, avg_diff, _count, chunk, axis, _sum_func)

        ctx[chunk.key] = (chunk_sum, chunk_count, var_square)


def _moment(ctx, chunk):
    from ..expressions.reduction import TensorNanMoment

    axis = _get_axis(chunk.op.axis)
    dtype = chunk.op.dtype
    _sum_func = nansum if type(chunk.op) is TensorNanMoment else sum

    (_data, _count, _var_square), device_id, xp = as_same_device(
        ctx[chunk.inputs[0].key], device=chunk.op.device, ret_extra=True)

    with device(device_id):
        chunk_count = _sum_func(_count, axis=axis, dtype=np.int64,
                                keepdims=True)
        chunk_sum = _sum_func(_data, axis=axis, dtype=dtype, keepdims=True)
        avg = xp.true_divide(chunk_sum, chunk_count, dtype=dtype)
        avg_diff = xp.true_divide(_data, _count, dtype=dtype) - avg
        var_square = _reduce_var_square(_var_square, avg_diff, _count, chunk, axis, _sum_func)

        ctx[chunk.key] = xp.true_divide(
            var_square,
            sum(chunk_count, axis=axis, dtype=dtype, keepdims=bool(chunk.op.keepdims)) - chunk.op.ddof,
            dtype=dtype)


def _var(ctx, chunk):
    from ..expressions.reduction import TensorNanVar

    axis = _get_axis(chunk.op.axis)
    (in_chunk,), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.op.device, ret_extra=True)

    with device(device_id):
        func = xp.nanvar if type(chunk.op) is TensorNanVar else xp.var
        ctx[chunk.key] = func(in_chunk, axis=axis, dtype=chunk.op.dtype, ddof=chunk.op.ddof,
                              keepdims=bool(chunk.op.keepdims))


def register_reduction_handler():
    from .core import register
    from ...operands import Sum, Prod, Min, Max, All, Any, Mean, MeanChunk, MeanCombine, \
        Argmax, ArgmaxChunk, ArgmaxCombine, Argmin, ArgminChunk, ArgminCombine, NanSum,\
        NanProd, NanMax, NanMin, NanMeanChunk, NanMean, NanArgmax, NanArgmaxChunk, NanArgmin,\
        NanArgminChunk, NanArgmaxCombine, NanArgminCombine, Moment, NanMomentChunk, MomentChunk, \
        MomentCombine, NanMomentCombine, NanMoment, Var, NanVar, CountNonzero, \
        Cumsum, Cumprod, NanCumsum, NanCumprod

    register(Sum, _handle_reduction(sum))
    register(NanSum, _handle_reduction(nansum))
    register(Prod, _handle_reduction(prod))
    register(NanProd, _handle_reduction(nanprod))
    register(Max, _handle_reduction(max))
    register(NanMax, _handle_reduction(nanmax))
    register(Min, _handle_reduction(min))
    register(NanMin, _handle_reduction(nanmin))
    register(All, _handle_reduction(all))
    register(Any, _handle_reduction(any))

    register(Mean, _mean)
    register(NanMean, _mean)
    register(Var, _var)
    register(NanVar, _var)
    register(MeanChunk, _mean_chunk)
    register(NanMeanChunk, _mean_chunk)
    register(MeanCombine, _mean_combine)

    register(Moment, _moment)
    register(NanMoment, _moment)
    register(MomentChunk, _moment_chunk)
    register(NanMomentChunk, _moment_chunk)
    register(MomentCombine, _moment_combine)
    register(NanMomentCombine, _moment_combine)

    register(CountNonzero, _count_nonzero)

    register(Argmax, _handle_arg(argmax))
    register(NanArgmax, _handle_arg(_nanargmax))
    register(ArgmaxChunk, _handle_arg_chunk(max, argmax))
    register(NanArgmaxChunk, _handle_arg_chunk(nanmax, _nanargmax))
    register(ArgmaxCombine, _handle_arg_combine(argmax))
    register(NanArgmaxCombine, _handle_arg_combine(_nanargmax))
    register(Argmin, _handle_arg(argmin))
    register(NanArgmin, _handle_arg(_nanargmin))
    register(ArgminChunk, _handle_arg_chunk(min, argmin))
    register(NanArgminChunk, _handle_arg_chunk(nanmin, _nanargmin))
    register(ArgminCombine, _handle_arg_combine(argmin))
    register(NanArgminCombine, _handle_arg_combine(_nanargmin))

    register(Cumsum, _handle_cum(np.cumsum))
    register(Cumprod, _handle_cum(np.cumprod))
    register(NanCumsum, _handle_cum(nancumsum))
    register(NanCumprod, _handle_cum(nancumprod))
