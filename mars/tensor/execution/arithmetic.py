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

import operator

import numpy as np

from .array import as_same_device, device, is_sparse_module, get_array_module
from ..expressions import arithmetic
from ...compat import reduce, six

OP_TO_HANDLER = {
    arithmetic.TensorAdd: 'add',
    arithmetic.TensorSubtract: 'subtract',
    arithmetic.TensorMultiply: 'multiply',
    arithmetic.TensorDivide: 'divide',
    arithmetic.TensorTrueDiv: 'true_divide',
    arithmetic.TensorFloorDiv: 'floor_divide',
    arithmetic.TensorPower: 'power',
    arithmetic.TensorFloatPower: 'float_power',
    arithmetic.TensorMod: 'mod',
    arithmetic.TensorFMod: 'fmod',
    arithmetic.TensorLogAddExp: 'logaddexp',
    arithmetic.TensorLogAddExp2: 'logaddexp2',
    arithmetic.TensorNegative: 'negative',
    arithmetic.TensorPositive: operator.pos,
    arithmetic.TensorAbsolute: 'absolute',
    arithmetic.TensorAbs: 'abs',
    arithmetic.TensorFabs: 'fabs',
    arithmetic.TensorRint: 'rint',
    arithmetic.TensorSign: 'sign',
    arithmetic.TensorConj: 'conj',
    arithmetic.TensorExp: 'exp',
    arithmetic.TensorExp2: 'exp2',
    arithmetic.TensorLog: 'log',
    arithmetic.TensorLog2: 'log2',
    arithmetic.TensorLog10: 'log10',
    arithmetic.TensorExpm1: 'expm1',
    arithmetic.TensorLog1p: 'log1p',
    arithmetic.TensorSqrt: 'sqrt',
    arithmetic.TensorSquare: 'square',
    arithmetic.TensorCbrt: 'cbrt',
    arithmetic.TensorReciprocal: 'reciprocal',
    arithmetic.TensorAround: 'around',
    arithmetic.TensorIsFinite: 'isfinite',
    arithmetic.TensorIsInf: 'isinf',
    arithmetic.TensorIsNan: 'isnan',
    arithmetic.TensorSignbit: 'signbit',
    arithmetic.TensorCopysign: 'copysign',
    arithmetic.TensorNextafter: 'nextafter',
    arithmetic.TensorSpacing: 'spacing',
    arithmetic.TensorLdexp: 'ldexp',
    arithmetic.TensorFloor: 'floor',
    arithmetic.TensorCeil: 'ceil',
    arithmetic.TensorTrunc: 'trunc',
    arithmetic.TensorDegrees: 'degrees',
    arithmetic.TensorRadians: 'radians',

    arithmetic.TensorEqual: 'equal',
    arithmetic.TensorNotEqual: 'not_equal',
    arithmetic.TensorLessThan: 'less',
    arithmetic.TensorLessEqual: 'less_equal',
    arithmetic.TensorGreaterThan: 'greater',
    arithmetic.TensorGreaterEqual: 'greater_equal',

    arithmetic.TensorSin: 'sin',
    arithmetic.TensorCos: 'cos',
    arithmetic.TensorTan: 'tan',
    arithmetic.TensorArcsin: 'arcsin',
    arithmetic.TensorArccos: 'arccos',
    arithmetic.TensorArctan: 'arctan',
    arithmetic.TensorArctan2: 'arctan2',
    arithmetic.TensorHypot: 'hypot',
    arithmetic.TensorSinh: 'sinh',
    arithmetic.TensorCosh: 'cosh',
    arithmetic.TensorTanh: 'tanh',
    arithmetic.TensorArcsinh: 'arcsinh',
    arithmetic.TensorArccosh: 'arccosh',
    arithmetic.TensorArctanh: 'arctanh',
    arithmetic.TensorDeg2rad: 'deg2rad',
    arithmetic.TensorRad2deg: 'rad2deg',
    arithmetic.TensorAngle: 'angle',

    arithmetic.TensorBitand: 'bitwise_and',
    arithmetic.TensorBitor: 'bitwise_or',
    arithmetic.TensorBitxor: 'bitwise_xor',
    arithmetic.TensorInvert: 'invert',

    arithmetic.TensorLshift: 'left_shift',
    arithmetic.TensorRshift: 'right_shift',

    arithmetic.TensorAnd: 'logical_and',
    arithmetic.TensorOr: 'logical_or',
    arithmetic.TensorXor: 'logical_xor',
    arithmetic.TensorNot: 'logical_not',

    arithmetic.TensorMaximum: 'maximum',
    arithmetic.TensorMinimum: 'minimum',
    arithmetic.TensorFMax: 'fmax',
    arithmetic.TensorFMin: 'fmin',

    arithmetic.TensorIsclose: 'isclose',

    arithmetic.TensorClip: 'clip',
    arithmetic.TensorIsReal: 'isreal',
    arithmetic.TensorIsComplex: 'iscomplex',
    arithmetic.TensorReal: 'real',
    arithmetic.TensorImag: 'imag',
    arithmetic.TensorFix: 'fix',
    arithmetic.TensorI0: 'i0',
    arithmetic.TensorSinc: 'sinc',
    arithmetic.TensorNanToNum: 'nan_to_num',
}


def _handle_out_dtype(val, dtype):
    if val.dtype != dtype:
        return val.astype(dtype)
    return val


def _build_elementwise(op):
    def _handle(ctx, chunk):
        inputs, device_id, xp = as_same_device(
            [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

        if isinstance(op, six.string_types):
            func = getattr(xp, op)
        else:
            func = op

        with device(device_id):
            kw = {'casting': chunk.op.casting} if chunk.op.out else {}

            if chunk.op.out and chunk.op.where:
                inputs, kw['out'], kw['where'] = inputs[:-2], inputs[-2].copy(), inputs[-1]
            elif chunk.op.out:
                inputs, kw['out'] = inputs[:-1], inputs[-1].copy()
            elif chunk.op.where:
                inputs, kw['where'] = inputs[:-1], inputs[-1]

            with np.errstate(**chunk.op.err):
                if len(inputs) == 1:
                    try:
                        ctx[chunk.key] = _handle_out_dtype(func(inputs[0], **kw), chunk.op.dtype)
                    except TypeError:
                        if kw.get('where') is None:
                            raise
                        out, where = kw.pop('out'), kw.pop('where')
                        ctx[chunk.key] = _handle_out_dtype(xp.where(where, func(inputs[0]), out),
                                                           chunk.op.dtype)
                else:
                    try:
                        if is_sparse_module(xp):
                            ctx[chunk.key] = _handle_out_dtype(reduce(lambda a, b: func(a, b, **kw), inputs),
                                                               chunk.op.dtype)
                        else:
                            if 'out' not in kw:
                                dest_value = xp.empty(chunk.shape, chunk.dtype)
                                kw['out'] = dest_value
                            ctx[chunk.key] = _handle_out_dtype(reduce(lambda a, b: func(a, b, **kw), inputs),
                                                               chunk.op.dtype)
                    except TypeError:
                        if kw.get('where') is None:
                            raise
                        out, where = kw.pop('out'), kw.pop('where')
                        ctx[chunk.key] = _handle_out_dtype(
                            xp.where(where, reduce(lambda a, b: func(a, b), inputs), out),
                            chunk.op.dtype)
    return _handle


def _const_elementwise(op):
    def _handle(ctx, chunk):
        if chunk.inputs is not None:
            try:
                _, device_id, xp = as_same_device(
                    [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)
            except KeyError:
                raise
        else:
            # all constants
            device_id, xp = -1, np

        if isinstance(op, six.string_types):
            func = getattr(xp, op)
        else:
            func = op

        get = lambda x: ctx[x.key] if hasattr(x, 'key') else x

        with device(device_id):
            with np.errstate(**chunk.op.err):
                ctx[chunk.key] = _handle_out_dtype(
                    reduce(func, (get(chunk.op.lhs), get(chunk.op.rhs))), chunk.op.dtype)
    return _handle


def _around(ctx, chunk):
    (a,), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    with device(device_id):
        ctx[chunk.key] = xp.around(a, decimals=chunk.op.decimals)


def _angle(ctx, chunk):
    (z,), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    with device(device_id):
        ctx[chunk.key] = xp.angle(z, deg=chunk.op.deg)


def _isclose(ctx, chunk):
    (a, b), device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    with device(device_id):
        ctx[chunk.key] = xp.isclose(a, b, atol=chunk.op.atol, rtol=chunk.op.rtol,
                                    equal_nan=chunk.op.equal_nan)


def _frexp(ctx, chunk):
    inputs, device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    with device(device_id):
        kw = {'casting': chunk.op.casting}

        inputs_iter = iter(inputs)
        input = next(inputs_iter)
        if chunk.op.out1 is not None:
            out1 = next(inputs_iter)
        else:
            out1 = None
        if chunk.op.out2 is not None:
            out2 = next(inputs_iter)
        else:
            out2 = None
        if chunk.op.where is not None:
            where = kw['where'] = next(inputs_iter)
        else:
            where = None

        try:
            args = [input]
            if out1 is not None:
                args.append(out1)
            if out2 is not None:
                args.append(out2)
            mantissa, exponent = xp.frexp(*args, **kw)
        except TypeError:
            if where is None:
                raise
            mantissa, exponent = xp.frexp(input)
            mantissa, exponent = xp.where(where, mantissa, out1), xp.where(where, exponent, out2)

        for c, res in zip(chunk.op.outputs, (mantissa, exponent)):
            ctx[c.key] = res


def _modf(ctx, chunk):
    inputs, device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    with device(device_id):
        kw = {'casting': chunk.op.casting}

        inputs_iter = iter(inputs)
        input = next(inputs_iter)
        if chunk.op.out1 is not None:
            out1 = next(inputs_iter)
        else:
            out1 = None
        if chunk.op.out2 is not None:
            out2 = next(inputs_iter)
        else:
            out2 = None
        if chunk.op.where is not None:
            where = kw['where'] = next(inputs_iter)
        else:
            where = None

        try:
            args = [input]
            if out1 is not None:
                args.append(out1.copy())
            if out2 is not None:
                args.append(out2.copy())
            y1, y2 = xp.modf(*args, **kw)
        except TypeError:
            if where is None:
                raise
            y1, y2 = xp.modf(input)
            y1, y2 = xp.where(where, y1, out1), xp.where(where, y2, out2)

        for c, res in zip(chunk.op.outputs, (y1, y2)):
            ctx[c.key] = res


def _set_real(ctx, chunk):
    inputs, device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    if len(inputs) == 1:
        val, real = inputs[0], chunk.op.rhs
    else:
        assert len(inputs) == 2
        val, real = inputs

    with device(device_id):
        val = val.copy()
        val.real = real

        ctx[chunk.key] = val


def _set_imag(ctx, chunk):
    inputs, device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    if len(inputs) == 1:
        val, imag = inputs[0], chunk.op.rhs
    else:
        assert len(inputs) == 2
        val, imag = inputs

    with device(device_id):
        val = val.copy()
        val.imag = imag

        ctx[chunk.key] = val


def _clip(ctx, chunk):
    inputs, device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    inputs_iter = iter(inputs)
    a = next(inputs_iter)
    a_min = next(inputs_iter) if isinstance(chunk.op.a_min, type(chunk)) else chunk.op.a_min
    a_max = next(inputs_iter) if isinstance(chunk.op.a_max, type(chunk)) else chunk.op.a_max
    out = next(inputs_iter).copy() if chunk.op.out is not None else None

    with device(device_id):
        kw = {}
        if out is not None:
            kw['out'] = out
        ctx[chunk.key] = xp.clip(a, a_min, a_max, **kw)


def _i0(ctx, chunk):
    x = ctx[chunk.inputs[0].key]
    xp = get_array_module(x)
    res = xp.i0(x)
    if not is_sparse_module(xp):
        res = res.reshape(chunk.shape)
    ctx[chunk.key] = res


def _tree_add(ctx, chunk):
    inputs, device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    with device(device_id):
        ctx[chunk.key] = reduce(xp.add, inputs)


def _tree_multiply(ctx, chunk):
    inputs, device_id, xp = as_same_device(
        [ctx[c.key] for c in chunk.inputs], device=chunk.device, ret_extra=True)

    with device(device_id):
        ctx[chunk.key] = reduce(xp.multiply, inputs)


def _tree_op_estimate_size(ctx, chunk):
    sum_inputs = sum(ctx[inp.key][0] for inp in chunk.inputs)
    if not chunk.is_sparse():
        calc_size = chunk_size = chunk.nbytes
        if np.isnan(calc_size):
            chunk_size = calc_size = sum_inputs
    else:
        calc_size = sum_inputs
        chunk_size = min(sum_inputs, chunk.nbytes + np.dtype(np.int64).itemsize * np.prod(chunk.shape) * chunk.ndim)
        if np.isnan(chunk_size):
            chunk_size = sum_inputs
    ctx[chunk.key] = (chunk_size, calc_size)


def register_arithmetic_handler():
    from ...executor import register

    for op, new_op in six.iteritems(OP_TO_HANDLER):
        register(op, _build_elementwise(new_op))
        if hasattr(op, 'constant_cls'):
            const_op = op.constant_cls()
            if const_op:
                register(const_op, _const_elementwise(OP_TO_HANDLER[op]))

    register(arithmetic.TensorSetReal, _set_real)
    register(arithmetic.TensorSetRealConstant, _set_real)
    register(arithmetic.TensorSetImag, _set_imag)
    register(arithmetic.TensorSetImagConstant, _set_imag)

    register(arithmetic.TensorTreeAdd, _tree_add, _tree_op_estimate_size)
    register(arithmetic.TensorTreeMultiply, _tree_multiply, _tree_op_estimate_size)
    register(arithmetic.TensorAround, _around)
    register(arithmetic.TensorAngle, _angle)
    register(arithmetic.TensorIsclose, _isclose)
    register(arithmetic.TensorFrexp, _frexp)
    register(arithmetic.TensorModf, _modf)
    register(arithmetic.TensorClip, _clip)
    register(arithmetic.TensorI0, _i0)
