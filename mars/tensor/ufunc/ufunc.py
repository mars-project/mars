#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2021 Alibaba Group Holding Ltd.
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

from numbers import Number

import numpy as np

from ..datasource import tensor as astensor
from .. import arithmetic as arith, reduction


class TensorUfuncDef:
    def __init__(
        self, method, aggregator=None, accumulator=None, pre_agg=None, post_agg=None
    ):
        self._method = method
        self._aggregator = aggregator
        self._accumulator = accumulator
        self._pre_agg = pre_agg
        self._post_agg = post_agg

    def __call__(self, *args, **kwargs):
        return self._method(*args, **kwargs)

    def at(self, a, indices, b=None):
        # todo handle setting duplicated keys, a separate operand may be needed
        if b is None:
            a[indices] = self(a[indices])
        else:
            a[indices] = self(a[indices], b)

    def accumulate(self, array, axis=0, dtype=None, out=None):
        if self._accumulator is None:
            raise NotImplementedError
        data = array if self._pre_agg is None else self._pre_agg(array)
        result = self._accumulator(data, axis=axis, dtype=dtype)
        result = result if self._post_agg is None else self._post_agg(result)
        if out is not None:
            out[0]._data = result._data
        else:
            return result

    def reduce(self, array, axis=0, dtype=None, out=None, keepdims=False):
        if self._aggregator is None:
            raise NotImplementedError
        data = array if self._pre_agg is None else self._pre_agg(array)
        result = self._aggregator(data, axis=axis, dtype=dtype, keepdims=keepdims)
        result = result if self._post_agg is None else self._post_agg(result)
        if out is not None:
            out[0]._data = result._data
        else:
            return result


UFUNC_TO_TENSOR_FUNCS = {
    np.add: TensorUfuncDef(
        arith.add,
        accumulator=reduction.cumsum,
        aggregator=reduction.sum,
    ),
    np.subtract: TensorUfuncDef(arith.subtract),
    np.multiply: TensorUfuncDef(
        arith.multiply,
        accumulator=reduction.cumprod,
        aggregator=reduction.prod,
    ),
    np.divide: TensorUfuncDef(arith.divide),
    np.logaddexp: TensorUfuncDef(
        arith.logaddexp,
        accumulator=reduction.cumsum,
        aggregator=reduction.sum,
        pre_agg=arith.exp,
        post_agg=arith.log,
    ),
    np.logaddexp2: TensorUfuncDef(
        arith.logaddexp2,
        accumulator=reduction.cumsum,
        aggregator=reduction.sum,
        pre_agg=lambda x: arith.power(2, x),
        post_agg=arith.log2,
    ),
    np.true_divide: TensorUfuncDef(arith.truediv),
    np.floor_divide: TensorUfuncDef(arith.floordiv),
    # unary
    np.negative: TensorUfuncDef(arith.negative),
    np.power: TensorUfuncDef(arith.power),
    np.float_power: TensorUfuncDef(arith.float_power),
    np.remainder: TensorUfuncDef(arith.remainder),
    np.mod: TensorUfuncDef(arith.mod),
    np.fmod: TensorUfuncDef(arith.fmod),
    np.conj: TensorUfuncDef(arith.conj),
    np.conjugate: TensorUfuncDef(arith.conjugate),
    np.exp: TensorUfuncDef(arith.exp),
    np.exp2: TensorUfuncDef(arith.exp2),
    np.log: TensorUfuncDef(arith.log),
    np.log2: TensorUfuncDef(arith.log2),
    np.log10: TensorUfuncDef(arith.log10),
    np.log1p: TensorUfuncDef(arith.log1p),
    np.expm1: TensorUfuncDef(arith.expm1),
    np.sqrt: TensorUfuncDef(arith.sqrt),
    np.square: TensorUfuncDef(arith.square),
    np.cbrt: TensorUfuncDef(arith.cbrt),
    np.reciprocal: TensorUfuncDef(arith.reciprocal),
    # trigonometric functions
    np.sin: TensorUfuncDef(arith.sin),
    np.cos: TensorUfuncDef(arith.cos),
    np.tan: TensorUfuncDef(arith.tan),
    np.arcsin: TensorUfuncDef(arith.arcsin),
    np.arccos: TensorUfuncDef(arith.arccos),
    np.arctan: TensorUfuncDef(arith.arctan),
    np.arctan2: TensorUfuncDef(arith.arctan2),
    np.hypot: TensorUfuncDef(arith.hypot),
    np.sinh: TensorUfuncDef(arith.sinh),
    np.cosh: TensorUfuncDef(arith.cosh),
    np.tanh: TensorUfuncDef(arith.tanh),
    np.arcsinh: TensorUfuncDef(arith.arcsinh),
    np.arccosh: TensorUfuncDef(arith.arccosh),
    np.arctanh: TensorUfuncDef(arith.arctanh),
    np.deg2rad: TensorUfuncDef(arith.deg2rad),
    np.rad2deg: TensorUfuncDef(arith.rad2deg),
    # comparison functions
    np.greater: TensorUfuncDef(arith.greater),
    np.greater_equal: TensorUfuncDef(arith.greater_equal),
    np.less: TensorUfuncDef(arith.less),
    np.less_equal: TensorUfuncDef(arith.less_equal),
    np.not_equal: TensorUfuncDef(arith.not_equal),
    np.equal: TensorUfuncDef(arith.equal),
    np.logical_and: TensorUfuncDef(arith.logical_and),
    np.logical_or: TensorUfuncDef(arith.logical_or),
    np.logical_xor: TensorUfuncDef(arith.logical_xor),
    np.logical_not: TensorUfuncDef(arith.logical_not),
    np.maximum: TensorUfuncDef(arith.maximum),
    np.minimum: TensorUfuncDef(arith.minimum),
    np.fmax: TensorUfuncDef(arith.fmax),
    np.fmin: TensorUfuncDef(arith.fmin),
    # floating functions
    np.isfinite: TensorUfuncDef(arith.isfinite),
    np.isinf: TensorUfuncDef(arith.isinf),
    np.isnan: TensorUfuncDef(arith.isnan),
    np.signbit: TensorUfuncDef(arith.signbit),
    np.copysign: TensorUfuncDef(arith.copysign),
    np.nextafter: TensorUfuncDef(arith.nextafter),
    np.spacing: TensorUfuncDef(arith.spacing),
    np.modf: TensorUfuncDef(arith.modf),
    np.ldexp: TensorUfuncDef(arith.ldexp),
    np.frexp: TensorUfuncDef(arith.frexp),
    np.floor: TensorUfuncDef(arith.floor),
    np.ceil: TensorUfuncDef(arith.ceil),
    np.trunc: TensorUfuncDef(arith.trunc),
    # more math functions
    np.degrees: TensorUfuncDef(arith.degrees),
    np.radians: TensorUfuncDef(arith.radians),
    np.rint: TensorUfuncDef(arith.rint),
    np.fabs: TensorUfuncDef(arith.fabs),
    np.sign: TensorUfuncDef(arith.sign),
    np.absolute: TensorUfuncDef(arith.absolute),
}


def _check_arg(arg):
    if isinstance(arg, Number):
        return True

    try:
        astensor(arg)
        return True
    except ValueError:
        return False


def _array_ufunc(_, ufunc, method, *inputs, **kwargs):
    out = kwargs.get("out", tuple())
    for x in inputs + out:
        if not _check_arg(x):
            return NotImplemented

    if ufunc.signature is not None:
        return NotImplemented
    if ufunc not in UFUNC_TO_TENSOR_FUNCS:
        return NotImplemented

    try:
        tensor_func = getattr(UFUNC_TO_TENSOR_FUNCS[ufunc], method)
        return tensor_func(*inputs, **kwargs)
    except (AttributeError, NotImplementedError):
        return NotImplemented
