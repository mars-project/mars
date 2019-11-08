#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2020 Alibaba Group Holding Ltd.
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
from .. import arithmetic as arith


UFUNC_TO_TENSOR_FUNC = {
    # binary
    np.add: arith.add,
    np.subtract: arith.subtract,
    np.multiply: arith.multiply,
    np.divide: arith.divide,
    np.logaddexp: arith.logaddexp,
    np.logaddexp2: arith.logaddexp2,
    np.true_divide: arith.truediv,
    np.floor_divide: arith.floordiv,
    # unary
    np.negative: arith.negative,
    np.power: arith.power,
    np.float_power: arith.float_power,
    np.remainder: arith.remainder,
    np.mod: arith.mod,
    np.fmod: arith.fmod,
    np.conj: arith.conj,
    np.conjugate: arith.conjugate,
    np.exp: arith.exp,
    np.exp2: arith.exp2,
    np.log: arith.log,
    np.log2: arith.log2,
    np.log10: arith.log10,
    np.log1p: arith.log1p,
    np.expm1: arith.expm1,
    np.sqrt: arith.sqrt,
    np.square: arith.square,
    np.cbrt: arith.cbrt,
    np.reciprocal: arith.reciprocal,
    # trigonometric functions
    np.sin: arith.sin,
    np.cos: arith.cos,
    np.tan: arith.tan,
    np.arcsin: arith.arcsin,
    np.arccos: arith.arccos,
    np.arctan: arith.arctan,
    np.arctan2: arith.arctan2,
    np.hypot: arith.hypot,
    np.sinh: arith.sinh,
    np.cosh: arith.cosh,
    np.tanh: arith.tanh,
    np.arcsinh: arith.arcsinh,
    np.arccosh: arith.arccosh,
    np.arctanh: arith.arctanh,
    np.deg2rad: arith.deg2rad,
    np.rad2deg: arith.rad2deg,
    # comparison functions
    np.greater: arith.greater,
    np.greater_equal: arith.greater_equal,
    np.less: arith.less,
    np.less_equal: arith.less_equal,
    np.not_equal: arith.not_equal,
    np.equal: arith.equal,
    np.logical_and: arith.logical_and,
    np.logical_or: arith.logical_or,
    np.logical_xor: arith.logical_xor,
    np.logical_not: arith.logical_not,
    np.maximum: arith.maximum,
    np.minimum: arith.minimum,
    np.fmax: arith.fmax,
    np.fmin: arith.fmin,
    # floating functions
    np.isfinite: arith.isfinite,
    np.isinf: arith.isinf,
    np.isnan: arith.isnan,
    np.signbit: arith.signbit,
    np.copysign: arith.copysign,
    np.nextafter: arith.nextafter,
    np.spacing: arith.spacing,
    np.modf: arith.modf,
    np.ldexp: arith.ldexp,
    np.frexp: arith.frexp,
    np.floor: arith.floor,
    np.ceil: arith.ceil,
    np.trunc: arith.trunc,
    # more math functions
    np.degrees: arith.degrees,
    np.radians: arith.radians,
    np.rint: arith.rint,
    np.fabs: arith.fabs,
    np.sign: arith.sign,
    np.absolute: arith.absolute,
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
    out = kwargs.get('out', tuple())
    for x in inputs + out:
        if not _check_arg(x):
            return NotImplemented

    if method == '__call__':
        if ufunc.signature is not None:
            return NotImplemented
        if ufunc not in UFUNC_TO_TENSOR_FUNC:
            return NotImplemented

        tensor_func = UFUNC_TO_TENSOR_FUNC[ufunc]
        return tensor_func(*inputs, **kwargs)

    return NotImplemented
