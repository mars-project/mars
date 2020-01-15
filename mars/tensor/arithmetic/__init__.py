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

from ...core import build_mode
from .add import add, TensorAdd, TensorTreeAdd
from .subtract import subtract, TensorSubtract
from .multiply import multiply, TensorMultiply, TensorTreeMultiply
from .divide import divide, TensorDivide
from .truediv import truediv, TensorTrueDiv
from .floordiv import floordiv, TensorFloorDiv
from .mod import mod, TensorMod
from .power import power, TensorPower
from .float_power import float_power, TensorFloatPower
from .fmod import fmod, TensorFMod
from .sqrt import sqrt, TensorSqrt
from .around import around, around as round_, TensorAround
from .logaddexp import logaddexp, TensorLogAddExp
from .logaddexp2 import logaddexp2, TensorLogAddExp2
from .negative import negative, TensorNegative
from .positive import positive, TensorPositive
from .absolute import absolute, TensorAbsolute
from .fabs import fabs, TensorFabs
from .abs import abs, TensorAbs
from .rint import rint, TensorRint
from .sign import sign, TensorSign
from .degrees import degrees, TensorDegrees
from .radians import radians, TensorRadians
from .conj import conj, conj as conjugate, TensorConj
from .exp import exp, TensorExp
from .exp2 import exp2, TensorExp2
from .log import log, TensorLog
from .log2 import log2, TensorLog2
from .log10 import log10, TensorLog10
from .expm1 import expm1, TensorExpm1
from .log1p import log1p, TensorLog1p
from .sqrt import sqrt, TensorSqrt
from .square import square, TensorSquare
from .cbrt import cbrt, TensorCbrt
from .reciprocal import reciprocal, TensorReciprocal
from .equal import equal, TensorEqual
from .not_equal import not_equal, TensorNotEqual
from .less import less, TensorLessThan
from .less_equal import less_equal, TensorLessEqual
from .greater import greater, TensorGreaterThan
from .greater_equal import greater_equal, TensorGreaterEqual
from .sin import sin, TensorSin
from .cos import cos, TensorCos
from .tan import tan, TensorTan
from .arcsin import arcsin, TensorArcsin
from .arccos import arccos, TensorArccos
from .arctan import arctan, TensorArctan
from .arctan2 import arctan2, TensorArctan2
from .hypot import hypot, TensorHypot
from .sinh import sinh, TensorSinh
from .cosh import cosh, TensorCosh
from .tanh import tanh, TensorTanh
from .arcsinh import arcsinh, TensorArcsinh
from .arccosh import arccosh, TensorArccosh
from .arctanh import arctanh, TensorArctanh
from .deg2rad import deg2rad, TensorDeg2rad
from .rad2deg import rad2deg, TensorRad2deg
from .bitand import bitand, TensorBitand
from .bitor import bitor, TensorBitor
from .bitxor import bitxor, TensorBitxor
from .invert import invert, TensorInvert
from .lshift import lshift, TensorLshift
from .rshift import rshift, TensorRshift
from .logical_and import logical_and, TensorAnd
from .logical_or import logical_or, TensorOr
from .logical_xor import logical_xor, TensorXor
from .logical_not import logical_not, TensorNot
from .maximum import maximum, TensorMaximum
from .minimum import minimum, TensorMinimum
from .floor import floor, TensorFloor
from .ceil import ceil, TensorCeil
from .trunc import trunc, TensorTrunc
from .mod import mod as remainder
from .fmax import fmax, TensorFMax
from .fmin import fmin, TensorFMin
from .isfinite import isfinite, TensorIsFinite
from .isinf import isinf, TensorIsInf
from .isnan import isnan, TensorIsNan
from .signbit import signbit, TensorSignbit
from .copysign import copysign, TensorCopysign
from .nextafter import nextafter, TensorNextafter
from .spacing import spacing, TensorSpacing
from .clip import clip, TensorClip
from .isclose import isclose, TensorIsclose
from .ldexp import ldexp, TensorLdexp
from .frexp import frexp, TensorFrexp
from .modf import modf, TensorModf
from .angle import angle, TensorAngle
from .isreal import isreal, TensorIsReal
from .iscomplex import iscomplex, TensorIsComplex
from .real import real, TensorReal
from .imag import imag, TensorImag
from .fix import fix, TensorFix
from .i0 import i0, TensorI0
from .sinc import sinc, TensorSinc
from .nan_to_num import nan_to_num, TensorNanToNum
from .setreal import TensorSetReal
from .setimag import TensorSetImag


def _wrap_iop(func):
    def inner(self, *args, **kwargs):
        kwargs['out'] = self
        return func(self, *args, **kwargs)
    return inner


def _install():
    from ..core import TENSOR_TYPE, Tensor, TensorData
    from ..datasource import tensor as astensor
    from .add import add, radd
    from .subtract import subtract, rsubtract
    from .multiply import multiply, rmultiply
    from .divide import divide, rdivide
    from .truediv import truediv, rtruediv
    from .floordiv import floordiv, rfloordiv
    from .power import power, rpower
    from .mod import mod, rmod
    from .lshift import lshift, rlshift
    from .rshift import rshift, rrshift
    from .bitand import bitand, rbitand
    from .bitor import bitor, rbitor
    from .bitxor import bitxor, rbitxor

    def _wrap_equal(func):
        def eq(x1, x2, **kwargs):
            if build_mode().is_build_mode:
                return astensor(x1)._equals(x2)
            return func(x1, x2, **kwargs)
        return eq

    for cls in TENSOR_TYPE:
        setattr(cls, '__add__', add)
        setattr(cls, '__iadd__', _wrap_iop(add))
        setattr(cls, '__radd__', radd)
        setattr(cls, '__sub__', subtract)
        setattr(cls, '__isub__', _wrap_iop(subtract))
        setattr(cls, '__rsub__', rsubtract)
        setattr(cls, '__mul__', multiply)
        setattr(cls, '__imul__', _wrap_iop(multiply))
        setattr(cls, '__rmul__', rmultiply)
        setattr(cls, '__div__', divide)
        setattr(cls, '__idiv__', _wrap_iop(divide))
        setattr(cls, '__rdiv__', rdivide)
        setattr(cls, '__truediv__', truediv)
        setattr(cls, '__itruediv__', _wrap_iop(truediv))
        setattr(cls, '__rtruediv__', rtruediv)
        setattr(cls, '__floordiv__', floordiv)
        setattr(cls, '__ifloordiv__', _wrap_iop(floordiv))
        setattr(cls, '__rfloordiv__', rfloordiv)
        setattr(cls, '__pow__', power)
        setattr(cls, '__ipow__', _wrap_iop(power))
        setattr(cls, '__rpow__', rpower)
        setattr(cls, '__mod__', mod)
        setattr(cls, '__imod__', _wrap_iop(mod))
        setattr(cls, '__rmod__', rmod)
        setattr(cls, '__lshift__', lshift)
        setattr(cls, '__ilshift__', _wrap_iop(lshift))
        setattr(cls, '__rlshift__', rlshift)
        setattr(cls, '__rshift__', rshift)
        setattr(cls, '__irshift__', _wrap_iop(rshift))
        setattr(cls, '__rrshift__', rrshift)

        setattr(cls, '__eq__', _wrap_equal(equal))
        setattr(cls, '__ne__', not_equal)
        setattr(cls, '__lt__', less)
        setattr(cls, '__le__', less_equal)
        setattr(cls, '__gt__', greater)
        setattr(cls, '__ge__', greater_equal)
        setattr(cls, '__and__', bitand)
        setattr(cls, '__iand__', _wrap_iop(bitand))
        setattr(cls, '__rand__', rbitand)
        setattr(cls, '__or__', bitor)
        setattr(cls, '__ior__', _wrap_iop(bitor))
        setattr(cls, '__ror__', rbitor)
        setattr(cls, '__xor__', bitxor)
        setattr(cls, '__ixor__', _wrap_iop(bitxor))
        setattr(cls, '__rxor__', rbitxor)

        setattr(cls, '__neg__', negative)
        setattr(cls, '__pos__', positive)
        setattr(cls, '__abs__', abs)

    setattr(Tensor, 'round', round_)
    setattr(Tensor, 'conj', conj)
    setattr(Tensor, 'conjugate', conjugate)
    setattr(TensorData, 'round', round_)
    setattr(TensorData, 'conj', conj)
    setattr(TensorData, 'conjugate', conjugate)


_install()
del _install


BIN_UFUNC = {add, subtract, multiply, divide, truediv, floordiv, power, mod, fmod, logaddexp, logaddexp2, equal,
             not_equal, less, less_equal, greater, greater_equal, arctan2, hypot, bitand, bitor, bitxor, lshift,
             rshift, logical_and, logical_or, logical_xor, maximum, minimum, float_power, remainder, fmax, fmin,
             copysign, nextafter, ldexp}

UNARY_UFUNC = {square, arcsinh, rint, sign, conj, tan, absolute, deg2rad, log, fabs, exp2, invert, negative,
               sqrt, arctan, positive, cbrt, log10, sin, rad2deg, log2, arcsin, expm1, arctanh, cosh, sinh,
               cos, reciprocal, tanh, log1p, exp, arccos, arccosh, around, logical_not, conjugate,
               isfinite, isinf, isnan, signbit, spacing, floor, ceil, trunc, degrees, radians, angle,
               isreal, iscomplex, real, imag, fix, i0, sinc, nan_to_num}

