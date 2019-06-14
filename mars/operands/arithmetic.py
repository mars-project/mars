#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2017 Alibaba Group Holding Ltd.
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

from .core import Operand
from ..core import BaseWithKey
from .. import opcodes as OperandDef
from ..serialize import ValueType, AnyField, KeyField, TupleField, \
    StringField, Int32Field, BoolField, DictField, Float64Field


class ElementWise(Operand):
    __slots__ = ()


class Constant(ElementWise):
    _lhs = AnyField('lhs')
    _rhs = AnyField('rhs')
    _casting = StringField('casting')
    _err = DictField('err', ValueType.string, ValueType.string)

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

    @property
    def constant(self):
        if isinstance(self._lhs, BaseWithKey):
            return [self._rhs]
        elif isinstance(self._rhs, BaseWithKey):
            return [self._lhs]
        return [self._lhs, self._rhs]

    @property
    def reverse(self):
        return isinstance(self._rhs, BaseWithKey)

    @property
    def casting(self):
        return getattr(self, '_casting', None)

    @property
    def err(self):
        return getattr(self, '_err', dict())

    def _set_inputs(self, inputs):
        super(Constant, self)._set_inputs(inputs)
        inputs_iter = iter(self._inputs)

        if not np.isscalar(self._lhs):
            self._lhs = next(inputs_iter)
        if not np.isscalar(self._rhs):
            self._rhs = next(inputs_iter)


class BinOp(ElementWise):
    _lhs = KeyField('lhs')
    _rhs = KeyField('rhs')
    _out = KeyField('out')
    _where = KeyField('where')
    _casting = StringField('casting')
    _err = DictField('err', ValueType.string, ValueType.string)

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

    @property
    def out(self):
        return getattr(self, '_out', None)

    @property
    def where(self):
        return getattr(self, '_where', None)

    @property
    def casting(self):
        return getattr(self, '_casting', None)

    @property
    def err(self):
        return getattr(self, '_err', dict())

    def _set_inputs(self, inputs):
        super(BinOp, self)._set_inputs(inputs)
        inputs_iter = iter(self._inputs)

        self._lhs = next(inputs_iter)
        self._rhs = next(inputs_iter)
        if getattr(self, '_out', None) is not None:
            self._out = next(inputs_iter)
        if getattr(self, '_where', None) is not None:
            self._where = next(inputs_iter)


class UnaryOp(ElementWise):
    _input = KeyField('input')
    _out = KeyField('out')
    _where = KeyField('where')
    _casting = StringField('casting')
    _err = DictField('err', ValueType.string, ValueType.string)

    @property
    def input(self):
        return self._input

    @property
    def out(self):
        return getattr(self, '_out', None)

    @property
    def where(self):
        return getattr(self, '_where', None)

    @property
    def casting(self):
        return getattr(self, '_casting', None)

    @property
    def err(self):
        return getattr(self, '_err', dict())

    def _set_inputs(self, inputs):
        super(UnaryOp, self)._set_inputs(inputs)
        inputs_iter = iter(self._inputs)

        self._input = next(inputs_iter)
        if getattr(self, '_out', None) is not None:
            self._out = next(inputs_iter)
        if getattr(self, '_where', None) is not None:
            self._where = next(inputs_iter)


class Add(BinOp):
    _op_type_ = OperandDef.ADD


class AddConstant(Constant):
    _op_type_ = OperandDef.ADD_CONSTANT


class Subtract(BinOp):
    _op_type_ = OperandDef.SUB


class SubConstant(Constant):
    _op_type_ = OperandDef.SUB_CONSTANT


class Multiply(BinOp):
    _op_type_ = OperandDef.MUL


class MulConstant(Constant):
    _op_type_ = OperandDef.MUL_CONSTANT


class Divide(BinOp):
    _op_type_ = OperandDef.DIV


class DivConstant(Constant):
    _op_type_ = OperandDef.DIV_CONSTANT


class TrueDiv(BinOp):
    _op_type_ = OperandDef.TRUEDIV


class TDivConstant(Constant):
    _op_type_ = OperandDef.TDIV_CONSTANT


class FloorDiv(BinOp):
    _op_type_ = OperandDef.FLOORDIV


class FDivConstant(Constant):
    _op_type_ = OperandDef.FDIV_CONSTANT


class Power(BinOp):
    _op_type_ = OperandDef.POW


class PowConstant(Constant):
    _op_type_ = OperandDef.POW_CONSTANT


class FloatPower(BinOp):
    _op_type_ = OperandDef.FLOAT_POWER


class FloatPowerConstant(Constant):
    _op_type_ = OperandDef.FLOAT_POWER_CONSTANT


class Mod(BinOp):
    _op_type_ = OperandDef.MOD


class ModConstant(Constant):
    _op_type_ = OperandDef.MOD_CONSTANT


class FMod(BinOp):
    _op_type_ = OperandDef.FMOD


class FModConstant(Constant):
    _op_type_ = OperandDef.FMOD_CONSTANT


class LogAddExp(BinOp):
    _op_type_ = OperandDef.LOGADDEXP


class LAEConstant(Constant):
    _op_type_ = OperandDef.LAE_CONSTANT


class LogAddExp2(BinOp):
    _op_type_ = OperandDef.LOGADDEXP2


class LAE2Constant(Constant):
    _op_type_ = OperandDef.LAE2_CONSTANT


class Isclose(BinOp):
    _op_type_ = OperandDef.ISCLOSE

    _rtol = Float64Field('rtol')
    _atol = Float64Field('atol')
    _equal_nan = BoolField('equal_nan')

    @property
    def rtol(self):
        return self._rtol

    @property
    def atol(self):
        return self._atol

    @property
    def equal_nan(self):
        return self._equal_nan


class IscloseConstant(Constant):
    _op_type_ = OperandDef.ISCLOSE_CONSTANT

    _rtol = Float64Field('rtol')
    _atol = Float64Field('atol')
    _equal_nan = BoolField('equal_nan')

    @property
    def rtol(self):
        return self._rtol

    @property
    def atol(self):
        return self._atol

    @property
    def equal_nan(self):
        return self._equal_nan


class Negative(UnaryOp):
    _op_type_ = OperandDef.NEGATIVE


class Positive(UnaryOp):
    _op_type_ = OperandDef.POSITIVE


class Absolute(UnaryOp):
    _op_type_ = OperandDef.ABSOLUTE


class Fabs(UnaryOp):
    _op_type_ = OperandDef.FABS


class Abs(UnaryOp):
    _op_type_ = OperandDef.ABS


class Rint(UnaryOp):
    _op_type_ = OperandDef.RINT


class Sign(UnaryOp):
    _op_type_ = OperandDef.SIGN


class Degrees(UnaryOp):
    _op_type_ = OperandDef.DEGREES


class Radians(UnaryOp):
    _op_type_ = OperandDef.RADIANS


class Conj(UnaryOp):
    _op_type_ = OperandDef.CONJ


class Exp(UnaryOp):
    _op_type_ = OperandDef.EXP


class Exp2(UnaryOp):
    _op_type_ = OperandDef.EXP2


class Log(UnaryOp):
    _op_type_ = OperandDef.LOG


class Log2(UnaryOp):
    _op_type_ = OperandDef.LOG2


class Log10(UnaryOp):
    _op_type_ = OperandDef.LOG10


class Expm1(UnaryOp):
    _op_type_ = OperandDef.EXPM1


class Log1p(UnaryOp):
    _op_type_ = OperandDef.LOG1P


class Sqrt(UnaryOp):
    _op_type_ = OperandDef.SQRT


class Square(UnaryOp):
    _op_type_ = OperandDef.SQUARE


class Cbrt(UnaryOp):
    _op_type_ = OperandDef.CBRT


class Around(UnaryOp):
    _op_type_ = OperandDef.AROUND

    _decimals = Int32Field('decimals')

    @property
    def decimals(self):
        return self._decimals


class Reciprocal(UnaryOp):
    _op_type_ = OperandDef.RECIPROCAL


class Equal(BinOp):
    _op_type_ = OperandDef.EQ


class EqConstant(Constant):
    _op_type_ = OperandDef.EQ_CONSTANT


class NotEqual(BinOp):
    _op_type_ = OperandDef.NE


class NeConstant(Constant):
    _op_type_ = OperandDef.NE_CONSTANT


class LessThan(BinOp):
    _op_type_ = OperandDef.LT


class LtConstant(Constant):
    _op_type_ = OperandDef.LT_CONSTANT


class LessEqual(BinOp):
    _op_type_ = OperandDef.LE


class LeConstant(Constant):
    _op_type_ = OperandDef.LE_CONSTANT


class GreaterThan(BinOp):
    _op_type_ = OperandDef.GT


class GtConstant(Constant):
    _op_type_ = OperandDef.GT_CONSTANT


class GreaterEqual(BinOp):
    _op_type_ = OperandDef.GE


class GeConstant(Constant):
    _op_type_ = OperandDef.GE_CONSTANT


class Sin(UnaryOp):
    _op_type_ = OperandDef.SIN


class Cos(UnaryOp):
    _op_type_ = OperandDef.COS


class Tan(UnaryOp):
    _op_type_ = OperandDef.TAN


class Arcsin(UnaryOp):
    _op_type_ = OperandDef.ARCSIN


class Arccos(UnaryOp):
    _op_type_ = OperandDef.ARCCOS


class Arctan(UnaryOp):
    _op_type_ = OperandDef.ARCTAN


class Arctan2(BinOp):
    _op_type_ = OperandDef.ARCTAN2


class Arct2Constant(Constant):
    _op_type_ = OperandDef.ARCT2_CONSTANT


class Hypot(BinOp):
    _op_type_ = OperandDef.HYPOT


class HypotConstant(Constant):
    _op_type_ = OperandDef.HYPOT_CONSTANT


class Sinh(UnaryOp):
    _op_type_ = OperandDef.SINH


class Cosh(UnaryOp):
    _op_type_ = OperandDef.COSH


class Tanh(UnaryOp):
    _op_type_ = OperandDef.TANH


class Arcsinh(UnaryOp):
    _op_type_ = OperandDef.ARCSINH


class Arccosh(UnaryOp):
    _op_type_ = OperandDef.ARCCOSH


class Arctanh(UnaryOp):
    _op_type_ = OperandDef.ARCTANH


class Deg2rad(UnaryOp):
    _op_type_ = OperandDef.DEG2RAD


class Rad2deg(UnaryOp):
    _op_type_ = OperandDef.RAD2DEG


class Bitand(BinOp):
    _op_type_ = OperandDef.BITAND


class BitandConstant(Constant):
    _op_type_ = OperandDef.BITAND_CONSTANT


class Bitor(BinOp):
    _op_type_ = OperandDef.BITOR


class BitorConstant(Constant):
    _op_type_ = OperandDef.BITOR_CONSTANT


class Bitxor(BinOp):
    _op_type_ = OperandDef.BITXOR


class BitxorConstant(Constant):
    _op_type_ = OperandDef.BITXOR_CONSTANT


class Invert(UnaryOp):
    _op_type_ = OperandDef.INVERT


class Lshift(BinOp):
    _op_type_ = OperandDef.LSHIFT


class LshiftConstant(Constant):
    _op_type_ = OperandDef.LSHIFT_CONSTANT


class Rshift(BinOp):
    _op_type_ = OperandDef.RSHIFT


class RshiftConstant(Constant):
    _op_type_ = OperandDef.RSHIFT_CONSTANT


class And(BinOp):
    _op_type_ = OperandDef.AND


class AndConstant(Constant):
    _op_type_ = OperandDef.AND_CONSTANT


class Or(BinOp):
    _op_type_ = OperandDef.OR


class OrConstant(Constant):
    _op_type_ = OperandDef.OR_CONSTANT


class Xor(BinOp):
    _op_type_ = OperandDef.XOR


class XorConstant(Constant):
    _op_type_ = OperandDef.XOR_CONSTANT


class Not(UnaryOp):
    _op_type_ = OperandDef.NOT


class Maximum(BinOp):
    _op_type_ = OperandDef.MAXIMUM


class MaximumConstant(Constant):
    _op_type_ = OperandDef.MAXIMUM_CONSTANT


class Minimum(BinOp):
    _op_type_ = OperandDef.MINIMUM


class MinimumConstant(Constant):
    _op_type_ = OperandDef.MINIMUM_CONSTANT


class Floor(UnaryOp):
    _op_type_ = OperandDef.FLOOR


class Ceil(UnaryOp):
    _op_type_ = OperandDef.CEIL


class Trunc(UnaryOp):
    _op_type_ = OperandDef.TRUNC


class FMax(BinOp):
    _op_type_ = OperandDef.FMAX


class FMaxConstant(Constant):
    _op_type_ = OperandDef.FMAX_CONSTANT


class FMin(BinOp):
    _op_type_ = OperandDef.FMIN


class FMinConstant(Constant):
    _op_type_ = OperandDef.FMIN_CONSTANT


class IsFinite(UnaryOp):
    _op_type_ = OperandDef.ISFINITE


class IsInf(UnaryOp):
    _op_type_ = OperandDef.ISINF


class IsNan(UnaryOp):
    _op_type_ = OperandDef.ISNAN


class Signbit(UnaryOp):
    _op_type_ = OperandDef.SIGNBIT


class Copysign(BinOp):
    _op_type_ = OperandDef.COPYSIGN


class CopysignConstant(Constant):
    _op_type_ = OperandDef.COPYSIGN_CONSTANT


class Nextafter(BinOp):
    _op_type_ = OperandDef.NEXTAFTER


class NextafterConstant(Constant):
    _op_type_ = OperandDef.NEXTAFTER_CONSTANT


class Spacing(UnaryOp):
    _op_type_ = OperandDef.SPACING


class Modf(ElementWise):
    _op_type_ = OperandDef.MODF

    _input = KeyField('input')
    _out1 = KeyField('out1')
    _out2 = KeyField('out2')
    _where = KeyField('where')
    _casting = StringField('casting')

    @property
    def output_limit(self):
        return 2

    @property
    def input(self):
        return self._input

    @property
    def out1(self):
        return getattr(self, '_out1', None)

    @property
    def out2(self):
        return getattr(self, '_out2', None)

    @property
    def where(self):
        return getattr(self, '_where', None)

    @property
    def casting(self):
        return getattr(self, '_casting', None)

    def _set_inputs(self, inputs):
        super(Modf, self)._set_inputs(inputs)
        inputs_iter = iter(self._inputs)

        self._input = next(inputs_iter)
        if getattr(self, '_out1', None) is not None:
            self._out1 = next(inputs_iter)
        if getattr(self, '_out2', None) is not None:
            self._out2 = next(inputs_iter)
        if getattr(self, '_where', None) is not None:
            self._where = next(inputs_iter)



class Ldexp(BinOp):
    _op_type_ = OperandDef.LDEXP


class LdexpConstant(Constant):
    _op_type_ = OperandDef.LDEXP_CONSTANT


class Frexp(ElementWise):
    _op_type_ = OperandDef.FREXP

    _input = KeyField('input')
    _out1 = KeyField('out1')
    _out2 = KeyField('out2')
    _where = KeyField('where')
    _casting = StringField('casting')

    @property
    def output_limit(self):
        return 2

    @property
    def input(self):
        return self._input

    @property
    def out1(self):
        return getattr(self, '_out1', None)

    @property
    def out2(self):
        return getattr(self, '_out2', None)

    @property
    def where(self):
        return getattr(self, '_where', None)

    @property
    def casting(self):
        return getattr(self, '_casting', None)

    def _set_inputs(self, inputs):
        super(Frexp, self)._set_inputs(inputs)
        inputs_iter = iter(self._inputs)

        self._input = next(inputs_iter)
        if getattr(self, '_out1', None) is not None:
            self._out1 = next(inputs_iter)
        if getattr(self, '_out2', None) is not None:
            self._out2 = next(inputs_iter)
        if getattr(self, '_where', None) is not None:
            self._where = next(inputs_iter)


class Clip(ElementWise):
    _op_type_ = OperandDef.CLIP

    _a = KeyField('a')
    _a_min = AnyField('a_min')
    _a_max = AnyField('a_max')
    _out = KeyField('out')

    @property
    def a(self):
        return self._a

    @property
    def a_min(self):
        return self._a_min

    @property
    def a_max(self):
        return self._a_max

    @property
    def out(self):
        return getattr(self, '_out', None)


class Angle(UnaryOp):
    _op_type_ = OperandDef.ANGLE

    _deg = BoolField('deg')

    @property
    def deg(self):
        return self._deg


class IsReal(UnaryOp):
    _op_type_ = OperandDef.ISREAL


class IsComplex(UnaryOp):
    _op_type_ = OperandDef.ISCOMPLEX


class Real(UnaryOp):
    _op_type_ = OperandDef.REAL


class Imag(UnaryOp):
    _op_type_ = OperandDef.IMAG


class SetReal(BinOp):
    _op_type_ = OperandDef.SET_REAL


class SetRealConstant(Constant):
    _op_type_ = OperandDef.SET_REAL_CONSTANT


class SetImag(BinOp):
    _op_type_ = OperandDef.SET_IMAG


class SetImagConstant(Constant):
    _op_type_ = OperandDef.SET_IMAG_CONSTANT


class Fix(UnaryOp):
    _op_type_ = OperandDef.FIX


class I0(UnaryOp):
    _op_type_ = OperandDef.I0


class Sinc(UnaryOp):
    _op_type_ = OperandDef.SINC


class NanToNum(UnaryOp):
    _op_type_ = OperandDef.NAN_TO_NUM


class TreeAdd(ElementWise):
    _op_type_ = OperandDef.TREE_ADD


class TreeMultiply(ElementWise):
    _op_type_ = OperandDef.TREE_MULTIPLY


class TensorDot(Operand):
    _op_type_ = OperandDef.TENSORDOT

    _a = KeyField('a')
    _b = KeyField('b')
    _a_axes = TupleField('a_axes', ValueType.int32)
    _b_axes = TupleField('b_axes', ValueType.int32)

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def a_axes(self):
        return self._a_axes

    @property
    def b_axes(self):
        return self._b_axes


class Dot(Operand):
    _op_type_ = OperandDef.DOT

    _a = KeyField('a')
    _b = KeyField('b')

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b


class Matmul(Operand):
    _op_type_ = OperandDef.MATMUL

    _a = KeyField('a')
    _b = KeyField('b')

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b
