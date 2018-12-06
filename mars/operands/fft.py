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

from .core import Operand
from .base import HasInput
from .. import opcodes as OperandDef
from ..serialize import ValueType, KeyField, Int32Field, Int64Field, Float64Field, StringField, ListField


class DiscreteFourierTransform(HasInput):
    __slots__ = ()


class StandardFFT(DiscreteFourierTransform):
    _input = KeyField('input')
    _n = Int64Field('n')
    _axis = Int32Field('axis')
    _norm = StringField('norm')

    @property
    def n(self):
        return self._n

    @property
    def axis(self):
        return self._axis

    @property
    def norm(self):
        return getattr(self, '_norm', None)


class FFT(StandardFFT):
    _op_type_ = OperandDef.FFT


class IFFT(StandardFFT):
    _op_type_ = OperandDef.IFFT


class StandardFFTN(DiscreteFourierTransform):
    _input = KeyField('input')
    _shape = ListField('shape', ValueType.int64)
    _axes = ListField('axes', ValueType.int32)
    _norm = StringField('norm')

    @property
    def shape(self):
        return self._shape

    @property
    def axes(self):
        return self._axes

    @property
    def norm(self):
        return getattr(self, '_norm', None)


class FFT2(StandardFFTN):
    _op_type_ = OperandDef.FFT2


class IFFT2(StandardFFTN):
    _op_type_ = OperandDef.IFFT2


class FFTN(StandardFFTN):
    _op_type_ = OperandDef.FFTN


class IFFTN(StandardFFTN):
    _op_type_ = OperandDef.IFFTN


class RealFFT(DiscreteFourierTransform):
    _input = KeyField('input')
    _n = Int64Field('n')
    _axis = Int32Field('axis')
    _norm = StringField('norm')

    @property
    def n(self):
        return self._n

    @property
    def axis(self):
        return self._axis

    @property
    def norm(self):
        return getattr(self, '_norm', None)


class RFFT(RealFFT):
    _op_type_ = OperandDef.RFFT


class IRFFT(RealFFT):
    _op_type_ = OperandDef.IRFFT


class RealFFTN(DiscreteFourierTransform):
    _input = KeyField('input')
    _shape = ListField('shape', ValueType.int64)
    _axes = ListField('axes', ValueType.int32)
    _norm = StringField('norm')

    @property
    def shape(self):
        return self._shape

    @property
    def axes(self):
        return self._axes

    @property
    def norm(self):
        return getattr(self, '_norm', None)


class RFFT2(RealFFTN):
    _op_type_ = OperandDef.RFFT2


class IRFFT2(RealFFTN):
    _op_type_ = OperandDef.IRFFT2


class RFFTN(RealFFTN):
    _op_type_ = OperandDef.RFFTN


class IRFFTN(RealFFTN):
    _op_type_ = OperandDef.IRFFTN


class HermitianFFT(DiscreteFourierTransform):
    _input = KeyField('input')
    _n = Int64Field('n')
    _axis = Int32Field('axis')
    _norm = StringField('norm')

    @property
    def n(self):
        return self._n

    @property
    def axis(self):
        return self._axis

    @property
    def norm(self):
        return getattr(self, '_norm', None)


class HFFT(HermitianFFT):
    _op_type_ = OperandDef.HFFT


class IHFFT(HermitianFFT):
    _op_type_ = OperandDef.IHFFT


class FFTFreq(Operand):
    _op_type_ = OperandDef.FFTFREQ

    _n = Int32Field('n')
    _d = Float64Field('d')

    @property
    def n(self):
        return self._n

    @property
    def d(self):
        return self._d


class FFTFreqChunk(HasInput):
    _op_type_ = OperandDef.FFTFREQ_CHUNK

    _input = KeyField('input')
    _n = Int32Field('n')
    _d = Float64Field('d')

    @property
    def n(self):
        return self._n

    @property
    def d(self):
        return self._d


class RFFTFreq(Operand):
    _op_type_ = OperandDef.RFFTFREQ

    _n = Int32Field('n')
    _d = Float64Field('d')

    @property
    def n(self):
        return self._n

    @property
    def d(self):
        return self._d


class FFTShiftBase(HasInput):
    _input = KeyField('input')
    _axes = ListField('axes', ValueType.int32)

    @property
    def axes(self):
        return self._axes


class FFTShift(FFTShiftBase):
    _op_type_ = OperandDef.FFTSHIFT


class IFFTShift(FFTShiftBase):
    _op_type_ = OperandDef.IFFTSHIFT
