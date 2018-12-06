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

from .base import HasInput
from .. import opcodes as OperandDef
from ..serialize import AnyField, KeyField, TupleField, \
    DataTypeField, Int32Field, Int64Field, BoolField


class Reduction(HasInput):
    _input = KeyField('input')
    _out = KeyField('out')
    _axis = AnyField('axis')  # can be None or int or tuple of ints, just infer the data
    _dtype = DataTypeField('dtype')
    _keepdims = BoolField('keepdims')
    _combine_size = Int32Field('combine_size')

    @property
    def axis(self):
        return getattr(self, '_axis', None)

    @property
    def dtype(self):
        return getattr(self, '_dtype', None)

    @property
    def keepdims(self):
        return getattr(self, '_keepdims', None)

    @property
    def combine_size(self):
        return getattr(self, '_combine_size', None)


class Sum(Reduction):
    _op_type_ = OperandDef.SUM


class NanSum(Reduction):
    _op_type_ = OperandDef.NANSUM


class Prod(Reduction):
    _op_type_ = OperandDef.PROD


class NanProd(Reduction):
    _op_type_ = OperandDef.NANPROD


class Max(Reduction):
    _op_type_ = OperandDef.MAX


class NanMax(Reduction):
    _op_type_ = OperandDef.NANMAX


class Min(Reduction):
    _op_type_ = OperandDef.MIN


class NanMin(Reduction):
    _op_type_ = OperandDef.NANMIN


class All(Reduction):
    _op_type_ = OperandDef.ALL


class Any(Reduction):
    _op_type_ = OperandDef.ANY


class MeanChunk(Reduction):
    _op_type_ = OperandDef.MEAN_CHUNK


class MeanCombine(Reduction):
    _op_type_ = OperandDef.MEAN_COMBINE


class Mean(Reduction):
    _op_type_ = OperandDef.MEAN


class NanMeanChunk(Reduction):
    _op_type_ = OperandDef.NANMEAN_CHUNK


class NanMean(Reduction):
    _op_type_ = OperandDef.NANMEAN


class Argmax(Reduction):
    _op_type_ = OperandDef.ARGMAX


class ArgmaxChunk(Reduction):
    _op_type_ = OperandDef.ARGMAX_CHUNK

    _offset = Int64Field('offset')
    _total_shape = TupleField('total_shape')

    @property
    def offset(self):
        return getattr(self, '_offset', None)

    @property
    def total_shape(self):
        return getattr(self, '_total_shape', None)


class ArgmaxCombine(Reduction):
    _op_type_ = OperandDef.ARGMAX_COMBINE


class NanArgmax(Reduction):
    _op_type_ = OperandDef.NANARGMAX


class NanArgmaxChunk(Reduction):
    _op_type_ = OperandDef.NANARGMAX_CHUNK

    _offset = Int64Field('offset')
    _total_shape = TupleField('total_shape')

    @property
    def offset(self):
        return getattr(self, '_offset', None)

    @property
    def total_shape(self):
        return getattr(self, '_total_shape', None)


class NanArgmaxCombine(Reduction):
    _op_type_ = OperandDef.NANARGMAX_COMBINE


class Argmin(Reduction):
    _op_type_ = OperandDef.ARGMIN


class ArgminChunk(Reduction):
    _op_type_ = OperandDef.ARGMIN_CHUNK

    _offset = Int64Field('offset')
    _total_shape = TupleField('total_shape')

    @property
    def offset(self):
        return getattr(self, '_offset', None)

    @property
    def total_shape(self):
        return getattr(self, '_total_shape', None)


class ArgminCombine(Reduction):
    _op_type_ = OperandDef.ARGMIN_COMBINE


class NanArgmin(Reduction):
    _op_type_ = OperandDef.NANARGMIN


class NanArgminChunk(Reduction):
    _op_type_ = OperandDef.NANARGMIN_CHUNK

    _offset = Int64Field('offset')
    _total_shape = TupleField('total_shape')

    @property
    def offset(self):
        return getattr(self, '_offset', None)

    @property
    def total_shape(self):
        return getattr(self, '_total_shape', None)


class NanArgminCombine(Reduction):
    _op_type_ = OperandDef.NANARGMIN_COMBINE


class CountNonzero(Reduction):
    _op_type_ = OperandDef.COUNT_NONZERO


class MomentChunk(Reduction):
    _op_type_ = OperandDef.MOMENT_CHUNK

    _moment = Int32Field('moment', default=2)

    @property
    def moment(self):
        return getattr(self, '_moment', 2)


class MomentCombine(Reduction):
    _op_type_ = OperandDef.MOMENT_COMBINE

    _moment = Int32Field('moment')

    @property
    def moment(self):
        return getattr(self, '_moment', 2)


class Moment(Reduction):
    _op_type_ = OperandDef.MOMENT

    _moment = Int32Field('moment', default=2)
    _ddof = Int32Field('ddof')

    @property
    def moment(self):
        return getattr(self, '_moment', 2)

    @property
    def ddof(self):
        return self._ddof


class Var(Reduction):
    _op_type_ = OperandDef.VAR

    _ddof = Int32Field('ddof')

    @property
    def ddof(self):
        return self._ddof


class NanVar(Reduction):
    _op_type_ = OperandDef.NANVAR

    _ddof = Int32Field('ddof')

    @property
    def ddof(self):
        return self._ddof


class NanMomentChunk(Reduction):
    _op_type_ = OperandDef.NANMOMENT_CHUNK

    _moment = Int32Field('moment', default=2)

    @property
    def moment(self):
        return getattr(self, '_moment', 2)


class NanMomentCombine(Reduction):
    _op_type_ = OperandDef.NANMOMENT_COMBINE

    _moment = Int32Field('moment')

    @property
    def moment(self):
        return getattr(self, '_moment', 2)


class NanMoment(Reduction):
    _op_type_ = OperandDef.NANMOMENT

    _moment = Int32Field('moment', default=2)
    _ddof = Int32Field('ddof')

    @property
    def moment(self):
        return getattr(self, '_moment', 2)

    @property
    def ddof(self):
        return self._ddof


class CumReduction(HasInput):
    _input = KeyField('input')
    _axis = Int32Field('axis')
    _out = KeyField('out')

    @property
    def axis(self):
        return getattr(self, '_axis', None)

    @property
    def dtype(self):
        return getattr(self, '_dtype', None)


class Cumsum(CumReduction):
    _op_type_ = OperandDef.CUMSUM


class Cumprod(CumReduction):
    _op_type_ = OperandDef.CUMPROD


class NanCumsum(CumReduction):
    _op_type_ = OperandDef.NANCUMSUM


class NanCumprod(CumReduction):
    _op_type_ = OperandDef.NANCUMPROD
