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

from .... import opcodes as OperandDef
from .core import TensorBinOp, TensorConstant
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='binary_and')
class TensorSetImag(TensorBinOp):
    _op_type_ = OperandDef.SET_IMAG

    @classmethod
    def constant_cls(cls):
        return TensorSetImagConstant


@arithmetic_operand(sparse_mode='binary_and_const')
class TensorSetImagConstant(TensorConstant):
    _op_type_ = OperandDef.SET_IMAG_CONSTANT


def set_imag(val, imag):
    op = TensorSetImag(dtype=val.dtype)
    return op(val, imag)
