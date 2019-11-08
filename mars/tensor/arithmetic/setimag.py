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

from ... import opcodes as OperandDef
from ..array_utils import as_same_device, device
from .core import TensorBinOp
from .utils import arithmetic_operand


@arithmetic_operand(sparse_mode='binary_and')
class TensorSetImag(TensorBinOp):
    _op_type_ = OperandDef.SET_IMAG

    @classmethod
    def execute(cls, ctx, op):
        inputs, device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True)

        if len(inputs) == 1:
            val, imag = inputs[0], op.rhs
        else:
            assert len(inputs) == 2
            val, imag = inputs

        with device(device_id):
            val = val.copy()
            val.imag = imag

            ctx[op.outputs[0].key] = val


def set_imag(val, imag):
    op = TensorSetImag(dtype=val.dtype)
    return op(val, imag)
