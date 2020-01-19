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
from ...serialize import KeyField, StringField
from ..utils import get_order
from .core import TensorHasInput
from .array import tensor


class SparseToDense(TensorHasInput):
    _op_type_ = OperandDef.SPARSE_TO_DENSE

    _input = KeyField('_input')
    _order = StringField('_order')

    def __init__(self, dtype=None, gpu=None, order=None, **kw):
        super().__init__(_dtype=dtype, _gpu=gpu, _sparse=False, _order=order, **kw)

    @property
    def order(self):
        return self._order

    @classmethod
    def execute(cls, ctx, op):
        ctx[op.outputs[0].key] = \
            ctx[op.inputs[0].key].toarray().astype(
                op.outputs[0].dtype, order=op.order, copy=False)


def fromsparse(a, order='C'):
    a = tensor(a)
    if not a.issparse():
        return a.astype(a.dtype, order=order, copy=False)

    tensor_order = get_order(order, None, available_options='CF',
                             err_msg="only 'C' or 'F' order is permitted")
    op = SparseToDense(dtype=a.dtype, gpu=a.op.gpu, order=order)
    return op(a, order=tensor_order)
