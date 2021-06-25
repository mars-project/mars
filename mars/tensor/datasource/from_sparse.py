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

from ... import opcodes as OperandDef
from ...serialization.serializables import KeyField, StringField, AnyField
from ..array_utils import as_same_device, device, get_array_module
from ..utils import get_order
from .core import TensorHasInput
from .array import tensor


class SparseToDense(TensorHasInput):
    _op_type_ = OperandDef.SPARSE_TO_DENSE

    _input = KeyField('input')
    _order = StringField('order')
    _fill_value = AnyField('fill_value')

    def __init__(self, fill_value=None, order=None, **kw):
        super().__init__(_fill_value=fill_value,
                         _sparse=False, _order=order, **kw)

    @property
    def order(self):
        return self._order

    @property
    def fill_value(self):
        return self._fill_value

    @classmethod
    def execute(cls, ctx, op):
        fill_value = op.fill_value
        out = op.outputs[0]
        (inp,), device_id, xp = as_same_device(
            [ctx[inp.key] for inp in op.inputs], device=op.device, ret_extra=True)

        with device(device_id):
            if fill_value is None:
                ctx[out.key] = inp.toarray().astype(
                    out.dtype, order=op.order, copy=False)
            else:
                xp = get_array_module(xp)
                spmatrix = inp.spmatrix
                inds = spmatrix.nonzero()
                ret = xp.full(inp.shape, fill_value, dtype=out.dtype,
                              order=op.order)
                ret[inds] = spmatrix.data
                ctx[out.key] = ret


def fromsparse(a, order='C', fill_value=None):
    a = tensor(a)
    if not a.issparse():
        return a.astype(a.dtype, order=order, copy=False)

    tensor_order = get_order(order, None, available_options='CF',
                             err_msg="only 'C' or 'F' order is permitted")
    op = SparseToDense(dtype=a.dtype, gpu=a.op.gpu,
                       order=order, fill_value=fill_value)
    return op(a, order=tensor_order)
