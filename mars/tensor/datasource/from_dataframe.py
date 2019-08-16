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

from ... import opcodes as OperandDef
from ...serialize import KeyField
from ...dataframe.utils import build_empty_df
from ..utils import to_numpy
from ..core import TensorOrder
from .core import TensorHasInput


class TensorDataFrameDataSource(TensorHasInput):
    """ represent tensor from DataFrame """

    _op_type_ = OperandDef.TENSOR_FROM_DATAFRAME
    _input = KeyField('_input')

    def __init__(self, dtype=None, gpu=None, **kw):
        super(TensorDataFrameDataSource, self).__init__(_dtype=dtype, _gpu=gpu,
                                                        _sparse=True, **kw)

    @classmethod
    def execute(cls, ctx, op):
        df = ctx[op.inputs[0].key]
        ctx[op.outputs[0].key] = to_numpy(df).astype(op.outputs[0].dtype, order='F')


def from_dataframe(in_df):
    empty_pdf = build_empty_df(in_df.dtypes)
    op = TensorDataFrameDataSource(dtype=empty_pdf.dtypes[0], gpu=in_df.op.gpu)
    return op(in_df, order=TensorOrder.F_ORDER)  # return tensor with F-order always
