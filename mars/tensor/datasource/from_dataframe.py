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
from ...serialize import BoolField, KeyField
from ..utils import to_numpy
from ..core import TensorOrder
from .core import TensorHasInput


class TensorFromDataFrame(TensorHasInput):
    """ represent tensor from DataFrame """

    _op_type_ = OperandDef.TENSOR_FROM_DATAFRAME
    _input = KeyField('_input')
    _extract_multi_index = BoolField('extract_multi_index')

    def __init__(self, dtype=None, gpu=None, sparse=None, extract_multi_index=False, **kw):
        super().__init__(_dtype=dtype, _gpu=gpu, _sparse=sparse,
                         _extract_multi_index=extract_multi_index, **kw)

    @classmethod
    def execute(cls, ctx, op: 'TensorFromDataFrame'):
        df = ctx[op.inputs[0].key]
        if op._extract_multi_index:
            df = df.to_frame()
        ctx[op.outputs[0].key] = to_numpy(df).astype(op.dtype, order='F')

    @classmethod
    def tile(cls, op: 'TensorFromDataFrame'):
        output = op.outputs[0]

        out_chunks = []
        for c in op.input.chunks:
            shape = (c.shape[0], output.shape[1]) if op._extract_multi_index else c.shape
            index = (c.index[0], 0) if op._extract_multi_index else c.index
            out_chunk = op.copy().reset_key().new_chunk(
                [c], shape=shape, index=index, order=output.order)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        nsplits = (op.input.nsplits[0], (output.shape[1],)) if op._extract_multi_index else op.input.nsplits
        return new_op.new_tensors(op.inputs, output.shape, order=output.order,
                                  chunks=out_chunks, nsplits=nsplits)

    def __call__(self, a, order=None):
        from ...dataframe.core import INDEX_TYPE, IndexValue

        if self._extract_multi_index and isinstance(a, INDEX_TYPE) \
                and isinstance(a.index_value.value, IndexValue.MultiIndex):
            order = a.order if order is None else order
            return self.new_tensor([a], (a.shape[0], len(a.index_value.value.names)), order=order)
        else:
            self._extract_multi_index = False

        return super().__call__(a, order=order)


def from_dataframe(in_df, dtype=None):
    from ...dataframe.utils import build_empty_df

    if dtype is None:
        empty_pdf = build_empty_df(in_df.dtypes)
        dtype = to_numpy(empty_pdf).dtype
    op = TensorFromDataFrame(dtype=dtype, gpu=in_df.op.gpu)
    return op(in_df, order=TensorOrder.F_ORDER)  # return tensor with F-order always


def from_series(in_series, dtype=None):
    op = TensorFromDataFrame(dtype=dtype or in_series.dtype, gpu=in_series.op.gpu)
    return op(in_series, order=TensorOrder.F_ORDER)  # return tensor with F-order always


def from_index(in_index, dtype=None, extract_multi_index=False):
    op = TensorFromDataFrame(dtype=dtype or in_index.dtype, gpu=in_index.op.gpu,
                             extract_multi_index=extract_multi_index)
    return op(in_index, order=TensorOrder.F_ORDER)  # return tensor with F-order always
