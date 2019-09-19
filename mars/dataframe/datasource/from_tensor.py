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

import numpy as np
import pandas as pd

from ...core import Base
from ...serialize import KeyField, SeriesField
from ... import opcodes as OperandDef
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import parse_index


class DataFrameFromTensor(DataFrameOperand, DataFrameOperandMixin):
    """
    Represents data from mars tensor
    """
    _op_type_ = OperandDef.DATAFRAME_FROM_TENSOR
    _dtypes = SeriesField('dtypes')
    _input = KeyField('input')

    def __init__(self, dtypes=None, gpu=None, sparse=None, **kw):
        super(DataFrameFromTensor, self).__init__(_dtypes=dtypes,
                                                  _gpu=gpu, _sparse=sparse,
                                                  _object_type=ObjectType.dataframe, **kw)

    @property
    def dtypes(self):
        return self._dtypes

    @property
    def input(self):
        return self._input

    def _set_inputs(self, inputs):
        super(DataFrameFromTensor, self)._set_inputs(inputs)
        self._input = inputs[0]

    def __call__(self, input_tensor, index, columns):
        if input_tensor.ndim != 1 and input_tensor.ndim != 2:
            raise ValueError('Must pass 1-d or 2-d input')

        if index is not None:
            if input_tensor.shape[0] != len(index):
                raise ValueError(
                    'index {0} should have the same shape with tensor: {1}'.format(index, input_tensor.shape[0]))
            if not isinstance(index, pd.Index):
                if isinstance(index, Base):
                    raise NotImplementedError('The index value cannot be a tileable')
                index = pd.Index(index)
            index_value = parse_index(index, store_data=True)
        else:
            index_value = parse_index(pd.RangeIndex(start=0, stop=input_tensor.shape[0]))

        if columns is not None:
            if input_tensor.shape[1] != len(columns):
                raise ValueError(
                    'columns {0} should have the same shape with tensor: {1}'.format(columns, input_tensor.shape[1]))
            if not isinstance(columns, pd.Index):
                if isinstance(index, Base):
                    raise NotImplementedError('The index value cannot be a tileable')
                columns = pd.Index(columns)
            columns_value = parse_index(columns, store_data=True)
        else:
            if input_tensor.ndim == 1:
                # convert to 1-d DataFrame
                columns_value = parse_index(pd.RangeIndex(start=0, stop=1), store_data=True)
            else:
                columns_value = parse_index(pd.RangeIndex(start=0, stop=input_tensor.shape[1]), store_data=True)

        return self.new_dataframe([input_tensor], input_tensor.shape, dtypes=self.dtypes,
                                  index_value=index_value,
                                  columns_value=columns_value)

    @classmethod
    def tile(cls, op):
        out_df = op.outputs[0]
        in_tensor = op.input
        out_chunks = []
        nsplits = in_tensor.nsplits
        if any(any(np.isnan(ns)) for ns in nsplits):
            raise NotImplementedError('NAN shape is not supported in DataFrame')

        cum_size = [np.cumsum(s) for s in nsplits]
        for in_chunk in in_tensor.chunks:
            out_op = op.copy().reset_key()
            if in_chunk.ndim == 1:
                i, = in_chunk.index
                column_stop = 1
                index = (in_chunk.index[0], 0)
                columns_value = parse_index(out_df.columns.to_pandas()[0:1], store_data=True)
            else:
                i, j = in_chunk.index
                column_stop = cum_size[1][j]
                index = in_chunk.index
                columns_value = parse_index(out_df.columns.to_pandas()[column_stop - in_chunk.shape[1]:column_stop],
                                            store_data=True)

            index_stop = cum_size[0][i]
            if out_df.index_value.has_value():
                index_value = parse_index(out_df.index_value.to_pandas()[index_stop - in_chunk.shape[0]:index_stop],
                                          store_data=True)
            else:
                index_value = parse_index(pd.RangeIndex(start=index_stop - in_chunk.shape[0], stop=index_stop))

            out_op.extra_params['index_stop'] = index_stop
            out_op.extra_params['column_stop'] = column_stop
            out_chunk = out_op.new_chunk([in_chunk], shape=in_chunk.shape, index=index,
                                         index_value=index_value, columns_value=columns_value)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(out_df.inputs, out_df.shape, dtypes=out_df.dtypes,
                                     index_value=out_df.index_value,
                                     columns_value=out_df.columns,
                                     chunks=out_chunks, nsplits=in_tensor.nsplits)

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        tensor_data = ctx[op.inputs[0].key]
        ctx[chunk.key] = pd.DataFrame(tensor_data, index=chunk.index_value.to_pandas(),
                                      columns=chunk.columns.to_pandas())


def from_tensor(tensor, index=None, columns=None, gpu=None, sparse=False):
    if tensor.ndim > 2 or tensor.ndim <= 0:
        raise TypeError('Not support create DataFrame from {0} dims tensor', format(tensor.ndim))
    try:
        col_num = tensor.shape[1]
    except IndexError:
        col_num = 1
    gpu = tensor.op.gpu if gpu is None else gpu
    op = DataFrameFromTensor(dtypes=pd.Series([tensor.dtype] * col_num), gpu=gpu, sparse=sparse)
    return op(tensor, index, columns)
