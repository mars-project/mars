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

import numpy as np
import pandas as pd

from ...serialize import BoolField, ListField, Int32Field
from ... import opcodes as OperandDef
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import parse_index
from ...tensor.core import TENSOR_TYPE


class DataFrameFromRecords(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_FROM_RECORDS

    _columns = ListField('columns')
    _exclude = ListField('exclude')
    _coerce_float = BoolField('coerce_float')
    _nrows = Int32Field('nrows')

    def __init__(self, index=None, columns=None, exclude=None, coerce_float=False, nrows=None,
                 gpu=False, sparse=False, **kw):
        if index is not None or columns is not None:
            raise NotImplementedError('Specifying index value is not supported for now')
        super().__init__(_exclude=exclude, _columns=columns, _coerce_float=coerce_float, _nrows=nrows,
                         _gpu=gpu, _sparse=sparse, _object_type=ObjectType.dataframe, **kw)

    @property
    def columns(self):
        return self._columns

    @property
    def exclude(self):
        return self._exclude

    @property
    def coerce_float(self):
        return self._coerce_float

    @property
    def nrows(self):
        return self._nrows

    def __call__(self, data):
        if self.nrows is None:
            nrows = data.shape[0]
        else:
            nrows = self.nrows
        index_value = parse_index(pd.RangeIndex(start=0, stop=nrows))
        dtypes = pd.Series(dict((k, np.dtype(v)) for k, v in data.dtype.descr))
        columns_value = parse_index(pd.Index(data.dtype.names), store_data=True)
        return self.new_dataframe([data], (data.shape[0], len(data.dtype.names)), dtypes=dtypes,
                                  index_value=index_value, columns_value=columns_value)

    @classmethod
    def tile(cls, op):
        df = op.outputs[0]
        tensor = op.inputs[0]

        nsplit_acc = np.cumsum(tensor.nsplits[0])
        out_chunks = []
        for chunk in tensor.chunks:
            begin_index = nsplit_acc[chunk.index[0]] - chunk.shape[0]
            end_index = nsplit_acc[chunk.index[0]]
            chunk_index_value = parse_index(pd.RangeIndex(start=begin_index, stop=end_index))

            # Here the `new_chunk` is tricky:
            #
            # We can construct tensor that have identifcal chunks, for example, from `mt.ones(...)`, we know
            # that after tiling the chunk of the same shape (but at different position) in `mt.ones` is indeed
            # the same chunk (has the same key)!
            #
            # Thus, when we construct dataframe from such tensor, we will have dataframe chunks that only differ
            # in `index_value`. However the `index_value` field won't be used to calculate the chunk key of
            # the dataframe chunk, thus `new_chunk` generated the same keys for those indeed different chunks
            # (they have different `index_values`).
            #
            # Here, we construct new chunk with some unique `_extra_params` to make the `new_chunk` work as
            # expected.
            chunk_op = op.copy().reset_key()
            chunk_op.extra_params['begin_index'] = begin_index
            chunk_op.extra_params['end_index'] = end_index
            out_chunk = chunk_op.new_chunk(
                [chunk], shape=(chunk.shape[0], df.shape[1]), index=(chunk.index[0], 0), dtypes=df.dtypes,
                index_value=chunk_index_value, columns_value=df.columns_value)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes([tensor], df.shape, dtypes=df.dtypes,
                                     index_value=df.index_value,
                                     columns_value=df.columns_value,
                                     chunks=out_chunks, nsplits=[tensor.nsplits[0], [df.shape[1]]])

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        ctx[chunk.key] = pd.DataFrame.from_records(
            ctx[op.inputs[0].key],
            index=chunk.index_value.to_pandas(), columns=chunk.columns_value.to_pandas(),
            exclude=op.exclude, coerce_float=op.coerce_float, nrows=op.nrows)


def from_records(data, index=None, exclude=None, columns=None, coerce_float=False, nrows=None,
                 gpu=False, sparse=False, **kw):
    if isinstance(data, np.ndarray):
        from .dataframe import from_pandas
        return from_pandas(pd.DataFrame.from_records(data, index=index, exclude=exclude, columns=columns,
                                                     coerce_float=coerce_float, nrows=nrows), **kw)
    elif isinstance(data, TENSOR_TYPE):
        if data.dtype.names is None:
            raise TypeError('Not a tensor with structured dtype {0}', data.dtype)
        if data.ndim != 1:
            raise ValueError('Not a tensor with non 1-D structured dtype {0}', data.shape)

        op = DataFrameFromRecords(index=None, exclude=exclude, columns=columns, coerce_float=coerce_float,
                                  nrows=nrows, gpu=gpu, sparse=sparse, **kw)
        return op(data)
    else:
        raise TypeError('Not support create DataFrame from {0}', type(data))
