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

from collections import OrderedDict

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...core import Base, Entity
from ...serialize import KeyField, SeriesField, DataTypeField, BoolField
from ...tensor.datasource import tensor as astensor
from ...tensor.utils import unify_chunks
from ...tiles import TilesError
from ...utils import check_chunks_unknown_shape
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import parse_index


class DataFrameFromTensor(DataFrameOperand, DataFrameOperandMixin):
    """
    Represents data from mars tensor
    """
    _op_type_ = OperandDef.DATAFRAME_FROM_TENSOR

    _input = KeyField('input')
    _index = KeyField('index')
    _dtypes = SeriesField('dtypes')
    _from_1d_tensors = BoolField('from_1d_tensors')

    def __init__(self, index=None, dtypes=None, from_1d_tensors=None,
                 gpu=None, sparse=None, **kw):
        super(DataFrameFromTensor, self).__init__(
            _index=index, _dtypes=dtypes, _from_1d_tensors=from_1d_tensors, _gpu=gpu,
            _sparse=sparse, _object_type=ObjectType.dataframe, **kw)

    @property
    def dtypes(self):
        return self._dtypes

    @property
    def input(self):
        return self._input

    @property
    def index(self):
        return self._index

    @property
    def from_1d_tensors(self):
        return self._from_1d_tensors

    def _set_inputs(self, inputs):
        super(DataFrameFromTensor, self)._set_inputs(inputs)
        if not self._from_1d_tensors:
            self._input = inputs[0]
        if self._index is not None:
            self._index = inputs[-1]

    def __call__(self, input_tensor, index, columns):
        if self._from_1d_tensors:
            return self._call_input_1d_tensors(input_tensor, index, columns)
        else:
            return self._call_input_tensor(input_tensor, index, columns)

    def _process_index(self, index, inputs):
        if not isinstance(index, pd.Index):
            if isinstance(index, (Base, Entity)):
                self._index = index
                index = astensor(index)
                if index.ndim != 1:
                    raise ValueError('index should be 1-d, got {}-d'.format(index.ndim))
                index_value = parse_index(pd.Index([], dtype=index.dtype), index, type(self).__name__)
                inputs.append(index)
            else:
                index = pd.Index(index)
                index_value = parse_index(index, store_data=True)
        else:
            index_value = parse_index(index, store_data=True)
        return index_value

    def _call_input_1d_tensors(self, input_1d_tensors, index, columns):
        if len({t.shape for t in input_1d_tensors}) != 1:
            raise ValueError('input 1-d tensors should have same shape')

        inputs = list(input_1d_tensors)
        if index is not None:
            if input_1d_tensors[0].shape[0] != len(index):
                raise ValueError('index {} should have '
                                 'the same shape with tensor: {}'.format(
                    index, input_1d_tensors[0].shape[0]))
            index_value = self._process_index(index, inputs)
        else:
            index_value = parse_index(pd.RangeIndex(0, input_1d_tensors[0].shape[0]))

        if columns is not None:
            if len(input_1d_tensors) != len(columns):
                raise ValueError(
                    'columns {0} should have size {1}'.format(columns, len(input_1d_tensors)))
            if not isinstance(columns, pd.Index):
                if isinstance(columns, Base):
                    raise NotImplementedError('The columns value cannot be a tileable')
                columns = pd.Index(columns)
            columns_value = parse_index(columns, store_data=True)
        else:
            columns_value = parse_index(pd.RangeIndex(0, len(input_1d_tensors)), store_data=True)

        shape = (input_1d_tensors[0].shape[0], len(input_1d_tensors))
        return self.new_dataframe(inputs, shape, dtypes=self.dtypes,
                                  index_value=index_value, columns_value=columns_value)

    def _call_input_tensor(self, input_tensor, index, columns):
        if input_tensor.ndim not in {1, 2}:
            raise ValueError('Must pass 1-d or 2-d input')
        inputs = [input_tensor]

        if index is not None:
            if input_tensor.shape[0] != len(index):
                raise ValueError(
                    'index {0} should have the same shape with tensor: {1}'.format(
                        index, input_tensor.shape[0]))
            index_value = self._process_index(index, inputs)
        else:
            index_value = parse_index(pd.RangeIndex(start=0, stop=input_tensor.shape[0]))

        if columns is not None:
            if not (input_tensor.ndim == 1 and len(columns) == 1 or
                    input_tensor.shape[1] == len(columns)):
                raise ValueError(
                    'columns {0} should have the same shape with tensor: {1}'.format(
                        columns, input_tensor.shape[1]))
            if not isinstance(columns, pd.Index):
                if isinstance(columns, Base):
                    raise NotImplementedError('The columns value cannot be a tileable')
                columns = pd.Index(columns)
            columns_value = parse_index(columns, store_data=True)
        else:
            if input_tensor.ndim == 1:
                # convert to 1-d DataFrame
                columns_value = parse_index(pd.RangeIndex(start=0, stop=1), store_data=True)
            else:
                columns_value = parse_index(pd.RangeIndex(start=0, stop=input_tensor.shape[1]), store_data=True)

        if input_tensor.ndim == 1:
            shape = (input_tensor.shape[0], 1)
        else:
            shape = input_tensor.shape

        return self.new_dataframe(inputs, shape, dtypes=self.dtypes,
                                  index_value=index_value, columns_value=columns_value)

    @classmethod
    def tile(cls, op):
        # make sure all tensor have known chunk shapes
        check_chunks_unknown_shape(op.inputs, TilesError)

        if op.from_1d_tensors:
            return cls._tile_input_1d_tensors(op)
        else:
            return cls._tile_input_tensor(op)

    @classmethod
    def _tile_input_1d_tensors(cls, op):
        out_df = op.outputs[0]
        in_tensors = op.inputs
        in_tensors = unify_chunks(*in_tensors)
        nsplit = in_tensors[0].nsplits[0]

        cum_sizes = [0] + np.cumsum(nsplit).tolist()
        out_chunks = []
        for i in range(in_tensors[0].chunk_shape[0]):
            chunk_op = op.copy().reset_key()
            columns_value = out_df.columns_value
            dtypes = out_df.dtypes
            chunk_index = (i, 0)
            if out_df.index_value.has_value():
                pd_index = out_df.index_value.to_pandas()[cum_sizes[i]: cum_sizes[i + 1]]
                index_value = parse_index(pd_index, store_data=True)
            else:
                assert op.index is not None
                index_chunk = in_tensors[-1].cix[i,]
                index_value = parse_index(pd.Index([], dtype=index_chunk.dtype),
                                          index_chunk, type(chunk_op).__name__)
            shape = (nsplit[i], len(out_df.dtypes))
            out_chunk = chunk_op.new_chunk([t.cix[i,] for t in in_tensors],
                                           shape=shape, index=chunk_index,
                                           dtypes=dtypes, index_value=index_value,
                                           columns_value=columns_value)
            out_chunks.append(out_chunk)

        nsplits = (nsplit, (len(out_df.dtypes),))
        new_op = op.copy()
        return new_op.new_dataframes(out_df.inputs, out_df.shape, dtypes=out_df.dtypes,
                                     index_value=out_df.index_value, columns_value=out_df.columns_value,
                                     chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def _tile_input_tensor(cls, op):
        out_df = op.outputs[0]
        in_tensor = op.input
        out_chunks = []
        nsplits = in_tensor.nsplits

        if op.index is not None:
            # rechunk index if it's a tensor
            index_tensor = op.index.rechunk([nsplits[0]])._inplace_tile()
        else:
            index_tensor = None

        cum_size = [np.cumsum(s) for s in nsplits]
        for in_chunk in in_tensor.chunks:
            out_op = op.copy().reset_key()
            chunk_inputs = [in_chunk]
            if in_chunk.ndim == 1:
                i, = in_chunk.index
                column_stop = 1
                chunk_index = (in_chunk.index[0], 0)
                dtypes = out_df.dtypes
                columns_value = parse_index(out_df.columns_value.to_pandas()[0:1],
                                            store_data=True)
            else:
                i, j = in_chunk.index
                column_stop = cum_size[1][j]
                chunk_index = in_chunk.index
                dtypes = out_df.dtypes[column_stop - in_chunk.shape[1]:column_stop]
                pd_columns = out_df.columns_value.to_pandas()
                chunk_pd_columns = pd_columns[column_stop - in_chunk.shape[1]:column_stop]
                columns_value = parse_index(chunk_pd_columns, store_data=True)

            index_stop = cum_size[0][i]
            if out_df.index_value.has_value():
                pd_index = out_df.index_value.to_pandas()
                chunk_pd_index = pd_index[index_stop - in_chunk.shape[0]:index_stop]
                index_value = parse_index(chunk_pd_index, store_data=True)
            else:
                assert op.index is not None
                index_chunk = index_tensor.cix[in_chunk.index[0],]
                chunk_inputs.append(index_chunk)
                index_value = parse_index(pd.Index([], dtype=index_tensor.dtype),
                                          index_chunk, type(out_op).__name__)

            out_op.extra_params['index_stop'] = index_stop
            out_op.extra_params['column_stop'] = column_stop
            out_chunk = out_op.new_chunk(chunk_inputs, shape=in_chunk.shape,
                                         index=chunk_index, dtypes=dtypes,
                                         index_value=index_value,
                                         columns_value=columns_value)
            out_chunks.append(out_chunk)

        if in_tensor.ndim == 1:
            nsplits = in_tensor.nsplits + ((1,),)
        else:
            nsplits = in_tensor.nsplits

        new_op = op.copy()
        return new_op.new_dataframes(out_df.inputs, out_df.shape, dtypes=out_df.dtypes,
                                     index_value=out_df.index_value, columns_value=out_df.columns_value,
                                     chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]

        if op.from_1d_tensors:
            d = OrderedDict()
            tensors = [ctx[inp.key] for inp in op.inputs]
            if op.index is not None:
                tensors_data, index_data = tensors[:-1], tensors[-1]
            else:
                tensors_data = tensors
                index_data = chunk.index_value.to_pandas()
            for name, data_1d in zip(chunk.dtypes.index, tensors_data):
                d[name] = data_1d
            ctx[chunk.key] = pd.DataFrame(d, index=index_data,
                                          columns=chunk.columns_value.to_pandas())
        else:
            tensor_data = ctx[op.inputs[0].key]
            if op.index is not None:
                # index is a tensor
                index_data = ctx[op.inputs[1].key]
            else:
                index_data = chunk.index_value.to_pandas()
            ctx[chunk.key] = pd.DataFrame(tensor_data, index=index_data,
                                          columns=chunk.columns_value.to_pandas())


def dataframe_from_tensor(tensor, index=None, columns=None, gpu=None, sparse=False):
    if tensor.ndim > 2 or tensor.ndim <= 0:
        raise TypeError('Not support create DataFrame from {0} dims tensor', format(tensor.ndim))
    try:
        col_num = tensor.shape[1]
    except IndexError:
        col_num = 1
    gpu = tensor.op.gpu if gpu is None else gpu
    op = DataFrameFromTensor(dtypes=pd.Series([tensor.dtype] * col_num, index=columns),
                             gpu=gpu, sparse=sparse)
    return op(tensor, index, columns)


def dataframe_from_1d_tensors(tensors, index=None, columns=None, gpu=None, sparse=False):
    gpu = tensors[0].op.gpu if gpu is None else gpu
    dtypes = pd.Series([t.dtype for t in tensors], index=columns)
    op = DataFrameFromTensor(dtypes=dtypes, from_1d_tensors=True,
                             gpu=gpu, sparse=sparse)
    return op(tensors, index, columns)


class SeriesFromTensor(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.SERIES_FROM_TENSOR

    _input = KeyField('input')
    _index = KeyField('index')
    _dtype = DataTypeField('dtype')

    def __init__(self, index=None, dtype=None, gpu=None, sparse=None, **kw):
        super(SeriesFromTensor, self).__init__(
            _index=index, _dtype=dtype, _gpu=gpu, _sparse=sparse,
            _object_type=ObjectType.series, **kw)

    @property
    def index(self):
        return self._index

    @property
    def dtype(self):
        return self._dtype

    def _set_inputs(self, inputs):
        super(SeriesFromTensor, self)._set_inputs(inputs)
        self._input = self._inputs[0]
        if self._index is not None:
            self._index = self._inputs[-1]

    @classmethod
    def tile(cls, op):
        # check all inputs to make sure no unknown chunk shape
        check_chunks_unknown_shape(op.inputs, TilesError)

        out_series = op.outputs[0]
        in_tensor = op.inputs[0]
        nsplits = in_tensor.nsplits

        if op.index is not None:
            index_tensor = op.index.rechunk([nsplits[0]])._inplace_tile()
        else:
            index_tensor = None

        index_start = 0
        out_chunks = []
        series_index = out_series.index_value.to_pandas()
        for in_chunk in in_tensor.chunks:
            new_op = op.copy().reset_key()
            new_op.extra_params['index_start'] = index_start
            chunk_inputs = [in_chunk]
            if index_tensor is not None:
                index_chunk = index_tensor.cix[in_chunk.index]
                chunk_inputs.append(index_chunk)
                index_value = parse_index(pd.Index([], dtype=in_chunk.dtype), index_chunk, type(new_op).__name__)
            else:
                chunk_pd_index = series_index[index_start: index_start + in_chunk.shape[0]]
                index_value = parse_index(chunk_pd_index, store_data=True)
            index_start += in_chunk.shape[0]
            out_chunk = new_op.new_chunk(chunk_inputs, shape=in_chunk.shape, index=in_chunk.index,
                                         index_value=index_value, name=out_series.name)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, out_series.shape, dtype=out_series.dtype,
                                     index_value=out_series.index_value, name=out_series.name,
                                     chunks=out_chunks, nsplits=in_tensor.nsplits)

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        tensor_data = ctx[op.inputs[0].key]
        if op.index is not None:
            index_data = ctx[op.inputs[1].key]
        else:
            index_data = chunk.index_value.to_pandas()
        ctx[chunk.key] = pd.Series(tensor_data, index=index_data, name=chunk.name)

    def __call__(self, input_tensor, index, name):
        inputs = [input_tensor]
        if index is not None:
            if not isinstance(index, pd.Index):
                if isinstance(index, (Base, Entity)):
                    self._index = index
                    index = astensor(index)
                    if index.ndim != 1:
                        raise ValueError('index should be 1-d, got {}-d'.format(index.ndim))
                    index_value = parse_index(pd.Index([], dtype=index.dtype), index, type(self).__name__)
                    inputs.append(index)
                else:
                    index = pd.Index(index)
                    index_value = parse_index(index, store_data=True)
            else:
                index_value = parse_index(index, store_data=True)
        else:
            index_value = parse_index(pd.RangeIndex(start=0, stop=input_tensor.shape[0]))
        return self.new_series(inputs, shape=input_tensor.shape, dtype=self.dtype,
                               index_value=index_value, name=name)


def series_from_tensor(tensor, index=None, name=None, gpu=None, sparse=False):
    if tensor.ndim > 1 or tensor.ndim <= 0:
        raise TypeError('Not support create DataFrame from {0} dims tensor', format(tensor.ndim))
    gpu = tensor.op.gpu if gpu is None else gpu
    op = SeriesFromTensor(dtype=tensor.dtype, gpu=gpu, sparse=sparse)
    return op(tensor, index, name)

