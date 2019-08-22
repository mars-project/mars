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

import itertools
from numbers import Integral

import numpy as np
import pandas as pd
from pandas.core.dtypes.cast import find_common_type

from ...tensor.core import TENSOR_TYPE
from ...tensor.datasource.empty import empty
from ...tensor.indexing.core import calc_shape, process_index
from ...serialize import AnyField, ListField
from ... import opcodes as OperandDef
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import indexing_index_value


class DataFrameIloc(object):
    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, indexes):
        op = DataFrameIlocGetItem(indexes=process_index(self._obj.ndim, indexes), object_type=ObjectType.dataframe)
        return op(self._obj)

    def __setitem__(self, indexes, value):
        if not np.isscalar(value):
            raise NotImplementedError('Only scalar value is supported to set by iloc')

        op = DataFrameIlocSetItem(indexes=process_index(self._obj.ndim, indexes), value=value, object_type=ObjectType.dataframe)
        ret = op(self._obj)
        self._obj.data = ret.data


class DataFrameIlocGetItem(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_ILOC_GETITEM

    _indexes = ListField('indexes')

    def __init__(self, indexes=None, gpu=False, sparse=False, object_type=None, **kw):
        super(DataFrameIlocGetItem, self).__init__(_indexes=indexes,
                                                   _gpu=gpu, _sparse=sparse,
                                                   _object_type=object_type, **kw)

    @property
    def indexes(self):
        return self._indexes

    def __call__(self, df):
        # Note [Fancy Index of Numpy and Pandas]
        #
        # The numpy and pandas.iloc have different semantic when processing fancy index:
        #
        # >>> np.ones((3,3))[[1,2],[1,2]]
        # array([1., 1.])
        #
        # >>> pd.DataFrame(np.ones((3,3))).iloc[[1,2],[1,2]]
        #    1    2
        # 1  1.0  1.0
        # 2  1.0  1.0
        #
        # Thus, we processing the index along two axis of DataFrame seperately.

        if isinstance(self.indexes[0], TENSOR_TYPE) or isinstance(self.indexes[1], TENSOR_TYPE):
            raise NotImplementedError('The index value cannot be unexecuted mars tensor')

        shape0 = tuple(calc_shape((df.shape[0],), (self.indexes[0],)))
        shape1 = tuple(calc_shape((df.shape[1],), (self.indexes[1],)))

        # NB: pandas only compresses the result to series when index on one of axis is integral
        if isinstance(self.indexes[1], Integral):
            shape = shape0
            dtype = df.dtypes.iloc[self.indexes[1]]
            index_value = indexing_index_value(df.index_value, self.indexes[0])
            self._object_type = ObjectType.series
            return self.new_series([df], shape=shape, dtype=dtype, index_value=index_value)
        elif isinstance(self.indexes[0], Integral):
            shape = shape1
            dtype = find_common_type(df.dtypes.iloc[self.indexes[1]].values)
            index_value = indexing_index_value(df.columns, self.indexes[1])
            self._object_type = ObjectType.series
            return self.new_series([df], shape=shape, dtype=dtype, index_value=index_value)
        else:
            return self.new_dataframe([df], shape=shape0 + shape1, dtypes=df.dtypes.iloc[self.indexes[1]],
                                      index_value=indexing_index_value(df.index_value, self.indexes[0]),
                                      columns_value=indexing_index_value(df.columns, self.indexes[1], store_data=True))

    # FIXME The view behavior of DataFrame.iloc
    #
    # The pandas's iloc has complicated behavior about whether to create a view or not, it depends
    # on the further operation on the view, as illustrated by the following example:
    #
    # >>> df = pd.DataFrame([[1,2], [3,4]])
    # >>> x = df.iloc[:]
    # >>> df
    #    0  1
    # 0  1  2
    # 1  3  4
    # >>> x
    #    0  1
    # 0  1  2
    # 1  3  4
    #
    # >>> x.iloc[:] = 1000
    # >>> x
    #       0     1
    # 0  1000  1000
    # 1  1000  1000
    # df
    #       0     1
    # 0  1000  1000
    # 1  1000  1000
    #
    # >>> x.iloc[:] = 2000.0
    # >>> x
    #         0       1
    # 0  2000.0  2000.0
    # 1  2000.0  2000.0
    # >>> df
    #       0     1
    # 0  1000  1000
    # 1  1000  1000

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        out_val = op.outputs[0]

        # See Note [Fancy Index of Numpy and Pandas]
        tensor0 = empty(in_df.shape[0], chunk_size=(in_df.nsplits[0],))[op.indexes[0]].tiles()
        tensor1 = empty(in_df.shape[1], chunk_size=(in_df.nsplits[1],))[op.indexes[1]].tiles()

        integral_index_on_index = isinstance(op.indexes[0], Integral)
        integral_index_on_column = isinstance(op.indexes[1], Integral)

        out_chunks = []
        for index_chunk, column_chunk in itertools.product(tensor0.chunks, tensor1.chunks):
            in_chunk = in_df.cix[index_chunk.inputs[0].index + column_chunk.inputs[0].index]

            chunk_op = op.copy().reset_key()
            chunk_op._indexes = (index_chunk.op.indexes[0], column_chunk.op.indexes[0])

            if integral_index_on_column:
                shape = index_chunk.shape
                index = index_chunk.index
                index_value = indexing_index_value(in_chunk.index_value, index_chunk.op.indexes[0])
                out_chunk = chunk_op.new_chunk([in_chunk], shape=shape, index=index,
                                               dtype=out_val.dtype, index_value=index_value)
            elif integral_index_on_index:
                shape = column_chunk.shape
                index = column_chunk.index
                index_value = indexing_index_value(in_chunk.columns, column_chunk.op.indexes[0])
                out_chunk = chunk_op.new_chunk([in_chunk], shape=shape, index=index,
                                               dtype=out_val.dtype, index_value=index_value)
            else:
                index_value = indexing_index_value(in_chunk.index_value, index_chunk.op.indexes[0])
                columns_value = indexing_index_value(in_chunk.columns, column_chunk.op.indexes[0], store_data=True)
                dtypes = in_chunk.dtypes.iloc[column_chunk.op.indexes[0]]
                out_chunk = chunk_op.new_chunk([in_chunk],
                                               shape=index_chunk.shape + column_chunk.shape,
                                               index=index_chunk.index + column_chunk.index,
                                               dtypes=dtypes, index_value=index_value, columns_value=columns_value)
            out_chunks.append(out_chunk)

        new_op = op.copy()
        if integral_index_on_column or integral_index_on_index:
            if integral_index_on_column:
                nsplits = tensor0.nsplits
            else:
                nsplits = tensor1.nsplits
            return new_op.new_seriess(op.inputs, out_val.shape, dtype=out_val.dtype,
                                      index_value=out_val.index_value, chunks=out_chunks, nsplits=nsplits)
        else:
            nsplits = tensor0.nsplits + tensor1.nsplits
            return new_op.new_dataframes(op.inputs, out_val.shape, dtypes=out_val.dtypes,
                                        index_value=out_val.index_value,
                                        columns_value=out_val.columns, chunks=out_chunks, nsplits=nsplits)

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        r = ctx[op.inputs[0].key].iloc[op.indexes]
        if isinstance(r, pd.Series) and r.dtype != chunk.dtype:
            r = r.astype(chunk.dtype)
        ctx[chunk.key] = r


class DataFrameIlocSetItem(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.DATAFRAME_ILOC_SETITEM

    _indexes = ListField('indexes')
    _value = AnyField('value')

    def __init__(self, indexes=None, value=None, gpu=False, sparse=False, object_type=None, **kw):
        super(DataFrameIlocSetItem, self).__init__(_indexes=indexes, _value=value,
                                                   _gpu=gpu, _sparse=sparse,
                                                   _object_type=object_type, **kw)

    @property
    def indexes(self):
        return self._indexes

    @property
    def value(self):
        return self._value

    def __call__(self, df):
        if isinstance(self.indexes[0], TENSOR_TYPE) or isinstance(self.indexes[1], TENSOR_TYPE):
            raise NotImplementedError('The index value cannot be unexecuted mars tensor')
        return self.new_dataframe([df], shape=df.shape, dtypes=df.dtypes,
                                  index_value=df.index_value, columns_value=df.columns)

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        # See Note [Fancy Index of Numpy and Pandas]
        tensor0 = empty(in_df.shape[0], chunk_size=(in_df.nsplits[0],))[op.indexes[0]].tiles()
        tensor1 = empty(in_df.shape[1], chunk_size=(in_df.nsplits[1],))[op.indexes[1]].tiles()

        chunk_mapping = {c0.inputs[0].index + c1.inputs[0].index: (c0, c1)
                         for c0, c1 in itertools.product(tensor0.chunks, tensor1.chunks)}

        out_chunks = []
        for chunk in in_df.chunks:
            if chunk.index not in chunk_mapping:
                out_chunks.append(chunk)
            else:
                chunk_op = op.copy().reset_key()
                index_chunk, column_chunk = chunk_mapping[chunk.index]
                chunk_op._indexes = (index_chunk.op.indexes[0], column_chunk.op.indexes[0])
                chunk_op._value = op.value
                out_chunk = chunk_op.new_chunk([chunk],
                                                shape=chunk.shape, index=chunk.index, dtypes=chunk.dtypes,
                                                index_value=chunk.index_value, columns_value=chunk.columns)
                out_chunks.append(out_chunk)

        new_op = op.copy()
        return new_op.new_dataframes(op.inputs, shape=out_df.shape, dtypes=out_df.dtypes,
                                     index_value=out_df.index_value, columns_value=out_df.columns,
                                     chunks=out_chunks, nsplits=in_df.nsplits)

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        r = ctx[op.inputs[0].key].copy(deep=True)
        r.iloc[op.indexes] = op.value
        ctx[chunk.key] = r


def iloc(df):
    return DataFrameIloc(df)
