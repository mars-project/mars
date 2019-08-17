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
from pandas.core.dtypes.cast import find_common_type

from ...serialize import AnyField, ListField
from ... import opcodes as OperandDef
from ..operands import DataFrameOperand, DataFrameOperandMixin, ObjectType
from ..utils import indexing_index_value


class DataFrameIloc(object):
    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, indexes):
        op = DataFrameIlocGetItem(indexes=_extend_indexes(indexes), object_type=ObjectType.dataframe)
        return op(self._obj)

    def __setitem__(self, indexes, value):
        if not np.isscalar(value):
            raise NotImplementedError('Only scalar value is supported to set by iloc')

        op = DataFrameIlocSetItem(indexes=_extend_indexes(indexes), value=value, object_type=ObjectType.dataframe)
        ret = op(self._obj)
        self._obj.data = ret.data


def _extend_indexes(indexes):
    # Extend the indexes to a 2-D indexes
    if isinstance(indexes, tuple):
        if len(indexes) == 1:
            return indexes + (slice(None, None, None),)
        else:
            return indexes
    else:
        return (indexes, slice(None, None, None))


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
        from ...tensor.indexing.getitem import _getitem
        from ...tensor.datasource.empty import empty

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

        tensor0 = _getitem(empty(df.shape[0]), self.indexes[0])
        tensor1 = _getitem(empty(df.shape[1]), self.indexes[1])

        if isinstance(self.indexes[0], Integral) or isinstance(self.indexes[1], Integral):
            # NB: pandas only compresses the result to series when index on one of axis is integral
            if isinstance(self.indexes[1], Integral):
                shape = tensor0.shape
                dtype = df.dtypes.iloc[self.indexes[1]]
                index_value = indexing_index_value(df.index_value, self.indexes[0])
            else:
                shape = tensor1.shape
                dtype = find_common_type(df.dtypes.iloc[self.indexes[1]].values)
                index_value = indexing_index_value(df.columns, self.indexes[1])
            self._object_type = ObjectType.series
            return self.new_series([df], shape=shape, dtype=dtype, index_value=index_value)
        else:
            return self.new_dataframe([df], shape=tensor0.shape + tensor1.shape, dtypes=df.dtypes.iloc[self.indexes[1]],
                                      index_value=indexing_index_value(df.index_value, self.indexes[0]),
                                      columns_value=indexing_index_value(df.columns, self.indexes[1], store_data=True))

    def on_output_modify(self, new_output):
        a = self.inputs[0]
        op = DataFrameIlocSetItem(indexes=self.indexes, value=new_output,
                                  dtypes=a.dtypes, gpu=a.gpu, sparse=a.issparse(), object_type=self.object_type)
        return op(a, self._indexes, new_output)

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        out_val = op.outputs[0]

        from ...tensor.indexing.getitem import TensorIndex, _getitem
        from ...tensor.datasource.empty import empty

        # See Note [Fancy Index of Numpy and Pandas]
        tensor0 = _getitem(empty(in_df.shape[0], chunk_size=(in_df.nsplits[0],)).tiles(), op.indexes[0]).single_tiles()
        tensor1 = _getitem(empty(in_df.shape[1], chunk_size=(in_df.nsplits[1],)).tiles(), op.indexes[1]).single_tiles()

        out_chunks = []
        for index_chunk, column_chunk in itertools.product(tensor0.chunks, tensor1.chunks):
            # The indexes itself cannot be a Tensor of mars, that requires more complicate logic
            if not isinstance(index_chunk.op, TensorIndex) or not isinstance(column_chunk.op, TensorIndex):
                raise NotImplementedError('The index value cannot be unexecuted mars tensor')

            in_chunk = in_df.cix[index_chunk.inputs[0].index + column_chunk.inputs[0].index]

            chunk_op = op.copy().reset_key()
            chunk_op._indexes = (index_chunk.op.indexes[0], column_chunk.op.indexes[0])

            if isinstance(op.indexes[0], Integral) or isinstance(op.indexes[1], Integral):
                # NB: pandas only compresses the result to series when index on one of axis is integral
                if isinstance(op.indexes[1], Integral):
                    shape = index_chunk.shape
                    index = index_chunk.index
                    dtype = in_chunk.dtypes.iloc[column_chunk.op.indexes[0]]
                    index_value = indexing_index_value(in_chunk.index_value, index_chunk.op.indexes[0])
                else:
                    shape = column_chunk.shape
                    index = column_chunk.index
                    dtype = find_common_type(in_chunk.dtypes.iloc[column_chunk.op.indexes[0]])
                    index_value = indexing_index_value(in_chunk.columns, column_chunk.op.indexes[0])
                out_chunk = chunk_op.new_chunk([in_chunk], shape=shape, index=index,
                                               dtype=dtype, index_value=index_value)
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
        if isinstance(op.indexes[0], Integral) or isinstance(op.indexes[1], Integral):
            if isinstance(op.indexes[1], Integral):
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
        ctx[chunk.key] = ctx[op.inputs[0].key].iloc[op.indexes]


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
        return self.new_dataframe([df], shape=df.shape, dtypes=df.dtypes,
                                  index_value=df.index_value, columns_value=df.columns)

    @classmethod
    def tile(cls, op):
        in_df = op.inputs[0]
        out_df = op.outputs[0]

        from ...tensor.indexing.getitem import TensorIndex, _getitem
        from ...tensor.datasource.empty import empty

        # See Note [Fancy Index of Numpy and Pandas]
        tensor0 = _getitem(empty(in_df.shape[0], chunk_size=(in_df.nsplits[0],)).tiles(), op.indexes[0]).single_tiles()
        tensor1 = _getitem(empty(in_df.shape[1], chunk_size=(in_df.nsplits[1],)).tiles(), op.indexes[1]).single_tiles()

        chunk_mapping = {c0.inputs[0].index + c1.inputs[0].index: (c0, c1)
                         for c0, c1 in itertools.product(tensor0.chunks, tensor1.chunks)}

        out_chunks = []
        for chunk in in_df.chunks:
            if chunk.index not in chunk_mapping:
                out_chunks.append(chunk)
            else:
                chunk_op = op.copy().reset_key()
                index_chunk, column_chunk = chunk_mapping[chunk.index]
                if not isinstance(index_chunk.op, TensorIndex) or not isinstance(column_chunk.op, TensorIndex):
                    raise NotImplementedError('The index value cannot be unexecuted mars tensor')
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
