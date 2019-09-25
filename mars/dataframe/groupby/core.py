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

import numpy as np
import pandas as pd

from ... import opcodes as OperandDef
from ...serialize import BoolField, Int32Field, AnyField
from ...compat import six
from ...utils import get_shuffle_input_keys_idxes
from ..utils import build_concated_rows_frame, hash_dataframe_on
from ..operands import DataFrameOperand, DataFrameOperandMixin, DataFrameShuffleMap, \
    DataFrameShuffleReduce, DataFrameShuffleProxy, ObjectType


class DataFrameGroupByMap(DataFrameShuffleMap, DataFrameOperandMixin):
    _op_type_ = OperandDef.GROUPBY_MAP

    _by = AnyField('by')
    _shuffle_size = Int32Field('shuffle_size')

    def __init__(self, by=None, shuffle_size=None, **kw):
        super(DataFrameGroupByMap, self).__init__(_by=by, _shuffle_size=shuffle_size,
                                                  _object_type=ObjectType.dataframe, **kw)

    @property
    def shuffle_size(self):
        return self._shuffle_size

    @property
    def by(self):
        return self._by

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        df = ctx[op.inputs[0].key]

        if isinstance(op.by, list):
            on = op.by
        else:
            on = None
        filters = hash_dataframe_on(df, on, op.shuffle_size)

        for index_idx, index_filter in enumerate(filters):
            group_key = ','.join([str(index_idx), str(chunk.index[1])])
            if index_filter is not None and index_filter is not list():
                ctx[(chunk.key, group_key)] = df.loc[index_filter]
            else:
                ctx[(chunk.key, group_key)] = None


class DataFrameGroupByReduce(DataFrameShuffleReduce, DataFrameOperandMixin):
    _op_type_ = OperandDef.GROUPBY_REDUCE

    def __init__(self, shuffle_key=None, **kw):
        super(DataFrameGroupByReduce, self).__init__(_shuffle_key=shuffle_key,
                                                     _object_type=ObjectType.dataframe, **kw)

    @classmethod
    def execute(cls, ctx, op):
        chunk = op.outputs[0]
        input_keys, input_idxes = get_shuffle_input_keys_idxes(op.inputs[0])
        input_idx_to_df = {idx: ctx[inp_key, ','.join(str(ix) for ix in chunk.index)]
                           for inp_key, idx in zip(input_keys, input_idxes)}
        row_idxes = sorted({idx[0] for idx in input_idx_to_df})

        res = []
        for row_idx in row_idxes:
            row_df = input_idx_to_df.get((row_idx, 0), None)
            if row_df is not None:
                res.append(row_df)
        r = pd.concat(res, axis=0)
        if chunk.index_value is not None:
            r.index.name = chunk.index_value.value._name
        ctx[chunk.key] = r


class DataFrameGroupByOperand(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.GROUPBY

    _by = AnyField('by')
    _as_index = BoolField('as_index')

    def __init__(self, by=None, as_index=None, object_type=ObjectType.groupby, **kw):
        super(DataFrameGroupByOperand, self).__init__(_by=by, _as_index=as_index,
                                                      _object_type=object_type, **kw)

    @property
    def by(self):
        return self._by

    @property
    def as_index(self):
        return self._as_index

    def __call__(self, df):
        return self.new_tileable([df])

    @classmethod
    def tile(cls, op):
        in_df = build_concated_rows_frame(op.inputs[0])

        # generate map chunks
        map_chunks = []
        chunk_shape = (in_df.chunk_shape[0], 1)
        for chunk in in_df.chunks:
            map_op = DataFrameGroupByMap(by=op.by, shuffle_size=chunk_shape[0])
            map_chunks.append(map_op.new_chunk([chunk], shape=(np.nan, np.nan), index=chunk.index))

        proxy_chunk = DataFrameShuffleProxy(object_type=ObjectType.dataframe).new_chunk(map_chunks, shape=())

        # generate reduce chunks
        reduce_chunks = []
        for out_idx in itertools.product(*(range(s) for s in chunk_shape)):
            reduce_op = DataFrameGroupByReduce(shuffle_key=','.join(str(idx) for idx in out_idx))
            reduce_chunks.append(
                reduce_op.new_chunk([proxy_chunk], shape=(np.nan, np.nan), index=out_idx))

        # generate groupby chunks
        out_chunks = []
        for chunk in reduce_chunks:
            groupby_op = op.copy().reset_key()
            groupby_op._object_type = ObjectType.dataframe
            out_chunks.append(groupby_op.new_chunk([chunk], shape=(np.nan, chunk.shape[1]),
                                                   index=chunk.index))

        new_op = op.copy()
        return new_op.new_tileables(op.inputs, chunks=out_chunks)

    @classmethod
    def execute(cls, ctx, op):
        df = ctx[op.inputs[0].key]
        ctx[op.outputs[0].key] = list(df.groupby(op.by))


def dataframe_groupby(df, by, as_index=True):
    if isinstance(by, six.string_types):
        by = [by]
    op = DataFrameGroupByOperand(by=by, as_index=as_index)
    return op(df)
