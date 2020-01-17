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

import itertools

import numpy as np

from ... import opcodes as OperandDef
from ...config import options
from ...operands import OperandStage
from ...serialize import BoolField, AnyField, StringField
from ...compat import six
from ..merge import DataFrameConcat
from ..operands import DataFrameOperand, DataFrameOperandMixin, DataFrameShuffleProxy, ObjectType
from ..core import GROUPBY_TYPE
from ..utils import build_empty_df, parse_index, build_concated_rows_frame
from .core import DataFrameGroupByOperand


class DataFrameGroupByAgg(DataFrameOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.GROUPBY_AGG

    _func = AnyField('func')
    _by = AnyField('by')
    _as_index = BoolField('as_index')
    _sort = BoolField('sort')
    _method = StringField('method')

    def __init__(self, func=None, by=None, as_index=None, sort=None, method=None, stage=None, **kw):
        super(DataFrameGroupByAgg, self).__init__(_func=func, _by=by, _as_index=as_index, _sort=sort, _method=method,
                                                  _stage=stage, _object_type=ObjectType.dataframe, **kw)

    @property
    def func(self):
        return self._func

    @property
    def by(self):
        return self._by

    @property
    def as_index(self):
        return self._as_index

    @property
    def sort(self):
        return self._sort

    @property
    def method(self):
        return self._method

    @property
    def stage(self):
        return self._stage

    def __call__(self, df):
        empty_df = build_empty_df(df.dtypes)
        agg_df = empty_df.groupby(self.by).agg(self.func)
        shape = (np.nan, agg_df.shape[1])
        index_value = parse_index(agg_df.index, df.key)
        index_value.value.should_be_monotonic = True
        return self.new_dataframe([df], shape=shape, dtypes=agg_df.dtypes,
                                  index_value=index_value,
                                  columns_value=parse_index(agg_df.columns, self))

    @classmethod
    def _gen_shuffle_chunks(cls, op, in_df, chunks):
        # generate map chunks
        map_chunks = []
        chunk_shape = (in_df.chunk_shape[0], 1)
        for chunk in chunks:
            if op.as_index:
                map_op = DataFrameGroupByOperand(stage=OperandStage.map, shuffle_size=chunk_shape[0])
            else:
                map_op = DataFrameGroupByOperand(stage=OperandStage.map, by=op.by, shuffle_size=chunk_shape[0])
            map_chunks.append(map_op.new_chunk([chunk], shape=(np.nan, np.nan), index=chunk.index,
                                               index_value=op.outputs[0].index_value))

        proxy_chunk = DataFrameShuffleProxy(object_type=ObjectType.dataframe).new_chunk(map_chunks, shape=())

        # generate reduce chunks
        reduce_chunks = []
        for out_idx in itertools.product(*(range(s) for s in chunk_shape)):
            reduce_op = DataFrameGroupByOperand(
                stage=OperandStage.reduce, shuffle_key=','.join(str(idx) for idx in out_idx))
            reduce_chunks.append(
                reduce_op.new_chunk([proxy_chunk], shape=(np.nan, np.nan), index=out_idx,
                                    index_value=op.outputs[0].index_value))

        return reduce_chunks

    @classmethod
    def _gen_map_chunks(cls, op, in_df, out_df):
        agg_chunks = []
        for chunk in in_df.chunks:
            agg_op = op.copy().reset_key()
            agg_op._stage = OperandStage.map
            agg_chunk = agg_op.new_chunk([chunk], shape=out_df.shape, index=chunk.index,
                                         index_value=out_df.index_value,
                                         columns_value=out_df.columns_value)
            agg_chunks.append(agg_chunk)
        return agg_chunks

    @classmethod
    def _tile_with_shuffle(cls, op):
        in_df = build_concated_rows_frame(op.inputs[0])
        out_df = op.outputs[0]

        # First, perform groupby and aggregation on each chunk.
        agg_chunks = cls._gen_map_chunks(op, in_df, out_df)

        # Shuffle the aggregation chunk.
        reduce_chunks = cls._gen_shuffle_chunks(op, in_df, agg_chunks)

        # Combine groups
        combine_chunks = []
        for chunk in reduce_chunks:
            combine_op = op.copy().reset_key()
            combine_op._stage = OperandStage.combine
            combine_chunk = combine_op.new_chunk([chunk], shape=out_df.shape, index=chunk.index,
                                                 index_value=out_df.index_value,
                                                 columns_value=out_df.columns_value)
            combine_chunks.append(combine_chunk)

        new_op = op.copy()
        return new_op.new_dataframes([in_df], shape=out_df.shape, index_value=out_df.index_value,
                                     columns_value=out_df.columns_value, chunks=combine_chunks,
                                     nsplits=((np.nan,) * len(combine_chunks), (out_df.shape[1],)))

    @classmethod
    def _tile_with_tree(cls, op):
        in_df = build_concated_rows_frame(op.inputs[0])
        out_df = op.outputs[0]

        combine_size = options.combine_size
        chunks = cls._gen_map_chunks(op, in_df, out_df)
        while len(chunks) > combine_size:
            new_chunks = []
            for idx, i in enumerate(range(0, len(chunks), combine_size)):
                chks = chunks[i: i + combine_size]
                if len(chks) == 1:
                    chk = chks[0]
                else:
                    concat_op = DataFrameConcat(object_type=ObjectType.dataframe)
                    # Change index for concatenate
                    for j, c in enumerate(chks):
                        c._index = (j, 0)
                    chk = concat_op.new_chunk(chks, dtypes=chks[0].dtypes)
                chunk_op = op.copy().reset_key()
                chunk_op._stage = OperandStage.combine
                new_chunks.append(chunk_op.new_chunk([chk], index=(idx, 0), shape=(np.nan, out_df.shape[1]),
                                                     index_value=chks[0].index_value,
                                                     columns_value=chks[0].columns_value,
                                                     dtypes=chks[0].dtypes))
            chunks = new_chunks

        concat_op = DataFrameConcat(object_type=ObjectType.dataframe)
        chk = concat_op.new_chunk(chunks, dtypes=chunks[0].dtypes)
        chunk_op = op.copy().reset_key()
        chunk_op._stage = OperandStage.combine
        chunk = chunk_op.new_chunk([chk], index=(0, 0), shape=out_df.shape, index_value=out_df.index_value,
                                   columns_value=out_df.columns_value, dtypes=out_df.dtypes)
        new_op = op.copy()
        nsplits = ((out_df.shape[0],), (out_df.shape[1],))
        return new_op.new_tileables(op.inputs, chunks=[chunk], nsplits=nsplits,
                                    dtypes=out_df.dtypes, shape=out_df.shape,
                                    index_value=out_df.index_value,
                                    columns_value=out_df.columns_value)

    @classmethod
    def tile(cls, op):
        if op.method == 'shuffle':
            return cls._tile_with_shuffle(op)
        elif op.method == 'tree':
            return cls._tile_with_tree(op)
        else:
            raise NotImplementedError

    @classmethod
    def execute(cls, ctx, op):
        df = ctx[op.inputs[0].key]
        if op.stage == OperandStage.map:
            ret = cls._execute_map(df, op)
        else:
            ret = cls._execute_combine(df, op)
        ctx[op.outputs[0].key] = ret

    @classmethod
    def _execute_map(cls, df, op):
        if isinstance(op.func, (six.string_types, dict)):
            return df.groupby(op.by, as_index=op.as_index, sort=False).agg(op.func)
        else:
            raise NotImplementedError

    @classmethod
    def _execute_combine(cls, df, op):
        if isinstance(op.func, (six.string_types, dict)):
            return df.groupby(level=0, as_index=op.as_index, sort=op.sort).agg(op.func)
        else:
            raise NotImplementedError


def agg(groupby, func, method='tree'):
    """
    Aggregate using one or more operations on grouped data.
    :param groupby: Groupby data.
    :param func: Aggregation functions.
    :param method: 'shuffle' or 'tree', 'tree' method provide a better performance, 'shuffle' is recommended
    if aggregated result is very large.
    :return: Aggregated result.
    """

    # When perform a computation on the grouped data, we won't shuffle
    # the data in the stage of groupby and do shuffle after aggregation.
    if not isinstance(groupby, GROUPBY_TYPE):
        raise TypeError('Input should be type of groupby, not %s' % type(groupby))
    elif isinstance(func, list):
        raise NotImplementedError('Function list is not supported now.')

    if method not in ['shuffle', 'tree']:
        raise NotImplementedError('Method %s has not been implemented' % method)

    if isinstance(func, six.string_types):
        funcs = [func]
    elif isinstance(func, dict):
        funcs = func.values()
    else:
        raise NotImplementedError('Type %s is not support' % type(func))
    for f in funcs:
        if f not in ['sum', 'prod', 'min', 'max']:
            raise NotImplementedError('Aggregation function %s has not been supported' % f)

    in_df = groupby.inputs[0]
    agg_op = DataFrameGroupByAgg(func=func, by=groupby.op.by, method=method,
                                 as_index=groupby.op.as_index, sort=groupby.op.sort)
    return agg_op(in_df)
