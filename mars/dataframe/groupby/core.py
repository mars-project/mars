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
import pandas as pd

from ... import opcodes as OperandDef
from ...lib.groupby_wrapper import wrapped_groupby
from ...operands import OperandStage
from ...serialize import BoolField, Int32Field, AnyField
from ...utils import get_shuffle_input_keys_idxes
from ..utils import build_concatenated_rows_frame, hash_dataframe_on, \
    build_empty_df, build_empty_series, parse_index
from ..operands import DataFrameOperandMixin, \
    DataFrameMapReduceOperand, DataFrameShuffleProxy, ObjectType


class DataFrameGroupByOperand(DataFrameMapReduceOperand, DataFrameOperandMixin):
    _op_type_ = OperandDef.GROUPBY

    _by = AnyField('by')
    _as_index = BoolField('as_index')
    _sort = BoolField('sort')

    _shuffle_size = Int32Field('shuffle_size')

    def __init__(self, by=None, as_index=None, sort=None, shuffle_size=None,
                 stage=None, shuffle_key=None, object_type=None, **kw):
        super().__init__(_by=by, _as_index=as_index, _sort=sort,
                         _shuffle_size=shuffle_size, _stage=stage, _shuffle_key=shuffle_key,
                         _object_type=object_type, **kw)
        if stage in (OperandStage.map, OperandStage.reduce):
            if object_type in (ObjectType.dataframe, ObjectType.dataframe_groupby):
                object_type = ObjectType.dataframe
            else:
                object_type = ObjectType.series
        else:
            if object_type in (ObjectType.dataframe, ObjectType.dataframe_groupby):
                object_type = ObjectType.dataframe_groupby
            elif object_type == ObjectType.series:
                object_type = ObjectType.series_groupby
        self._object_type = object_type

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
    def shuffle_size(self):
        return self._shuffle_size

    @property
    def is_dataframe_obj(self):
        return self._object_type in (ObjectType.dataframe_groupby, ObjectType.dataframe)

    @property
    def groupby_params(self):
        return dict(by=self.by, as_index=self.as_index, sort=self.sort)

    def build_mock_groupby(self, **kwargs):
        in_df = self.inputs[0]
        if self.is_dataframe_obj:
            empty_df = build_empty_df(in_df.dtypes, index=pd.RangeIndex(2))
            obj_dtypes = in_df.dtypes[in_df.dtypes == np.dtype('O')]
            empty_df[obj_dtypes.index] = 'O'
        else:
            if in_df.dtype == np.dtype('O'):
                empty_df = pd.Series('O', index=pd.RangeIndex(2), name=in_df.name, dtype=np.dtype('O'))
            else:
                empty_df = build_empty_series(in_df.dtype, index=pd.RangeIndex(2), name=in_df.name)

        new_kw = dict(by=self.by, as_index=self.as_index, sort=self.sort)
        new_kw.update(kwargs)
        return empty_df.groupby(**new_kw)

    def __call__(self, df):
        params = df.params.copy()
        params['index_value'] = parse_index(None, df.key, df.index_value.key)
        if df.ndim == 2:
            if isinstance(self.by, str):
                params['key_columns'] = [self.by]
            elif isinstance(self.by, list):
                params['key_columns'] = self.by
        return self.new_tileable([df], **params)

    @classmethod
    def tile(cls, op):
        is_dataframe_obj = op.is_dataframe_obj
        if is_dataframe_obj:
            in_df = build_concatenated_rows_frame(op.inputs[0])
            out_object_type = ObjectType.dataframe
        else:
            in_df = op.inputs[0]
            out_object_type = ObjectType.series

        # generate map chunks
        map_chunks = []
        chunk_shape = (in_df.chunk_shape[0], 1)
        for chunk in in_df.chunks:
            map_op = DataFrameGroupByOperand(stage=OperandStage.map, by=op.by,
                                             shuffle_size=chunk_shape[0], object_type=op.object_type)
            map_chunks.append(map_op.new_chunk([chunk], shape=(np.nan, np.nan), index=chunk.index))

        proxy_chunk = DataFrameShuffleProxy(object_type=out_object_type).new_chunk(map_chunks, shape=())

        # generate reduce chunks
        reduce_chunks = []
        for out_idx in itertools.product(*(range(s) for s in chunk_shape)):
            shuffle_key = ','.join(str(idx) for idx in out_idx)
            reduce_op = DataFrameGroupByOperand(stage=OperandStage.reduce, shuffle_key=shuffle_key,
                                                object_type=op.object_type)
            reduce_chunks.append(
                reduce_op.new_chunk([proxy_chunk], shape=(np.nan, np.nan), index=out_idx))

        # generate groupby chunks
        out_chunks = []
        for chunk in reduce_chunks:
            groupby_op = op.copy().reset_key()
            if is_dataframe_obj:
                new_shape = (np.nan, chunk.shape[1])
            else:
                new_shape = (np.nan,)
            params = dict(shape=new_shape, index=chunk.index)
            if op.is_dataframe_obj:
                params.update(dict(dtypes=in_df.dtypes, columns_value=in_df.columns_value,
                                   index_value=parse_index(None, chunk.key, proxy_chunk.key)))
            else:
                params.update(dict(name=in_df.name, dtype=in_df.dtype,
                                   index_value=parse_index(None, chunk.key, proxy_chunk.key)))
            out_chunks.append(groupby_op.new_chunk([chunk], **params))

        new_op = op.copy()
        params = op.outputs[0].params.copy()
        if is_dataframe_obj:
            params['nsplits'] = ((np.nan,) * len(out_chunks), (in_df.shape[1],))
        else:
            params['nsplits'] = ((np.nan,) * len(out_chunks),)
        params['chunks'] = out_chunks
        return new_op.new_tileables(op.inputs, **params)

    @classmethod
    def execute_map(cls, ctx, op):
        is_dataframe_obj = op.is_dataframe_obj
        by = op.by
        chunk = op.outputs[0]
        df = ctx[op.inputs[0].key]

        if isinstance(by, list) or callable(by):
            on = by
        else:
            on = None
        filters = hash_dataframe_on(df, on, op.shuffle_size)

        for index_idx, index_filter in enumerate(filters):
            if is_dataframe_obj:
                group_key = ','.join([str(index_idx), str(chunk.index[1])])
            else:
                group_key = str(index_idx) + ',0'
            if index_filter is not None and index_filter is not list():
                ctx[(chunk.key, group_key)] = df.loc[index_filter]
            else:
                ctx[(chunk.key, group_key)] = None

    @classmethod
    def execute_reduce(cls, ctx, op):
        is_dataframe_obj = op.inputs[0].op.object_type == ObjectType.dataframe
        chunk = op.outputs[0]
        input_keys, input_idxes = get_shuffle_input_keys_idxes(op.inputs[0])
        input_idx_to_df = {idx: ctx[inp_key, ','.join(str(ix) for ix in chunk.index)]
                           for inp_key, idx in zip(input_keys, input_idxes)}
        row_idxes = sorted({idx[0] for idx in input_idx_to_df})

        res = []
        for row_idx in row_idxes:
            if is_dataframe_obj:
                row_df = input_idx_to_df.get((row_idx, 0), None)
            else:
                row_df = input_idx_to_df.get((row_idx,), None)
            if row_df is not None:
                res.append(row_df)
        r = pd.concat(res, axis=0)
        if chunk.index_value is not None:
            r.index.name = chunk.index_value.value._name
        ctx[chunk.key] = r

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls.execute_map(ctx, op)
        elif op.stage == OperandStage.reduce:
            cls.execute_reduce(ctx, op)
        else:
            df = ctx[op.inputs[0].key]
            ctx[op.outputs[0].key] = wrapped_groupby(df, op.by)


def groupby(df, by, as_index=True, sort=True):
    if not as_index and df.op.object_type == ObjectType.series:
        raise TypeError('as_index=False only valid with DataFrame')

    if isinstance(by, str):
        by = [by]
    op = DataFrameGroupByOperand(by=by, as_index=as_index, sort=sort,
                                 object_type=df.op.object_type)
    return op(df)
